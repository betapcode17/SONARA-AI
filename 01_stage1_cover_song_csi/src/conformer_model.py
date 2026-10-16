import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class Swish(nn.Module):
    """Hàm kích hoạt Swish / SiLU chuẩn cho Conformer."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class FeedForwardModule(nn.Module):
    """Khối Feed-Forward bán bước (Macaron-style Half-Step FFN)."""
    def __init__(self, d_model: int, d_ff: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.swish = Swish()
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        residual = x
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.swish(x)
        x = self.dropout1(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        return residual + 0.5 * x


class ConformerConvModule(nn.Module):
    """Khối Tích chập 1D theo chiều sâu (Depthwise Separable Convolution) của Conformer."""
    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size phải là số lẻ để padding cân bằng!"
        self.layer_norm = nn.LayerNorm(d_model)
        # Pointwise Conv 1 (Tăng gấp đôi kênh cho Gated Linear Unit - GLU)
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        # Depthwise Conv (Quét dọc theo trục thời gian)
        self.depthwise_conv = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            stride=1, padding=(kernel_size - 1) // 2, groups=d_model
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.swish = Swish()
        # Pointwise Conv 2
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        residual = x
        x = self.layer_norm(x)
        # Chuyển về (B, C, T) để áp dụng Conv1D
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        # GLU (Gated Linear Unit)
        x = F.glu(x, dim=1)
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = self.swish(x)
        x = self.pointwise_conv2(x)
        x = self.dropout(x)
        # Chuyển lại về (B, T, C)
        x = x.transpose(1, 2)
        return residual + x


class ConformerBlock(nn.Module):
    """Một khối Conformer tiêu chuẩn gồm: FFN1 (1/2) -> MultiHeadAttention -> Conv -> FFN2 (1/2) -> LayerNorm."""
    def __init__(self, d_model: int = 256, n_heads: int = 4, d_ff: int = 1024, conv_kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        self.ffn1 = FeedForwardModule(d_model, d_ff, dropout=dropout)
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.conv_module = ConformerConvModule(d_model, kernel_size=conv_kernel_size, dropout=dropout)
        self.ffn2 = FeedForwardModule(d_model, d_ff, dropout=dropout)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. FFN 1
        x = self.ffn1(x)
        # 2. Multi-Head Self-Attention
        residual = x
        norm_x = self.self_attn_norm(x)
        attn_out, _ = self.self_attn(norm_x, norm_x, norm_x)
        x = residual + self.dropout(attn_out)
        # 3. Convolution Module
        x = self.conv_module(x)
        # 4. FFN 2
        x = self.ffn2(x)
        # 5. Final Norm
        return self.final_norm(x)


class AttentiveStatisticsPooling(nn.Module):
    """
    Tầng Gom Cụm Thời Gian Theo Cơ Chế Chú Ý (Attentive Statistics Pooling - ASP):
    Tính toán trung bình có trọng số (weighted mean) và độ lệch chuẩn có trọng số (weighted std)
    qua toàn bộ trục thời gian, sau đó chiếu về chiều vector đặc trưng mong muốn.
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.attention_conv = nn.Sequential(
            nn.Conv1d(in_features, in_features // 2, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(in_features // 2, in_features, kernel_size=1)
        )
        # Nối mean (C) và std (C) -> 2 * in_features
        self.projection = nn.Linear(2 * in_features, out_features, bias=False)
        nn.init.orthogonal_(self.projection.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        x_trans = x.transpose(1, 2) # (B, C, T)
        attn = F.softmax(self.attention_conv(x_trans), dim=-1) # (B, C, T)

        # Weighted Mean
        mean = torch.sum(x_trans * attn, dim=-1) # (B, C)
        # Weighted Std
        var = torch.sum(attn * ((x_trans - mean.unsqueeze(-1)) ** 2), dim=-1)
        std = torch.sqrt(torch.clamp(var, min=1e-8)) # (B, C)

        stats = torch.cat([mean, std], dim=-1) # (B, 2*C)
        emb = self.projection(stats) # (B, out_features)
        return F.normalize(emb, p=2, dim=-1)


class CoverHunterConformer(nn.Module):
    """
    Mô hình CoverHunter chuẩn quốc tế (Conformer Backbone + Attentive Statistics Pooling):
    - Đầu vào: Ma trận phổ CQT (B, 1, 84, T) hoặc (B, 84, T)
    - Tầng Front-end Conv2D / Subsampling: Nén chiều tần số 84 bins về d_model
    - Thân mạng: Chuỗi N khối ConformerBlocks (Mặc định N=4 hoặc N=6)
    - Đầu ra: Attentive Statistics Pooling -> Vector nhúng 512 chiều (L2 normalized)
    """
    def __init__(
        self,
        input_dim: int = 84,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        d_ff: int = 1024,
        conv_kernel_size: int = 15,
        embedding_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.embedding_dim = embedding_dim

        # 1. Front-end chiếu phổ CQT 84 bins vào không gian d_model (256)
        self.input_projection = nn.Sequential(
            nn.Conv1d(input_dim, d_model, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # 2. Chuỗi các khối Conformer
        self.conformer_layers = nn.ModuleList([
            ConformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                conv_kernel_size=conv_kernel_size,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])

        # 3. Tầng Pooling Attentive Statistics Pooling ra vector embedding
        self.asp_pooling = AttentiveStatisticsPooling(in_features=d_model, out_features=embedding_dim)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        # Hỗ trợ input dạng (B, 1, F, T) hoặc (B, F, T)
        if x.dim() == 4:
            x = x.squeeze(1) # (B, F, T)
        # Chiếu đầu vào
        x = self.input_projection(x) # (B, d_model, T)
        x = x.transpose(1, 2)        # (B, T, d_model)

        # Lan truyền qua chuỗi khối Conformer
        for layer in self.conformer_layers:
            x = layer(x)

        return x # (B, T, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        embeddings = self.asp_pooling(features) # (B, embedding_dim)
        return embeddings


# Alias tiện ích
CoverHunterModel = CoverHunterConformer
