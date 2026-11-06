# CHƯƠNG 3: THIẾT KẾ KIẾN TRÚC MÔ HÌNH CONFORMER-BYTEHUM QBH V6.2

---

## 3.1. Tổng quan Kiến trúc Mạng Nơ-ron (CoverHunter Conformer 9.91M Tham số)

Để giải quyết bài toán nhận diện giai điệu từ tiếng ngân nga, mô hình học sâu cần giải quyết đồng thời hai thách thức:
1. **Nắm bắt tương quan cục bộ (Local Dependencies)**: Các nốt nhạc luyến láy kề cận, nhịp điệu chuyển tiếp tức thời giữa các âm tiết. Đây là thế mạnh truyền thống của mạng nơ-ron tích chập (Convolutional Neural Networks - CNN).
2. **Nắm bắt tương quan toàn cục dài hạn (Global Long-range Context)**: Cấu trúc lặp lại của các phân đoạn giai điệu, cấu trúc phiên khúc (Verse) và điệp khúc (Chorus) cách xa nhau nhiều giây trong dòng thời gian. Đây là thế mạnh của cơ chế Tự chú ý đa đầu (Multi-Head Self-Attention - MHSA) trong kiến trúc Transformer.

Trong đề tài này, chúng tôi xây dựng mạng nơ-ron kết hợp dựa trên kiến trúc **Conformer** (Convolution-augmented Transformer for Speech and Audio) kết hợp với các mô đun chuyên biệt của **ByteHum** (ICASSP 2024). Toàn bộ mô hình có quy mô **9.91 triệu tham số** (9.91M parameters) tối ưu hóa, đảm bảo độ chính xác vượt trội đồng thời có thể thực thi suy luận trực tiếp theo thời gian thực trên GPU/CPU thương mại.

Quy trình xử lý của mô hình bao gồm:

![Kiến trúc mạng Conformer-ByteHum V6.2](docs/figures/fig_ch3_conformer_arch.png)
*Hình 3.1: Kiến trúc tổng thể mạng nơ-ron Conformer-ByteHum V6.2 (9.91 triệu tham số) với cơ chế Dual Embeddings và các mô đun LAT, LAC, GRL.*

---

## 3.2. Cấu trúc Khối Conformer

Khối Conformer kế thừa thiết kế "Macaron-Net" chuẩn mực, trong đó một khối gồm hai nửa lớp Feed-Forward xen kẽ xung quanh cơ chế Self-Attention và Convolution:

$$y_i^{(1)} = x_i + \frac{1}{2} \text{FFN}(x_i)$$
$$y_i^{(2)} = y_i^{(1)} + \text{MHSA}(y_i^{(1)})$$
$$y_i^{(3)} = y_i^{(2)} + \text{Conv}(y_i^{(2)})$$
$$\text{Output}_i = \text{LayerNorm}\left(y_i^{(3)} + \frac{1}{2} \text{FFN}(y_i^{(3)})\right)$$

### 1. Convolutional Subsampling Module (Khối nén đầu vào)
Đầu vào ảnh phổ CQT có số kênh ban đầu là 1, kích thước $96 \times 83$. Để giảm độ phức tạp tính toán của cơ chế Self-Attention (vốn có độ phức tạp bậc hai $O(T^2)$), tín hiệu trước tiên đi qua 2 tầng tích chập 2D (Conv2d) với kernel $(3, 3)$, stride $(2, 2)$, kết hợp hàm kích hoạt ReLU và chiếu tuyến tính (Linear Projection) lên chiều ẩn $D = 144$.

### 2. Mô đun Multi-Head Self-Attention (MHSA)
MHSA cho phép mỗi vị trí thời gian chú ý đến toàn bộ các vị trí khác trong toàn bộ chuỗi giai điệu:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + S_{rel}\right) V$$
trong đó $S_{rel}$ là ma trận nhúng vị trí tương đối (Relative Positional Encoding), giúp mô hình bất biến trước việc giai điệu bắt đầu sớm hay muộn vài giây.
- **Số đầu chú ý (Attention Heads)**: $H = 4$.
- **Chiều mỗi đầu**: $d_k = 144 / 4 = 36$.

### 3. Mô đun Convolution Tích chập theo chiều sâu (Depthwise Convolution)
Mô đun tích chập được bố trí theo chuỗi:
1. LayerNorm và Pointwise Conv1d (chiếu tăng kênh lên $2 \times D = 288$).
2. Hàm kích hoạt Gated Linear Unit (GLU).
3. **1D Depthwise Convolution**: Kernel size lớn $K = 31$, giúp mô hình có trường tiếp nhận bao phủ rộng các biến thiên âm vực cục bộ mà không làm bùng nổ số lượng trọng số.
4. BatchNorm (tăng tính ổn định gradient âm thanh).
5. Hàm kích hoạt Swish (SiLU).
6. Pointwise Conv1d (chiếu ngược về chiều $D = 144$) kết hợp Dropout tỷ lệ $0.1$.

### 4. Mô đun Feed-Forward (FFN)
Áp dụng cấu trúc 2 tầng Linear biến đổi với hệ số mở rộng kênh bằng 4:
$$\text{FFN}(x) = W_2 \cdot \text{Swish}(W_1 x + b_1) + b_2$$

---

## 3.3. Cơ chế Trích xuất Dual Embeddings và Attentive Statistics Pooling (ASP)

Để phục vụ tìm kiếm hai chặng, mô hình trích xuất ra hai mức độ biểu diễn song song (Dual Embeddings):
1. **Biểu diễn Cục bộ (Local Frame Sequence)**: Chuỗi các vector đặc trưng theo thời gian sau khối Conformer, được định hình thành $T = 20$ lát cắt thời gian với chiều nén $d = 128$:
   $$\mathbf{E}_{local} \in \mathbb{R}^{20 \times 128}$$
2. **Biểu diễn Toàn cục (Global Embedding)**: Được tổng hợp thông qua cơ chế **Attentive Statistics Pooling (ASP)**.

### Toán học của Attentive Statistics Pooling (ASP):
Thay vì sử dụng phép lấy trung bình giản đơn (Global Average Pooling) làm nhòe các đỉnh năng lượng giai điệu, ASP tự động tính toán trọng số quan trọng $\alpha_t$ của từng khung thời gian:
$$e_t = \mathbf{v}^T \tanh(\mathbf{W} h_t + \mathbf{b})$$
$$\alpha_t = \frac{\exp(e_t)}{\sum_{\tau=1}^T \exp(e_\tau)}$$

Trọng số $\alpha_t$ được sử dụng để tính đồng thời **Vector Kỳ vọng có trọng số (Mean)** và **Vector Độ lệch chuẩn có trọng số (Standard Deviation)**:
$$\boldsymbol{\mu} = \sum_{t=1}^T \alpha_t h_t$$
$$\boldsymbol{\sigma} = \sqrt{\sum_{t=1}^T \alpha_t (h_t \odot h_t) - (\boldsymbol{\mu} \odot \boldsymbol{\mu}) + \epsilon}$$

Hai vector này được ghép nối (concatenation) thành vector đặc trưng thống kê toàn cục 512 chiều:
$$\mathbf{h}_{stat} = [\boldsymbol{\mu} \,\|\, \boldsymbol{\sigma}] \in \mathbb{R}^{512}$$
Sau đó, $\mathbf{h}_{stat}$ được chiếu qua một lớp Linear Dense và chuẩn hóa $L_2$ trên siêu mặt cầu (Hypersphere Normalization) để tạo thành **Global Embedding**:
$$\mathbf{e}_{global} = \frac{\mathbf{W}_p \mathbf{h}_{stat}}{\|\mathbf{W}_p \mathbf{h}_{stat}\|_2} \in \mathbb{R}^{128}$$

Chi tiết mã nguồn cài đặt khối `ConformerEncoderLayer` theo cấu trúc Macaron-Net và lớp gộp thống kê chú ý `AttentiveStatisticsPooling` (`01_stage1_cover_song_csi/src/coverhunter/module/conformer.py` & `model.py`):

```python
class ConformerEncoderLayer(torch.nn.Module):
    """Khối Conformer Macaron-Net: 1/2 FFN -> RelPos MHSA -> Depthwise Conv1d -> 1/2 FFN."""
    def __init__(self, size: int, self_attn: nn.Module, feed_forward: nn.Module,
                 feed_forward_macaron: nn.Module, conv_module: nn.Module, dropout_rate: float = 0.1):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module
        self.norm_ff_macaron = nn.LayerNorm(size, eps=1e-5)
        self.norm_mha = nn.LayerNorm(size, eps=1e-5)
        self.norm_conv = nn.LayerNorm(size, eps=1e-5)
        self.norm_ff = nn.LayerNorm(size, eps=1e-5)
        self.norm_final = nn.LayerNorm(size, eps=1e-5)
        self.ff_scale = 0.5
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, pos_emb: torch.Tensor, mask_pad: torch.Tensor):
        # 1. Half-step Macaron Feed-Forward Module
        x = x + self.ff_scale * self.dropout(self.feed_forward_macaron(self.norm_ff_macaron(x)))
        # 2. Relative Positional Multi-Head Self-Attention (MHSA)
        x_att, new_att_cache = self.self_attn(self.norm_mha(x), self.norm_mha(x), self.norm_mha(x), mask, pos_emb)
        x = x + self.dropout(x_att)
        # 3. Depthwise Separable Convolution Module (GLU + 1D Depthwise Conv + BatchNorm + Swish)
        x_conv, new_cnn_cache = self.conv_module(self.norm_conv(x), mask_pad)
        x = x + self.dropout(x_conv)
        # 4. Second Half-step Feed-Forward & Final LayerNorm
        x = x + self.ff_scale * self.dropout(self.feed_forward(self.norm_ff(x)))
        return self.norm_final(x), mask, new_att_cache, new_cnn_cache

class AttentiveStatisticsPooling(torch.nn.Module):
    """Gộp thống kê tự chú ý (ASP): Kết hợp có trọng số kỳ vọng (Mean) và độ lệch chuẩn (Std)."""
    def __init__(self, channels: int = 128, output_channels: int = 128):
        super().__init__()
        self._eps = 1e-12
        self._linear = Linear(channels * 3, channels)
        self._tanh = torch.nn.Tanh()
        self._conv = Conv1d(in_channels=channels, out_channels=channels, kernel_size=1)
        self._final_layer = torch.nn.Linear(channels * 2, output_channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # [B, C=128, T=20]
        L = x.shape[-1]
        mean, std = self._compute_statistics(x, torch.ones_like(x[:, :1, :]) / L, self._eps)
        attn = torch.cat([x, mean.unsqueeze(2).repeat(1, 1, L), std.unsqueeze(2).repeat(1, 1, L)], dim=1)
        attn = F.softmax(self._conv(self._tanh(self._linear(attn.transpose(1, 2)).transpose(1, 2))), dim=2)
        mean, std = self._compute_statistics(x, attn, self._eps)
        return self._final_layer(torch.cat((mean, std), dim=1))  # [B, 128]
```

---

## 3.4. Khung Kiến trúc ByteHum (ICASSP 2024)

Khung kiến trúc **ByteHum** được nhóm nghiên cứu của ByteDance công bố tại hội nghị ICASSP 2024 nhằm giải quyết hai điểm yếu chí tử của mô hình QBH truyền thống: hiện tượng sai lệch độ dài phân đoạn (Short Query vs Long Song) và khoảng cách miền thanh âm (Domain Gap). Chúng tôi tích hợp trọn vẹn 3 mô đun đột phá của ByteHum:

### 3.4.1. Learned Alignment Transformer (LAT) với MaxMean Pooling
Người dùng thường chỉ ngân nga một đoạn ngắn (5 đến 15 giây) của một bài hát kéo dài 3 đến 5 phút. Khi so sánh Global Embedding, thông tin của cả bài hát dài bị nén lại làm giảm độ tương đồng với đoạn ngân nga ngắn.

Mô đun LAT giải quyết vấn đề này bằng cách tính ma trận tương đồng chéo (Cross-Attention Correlation Matrix) giữa chuỗi đặc trưng cục bộ của đoạn truy vấn $\mathbf{Q} \in \mathbb{R}^{T_q \times d}$ và đoạn bài hát thư viện $\mathbf{K} \in \mathbb{R}^{T_k \times d}$:
$$\mathbf{M}_{ij} = \frac{\mathbf{Q}_i \cdot \mathbf{K}_j^T}{\|\mathbf{Q}_i\|_2 \|\mathbf{K}_j\|_2} \in [-1, 1]$$

Sau đó, áp dụng cơ chế **MaxMean Pooling**:
1. Với mỗi khung thời gian thứ $i$ của tiếng ngân nga, tìm điểm tương đồng lớn nhất dọc theo chiều thời gian của bài hát đích:
   $$s_i = \max_{1 \le j \le T_k} \mathbf{M}_{ij}$$
   (Thao tác này tìm xem nốt nhạc người dùng đang ngân nga khớp chuẩn nhất vào giây thứ mấy trong bài hát).
2. Lấy trung bình cộng các giá trị cực đại trên toàn bộ độ dài truy vấn để cho ra điểm tương đồng căn chỉnh cuối cùng:
   $$S_{LAT}(\mathbf{Q}, \mathbf{K}) = \frac{1}{T_q} \sum_{i=1}^{T_q} s_i$$

### 3.4.2. Learned Alignment Classifier (LAC) với L2 Prototypes
Mô đun LAC duy trì một ma trận các vector tâm cụm đặc trưng (Prototypes) đại diện cho từng bài hát trong không gian đa chiều:
$$\mathbf{P} \in \mathbb{R}^{C \times d}$$
với $C$ là số lượng bài hát và $d = 128$.

Xác suất phân loại bài hát được tính dựa trên khoảng cách Euclidean thu nhỏ kết hợp hệ số tỷ lệ nhiệt độ (Temperature scale) $\tau$:
$$P(y = c \mid \mathbf{e}) = \frac{\exp(-\|\mathbf{e} - \mathbf{p}_c\|_2^2 / \tau)}{\sum_{j=1}^C \exp(-\|\mathbf{e} - \mathbf{p}_j\|_2^2 / \tau)}$$
Hàm mất mát $\mathcal{L}_{LAC}$ ép buộc biểu diễn của đoạn ngân nga phải hội tụ về đúng tâm cụm nguyên mẫu của bài hát đó.

### 3.4.3. Huấn luyện Đối kháng Miền với Gradient Reversal Layer (GRL)
Khoảng cách miền (Domain Gap) giữa giọng ngân nga của người dùng nghiệp dư và giọng ca sĩ phòng thu là nguyên nhân gây tụt giảm hiệu năng lớn.

Để buộc bộ mã hóa Conformer học được các đặc trưng "bất biến theo miền" (Domain-Invariant Representations), chúng tôi tích hợp một nhánh Phân loại Miền (Domain Classifier) kết hợp với **Lớp Đảo ngược Gradient (Gradient Reversal Layer - GRL)**:
- Ở lượt truyền xuôi (Forward pass): GRL hoạt động như một hàm đồng nhất:
  $$\text{GRL}(x) = x$$
- Ở lượt truyền ngược (Backward pass): GRL tự động đảo ngược dấu của gradient và nhân với hệ số điều tiết $\lambda$:
  $$\frac{\partial \mathcal{L}}{\partial x} = -\lambda \frac{\partial \mathcal{L}_{domain}}{\partial x}$$

Khi nhánh phân loại miền cố gắng tối đa hóa khả năng nhận diện xem đặc trưng là "giọng ngân nga" hay "giọng ca sĩ", thì qua lớp GRL, bộ mã hóa Conformer lại được tối ưu hóa để **đánh lừa bộ phân loại miền**. Kết quả là các vector embedding sinh ra hoàn toàn không còn mang dấu ấn âm sắc của miền, chỉ lưu giữ thuần túy cấu trúc giai điệu bài hát.

Đoạn mã nguồn cài đặt lớp đảo ngược đạo hàm `GradientReversalFunction`, mô đun phân loại nguyên mẫu `LocalAlignmentClassifier` (LAC) và lớp kiến trúc tổng thể `ConformerByteHumModel` (9.91 triệu tham số):

```python
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_val: float):
        ctx.lambda_val = lambda_val
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Đảo ngược dấu gradient nhân với hệ số điều tiết lambda_val trong lượt lan truyền ngược
        return grad_output.neg() * ctx.lambda_val, None

class LocalAlignmentClassifier(nn.Module):
    """Local Alignment Classifier (LAC - ByteHum ICASSP 2024) với L=4 L2 Prototypes & MaxMean."""
    def __init__(self, num_classes: int = 244, feature_dim: int = 128, num_prototypes: int = 4):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.num_prototypes = num_prototypes
        self.prototypes = nn.Parameter(torch.randn(num_classes, num_prototypes, feature_dim) * 0.05)
        self.temperature = nn.Parameter(torch.tensor(10.0))

    def forward(self, local_feat: torch.Tensor) -> torch.Tensor:
        x_norm = F.normalize(local_feat, p=2, dim=-1)          # [B, T=20, 128]
        w_norm = F.normalize(self.prototypes, p=2, dim=-1)     # [K, L=4, 128]
        sim = torch.matmul(x_norm, w_norm.view(-1, self.feature_dim).t())
        sim = sim.view(local_feat.size(0), local_feat.size(1), self.num_classes, self.num_prototypes)
        logits = sim.max(dim=1).values.mean(dim=-1)            # Max theo T=20, Mean theo L=4
        return logits * self.temperature.clamp(min=1.0, max=100.0)

class ConformerByteHumModel(nn.Module):
    """Kiến trúc tổng thể Conformer-ByteHum V6.2 (9.91M tham số - Dual Embeddings + LAT/LAC/GRL)."""
    def __init__(self, ckpt_path: str = None, input_dim: int = 96, embedding_dim: int = 128, num_classes: int = 244):
        super().__init__()
        self.feature_extractor = build_coverhunter_model(ckpt_path=ckpt_path)
        self.grl = GradientReversalLayer(lambda_val=0.2)
        self.domain_classifier = LocalDomainClassifier(feature_dim=embedding_dim, hidden_dim=64)
        self.lac = LocalAlignmentClassifier(num_classes=num_classes, feature_dim=embedding_dim, num_prototypes=4)

    def forward(self, x: torch.Tensor, compute_lac: bool = True, compute_domain: bool = True):
        backbone_feat = self.extract_backbone_features(x)              # [B, T=20, 128]
        local_feat = self.local_projection(backbone_feat)              # Local Sequence [B, 20, 128]
        global_emb = self.extract_global_embedding(backbone_feat)      # Global ASP L2 Embedding [B, 128]
        lac_logits = self.lac(local_feat) if compute_lac else None
        domain_logits = self.domain_classifier(self.grl(local_feat)) if compute_domain else None
        return global_emb, local_feat, lac_logits, domain_logits
```

---

## 3.5. Hàm Mất mát Đa Mục tiêu Hợp nhất (Joint Multi-Task Loss)

Toàn bộ hệ thống Conformer-ByteHum V6.2 được tối ưu hóa đồng thời bằng một hàm mục tiêu đa nhiệm liên kết chặt chẽ:
$$\mathcal{L}_{total} = \mathcal{L}_{metric} + \lambda_1 \mathcal{L}_{LAT} + \lambda_2 \mathcal{L}_{LAC} + \lambda_3 \mathcal{L}_{domain}$$

trong đó:
1. **$\mathcal{L}_{metric}$**: Hàm mất mát bộ ba thích ứng (Multi-Similarity / Triplet Margin Loss with Hard Negative Mining) trên Global Embeddings:
   $$\mathcal{L}_{metric} = \max\left(0, \|\mathbf{e}_a - \mathbf{e}_p\|_2^2 - \|\mathbf{e}_a - \mathbf{e}_n\|_2^2 + \alpha\right)$$
   với biên lề (margin) $\alpha = 0.3$.
2. **$\mathcal{L}_{LAT}$**: Hàm mất mát Cross-Entropy trên điểm tương đồng căn chỉnh MaxMean.
3. **$\mathcal{L}_{LAC}$**: Hàm mất mát nguyên mẫu L2 Prototype Loss.
4. **$\mathcal{L}_{domain}$**: Hàm mất mát phân loại miền nhị phân (Binary Cross-Entropy) truyền ngược qua GRL.
5. **Hệ số cân bằng**: Được tinh chỉnh thực nghiệm tối ưu tại $\lambda_1 = 0.5$, $\lambda_2 = 0.3$, và $\lambda_3 = 0.1$.

Chi tiết mã nguồn hàm căn chỉnh cục bộ `local_alignment_triplet` (LAT) và hàm mục tiêu huấn luyện đa nhiệm `training_objective` trong dự án (`02_stage2_humming_qbh/scripts/notebook_builders/templates/final_solution_runtime.py`):

```python
def local_alignment_triplet(anchor, positive, negative, song_ids, negative_song_ids,
                            margin=0.2, hardest_ratio=0.7, top_k=5):
    """Hàm mất mát căn chỉnh cục bộ ByteHum LAT (MaxMean Cross-Attention Triplet Loss)."""
    anchor, positive, negative = [F.normalize(x, dim=-1) for x in (anchor, positive, negative)]
    positive_scores = torch.bmm(anchor, positive.transpose(1, 2)).max(-1).values.mean(-1, keepdim=True)
    candidates = torch.cat([positive, negative], dim=0)
    labels = torch.cat([song_ids, negative_song_ids], dim=0)
    valid = song_ids.unsqueeze(1) != labels.unsqueeze(0)

    # Tính ma trận tương quan chéo M_ij và áp dụng MaxMean Pooling dọc trục thời gian
    similarities = torch.einsum('btc,nsc->bnts', anchor, candidates).max(dim=-1).values.mean(dim=-1)
    hard_score, _ = _topk_masked_mean(similarities, valid, top_k)
    semi_mask = valid & (similarities < positive_scores) & (similarities > positive_scores - margin)
    semi_score, has_semi = _topk_masked_mean(similarities, semi_mask, top_k)
    selected = (1.0 - hardest_ratio) * torch.where(has_semi, semi_score, hard_score) + hardest_ratio * hard_score
    return F.relu(selected - positive_scores + margin).mean()

def training_objective(outputs, labels, negative_labels, epoch, memory_bank):
    """Hàm mất mát đa mục tiêu hợp nhất (Joint Multi-Task Objective V6.2)."""
    ga, gp, gn = [F.normalize(out[0].float(), dim=-1) for out in outputs]
    la, lp, ln = [out[1].float() for out in outputs]
    curriculum = mining_hardest_ratio(epoch)

    triplet = in_batch_cosine_triplet_loss(
        ga, gp, gn, labels, negative_labels,
        margin=RUN_CONFIG['margin'], memory_bank=memory_bank,
        hardest_ratio=curriculum, top_k=RUN_CONFIG['mining_top_k']
    )
    lat = local_alignment_triplet(
        la, lp, ln, labels, negative_labels,
        margin=RUN_CONFIG['margin'], hardest_ratio=curriculum, top_k=RUN_CONFIG['mining_top_k']
    )
    lac = torch.stack([F.cross_entropy(out[2].float(), t) for out, t in zip(outputs, (labels, labels, negative_labels))]).mean()
    domain = torch.stack([F.binary_cross_entropy_with_logits(out[3].float(), torch.full_like(out[3].float(), t))
                          for out, t in zip(outputs, (1.0, 0.0, 0.0))]).mean()
    prototype = prototype_diversity_loss(model.lac.prototypes.float())

    total = (RUN_CONFIG['global_triplet'] * triplet + RUN_CONFIG['lat'] * lat
             + RUN_CONFIG['lac'] * lac + domain + RUN_CONFIG['prototype_diversity'] * prototype)
    return dict(global_triplet=triplet, lat=lat, lac=lac, domain=domain, prototype=prototype, total=total)
```

---

## 3.6. Sơ đồ Kiến trúc Mạng Nơ-ron Chi tiết

Sơ đồ chi tiết các khối kết nối, kích thước tensor và đường đi gradient của mô hình Conformer-ByteHum V6.2 được lưu trữ tại file vector chuẩn Draw.io:
📁 **Tham chiếu sơ đồ**: [`docs/figures/conformer_bytehum_detailed.drawio`](file:///e:/HOCKY7/PBL6/code/docs/figures/conformer_bytehum_detailed.drawio)

*Mô tả luồng trên sơ đồ*:
- Khối đầu vào: Log-CQT $96 \times 83$ $\rightarrow$ Convolutional Subsampling.
- Khối thân mạng: 6 tầng Conformer Encoder với liên kết Macaron Feed-Forward và Depthwise Conv.
- Khối phân nhánh: Nhánh ASP sinh Global Metric Embedding (128D) đi vào Triplet Loss; nhánh Local Sequence đi vào bộ đôi ByteHum LAT/LAC; nhánh trung gian đi qua GRL vào Domain Classifier.
- Khối hàm mất mát: Hợp nhất 4 thành phần loss thành $\mathcal{L}_{total}$ duy nhất.

---

## 3.7. Kết luận chương

Chương 3 đã làm rõ kiến trúc học sâu tiên tiến nhất trong đề tài:
- Khắc phục triệt để nhược điểm của mạng CNN nông truyền thống bằng **CoverHunter Conformer 9.91M tham số**, kết hợp hài hòa giữa khả năng nắm bắt nhịp điệu cục bộ (Depthwise Conv) và quan hệ cấu trúc bài hát toàn cục (MHSA).
- Cơ chế **Attentive Statistics Pooling (ASP)** cung cấp vector đại diện thống kê bền vững hơn nhiều so với trung bình số học thông thường.
- Sự kết hợp của bộ ba công nghệ **ByteHum (LAT, LAC, GRL)** đã giải quyết đồng thời hai bài toán hóc búa nhất của lĩnh vực QBH: căn chỉnh độ dài bài hát bất đối xứng và triệt tiêu khoảng cách miền thanh âm giữa người dùng và ca sĩ phòng thu.

