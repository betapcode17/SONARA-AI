import torch
import torch.nn as nn
import torch.nn.functional as F


def batch_pairwise_distance(embeddings: torch.Tensor) -> torch.Tensor:
    """
    Tính ma trận khoảng cách tương quan cặp Euclidean giữa tất cả các vector trong batch:
    dist[i, j] = ||e_i - e_j||_2
    """
    dot_product = torch.matmul(embeddings, embeddings.t())
    square_norm = torch.diag(dot_product)
    distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
    distances = torch.clamp(distances, min=0.0)
    # Tránh đạo hàm vô cực tại 0 bằng cách thêm epsilon
    mask = (distances == 0.0).float()
    distances = distances + mask * 1e-16
    distances = torch.sqrt(distances)
    distances = distances * (1.0 - mask)
    return distances


class OnlineTripletLoss(nn.Module):
    """
    Hàm mất mát Online Triplet Loss tích hợp 3 chiến lược đào mẫu chuẩn học thuật:
    1. 'hard': Chọn Negative gần nhất và Positive xa nhất trong batch.
    2. 'semi_hard': Chọn Negative nằm trong dải Margin: D_ap < D_an < D_ap + margin.
    3. 'random': Bốc ngẫu nhiên Positive và Negative hợp lệ.
    """
    def __init__(self, margin: float = 0.2, mining_type: str = "hard"):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.mining_type = mining_type

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> dict:
        embeddings = F.normalize(embeddings, p=2, dim=-1)
        pairwise_dist = batch_pairwise_distance(embeddings)
        device = embeddings.device
        B = embeddings.size(0)

        # Mặt nạ: mask_pos = 1 nếu cùng bài hát, mask_neg = 1 nếu khác bài hát
        mask_pos = (labels.unsqueeze(0) == labels.unsqueeze(1))
        mask_neg = ~mask_pos

        # Bỏ đi chính nó (i == j) khỏi positive mask
        diag_mask = torch.eye(B, dtype=torch.bool, device=device)
        mask_pos = mask_pos & (~diag_mask)

        if self.mining_type == "hard":
            # 1. Hard Positive: Cặp cùng bài có khoảng cách xa nhất
            dist_pos_masked = pairwise_dist * mask_pos.float()
            hardest_pos, _ = torch.max(dist_pos_masked, dim=1)

            # 2. Hard Negative: Cặp khác bài có khoảng cách gần nhất
            max_dist = torch.max(pairwise_dist).item()
            dist_neg_masked = pairwise_dist + (mask_pos | diag_mask).float() * (max_dist + 1.0)
            hardest_neg, _ = torch.min(dist_neg_masked, dim=1)

            triplet_loss = F.relu(hardest_pos - hardest_neg + self.margin)
            # Chỉ lấy trung bình trên các mẫu có cả positive và negative hợp lệ
            valid_anchors = (mask_pos.sum(dim=1) > 0) & (mask_neg.sum(dim=1) > 0)
            if valid_anchors.any():
                final_loss = triplet_loss[valid_anchors].mean()
            else:
                final_loss = torch.tensor(0.0, device=device, requires_grad=True)

        elif self.mining_type == "semi_hard":
            # Semi-hard Negative: D_ap < D_an < D_ap + margin
            loss_list = []
            for i in range(B):
                pos_idx = torch.where(mask_pos[i])[0]
                neg_idx = torch.where(mask_neg[i])[0]
                if len(pos_idx) == 0 or len(neg_idx) == 0:
                    continue
                d_ap = pairwise_dist[i, pos_idx[torch.randint(len(pos_idx), (1,)).item()]]
                d_an_all = pairwise_dist[i, neg_idx]
                semi_hard = neg_idx[(d_an_all > d_ap) & (d_an_all < d_ap + self.margin)]
                if len(semi_hard) > 0:
                    d_an = pairwise_dist[i, semi_hard[0]]
                else:
                    d_an = d_an_all.min()
                loss_list.append(F.relu(d_ap - d_an + self.margin))
            final_loss = torch.stack(loss_list).mean() if loss_list else torch.tensor(0.0, device=device, requires_grad=True)

        else: # "random"
            loss_list = []
            for i in range(B):
                pos_idx = torch.where(mask_pos[i])[0]
                neg_idx = torch.where(mask_neg[i])[0]
                if len(pos_idx) > 0 and len(neg_idx) > 0:
                    p = pos_idx[torch.randint(len(pos_idx), (1,)).item()]
                    n = neg_idx[torch.randint(len(neg_idx), (1,)).item()]
                    loss_list.append(F.relu(pairwise_dist[i, p] - pairwise_dist[i, n] + self.margin))
            final_loss = torch.stack(loss_list).mean() if loss_list else torch.tensor(0.0, device=device, requires_grad=True)

        return {
            'loss': final_loss,
            'mean_pos_dist': pairwise_dist[mask_pos].mean().item() if mask_pos.any() else 0.0,
            'mean_neg_dist': pairwise_dist[mask_neg].mean().item() if mask_neg.any() else 0.0
        }
