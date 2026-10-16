import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple


def evaluate_csi_metrics(
    query_embeddings: torch.Tensor,
    reference_embeddings: torch.Tensor,
    query_labels: List[str],
    reference_labels: List[str],
    top_k_list: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    Đánh giá các độ đo xếp hạng chuẩn quốc tế cho bài toán Cover Song Identification:
    - Mean Rank (MR1)
    - Mean Reciprocal Rank (MRR1)
    - Precision @ K (P@1, P@5, P@10)
    - Mean Average Precision (mAP)
    """
    q_norm = F.normalize(query_embeddings, p=2, dim=-1)
    r_norm = F.normalize(reference_embeddings, p=2, dim=-1)

    # Tính độ tương đồng Cosine (hoặc khoảng cách Euclidean âm)
    sim_matrix = torch.mm(q_norm, r_norm.t()).detach().cpu().numpy()

    num_queries = len(query_labels)
    rank_list = []
    rr_list = []
    precision_at_k = {k: [] for k in top_k_list}
    ap_list = []

    for i in range(num_queries):
        q_label = query_labels[i]
        sims = sim_matrix[i].copy()

        # Sắp xếp bài tham chiếu theo thứ tự độ tương đồng giảm dần
        sorted_indices = np.argsort(-sims)
        ranked_labels = [reference_labels[idx] for idx in sorted_indices]

        # Kiểm tra sự xuất hiện của bài cover cùng work-id
        relevance = np.array([1 if r_lbl == q_label else 0 for r_lbl in ranked_labels])

        if np.sum(relevance) == 0:
            continue

        first_hit_idx = np.where(relevance == 1)[0][0]
        rank = first_hit_idx + 1
        rank_list.append(rank)
        rr_list.append(1.0 / rank)

        for k in top_k_list:
            p_k = np.sum(relevance[:k]) / float(k)
            precision_at_k[k].append(p_k)

        # Average Precision cho query i
        hit_ranks = np.where(relevance == 1)[0] + 1
        ap = np.mean([(idx + 1) / r for idx, r in enumerate(hit_ranks)])
        ap_list.append(ap)

    return {
        'mean_rank': float(np.mean(rank_list)) if rank_list else 0.0,
        'mrr': float(np.mean(rr_list)) if rr_list else 0.0,
        'p@1': float(np.mean(precision_at_k[1])) if 1 in precision_at_k and precision_at_k[1] else 0.0,
        'p@5': float(np.mean(precision_at_k[5])) if 5 in precision_at_k and precision_at_k[5] else 0.0,
        'p@10': float(np.mean(precision_at_k[10])) if 10 in precision_at_k and precision_at_k[10] else 0.0,
        'map': float(np.mean(ap_list)) if ap_list else 0.0
    }
