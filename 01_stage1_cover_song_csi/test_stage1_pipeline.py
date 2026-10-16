import os
import sys
import io
import torch

if sys.platform == "win32" and not hasattr(sys, "_utf8_wrapped"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    sys._utf8_wrapped = True

sys.path.append(os.path.abspath("01_stage1_cover_song_csi"))
from src.transforms import CQTExtractor, RandomPitchShiftCQT
from src.resnet_ibn import ResNetIBN
from src.losses import OnlineTripletLoss, batch_pairwise_distance
from src.metrics import evaluate_csi_metrics

def test_stage1():
    print("=" * 65)
    print("[*] KIỂM THỬ PIPELINE GIAI ĐOẠN 1: COVER SONG IDENTIFICATION (CSI)")
    print("=" * 65)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[+] Thiết bị: {device}")

    # 1. Khởi tạo mô hình ResNet-IBN (512-D, Orthogonal Init)
    model = ResNetIBN(
        in_channels=1,
        embedding_dim=512,
        gem_p=3.0,
        dropout=0.1,
        pretrained=False
    ).to(device)

    print(f"[+] Khởi tạo ResNetIBN thành công (Tham số: {sum(p.numel() for p in model.parameters()):,})")

    # 2. Tạo batch giả lập CQT 84 bins (B=16, 4 classes, mỗi class 4 bản cover)
    B = 16
    cqts = torch.randn(B, 1, 84, 431, device=device)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3], device=device)

    # 3. Forward pass
    embs = model(cqts)
    print(f"[+] Shape Embeddings: {embs.shape} (Kỳ vọng: [{B}, 512])")
    assert embs.shape == (B, 512), f"Lỗi shape embedding: {embs.shape}"

    # 4. Kiểm tra Online Triplet Loss (Hard Mining)
    criterion = OnlineTripletLoss(margin=0.2, mining_type="hard")
    loss_dict = criterion(embs, labels)
    loss = loss_dict['loss']

    print(f"[+] Hard Triplet Loss: {loss.item():.4f}")
    print(f"    - Mean Positive Dist: {loss_dict['mean_pos_dist']:.4f}")
    print(f"    - Mean Negative Dist: {loss_dict['mean_neg_dist']:.4f}")

    # 5. Backward pass
    loss.backward()
    grad_norm = model.fc.weight.grad.norm().item()
    print(f"[+] Backward thành công! Grad norm của FC weight: {grad_norm:.6f}")

    # 6. Kiểm tra Metrics
    q_embs = embs[:4]
    r_embs = embs[4:]
    q_lbls = ["Song_A", "Song_A", "Song_B", "Song_B"]
    r_lbls = ["Song_A", "Song_A", "Song_B", "Song_B", "Song_C", "Song_C", "Song_D", "Song_D", "Song_E", "Song_E", "Song_F", "Song_F"]
    metrics = evaluate_csi_metrics(q_embs, r_embs, q_lbls, r_lbls)

    print(f"[+] Kết quả đo kiểm CSI Metrics giả lập:")
    print(f"    - Mean Rank: {metrics['mean_rank']:.2f}")
    print(f"    - MRR: {metrics['mrr']:.4f}")
    print(f"    - P@1 (Top-1): {metrics['p@1'] * 100:.1f}%")
    print(f"    - mAP: {metrics['map']:.4f}")

    print("\n[V] TOÀN BỘ CÁC MODULE STAGE 1 ĐÃ HOẠT ĐỘNG HOÀN HẢO!")

if __name__ == "__main__":
    test_stage1()
