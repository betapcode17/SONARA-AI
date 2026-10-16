import os
import sys
import io
import shutil
import tarfile
import torch

if sys.platform == "win32" and not hasattr(sys, "_utf8_wrapped"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    sys._utf8_wrapped = True

try:
    import gdown
except ImportError:
    gdown = None


def download_coverhunter_pretrained():
    print("=" * 70)
    print("[*] TẢI TRỌNG SỐ PRE-TRAINED SHS100K CỦA COVERHUNTER (SOTA CONFORMER)")
    print("=" * 70)

    ckpt_dir = os.path.abspath("01_stage1_cover_song_csi/checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    dest_pth = os.path.join(ckpt_dir, "coverhunter_shs100k_pretrained.pth")
    dest_pt = os.path.join(ckpt_dir, "coverhunter_shs100k.pt")

    if os.path.exists(dest_pth) and os.path.getsize(dest_pth) > 10 * 1024 * 1024:
        print(f"[+] Checkpoint CoverHunter chuẩn đã tồn tại tại:\n    -> {dest_pth} ({os.path.getsize(dest_pth)/(1024**2):.1f} MB)")
        return dest_pth

    tar_path = os.path.join(ckpt_dir, "pretrain_model.tar.gz")
    file_id = "1rDZ9CDInpxQUvXRLv87mr-hfDfnV7Y-j"

    if not os.path.exists(tar_path) or os.path.getsize(tar_path) < 10 * 1024 * 1024:
        print(f"[*] Đang tải pretrain_model.tar.gz từ Google Drive (File ID: {file_id})...")
        if gdown is not None:
            gdown.download(id=file_id, output=tar_path, quiet=False)
        else:
            print("[!] Cần cài đặt gdown để tải Google Drive: pip install gdown")
            return None

    # Giải nén archive
    print("[*] Đang giải nén pretrain_model.tar.gz...")
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(ckpt_dir)

    raw_model_path = os.path.join(ckpt_dir, "pretrain_model/pt_model/g_00000040")
    if os.path.exists(raw_model_path):
        print(f"[+] Tìm thấy file trọng số gốc: {raw_model_path}")
        ckpt_data = torch.load(raw_model_path, map_location='cpu', weights_only=False)
        torch.save(ckpt_data, dest_pth)
        torch.save(ckpt_data, dest_pt)

        # Sao chép sang Stage 2 checkpoints
        stage2_ckpt_dir = os.path.abspath("02_stage2_humming_qbh/checkpoints")
        os.makedirs(stage2_ckpt_dir, exist_ok=True)
        shutil.copy2(dest_pth, os.path.join(stage2_ckpt_dir, "coverhunter_shs100k_pretrained.pth"))

        print(f"[+] Đã hoàn tất cài đặt trọng số CoverHunter pre-trained tại:")
        print(f"    - {dest_pth}")
        print(f"    - {dest_pt}")
        print(f"    - {os.path.join(stage2_ckpt_dir, 'coverhunter_shs100k_pretrained.pth')}")
        return dest_pth
    else:
        print(f"[!] Không tìm thấy file g_00000040 trong thư mục giải nén!")
        return None


if __name__ == "__main__":
    download_coverhunter_pretrained()
