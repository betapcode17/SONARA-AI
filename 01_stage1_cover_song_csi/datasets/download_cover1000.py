import os
import sys
import io
import urllib.request
import zipfile
from tqdm import tqdm

if sys.platform == "win32" and not hasattr(sys, "_utf8_wrapped"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    sys._utf8_wrapped = True

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_cover1000(target_dir="01_stage1_cover_song_csi/datasets/cover1000"):
    print("=" * 65)
    print("[*] TẢI ĐẶC TRƯNG COVER-1000 CHO BASELINE MOVE (GS. PEETERS)")
    print("=" * 65)

    os.makedirs(target_dir, exist_ok=True)
    root_url = "https://perso.telecom-paristech.fr/gpeeters/tuto_DL101forMIR/"
    files = ["cover1000_feat.hdf5.zip", "cover1000.pyjama"]

    for fname in files:
        fpath = os.path.join(target_dir, fname)
        if not os.path.exists(fpath):
            url = root_url + fname
            print(f"[*] Đang tải {fname}...")
            try:
                with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=fname) as t:
                    urllib.request.urlretrieve(url, filename=fpath, reporthook=t.update_to)
                print(f"[+] Đã tải {fname}")
            except Exception as e:
                print(f"[!] Lỗi tải {fname}: {e}")

        if fname.endswith(".zip") and os.path.exists(fpath):
            unzip_target = os.path.join(target_dir, "cover1000_feat.hdf5")
            if not os.path.exists(unzip_target):
                print(f"[*] Đang giải nén {fname}...")
                with zipfile.ZipFile(fpath, 'r') as zf:
                    zf.extractall(target_dir)
                print(f"[+] Đã giải nén cover1000_feat.hdf5 vào {target_dir}")

    print("[+] Hoàn tất chuẩn bị tập dữ liệu Cover-1000!")

if __name__ == "__main__":
    download_cover1000()
