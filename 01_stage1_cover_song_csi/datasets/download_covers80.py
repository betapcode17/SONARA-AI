import os
import sys
import io
import urllib.request
import tarfile
from tqdm import tqdm

if sys.platform == "win32" and not hasattr(sys, "_utf8_wrapped"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    sys._utf8_wrapped = True

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_covers80(target_dir="01_stage1_cover_song_csi/datasets/covers80"):
    print("=" * 65)
    print("[*] TỰ ĐỘNG TẢI VÀ GIẢI NÉN BỘ DỮ LIỆU BENCHMARK COVERS80")
    print("=" * 65)

    os.makedirs(target_dir, exist_ok=True)
    tar_path = os.path.join(target_dir, "covers80.tgz")
    url = "https://labrosa.ee.columbia.edu/projects/coversongs/covers80/covers80.tgz"

    if not os.path.exists(tar_path):
        print(f"[*] Đang tải covers80.tgz từ Columbia LabROSA:\n    {url}")
        try:
            with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="covers80.tgz") as t:
                urllib.request.urlretrieve(url, filename=tar_path, reporthook=t.update_to)
            print("[+] Tải thành công file nén!")
        except Exception as e:
            print(f"[!] Không thể tải trực tiếp từ URL ({e}).")
            print("[*] Hướng dẫn thủ công: Tải covers80.tgz và đặt vào thư mục:", target_dir)
            return False

    print(f"[*] Đang giải nén {tar_path}...")
    try:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(path=target_dir)
        print(f"[+] Giải nén thành công! Dữ liệu đã sẵn sàng tại: {target_dir}")
        return True
    except Exception as e:
        print(f"[!] Lỗi giải nén: {e}")
        return False

if __name__ == "__main__":
    download_covers80()
