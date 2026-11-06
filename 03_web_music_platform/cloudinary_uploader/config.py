import os
from pathlib import Path

# Thư mục gốc của uploader
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent

# Cấu hình tài khoản Cloudinary
CLOUD_NAME = "hm8hhw5o"
API_KEY = "438676921963891"
API_SECRET = "XdVlD5EjLkLlaM5wvMVwFPgvDM0"
CLOUDINARY_FOLDER = "pbl6_music_database"

# Đường dẫn dữ liệu âm thanh gốc
LOCAL_MUSIC_DIR = PROJECT_ROOT / "02_stage2_humming_qbh" / "datasets" / "2_music_database"
SONG_NAME_MAPPING_FILE = PROJECT_ROOT / "02_stage2_humming_qbh" / "datasets" / "song_name_mapping.json"

# File lưu trữ danh mục output sau khi upload
OUTPUT_CATALOG_JSON = CURRENT_DIR / "cloudinary_songs_catalog.json"
OUTPUT_CATALOG_CSV = CURRENT_DIR / "cloudinary_songs_catalog.csv"

# Cấu hình upload đa luồng
DEFAULT_WORKERS = 3
DEFAULT_RETRY_COUNT = 3

