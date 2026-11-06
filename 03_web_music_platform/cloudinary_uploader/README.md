# PBL6 - Cloudinary Audio Uploader & Media Catalog

Module quản lý và tự động hóa upload kho bài hát âm thanh gốc (300 bài full recordings trong `2_music_database`) lên nền tảng đám mây **Cloudinary** nhằm phục vụ streaming audio cho Website Âm nhạc PBL6.

---

## 1. Cấu trúc thư mục

```
03_web_music_platform/cloudinary_uploader/
├── .env                              # Thông tin tài khoản và API Key Cloudinary
├── config.py                         # Cấu hình đường dẫn, tài khoản và tham số đa luồng
├── test_connection.py                # Script kiểm tra kết nối & dung lượng Cloudinary
├── upload_audio_gallery.py           # Script upload tự động, đa luồng, hỗ trợ resume
├── cloudinary_songs_catalog.json     # Danh mục URL HTTPS CDN từng bài (cho Backend/API)
├── cloudinary_songs_catalog.csv      # Bảng danh mục định dạng CSV (cho Excel / báo cáo)
└── README.md                         # Hướng dẫn sử dụng
```

---

## 2. Cách chạy script

### A. Kiểm tra kết nối tài khoản
```bash
python 03_web_music_platform/cloudinary_uploader/test_connection.py
```

### B. Chạy thử nghiệm 3 bài hát đầu tiên
```bash
python 03_web_music_platform/cloudinary_uploader/upload_audio_gallery.py --limit 3
```

### C. Chạy upload toàn bộ kho nhạc (300 bài hát)
```bash
python 03_web_music_platform/cloudinary_uploader/upload_audio_gallery.py --workers 3
```

> **Tính năng Resumable (Tự động tiếp tục):**  
> Script lưu tiến độ ngay sau mỗi bài hát được upload thành công vào `cloudinary_songs_catalog.json`. Nếu mạng rớt hoặc bạn tắt máy giữa chừng, lần chạy tiếp theo script sẽ **tự động bỏ qua** các bài đã upload và chỉ upload những bài còn lại!

---

## 3. Cách tích hợp vào Website Âm nhạc

### Trong Backend (FastAPI / Flask / Express.js)
Khi mô hình QBH (Query by Humming) nhận diện ra Top bài hát phù hợp (ví dụ: `song_0001`), Backend chỉ cần đọc từ `cloudinary_songs_catalog.json`:

```python
import json

with open("03_web_music_platform/cloudinary_uploader/cloudinary_songs_catalog.json", "r", encoding="utf-8") as f:
    catalog = json.load(f)

# Giả sử mô hình dự đoán ra Top 1 là song_0001
top1_id = "song_0001"
song_data = catalog.get(top1_id)

print(song_data["title"])       # 2 Unlimited - Get Ready For This
print(song_data["secure_url"])  # https://res.cloudinary.com/.../song_0001.mp3
print(song_data["duration"])    # 220.29 giây
```

### Trong Frontend (HTML5 / React / Vue)
Gắn trực tiếp `secure_url` vào thẻ phát nhạc HTML5:
```html
<audio controls preload="none">
  <source src="https://res.cloudinary.com/hm8hhw5o/video/upload/v1793702370/pbl6_music_database/song_0001.mp3" type="audio/mpeg">
  Trình duyệt của bạn không hỗ trợ phát âm thanh.
</audio>
```
Cloudinary hỗ trợ streaming byte-range, cho phép người dùng tua nhanh (seek) đến bất kỳ giây nào mà không cần tải hết cả bài hát!

