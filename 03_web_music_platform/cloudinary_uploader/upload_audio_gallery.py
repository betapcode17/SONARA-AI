import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import os
import json
import time
import csv
import shutil
import argparse
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import cloudinary
import cloudinary.uploader

from config import (
    CLOUD_NAME, API_KEY, API_SECRET, CLOUDINARY_FOLDER,
    LOCAL_MUSIC_DIR, SONG_NAME_MAPPING_FILE,
    OUTPUT_CATALOG_JSON, OUTPUT_CATALOG_CSV,
    DEFAULT_WORKERS, DEFAULT_RETRY_COUNT
)

# Cấu hình Cloudinary
cloudinary.config(
    cloud_name=CLOUD_NAME,
    api_key=API_KEY,
    api_secret=API_SECRET,
    secure=True
)

file_lock = threading.Lock()

def safe_replace(temp_path: Path, target_path: Path, max_attempts=5):
    """Thay thế file an toàn trên Windows chống lỗi PermissionError khi có indexer/antivirus đọc file"""
    for attempt in range(max_attempts):
        try:
            temp_path.replace(target_path)
            return
        except PermissionError:
            time.sleep(0.15 * (attempt + 1))
        except Exception:
            break
            
    # Fallback dự phòng: sao chép đè nội dung
    try:
        shutil.copyfile(temp_path, target_path)
        if temp_path.exists():
            temp_path.unlink()
    except Exception as e:
        # Nếu file đích đang bị Excel mở khóa độc quyền, ghi ra file phụ để không mất dữ liệu
        backup_path = target_path.with_name(f"{target_path.stem}_backup{target_path.suffix}")
        try:
            shutil.copyfile(temp_path, backup_path)
        except Exception:
            pass

def load_catalog(json_path):
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[WARNING] Không thể đọc catalog cũ ({e}), khởi tạo catalog mới.")
    return {}

def save_catalog(catalog, json_path, csv_path, save_csv=True):
    with file_lock:
        # 1. Lưu JSON
        temp_json = json_path.with_suffix(".json.tmp")
        try:
            with open(temp_json, "w", encoding="utf-8") as f:
                json.dump(catalog, f, indent=2, ensure_ascii=False)
            safe_replace(temp_json, json_path)
        except Exception as e:
            print(f"[WARNING] Lỗi ghi JSON: {e}")

        # 2. Lưu CSV
        if save_csv:
            temp_csv = csv_path.with_suffix(".csv.tmp")
            headers = ["song_id", "title", "file_name", "public_id", "secure_url", "duration", "format", "bytes", "created_at"]
            try:
                with open(temp_csv, "w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=headers)
                    writer.writeheader()
                    for item in sorted(catalog.values(), key=lambda x: x.get("song_id", "")):
                        row = {k: item.get(k, "") for k in headers}
                        writer.writerow(row)
                safe_replace(temp_csv, csv_path)
            except Exception as e:
                # Không bao giờ để lỗi CSV làm gián đoạn upload
                pass

def upload_single_song(song_info, retry_count=DEFAULT_RETRY_COUNT, force=False):
    song_id = song_info["song_id"]
    title = song_info["title"]
    file_path = song_info["file_path"]

    for attempt in range(1, retry_count + 1):
        try:
            res = cloudinary.uploader.upload(
                str(file_path),
                folder=CLOUDINARY_FOLDER,
                public_id=song_id,
                resource_type="video",  # BẮT BUỘC để Cloudinary nhận diện luồng audio streaming
                overwrite=force,
                tags=["pbl6", "qbh_gallery", "original_audio"],
                context={"title": title, "song_id": song_id}
            )

            record = {
                "song_id": song_id,
                "title": title,
                "file_name": file_path.name,
                "public_id": res.get("public_id"),
                "secure_url": res.get("secure_url"),
                "duration": res.get("duration"),
                "format": res.get("format"),
                "bytes": res.get("bytes"),
                "created_at": res.get("created_at")
            }
            return (True, song_id, record, None)

        except Exception as e:
            if attempt < retry_count:
                time.sleep(2 * attempt)
            else:
                return (False, song_id, None, str(e))

def main():
    parser = argparse.ArgumentParser(description="PBL6 - Upload kho nhạc gốc lên Cloudinary")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS, help="Số luồng upload đồng thời")
    parser.add_argument("--limit", type=int, default=None, help="Giới hạn số bài upload (dùng khi test thử)")
    parser.add_argument("--force", action="store_true", help="Ghi đè bài hát nếu đã upload trước đó")
    args = parser.parse_args()

    print("=" * 70)
    print("PBL6 MUSIC WEBSITE - HỆ THỐNG UPLOAD AUDIO LÊN CLOUDINARY")
    print("=" * 70)
    print(f"Cloud Name         : {CLOUD_NAME}")
    print(f"Folder Cloudinary  : {CLOUDINARY_FOLDER}")
    print(f"Thư mục nhạc gốc   : {LOCAL_MUSIC_DIR}")
    print(f"Số luồng (workers) : {args.workers}")
    print("=" * 70)

    if not LOCAL_MUSIC_DIR.exists():
        print(f"[ERROR] Không tìm thấy thư mục nhạc: {LOCAL_MUSIC_DIR}")
        return

    # 1. Đọc mapping tên bài hát
    song_mapping = {}
    if SONG_NAME_MAPPING_FILE.exists():
        with open(SONG_NAME_MAPPING_FILE, "r", encoding="utf-8") as f:
            song_mapping = json.load(f)
        print(f"[INFO] Đã nạp mapping cho {len(song_mapping)} bài hát.")

    # 2. Quét toàn bộ file âm thanh gốc
    all_files = sorted([f for f in LOCAL_MUSIC_DIR.glob("*.mp3")])
    print(f"[INFO] Tìm thấy {len(all_files)} file MP3 trong thư mục gốc.")

    # 3. Đọc catalog hiện có
    catalog = load_catalog(OUTPUT_CATALOG_JSON)
    print(f"[INFO] Catalog hiện tại có {len(catalog)} bài đã upload thành công.")

    # 4. Lọc các bài cần upload
    tasks = []
    for f in all_files:
        song_id = f.stem
        title = song_mapping.get(song_id, song_id)
        if not args.force and song_id in catalog:
            continue
        tasks.append({
            "song_id": song_id,
            "title": title,
            "file_path": f
        })

    if args.limit:
        tasks = tasks[:args.limit]
        print(f"[INFO] Giới hạn test: chỉ upload {len(tasks)} bài.")

    print(f"[INFO] Số bài hát còn lại cần upload: {len(tasks)} bài.")
    if not tasks:
        print("[SUCCESS] Tất cả 300 bài hát đã được upload hoàn tất lên Cloudinary!")
        print(f"Xem danh mục tại : {OUTPUT_CATALOG_JSON}")
        return

    # 5. Tiến hành upload đa luồng
    success_count = 0
    failed_count = 0
    failed_songs = []

    start_time = time.time()
    batch_counter = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(upload_single_song, t, DEFAULT_RETRY_COUNT, args.force): t for t in tasks}

        with tqdm(total=len(tasks), desc="Upload bài hát lên Cloudinary", unit="song") as pbar:
            for future in as_completed(futures):
                task_info = futures[future]
                ok, song_id, record, err = future.result()

                if ok:
                    success_count += 1
                    catalog[song_id] = record
                    batch_counter += 1
                    # Lưu JSON liên tục, lưu CSV mỗi 5 bài
                    save_catalog(catalog, OUTPUT_CATALOG_JSON, OUTPUT_CATALOG_CSV, save_csv=(batch_counter % 5 == 0))
                    pbar.set_postfix({"Vừa xong": song_id, "Tổng xong": len(catalog)})
                else:
                    failed_count += 1
                    failed_songs.append((song_id, err))
                    print(f"\n[FAIL] {song_id} ({task_info['title']}): {err}")

                pbar.update(1)

    # Lưu bản cuối cùng đầy đủ cả JSON và CSV
    save_catalog(catalog, OUTPUT_CATALOG_JSON, OUTPUT_CATALOG_CSV, save_csv=True)

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("KẾT QUẢ UPLOAD")
    print("=" * 70)
    print(f"Số bài upload mới thành công : {success_count}/{len(tasks)}")
    print(f"Số bài thất bại              : {failed_count}")
    print(f"Thời gian thực thi           : {elapsed / 60:.2f} phút")
    print(f"Tổng số bài trong Catalog    : {len(catalog)}/300 bài")
    print(f"File Catalog JSON            : {OUTPUT_CATALOG_JSON}")
    print(f"File Catalog CSV             : {OUTPUT_CATALOG_CSV}")

    if failed_songs:
        print("\n[DANH SÁCH BÀI LỖI]:")
        for sid, err in failed_songs:
            print(f"  - {sid}: {err}")
    else:
        print("\n[CHÚC MỪNG] Tất cả các bài hát đã được đồng bộ lên Cloudinary an toàn 100%!")

if __name__ == "__main__":
    main()
