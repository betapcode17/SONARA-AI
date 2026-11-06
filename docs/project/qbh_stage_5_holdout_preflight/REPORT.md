# Holdout sạch — preflight trước lần train/đánh giá cuối

**Trạng thái:** Preflight holdout hoàn tất; dừng trước train và model evaluation theo quy trình one-time.

## Kết quả khóa test candidate

Run gốc có 45 bài test, 809 query. Đã đối chiếu SHA256 với benchmark outputs và worker logs trong workspace. **187 query hashes** đã xuất hiện trước, thuộc **9 bài**; loại toàn bộ 9 bài để giữ song-disjoint. Không hash humming query nào trùng giữa các split nguồn theo manifest.

| Tập | Bài | Query | Tình trạng file trong workspace |
|---|---:|---:|---|
| Original test bị loại theo song vì có query đã xuất hiện trong benchmark | 9 | 191 query rows tổng; 187 query hashes đã dùng trước | Không dùng lại cả 9 bài |
| Test candidate sạch theo split/hash | **36** | **615 khóa vào test cuối** | 615 file có mặt và SHA256 khớp; 3 query rows thiếu được loại khỏi manifest khóa; 0 hash mismatch |

Mỗi bài trong 36 candidate có ít nhất một query file đã hash khớp. Đã tìm thấy full-song MP3 gallery cho **36/36** bài candidate (đối chiếu tên theo mapping; chưa đọc/hash các file gallery). File manifest test ghi expected/actual SHA và trạng thái từng query.

## File cần review

- [`clean_test_song_manifest.csv`](clean_test_song_manifest.csv): 36 bài, số query sạch/thiếu và đường dẫn full-song gallery.
- [`locked_final_test_manifest.csv`](locked_final_test_manifest.csv): manifest đã khóa, đúng 615 query hash-verified trên 36 bài.
- [`clean_test_query_manifest.csv`](clean_test_query_manifest.csv): inventory 618 rows; 615 `verified_sha256_match`, 3 `missing_file`.
- [`excluded_missing_queries.csv`](excluded_missing_queries.csv): 3 query rows bị loại vì file không có trong workspace. Không thêm lại sau khi bắt đầu model selection.
- [`clean_test_gallery_manifest.csv`](clean_test_gallery_manifest.csv): full-song path cho 36 candidate.
- [`excluded_test_songs.csv`](excluded_test_songs.csv): 9 bài và lý do loại.
- [`preflight_summary.json`](preflight_summary.json): số liệu kiểm tra và artifact nguồn.

## Điều kiện trước khi train/evaluate

- Lần train tới phải dùng **split gốc của checkpoint**: 210 train, 45 validation, 45 test. Split hiện tại trong `dataset_split/` đã bị xáo khác với run gốc; không dùng trực tiếp để train. Dùng mapping đã lưu ở `docs/project/qbh_stage_4_provenance/source_split_mapping.csv` để dựng lại phân hoạch đúng theo song ID.
- Ba query file thiếu (`1b474bae1557a0a6`, `c8f211bf97d38ddb`, `e8a16fee7f810f17`) đã được loại khỏi manifest test khóa. Cả 36 bài còn ít nhất một query đã xác minh.
- Giữ 36 bài candidate ngoài train và model selection. Sau khi checkpoint cuối được chọn trên validation, chạy một lần test đúng 615 query đã khóa. Ba query thiếu đã loại; không thêm sau khi model selection bắt đầu.
- Các benchmark/log remote ngoài workspace chưa kiểm tra; trước lần test cuối cần bảo đảm không có artifact khác đã dùng 36 bài này.

Đã tính SHA256 bằng cách đọc byte theo luồng; không decode, nghe, tạo feature/embedding, inference hoặc tính metric. **Chưa train và chưa chạy evaluation.**

## Cập nhật khi dựng lại dataset

Đã tạo một archive riêng theo split nguồn, không ghi đè `02_stage2_humming_qbh/datasets.zip`. Tuy nhiên archive nguồn chỉ chứa **5.148/5.187** query WAV trong manifest: thiếu 29 train, 3 validation và 7 test. 615 query holdout vẫn đủ và khớp SHA256. Vì thiếu query train/validation, archive dựng lại được đánh dấu **chưa sẵn sàng cho lần train cuối**; xem [`REBUILT_DATASET_REPORT.md`](REBUILT_DATASET_REPORT.md) và [`missing_source_queries.csv`](missing_source_queries.csv) để biết artifact và các file cần phục hồi.
