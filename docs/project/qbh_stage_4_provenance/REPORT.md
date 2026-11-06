# Bước 4 — Provenance checkpoint và kiểm tra rò rỉ split

**Trạng thái:** Hoàn tất audit provenance. Đã khôi phục mapping train ID→bài hát từ artifact do người dùng cung cấp; đồng thời phát hiện benchmark 773 query hiện tại trộn query thuộc cả train/validation/test của run gốc. Dừng tại Bước 4, chưa train và chưa chạy đánh giá mới.

## Chuỗi provenance đã xác minh

- `best.zip` SHA256: `94943558a9e4368ca2d4a1b02fde29541255477e2fdad08514ce41586026589b`. Hash này **khớp chính xác** trường `source_checkpoint_sha256` trong `core/checkpoints/qbh_v3_inference.pt`. Checkpoint inference và `best.zip` cùng epoch 26; `best_val_mrr=0.353322126351`.
- `last.zip` SHA256: `ecbeb0e200fb21a2829e7533e2fe2d0d70d89dca0b65e946fdc6cb1bc04c00b5`; epoch 30. `best.zip`, `last.zip`, contract người dùng gửi và contract trong checkpoint inference có cùng contract. Hai checkpoint được đọc với `weights_only=True` và allowlist giới hạn cho numpy RNG; không dùng pickle không an toàn.
- `dataset_fingerprint` và `split_fingerprint` trong contract lần lượt khớp `72c06606f969a1e3cd1cba86b12ee2a0b0f151530c3e6cc938cd93722318e99e` và `4d3b1cd647b4025f6477731f207611bc5c09e2fc8990b716cc623c86a32640b3` trong `fingerprints.json`.
- `splits.json`: 210 train / 45 val / 45 test. `manifest.csv`: 300 unique `song_id`, 300 tên bài duy nhất, không có song ID đổi tên hoặc split giữa các hàng. Set 210 train ID trùng contract; class index 0–209 khớp thứ tự ID đã sắp xếp.
- Mapping đầy đủ được lưu tại [`source_split_mapping.csv`](source_split_mapping.csv). Hash và số dòng của artifact đầu vào có trong JSON audit.

## Phát hiện về benchmark hiện tại

Đối chiếu **SHA256 của audio query** trong manifest benchmark với `humming_sha256` trong manifest run gốc cho kết quả **773/773 query khớp byte chính xác**. Phân bố theo split gốc:

| Split trong run tạo checkpoint | Query của benchmark hiện tại | Số bài trong 45 bài validation hiện tại |
|---|---:|---:|
| Train | 428 | 27 bài train gốc nằm trong validation hiện tại |
| Validation | 158 | 9 |
| Test | 187 | 9 |

Phân hoạch benchmark hiện tại khác đáng kể với split của checkpoint. Kết luận và số đếm ở trên đều dựa trên hash nội dung query khớp chính xác với từng dòng manifest gốc.

**Hệ quả:** metric 773 query hiện có vẫn hợp lệ cho kiểm tra kỹ thuật/parity trên tập cố định, nhưng **không phải** kết quả validation độc lập trên bài chưa thấy và không nên dùng để chọn cấu hình như một holdout song-disjoint. Có **187 query thuộc source-test** đã được xử lý trong benchmark trước đây vì lúc đó chúng nằm dưới thư mục `val` của split hiện tại. Đây là phát hiện hồi cứu từ metadata/hash; trong audit này không mở audio. Không tái sử dụng 187 query đó làm test cuối.

## Giới hạn và bước tiếp theo

- Provenance của checkpoint và mapping 210 train ID đã được giải quyết.
- Preflight Bước 5 sau đó rà log theo hash, loại 9 bài original-test đã xuất hiện trong benchmark và giữ candidate 36 bài còn lại (615 query file hash-khớp; 3 file thiếu). Chỉ dùng các bài này nếu kiểm tra log ngoài workspace cũng xác nhận chưa sử dụng. Chi tiết ở `docs/project/qbh_stage_5_holdout_preflight/REPORT.md`.
- Giữ nguyên checkpoint và benchmark hiện có; không train/test trong audit Bước 4.
- Không thay checkpoint, frozen benchmark manifest hay số đo. Không mở test audio trong audit.

## Artifact

- Báo cáo này: `docs/project/qbh_stage_4_provenance/REPORT.md`.
- Bảng mapping 300 bài: `docs/project/qbh_stage_4_provenance/source_split_mapping.csv`.
- Audit JSON: `core/.cache/benchmarks/v3_val_full_rss_v2/checkpoint_provenance_audit.json`.
- Benchmark manifest chữ ký `a36f0090dc6f68766de68d9ffedd69c7fd6f75fe03a7bba2ba8c6d6432181407` được giữ nguyên.
