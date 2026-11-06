# Thực hiện cải thiện QbH theo từng bước

Yêu cầu ngày 25/10/2026: thực hiện theo thứ tự; xong mỗi bước phải dừng và báo cáo. Chỉ bắt đầu bước kế tiếp khi người dùng yêu cầu tiếp tục. Không tự chuyển từ benchmark sang tối ưu hoặc train.

| Bước | Công việc | Trạng thái |
|---|---|---|
| 1 | Hoàn tất benchmark đủ 773 query × 4 chế độ × gallery 90/300 và phân tích lỗi | Hoàn tất; báo cáo đã tạo |
| 2 | Scorer ma trận exact và index tăng dần, kiểm tra parity | Hoàn tất; full validation scorer parity 773 query đã đo |
| 3 | Candidate shortlist và căn chỉnh theo cửa sổ thời gian của Cascade | Hoàn tất; K=20 giữ nguyên, window DTW tích hợp và xác nhận 773 query/gallery |
| 4 | Hoàn thiện provenance checkpoint, đối chiếu train IDs/validation | Hoàn tất; mapping được xác minh, phát hiện benchmark split bị trộn |
| 5 | Chốt dữ liệu/cấu hình và fine-tune trên Kaggle T4 khi có đủ căn cứ | Chưa bắt đầu |

## Bước 2 — Hoàn tất

- `core/engine.py` dùng `ExactEmbeddingIndex`: ma trận float32 liên tục, slice riêng mỗi bài, append với tăng dung lượng theo lũy thừa hai; embedding query chấm toàn gallery bằng một phép nhân ma trận chính xác.
- MaxMean/BPWR, giới hạn cosine, candidate order và tie-break theo song ID được giữ nguyên. Nhóm score cách nhau không quá `1e-5` được tính lại bằng hình dạng per-song cũ để hạn chế khác biệt ULP của BLAS ở trường hợp sát hòa.
- Bộ kiểm thử parity: 17/17 pass, gồm so score và thứ hạng old-vs-matrix trên fixture, append/reindex guard, thứ hạng embedding/hybrid và shortlist Cascade cho cả bốn chế độ.
- Microbenchmark NumPy synthetic, BLAS một luồng, 6 query chunks × 128 chiều, 12 reference chunks/bài: gallery 300 là 3.136 ms legacy / 1.880 ms matrix (1.67×); 1.000 là 11.858 / 11.924 ms (0.99×); 10.000 là 404.731 / 217.467 ms (1.86×). Đây là phép đo tổng hợp, không phải thời gian demo hay độ chính xác trên query thật.
- Full validation scorer parity sau đó hoàn tất: toàn thứ hạng và Cascade Top‑20 khớp 773/773 cho gallery 90/300. Gallery 300 giảm scorer p50 30.87%, p95 15.81%; gallery 90 giảm p50 27.35% nhưng p95 tăng 26.02%. Chi tiết: `docs/project/qbh_stage_2_exact_index/PARITY_773.md`.
- Đây là phép đo embedding scorer/shortlist, chưa đo DTW hoặc latency end-to-end. Giữ checkpoint, gallery và benchmark bước 1 nguyên trạng.
- Đầu ra: `docs/project/qbh_stage_2_exact_index/REPORT.md`. Dừng sau bước này theo yêu cầu; bước 3 chưa bắt đầu.

## Bước 3 — Hoàn tất

- Candidate Recall sweep K=20/40/60/100 (gallery 90/300). K lớn nâng candidate recall nhưng giảm MRR/Recall sau DTW; giữ production K=20.
- Thêm cửa sổ alignment Cascade 20s quanh offset chunk embedding tốt nhất; fallback full-song nếu cửa sổ không hợp lệ/không đủ pitch. Melody/Hybrid vẫn full-song.
- Offline sweep xong đủ 773 query mỗi gallery. Production `search_wave` khớp thứ tự Top‑20 và metrics 773/773 với kết quả offline ở cả hai gallery; không query nào fallback.
- Window K=20 MRR: gallery 90 0.3437, gallery 300 0.2762. Search p50/p95 đã cache query features: 0.435/0.761s và 0.488/0.798s; peak RSS 408.2 MiB toàn tiến trình. Không gọi đây là end-to-end vì decode/encode/CQT/pYIN không nằm trong timing production check.
- Chi tiết: `docs/project/qbh_stage_3_cascade/REPORT.md`; dữ liệu và runner trong `core/.cache/benchmarks/stage3_cascade_v1/`, `core/.cache/benchmarks/stage3_production_check/`, `core/benchmarks/stage3_cascade.py`.
- Tại thời điểm chạy Bước 3, provenance chưa xác minh. Audit Bước 4 sau đó khôi phục mapping và phát hiện 187/773 query trong benchmark khớp source-test; xem báo cáo Bước 4 để biết giới hạn cập nhật. Checkpoint không đổi.

## Bước 4 — Provenance checkpoint và kiểm tra split

- `best.zip` SHA256 khớp `source_checkpoint_sha256` của checkpoint inference; epoch và contract cũng khớp. Contract/split/fingerprint được đối chiếu với các artifact người dùng gửi.
- Khôi phục mapping đủ 210 train ID→tên bài; mapping kiểm tra được lưu ở `docs/project/qbh_stage_4_provenance/source_split_mapping.csv`.
- Đối chiếu SHA256 query theo từng file: 773/773 query của benchmark khớp manifest gốc; 428 thuộc original train, 158 original val, 187 original test. Vì vậy benchmark validation hiện tại không phải holdout song-disjoint; không dùng các metric này để kết luận tổng quát hóa trên bài chưa thấy.
- 187 query gốc-test đã được xử lý trong benchmark trước đây do split hiện tại gán chúng vào val. Audit Bước 4 chỉ đọc metadata/hash; không mở audio. Không tái sử dụng các query đó làm test cuối.
- Báo cáo: `docs/project/qbh_stage_4_provenance/REPORT.md`; audit: `core/.cache/benchmarks/v3_val_full_rss_v2/checkpoint_provenance_audit.json`.
- Bước 4 hoàn tất. Dừng tại đây; chưa train hoặc đánh giá mới. Preflight Bước 5 sau đó giữ lại 36 bài original-test không thấy trong benchmark cũ; 615 query hash đã xác minh và 3 query file còn thiếu.

## Điều kiện hoàn tất bước 1

- Run: `core/.cache/benchmarks/v3_val_full_rss_v2`.
- Dùng đúng manifest `a36f0090dc6f68766de68d9ffedd69c7fd6f75fe03a7bba2ba8c6d6432181407`.
- Đủ 773 query ở từng case, tổng 6.184 hàng kết quả; query ID/ground truth/SHA256 giống nhau giữa các case.
- Sampler `cached_psapi_reader_v2`; không dùng số đo RAM/latency của run v1.
- Worker ghi `complete_available_sizes` và đã nhả lock.
- Xuất bảng MRR/Recall@1/5/10, p50/p95/RSS; delta 90→300 và CI95; danh sách nhầm bài, mất bài ở shortlist, query thiếu pitch.
- 1.000/10.000 thiếu dữ liệu thật được ghi rõ, không tạo metric giả và không chặn việc báo cáo hai mốc có dữ liệu.
- Provenance train IDs chưa giải quyết phải được giữ là giới hạn; hoàn tất benchmark không tự biến nó thành validation trên bài chưa từng thấy hoặc đủ điều kiện chọn cấu hình.

Sau khi benchmark hoàn thành:

```powershell
cd E:\HOCKY7\PBL6\code\core
.\.venv\Scripts\python.exe -X utf8 benchmarks/analyze.py
```

Script chỉ đọc kết quả, không inference/train và không sửa code của worker. Nếu chưa xong, script trả `waiting_for_benchmark` và không xuất báo cáo hoàn tất.

Đầu ra bước 1: `docs/project/qbh_stage_1_benchmark/REPORT.md`, `stage_1_summary.json`, `query_diagnostics.csv`, `per_song.csv`, `confusions.csv`.

Sau khi xuất báo cáo, dừng ở bước 1. Bước 2 không được tự khởi chạy bởi job hoặc automation theo dõi.

## Theo dõi hoàn tất

Heartbeat của chat kiểm tra mỗi 30 phút, automation ID `ho-n-t-t-v-b-o-c-o-b-c-1-qbh`. Chỉ theo dõi và tổng hợp bước 1. Khi đã gửi báo cáo hoàn tất, tạm dừng automation; không tự chuyển bước. Những lượt kiểm tra job vẫn chạy bình thường không lặp lại báo cáo chưa hoàn tất.


## Bước 5 — Preflight holdout trước lần train cuối

- Rà soát manifest, per-query CSV và worker logs trong workspace theo hash. Loại cả 9 bài original-test có 187 query hashes từng xuất hiện trong benchmark cũ.
- Còn 36 bài/618 query rows; 615 query file đã hash khớp manifest nguồn, 3 file không có trong workspace và đã loại khỏi manifest test khóa. Full-song gallery MP3 có path cho cả 36 bài candidate.
- Candidate manifests và exclusion ledger: `docs/project/qbh_stage_5_holdout_preflight/`.
- Chưa train hoặc evaluate. Manifest test khóa gồm 615 query/36 bài; 3 query file thiếu đã loại và không được thêm lại sau khi model selection bắt đầu. Final run cần dựng lại split gốc 210/45/45 (split hiện tại khác) và xác nhận không có log remote đã dùng candidate.
- Đã tạo `02_stage2_humming_qbh/dist/qbh_v5_source_split_rebuilt_incomplete.zip`, sắp xếp 300 thư mục bài theo song ID/audio ID từ mapping nguồn và giữ manifest 615 query trong split test. Archive nguồn thiếu thêm 39 query so với manifest gốc: 29 train, 3 validation, 7 test; 615 holdout vẫn đủ và SHA256 khớp. Vì thiếu 32 query train/validation, gói được đánh dấu **chưa sẵn sàng cho train cuối**. Danh sách cần phục hồi: `docs/project/qbh_stage_5_holdout_preflight/missing_source_queries.csv`; báo cáo: `docs/project/qbh_stage_5_holdout_preflight/REBUILT_DATASET_REPORT.md`.

