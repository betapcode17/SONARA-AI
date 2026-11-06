# Bước 3 — Cascade shortlist và alignment theo cửa sổ

**Trạng thái:** Hoàn tất phép đo 773 query trên gallery 90/300; đã tích hợp Cascade window vào production và đối chiếu lại đủ 773 query mỗi gallery. Dừng tại Bước 3 theo yêu cầu làm từng bước.

## Phạm vi đóng băng

- Manifest validation: `a36f0090dc6f68766de68d9ffedd69c7fd6f75fe03a7bba2ba8c6d6432181407`; 773/773 query ở mỗi gallery.
- Checkpoint SHA256: `85d76048ba92cd1353036851f88b8c056f560a281949c5e8aa157ddc75aa47a9`.
- Cùng recordings, gallery lồng nhau 90/300, cùng embedding scorer và cùng local DTW. Không mở test audio, không train, không đổi checkpoint.
- Vocal tắt theo benchmark Bước 1. Kết quả được chọn trên validation để nghiên cứu; `selection_eligible=false`. Audit Bước 4 sau đó khôi phục train-ID mapping và xác nhận benchmark query bị trộn original train/val/test.

## Shortlist sweep

Tính candidate recall theo thứ hạng embedding; sau đó rerank bằng DTW cho shortlist K. Mỗi candidate chỉ căn chỉnh một lần và kết quả được dùng lại cho các K lớn hơn. Mốc K=100 ở gallery 90 tương đương tất cả 90 bài.

| Gallery | K | Candidate Recall | MRR với window | Recall@1 với window |
|---:|---:|---:|---:|---:|
| 90 | 20 | 80.60% | 0.3437 | 22.12% |
| 90 | 40 | 92.24% | 0.3037 | 18.89% |
| 90 | 60 | 96.77% | 0.2732 | 16.17% |
| 90 | 90 | 100.00% | 0.2483 | 14.75% |
| 300 | 20 | 58.60% | 0.2762 | 19.15% |
| 300 | 40 | 71.41% | 0.2728 | 19.40% |
| 300 | 60 | 79.04% | 0.2605 | 18.24% |
| 300 | 100 | 87.97% | 0.2387 | 16.04% |

K lớn tăng cơ hội đưa bài đúng vào candidate set nhưng làm giảm MRR và phần lớn Recall@K sau DTW. K=40 tăng Recall@1 gallery 300 khoảng 0.25 điểm phần trăm so với window K=20, nhưng MRR giảm 0.0034, Recall@5 giảm 2.85 điểm và Recall@10 giảm 3.63 điểm. Vì mục tiêu chính là thứ hạng toàn phần, production giữ shortlist **K=20**; không tăng K chỉ để làm candidate recall đẹp hơn.

## Full-song DTW so với window DTW tại K=20

Window lấy offset từ cặp chunk embedding cosine cao nhất, cắt reference từ `offset - 20s` tới `offset + 2 × query_duration + 20s`, rồi gọi cùng `align_melody`. Nếu cửa sổ không đủ frame hoặc alignment không khả dụng thì chạy lại full-song DTW.

| Gallery | Cách align | MRR | Recall@1 | Recall@5 | Recall@10 | Candidate Recall |
|---:|---|---:|---:|---:|---:|---:|
| 90 | Full-song | 0.3386 | 21.60% | 46.96% | 61.71% | 80.60% |
| 90 | Window 20s | 0.3437 | 22.12% | 47.74% | 62.61% | 80.60% |
| 300 | Full-song | 0.2734 | 19.28% | 35.45% | 46.05% | 58.60% |
| 300 | Window 20s | 0.2762 | 19.15% | 37.52% | 47.87% | 58.60% |

Window cải thiện MRR **+0.0050** ở gallery 90 và **+0.0028** ở gallery 300. Recall@1 gallery 300 giảm nhẹ **0.13 điểm phần trăm**, trong khi Recall@5 tăng **2.07 điểm** và Recall@10 tăng **1.81 điểm**. Candidate Recall không đổi vì shortlist vẫn là cùng Top‑20. Không query nào cần fallback trong 773 query/gallery.

## Đối chiếu production đủ 773 query

Sau khi tích hợp vào `core/engine.py`, gọi `DemoEngine.search_wave` cho từng query với feature đã cache. Danh sách và thứ tự Cascade Top‑20 khớp kết quả window offline **773/773** ở cả hai gallery; metrics khớp chính xác. Đây là đo CPU, đã precompute query embedding và query melody, do đó thời gian sau đây là scorer + alignment search, chưa gồm decode, model encode, CQT/pYIN.

| Gallery | Search p50 | Search p95 | Peak RSS toàn lượt |
|---:|---:|---:|---:|
| 90 | 0.435 s | 0.761 s | 408.2 MiB |
| 300 | 0.488 s | 0.798 s | 408.2 MiB |

Peak RSS là của toàn tiến trình chứa model, runtime và gallery; không phải RAM riêng của cửa sổ. Không kết luận latency end-to-end vì query feature extraction đã được cache ở lượt kiểm tra production. Để tham khảo, Stage 1 full-song Cascade có alignment p50/p95 là 0.774/1.711 s (gallery 90) và 0.799/2.234 s (gallery 300); đây là component alignment riêng nên không so trực tiếp như một end-to-end paired timing.

## Thay đổi code

- `core/engine.py`: thêm `cascade_alignment_window`, tính offset từ cặp chunk embedding tốt nhất, giới hạn alignment chỉ ở mode Cascade; mode Melody/Hybrid giữ nguyên full-song alignment.
- Cửa sổ 20 giây có fallback về full-song khi thiếu frame hoặc alignment không khả dụng. Số fallback đo được: **0/773** mỗi gallery.
- `CASCADE_CANDIDATES` giữ ở 20; chưa đổi thứ tự hoặc thuật toán scorer.
- Runner và workers có thể chạy lại: `core/benchmarks/stage3_cascade.py`, `core/benchmarks/stage3_cascade_worker.py`, `core/benchmarks/stage3_production_check.py`.

## Artifacts và giới hạn

- Sweep 16 tổ hợp gallery/K/alignment: `core/.cache/benchmarks/stage3_cascade_v1/summary.json` và `per_query.csv`.
- Đối chiếu production: `core/.cache/benchmarks/stage3_production_check/summary.json` và `per_query.csv`.
- Audit Bước 4 sau đó xác minh mapping: 428/773 query benchmark thuộc original train và 187/773 thuộc original test theo SHA256 manifest. Kết quả Cascade vẫn là parity/engineering benchmark trên tập cố định, không phải đánh giá tổng quát hóa độc lập. Trong audit Bước 4 không mở audio; không tái sử dụng original-test queries làm test cuối.
- Gallery 1.000/10.000 chưa có dữ liệu phù hợp; không suy diễn kết quả mở rộng từ hai gallery hiện tại.
