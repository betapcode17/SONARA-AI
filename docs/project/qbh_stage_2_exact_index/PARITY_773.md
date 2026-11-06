# Đối chiếu scorer cũ/mới trên 773 query

**Ngày chạy:** 25/10/2026  
**Trạng thái:** Hoàn tất cho embedding scorer và Cascade shortlist trên gallery 90/300.

## Phạm vi và tính công bằng

- Cùng manifest validation `a36f0090dc6f68766de68d9ffedd69c7fd6f75fe03a7bba2ba8c6d6432181407`, đủ 773 query, cùng ground truth và cùng gallery bản ghi thật ở 90/300.
- Checkpoint giống nhau: SHA256 `85d76048ba92cd1353036851f88b8c056f560a281949c5e8aa157ddc75aa47a9`.
- Scorer cũ lấy phép tính per-song từ bản lưu trước Stage 2: `core/.cache/engine.py.pre_stage2_20261025.bak` (SHA256 `92754221EC7832B1E7F949896CC25E8787E80290225D75BBFFF48BFA87863ACA`). Scorer mới dùng phép nhân ma trận toàn gallery và quy tắc fallback gần hòa như `core/engine.py` hiện tại.
- Mỗi query được decode/tiền xử lý và encode đúng một lần. Cùng vector query được đưa qua scorer cũ và scorer mới; vì vậy thời gian scorer không tính thời gian encode. Gallery embeddings được đọc từ feature cache đã đóng băng trong benchmark. Index matrix được dựng trước khi bấm giờ.
- CPU, NumPy/BLAS và môi trường giống runtime benchmark Stage 1. Không chạy DTW trong phép đo này; Cascade shortlist là 20 bài đầu theo embedding score.

## Kết quả parity

| Gallery | Query | Query có cùng rank bài đúng | Toàn thứ hạng giống hệt | Cascade Top‑20 giống hệt | Max chênh điểm mọi bài | Chênh điểm bài đúng: max / trung bình |
|---:|---:|---:|---:|---:|---:|---:|
| 90 | 773 | 773/773 | 773/773 | 773/773 | `1.1921e-7` | `5.9605e-8` / `8.0964e-10` |
| 300 | 773 | 773/773 | 773/773 | 773/773 | `1.1921e-7` | `5.9605e-8` / `8.0964e-10` |

Không có query nào đổi thứ hạng bài đúng. Không có query nào đổi thứ tự toàn gallery hoặc thành viên/thứ tự Cascade Top‑20. Số query có bài đúng nằm trong Top‑20 Cascade là 623/773 ở gallery 90 và 453/773 ở gallery 300; hai scorer cho đúng cùng tập ứng viên nên các số này không đổi. Do danh sách ứng viên không đổi, đầu vào cho bước DTW của Cascade cũng không đổi; phép đo này không chạy lại DTW hay đo thời gian DTW.

## Thời gian scorer embedding

Thời gian gồm chấm similarity và local reduction cho toàn bộ gallery, kể cả tính lại các nhóm score sát nhau; không gồm encode query, dựng index, DTW hoặc HTTP.

| Gallery | Scorer | p50 ms/query | p95 ms/query | Tổng 773 query (s) |
|---:|---|---:|---:|---:|
| 90 | Cũ | 4.102 | 7.707 | 3.484 |
| 90 | Mới | 2.980 | 9.712 | 3.055 |
| 300 | Cũ | 13.721 | 28.666 | 11.621 |
| 300 | Mới | 9.485 | 24.134 | 8.829 |

Ở gallery 90, scorer mới giảm p50 27.35% và tổng thời gian 12.30%, nhưng p95 tăng 26.02%; do đó chưa thể nói mọi truy vấn đều nhanh hơn ở gallery nhỏ. Ở gallery 300, p50 giảm 30.87%, p95 giảm 15.81%, tổng thời gian giảm 24.03%.

Encode query (dùng chung cho hai scorer): p50 **0.340 s/query**, p95 **0.777 s/query**, trung bình **0.387 s/query**. Encode lâu hơn phần scorer nhiều lần; tối ưu scorer không làm đổi thời gian encode. Các số này là CPU local, không đại diện latency GPU/server.

## Kết luận

Trên tập đo đầy đủ này, scorer ma trận giữ nguyên điểm xếp hạng thực tế: sai khác float32 tối đa `1.2e-7`, nhưng thứ hạng toàn gallery và Cascade shortlist Top‑20 khớp 100% ở cả hai kích thước. Tốc độ scorer tốt hơn rõ ở gallery 300. Với gallery 90 có p95 regression nên cần giữ lại theo dõi tail latency. Đây là parity và latency của embedding scorer; không phải một lần chạy lại toàn bộ bốn chế độ/DTW hay đánh giá chất lượng trên dữ liệu mới.

## Artifact

- Từng query × gallery, score bài đúng, rank trước/sau, cờ rank parity, danh sách Top‑20 và thời gian scorer: `core/.cache/benchmarks/scorer_parity_773/per_query.csv`.
- Số liệu tổng: `core/.cache/benchmarks/scorer_parity_773/summary.json`.
- Runner có thể tái lập phép so sánh: `core/.cache/compare_qbh_scorers.py`.
