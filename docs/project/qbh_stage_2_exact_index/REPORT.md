# Bước 2 — Exact scorer và index tăng dần

**Trạng thái: hoàn tất phần triển khai và kiểm tra parity trên fixture; dừng tại đây.**

## Thay đổi

- `core/engine.py` thêm `ExactEmbeddingIndex`, giữ chunk embedding float32 trong vùng nhớ liên tục, đánh dấu slice theo song ID, kiểm tra shape/NaN/Inf và append với capacity tăng gấp đôi.
- Rebuild thêm từng bài trong lúc đọc cache/tính feature. Nếu lần reindex chỉ thêm bài và mọi nguồn cũ còn nguyên hash, nó giữ index đang có và append riêng bài mới. Nếu bài bị xóa/đổi hash, nó dựng index lại để loại dữ liệu cũ. Search tính cosine query với toàn ma trận gallery một lần, rồi áp dụng scorer checkpoint gốc trong slice từng bài.
- Vocal embedding cũng dùng index ma trận khi vocal đã sẵn cho toàn thư viện. Gallery vocal một phần tiếp tục dùng đường per-song để không đổi điều kiện cộng nhánh.
- Do BLAS có thể làm tròn khác nhau vài ULP khi đổi shape GEMM, các nhóm điểm embedding/vocal cách nhau tối đa `1e-5` được tính lại bằng shape per-song cũ trước khi xếp hạng. Mục đích là ổn định tie/shortlist; thuật toán cosine và MaxMean/BPWR không đổi.
- Bản gốc `engine.py` được giữ tại `core/.cache/engine.py.pre_stage2_20261025.bak`, SHA256 `92754221EC7832B1E7F949896CC25E8787E80290225D75BBFFF48BFA87863ACA`.

## Kiểm tra

- Lệnh: `core/.venv/Scripts/python.exe -X utf8 -m unittest discover -s tests -p test_benchmark.py -v`.
- Kết quả: **17/17 tests pass**. Kiểm tra append, shape/finite validation, score tolerance đến `2e-7`, điều kiện dùng lại index khi chỉ thêm bài, parity thứ hạng giữa per-song và ma trận, bốn mode production, cùng shortlist Cascade.
- Microbenchmark NumPy tổng hợp với `OPENBLAS_NUM_THREADS=1`, query 6 × 128, mỗi bài 12 × 128, sau warmup:

| Gallery synthetic | Legacy per-song (ms/query) | Matrix (ms/query) | Tỉ lệ legacy/matrix |
|---:|---:|---:|---:|
| 300 | 3.136 | 1.880 | 1.67× |
| 1.000 | 11.858 | 11.924 | 0.99× |
| 10.000 | 404.731 | 217.467 | 1.86× |

Các con số trên chỉ là microbenchmark trên vector tổng hợp; không đại diện thời gian server, audio thật hay metric QbH.

## Giới hạn và điểm dừng

- Tại thời điểm lập báo cáo ban đầu, chưa chạy lại toàn bộ 773 query validation bằng engine đã sửa. Phép đối chiếu đầy đủ sau đó được ghi riêng trong [`PARITY_773.md`](PARITY_773.md); kết quả chỉ xác nhận parity của embedding scorer/shortlist, không phải end-to-end latency.
- Kết quả bước 1, manifest, checkpoint và dữ liệu không đổi. Run v2 vẫn là baseline đo trước thay đổi này.
- Tại thời điểm báo cáo này, provenance của 210 train IDs chưa được xác minh; audit Bước 4 sau đó khôi phục mapping và phát hiện benchmark query bị trộn split gốc. Xem `docs/project/qbh_stage_4_provenance/REPORT.md` trước khi diễn giải metric.

## Bổ sung: full validation scorer parity

- Ngày 25/10/2026, scorer cũ/mới được so sánh trên cùng 773 query và gallery 90/300. Toàn thứ hạng và Cascade Top‑20 khớp **773/773** ở cả hai gallery; sai khác score tối đa `1.1921e-7`.
- Gallery 300: scorer mới giảm p50 30.87%, p95 15.81% và tổng thời gian chấm 24.03%. Gallery 90: p50 giảm 27.35%, còn p95 tăng 26.02%.
- Phép đo không chạy lại DTW và không thay checkpoint. Audit provenance về sau xác định 187/773 query khớp original-test audio; điều này không làm mất giá trị parity kỹ thuật giữa scorer cũ/mới, nhưng không cho phép xem các query đó là test độc lập. Chi tiết từng query ở [`PARITY_773.md`](PARITY_773.md), split audit ở `docs/project/qbh_stage_4_provenance/REPORT.md`.
