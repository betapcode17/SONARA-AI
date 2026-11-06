# Bước 1 — Benchmark và phân tích kết quả

**Đã hoàn tất các mốc có dữ liệu. Dừng tại đây, chờ người dùng yêu cầu bước tiếp theo.**

Cùng 773 query validation, checkpoint và full recordings ở mọi mốc. Gallery 90 bài có kiểm soát chứa đủ 45 bài validation, khác thư viện demo.

| Gallery | Chế độ | MRR | R@1 | R@5 | R@10 | p50 (s) | p95 (s) | RSS peak MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 90 | embedding | 0.3993 | 26.78% | 52.91% | 67.40% | 0.227 | 0.691 | 303.0 |
| 90 | melody | 0.2569 | 15.78% | 33.89% | 46.44% | 4.711 | 9.648 | 328.4 |
| 90 | hybrid | 0.4365 | 31.37% | 57.70% | 68.95% | 4.729 | 9.448 | 351.5 |
| 90 | cascade | 0.3386 | 21.60% | 46.96% | 61.71% | 2.256 | 4.796 | 332.2 |
| 300 | embedding | 0.2532 | 16.04% | 33.89% | 45.54% | 0.240 | 1.001 | 286.7 |
| 300 | melody | 0.1884 | 12.42% | 22.38% | 31.05% | 12.328 | 34.233 | 319.9 |
| 300 | hybrid | 0.3270 | 22.83% | 41.91% | 50.71% | 12.268 | 34.994 | 315.6 |
| 300 | cascade | 0.2734 | 19.28% | 35.45% | 46.05% | 2.215 | 6.539 | 319.9 |

## Thay đổi khi thêm distractor

| Mốc | Chế độ | Delta MRR | Delta R@1 (điểm %) | CI95 delta R@1 |
|---|---|---:|---:|---|
| 90 → 300 | embedding | -0.1461 | -10.74 | [-13.24, -7.91] |
| 90 → 300 | melody | -0.0685 | -3.36 | [-5.07, -1.91] |
| 90 → 300 | hybrid | -0.1095 | -8.54 | [-11.66, -5.14] |
| 90 → 300 | cascade | -0.0652 | -2.33 | [-5.43, +0.00] |

## Dấu hiệu lỗi đo được

| Gallery | Chế độ | Dấu hiệu | Query | Tỷ lệ |
|---|---|---|---:|---:|
| 90 | cascade | ground_truth_outside_shortlist | 150 | 19.40% |
| 90 | cascade | ground_truth_retrieved_below_top5 | 260 | 33.64% |
| 90 | cascade | low_active_audio_fraction | 2 | 0.26% |
| 90 | cascade | wrong_top1_runtime_tiebreak | 606 | 78.40% |
| 90 | embedding | ground_truth_retrieved_below_top5 | 364 | 47.09% |
| 90 | embedding | low_active_audio_fraction | 2 | 0.26% |
| 90 | embedding | wrong_top1_runtime_tiebreak | 566 | 73.22% |
| 90 | hybrid | ground_truth_retrieved_below_top5 | 327 | 42.30% |
| 90 | hybrid | low_active_audio_fraction | 2 | 0.26% |
| 90 | hybrid | wrong_top1_runtime_tiebreak | 531 | 68.69% |
| 90 | melody | ground_truth_retrieved_below_top5 | 511 | 66.11% |
| 90 | melody | low_active_audio_fraction | 2 | 0.26% |
| 90 | melody | wrong_top1_runtime_tiebreak | 651 | 84.22% |
| 300 | cascade | ground_truth_outside_shortlist | 320 | 41.40% |
| 300 | cascade | ground_truth_retrieved_below_top5 | 179 | 23.16% |
| 300 | cascade | low_active_audio_fraction | 2 | 0.26% |
| 300 | cascade | wrong_top1_runtime_tiebreak | 624 | 80.72% |
| 300 | embedding | ground_truth_retrieved_below_top5 | 511 | 66.11% |
| 300 | embedding | low_active_audio_fraction | 2 | 0.26% |
| 300 | embedding | wrong_top1_runtime_tiebreak | 649 | 83.96% |
| 300 | hybrid | ground_truth_retrieved_below_top5 | 449 | 58.09% |
| 300 | hybrid | low_active_audio_fraction | 2 | 0.26% |
| 300 | hybrid | wrong_top1_runtime_tiebreak | 597 | 77.23% |
| 300 | melody | ground_truth_retrieved_below_top5 | 600 | 77.62% |
| 300 | melody | low_active_audio_fraction | 2 | 0.26% |
| 300 | melody | wrong_top1_runtime_tiebreak | 677 | 87.58% |

Các dấu hiệu trên không tự chứng minh nguyên nhân âm học. Cần nghe lại query/đoạn nhạc để kết luận nhầm nhạc nền, hát sai hay sai nhãn. Top1 lỗi dùng tie-break thực tế của runtime; Recall/MRR dùng expected ties của protocol.

## Giới hạn và bước tiếp theo

- Mốc 1.000/10.000 thiếu bản ghi thật; không có metric cho các mốc đó.
- Bổ sung audit Bước 4 đã khôi phục mapping train ID và phát hiện 773/773 query khớp audio hash với manifest run gốc: 428 query thuộc original train, 158 original val, 187 original test. Do đó các metric benchmark này vẫn dùng cho baseline/parity trên query cố định, nhưng không phải kết quả song-disjoint trên bài chưa thấy; `selection_eligible=false`.
- 187 query được gán original test đã bị xử lý trong benchmark trước đây vì split hiện tại phân loại chúng vào val. Không tái sử dụng chúng làm test cuối. Chi tiết ở `docs/project/qbh_stage_4_provenance/REPORT.md`.
- Vocal tắt ở mọi mốc. Không sử dụng số đo RAM/latency từ run cũ có lỗi sampler.
- RSS là tổng tiến trình lấy mẫu 10 ms. Latency loại trừ indexing/JIT/HTTP/feedback; không đồng nhất với latency server dưới tải.
- Không dùng query test để chọn cấu hình. Query diagnostics validation không tự trở thành dữ liệu train.
- Bước 2 dự kiến: scorer ma trận exact và index tăng dần; cần giữ parity thứ hạng và đo lại cùng protocol. **Chưa bắt đầu bước 2.**

Run signature: `52a4c73dd25194f184715a1ba2f1036a288e0c59030be5d9481253073b2d66f4`
Manifest: `a36f0090dc6f68766de68d9ffedd69c7fd6f75fe03a7bba2ba8c6d6432181407`
