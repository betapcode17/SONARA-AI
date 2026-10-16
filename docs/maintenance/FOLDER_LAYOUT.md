# Cấu trúc thư mục sau sắp xếp

Notebook Stage 2 được chia theo nền tảng (`kaggle`, `local_4gb`, `colab`) rồi theo công việc (`train`, `evaluate`, `experiments`). `templates` giữ notebook nguồn; `archive` giữ các bản cũ còn cần để tham chiếu hoặc sinh notebook mới. Stage 1 có notebook local tại `notebooks/local/`.

Script builder chuyển từ `notebooks/` sang `scripts/notebook_builders/`. Runtime template chuyển cùng nhóm script. Tài liệu dự án và paper chuyển từ gốc vào `docs/`; ảnh notebook chuyển vào `reports/figures/`; ZIP đóng gói chuyển vào `dist/`.

Đã dọn các file có thể xác định rõ mục đích tạm thời: `scratch_inspect.py`, `scratch_dump.txt`, thư mục Python `__pycache__`, và thư mục `checkpoints/` rỗng ở gốc. Giữ dữ liệu, trọng số, CQT cache và kết quả chạy tại các đường dẫn cũ.

Danh sách đầy đủ đường dẫn cũ → mới, SHA-256 trước/sau và kiểm kê dữ liệu được lưu tại [folder_migration_2026-10-16.json](folder_migration_2026-10-16.json). Những thay đổi nội dung notebook local chỉ phục vụ tìm đúng thư mục dự án sau khi di chuyển.

Kiểm tra sau sắp xếp: 228 cell notebook biên dịch thành công; 21 regression test đạt, 2 test PyTorch được bỏ qua vì runtime kiểm tra thiếu PyTorch. Năm builder chạy trong bản sao tạm; notebook V5 sinh lại khớp từng byte với bản hiện có. Đường dẫn nguồn/đích của tám builder và sáu tình huống tìm root dự án đều được kiểm tra. Chi tiết: [layout_validation_2026-10-16.json](layout_validation_2026-10-16.json).
