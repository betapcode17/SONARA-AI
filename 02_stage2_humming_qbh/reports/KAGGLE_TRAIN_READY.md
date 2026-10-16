# Notebook để train trên Kaggle

## File sử dụng

Upload [notebooks/kaggle/train/conformer_bytehum_qbh_final_train_kaggle.ipynb](../notebooks/kaggle/train/conformer_bytehum_qbh_final_train_kaggle.ipynb) lên Kaggle.
Đây là bản đóng gói để chạy V5: giữ nguyên toàn bộ code huấn luyện của
`conformer_bytehum_qbh_final_train.ipynb`, thêm kiểm tra môi trường Kaggle/GPU,
cài thư viện còn thiếu và xuất ZIP training outputs. Không cần upload file Python.

## Inputs cần gắn

- `02_stage2_humming_qbh/datasets.zip`, khoảng 3.7 GB: chứa `datasets/dataset_split/`
  và `datasets/2_music_database/`.
- `02_stage2_humming_qbh/checkpoints/coverhunter_shs100k_pretrained.pth`, khoảng
  39.8 MB: Stage 1 pretrained, không phải checkpoint V4 fine-tuned.

Nếu Kaggle hiển thị dataset đã giải nén thì vẫn chạy được. Nếu có nhiều bản
dataset/checkpoint, đặt đường dẫn chính xác ở cell cấu hình; notebook không đoán
chọn một file. Không điền đường dẫn `E:/...` vào notebook trên Kaggle.

## Thao tác

1. Bật Accelerator GPU T4 và Internet trong Settings. T4 x2 dùng GPU 0, không DDP.
2. Attach hai Inputs trên rồi Run All từ đầu.
3. Chờ kiểm tra/hash dữ liệu. GPU preflight chạy trước bước tạo toàn bộ CQT cache:
   train mode, loss thực tế, AMP accumulation, backward, AdamW update và scoring.
   Sau đó khôi phục weights, buffers, RNG và chế độ model; optimizer/scaler/scheduler
   thật không bị cập nhật. Tiếp theo cache CQT bằng CPU rồi train tối đa 30 epoch.
4. Giữ nguyên cấu hình model/loss/mining/scoring. Best checkpoint được chọn bằng
   fused validation MRR, early stopping patience 5; không chạy test trong train.
5. Lưu Kaggle version và outputs sau khi train hoàn tất.

Nếu thiếu CUDA, notebook dừng với hướng dẫn bật GPU, không chạy train trên CPU.
Nếu `typeguard` không tương thích, cell imports cài `typeguard==2.13.3`; cần
Internet. Notebook không cài lại hoặc thay bản PyTorch GPU của Kaggle.

## Kết quả cần giữ

- `/kaggle/working/qbh_final_v5/train/qbh_final_v5_best.pth`: model để đánh giá.
- `/kaggle/working/qbh_final_v5/train/qbh_final_v5_last.pth`: model/optimizer/RNG để resume.
- `/kaggle/working/qbh_final_v5/train/final_manifest.json`: bằng chứng train đã finalize.
- `history.json`, `run_contract.json`, validation metrics và `training_curves.png`.
- `/kaggle/working/qbh_final_v5_train_outputs.zip`: gói training outputs, không chứa
  dataset hoặc CQT cache. Giải nén nếu cần upload riêng checkpoint; notebook
  evaluation không tự giải nén ZIP checkpoint.

Đánh giá tiếp bằng [conformer_bytehum_qbh_final_evaluation.ipynb](../notebooks/kaggle/evaluate/conformer_bytehum_qbh_final_evaluation.ipynb), cùng dataset và
runtime như train. Có thể attach notebook output trực tiếp để evaluator tìm best.
Không nhầm `qbh_final_v5_train_outputs.zip` với ZIP notebook/source hoặc datasets.zip.

## Resume khi bị ngắt

Attach `qbh_final_v5_last.pth` từ cùng phiên bản notebook, đặt `RESUME_LAST_OVERRIDE`
vào cell cấu hình; giữ `RESUME=True`, để `RESUME_BEST_OVERRIDE=None`. LAST schema 2
chứa cả snapshot BEST nên tự dựng lại BEST bị thiếu/cũ. LAST được ghi atomically
trước khi xuất BEST và báo cáo. BEST riêng lẻ không đủ resume optimizer/RNG.
Nếu chưa có checkpoint
epoch nào hoàn thành, bắt đầu lại từ Stage 1. Resume ở ranh giới epoch, không phải
giữa một minibatch. Dataset, code/config và runtime phải khớp hợp đồng run cũ.

**Bản sửa reliability thay đổi code digest và định dạng LAST.** Không resume
checkpoint tạo bằng notebook V5 cũ trong notebook mới. Với checkpoint cũ, dùng
notebook gốc đã backup; muốn chạy bản mới hãy bắt đầu từ Stage 1 và chọn
`OUTPUT_ROOT` trống. Luôn dùng evaluator mới đi kèm, không trộn hai phiên bản.

## Giới hạn kiểm chứng

Đã kiểm tra cấu trúc/compile, giữ nguyên code V5 và các bước wrapper bằng mock/CPU.
Mock GPU không thay thế T4 thật. Chưa tuyên bố train thành công, chưa có MRR V5 từ
Kaggle. GPU preflight và đợt train/evaluation thực tế vẫn cần chạy trên tài khoản
Kaggle của bạn.

Chi tiết thay đổi và phạm vi kiểm tra: [V5_CODE_IMPROVEMENTS.md](V5_CODE_IMPROVEMENTS.md).
