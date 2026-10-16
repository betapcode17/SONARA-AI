# Stage 2: Query-by-Humming

Chọn notebook theo môi trường và công việc trong bảng dưới đây. Các notebook V5 là luồng train/evaluation hiện tại; V4 được giữ để đối chiếu kết quả và thử nghiệm trên máy local.

| Công việc | File |
| --- | --- |
| Train V5 trên Kaggle | [conformer_bytehum_qbh_final_train_kaggle.ipynb](notebooks/kaggle/train/conformer_bytehum_qbh_final_train_kaggle.ipynb) |
| Đánh giá V5 trên Kaggle | [conformer_bytehum_qbh_final_evaluation.ipynb](notebooks/kaggle/evaluate/conformer_bytehum_qbh_final_evaluation.ipynb) |
| Train V4 trên GPU local 4 GB | [conformer_bytehum_qbh_local_4gb.ipynb](notebooks/local_4gb/train/conformer_bytehum_qbh_local_4gb.ipynb) |
| Fine-tune V4 trên local 4 GB | [conformer_bytehum_qbh_accuracy_finetune_local_4gb.ipynb](notebooks/local_4gb/train/conformer_bytehum_qbh_accuracy_finetune_local_4gb.ipynb) |
| Đánh giá V4 trên local | [conformer_bytehum_qbh_evaluation_local_4gb.ipynb](notebooks/local_4gb/evaluate/conformer_bytehum_qbh_evaluation_local_4gb.ipynb) |
| So sánh scoring/candidate | [Danh mục notebook](notebooks/README.md) → `local_4gb/experiments/` |

```text
02_stage2_humming_qbh/
|-- notebooks/
|   |-- kaggle/train/           Train Kaggle hiện tại
|   |-- kaggle/evaluate/        Evaluation Kaggle hiện tại
|   |-- kaggle/archive/         Kaggle V4 làm tham chiếu
|   |-- local_4gb/train/        Train và fine-tune trên local
|   |-- local_4gb/evaluate/     Evaluation local
|   |-- local_4gb/experiments/  Ablation scoring/candidate
|   |-- colab/                 Vị trí dành cho notebook Colab khi bổ sung
|   |-- templates/             Notebook nguồn V5 dùng để sinh bản Kaggle
|   `-- archive/               Notebook QbH đời đầu
|-- scripts/notebook_builders/ Script sinh notebook
|-- configs/                   Cấu hình tham khảo
|-- experiments/               Engine cho retrieval ablation
|-- tests/                     Unit/regression tests
|-- docs/                      Ghi chú phương pháp/ablation
|-- reports/                   Báo cáo, validation JSON và hình
|-- dist/                      Bundle ZIP để chuyển/upload Kaggle
|-- datasets/                  Dữ liệu đang sử dụng
|-- datasets_unused/           Dữ liệu/thử nghiệm lưu trữ
|-- checkpoints/               Trọng số và các index cũ
|-- runs/                      Checkpoint, metric và output từng run
`-- cache/                     Cache CQT và Numba
```

Hướng dẫn Kaggle: [KAGGLE_TRAIN_READY.md](reports/KAGGLE_TRAIN_READY.md). Protocol V5: [FINAL_SOLUTION.md](reports/FINAL_SOLUTION.md). Hướng dẫn script sinh notebook: [scripts/README.md](scripts/README.md).

Các đường dẫn dữ liệu, checkpoint và `runs/` được giữ ổn định để notebook và checkpoint resume tiếp tục nhận diện đúng run. Các file trong `archive/` vẫn được dùng làm nguồn cho một số builder và regression test.

Chạy test từ thư mục Stage 2:

```powershell
python -B -X utf8 -m unittest discover -s tests -p test_final_solution.py -v
python -B -X utf8 -m unittest discover -s tests -p test_kaggle_ready_notebook.py -v
```

Các kiểm tra có dùng PyTorch cần môi trường đã cài PyTorch. Kết quả kiểm tra cấu trúc và đường dẫn được ghi riêng với kết quả train/evaluation GPU.
