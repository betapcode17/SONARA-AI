# Script sinh notebook

`notebook_builders/` chứa các script chuẩn bị notebook; notebook xuất ra chứa code để chạy theo môi trường tương ứng. Các script định vị đường dẫn theo vị trí file nên có thể được gọi từ gốc dự án hoặc thư mục Stage 2.

| Script | Đầu ra trong `notebooks/` |
| --- | --- |
| `build_final_solution_notebooks.py` | `templates/conformer_bytehum_qbh_final_train.ipynb`, `kaggle/evaluate/conformer_bytehum_qbh_final_evaluation.ipynb` |
| `build_kaggle_ready_notebook.py` | `kaggle/train/conformer_bytehum_qbh_final_train_kaggle.ipynb` |
| `build_local_4gb_notebook.py` | `local_4gb/train/conformer_bytehum_qbh_local_4gb.ipynb` |
| `build_local_evaluation_notebook.py` | `local_4gb/evaluate/conformer_bytehum_qbh_evaluation_local_4gb.ipynb` |
| `build_accuracy_finetune_notebook.py` | `local_4gb/train/conformer_bytehum_qbh_accuracy_finetune_local_4gb.ipynb` |
| `build_accuracy_finetune_kaggle_notebook.py` | `kaggle/train/conformer_bytehum_qbh_accuracy_finetune_kaggle.ipynb` (chưa sinh/kiểm chứng notebook đích) |
| `build_retrieval_ablation_notebook.py` | `local_4gb/experiments/conformer_bytehum_qbh_retrieval_ablation_local_4gb.ipynb` |
| `build_retrieval_candidate_ablation_notebook.py` | `local_4gb/experiments/conformer_bytehum_qbh_candidate_ablation_local_4gb.ipynb` |

`notebook_builders/templates/final_solution_runtime.py` là nguồn code được nhúng vào notebook V5 trong bước build. Giữ các notebook trong `notebooks/kaggle/archive/` vì builder vẫn lấy định nghĩa kiến trúc từ chúng.

Ví dụ sinh lại V5, chạy từ thư mục Stage 2:

```powershell
python scripts/notebook_builders/build_final_solution_notebooks.py
python scripts/notebook_builders/build_kaggle_ready_notebook.py
```

Builder có thể ghi lại notebook đích và xóa output hiển thị trong notebook đó. Chỉ chạy khi chủ động muốn sinh lại file; việc chuyển thư mục không yêu cầu chạy lại training hoặc các builder này.

`validate_final_notebooks.py` chạy các test V5 có phạm vi, kiểm tra compile/symbol,
cập nhật hai báo cáo JSON và bundle Kaggle trong `dist/` khi test thành công:

```powershell
python scripts/validate_final_notebooks.py
```

Test Torch được skip nếu môi trường không có PyTorch; script không train model.
