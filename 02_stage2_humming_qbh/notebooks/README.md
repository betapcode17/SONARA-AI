# Danh mục notebook

## Kaggle

| Nhóm | Notebook | Vai trò |
| --- | --- | --- |
| `kaggle/train/` | [conformer_bytehum_qbh_final_train_kaggle.ipynb](kaggle/train/conformer_bytehum_qbh_final_train_kaggle.ipynb) | Train V5, kèm kiểm tra GPU và xuất training outputs. |
| `kaggle/evaluate/` | [conformer_bytehum_qbh_final_evaluation.ipynb](kaggle/evaluate/conformer_bytehum_qbh_final_evaluation.ipynb) | Đánh giá checkpoint V5 sau khi chọn trên validation. |
| `kaggle/archive/` | [conformer_bytehum_qbh_kaggle.ipynb](kaggle/archive/conformer_bytehum_qbh_kaggle.ipynb) | Train V4 và nguồn kiến trúc cho builder. |
| `kaggle/archive/` | [conformer_bytehum_qbh_evaluation_kaggle.ipynb](kaggle/archive/conformer_bytehum_qbh_evaluation_kaggle.ipynb) | Evaluator V4 và nguồn cho evaluator local. |

## Local GPU 4 GB

| Nhóm | Notebook | Vai trò |
| --- | --- | --- |
| `local_4gb/train/` | [conformer_bytehum_qbh_local_4gb.ipynb](local_4gb/train/conformer_bytehum_qbh_local_4gb.ipynb) | Huấn luyện V4 trên local. |
| `local_4gb/train/` | [conformer_bytehum_qbh_accuracy_finetune_local_4gb.ipynb](local_4gb/train/conformer_bytehum_qbh_accuracy_finetune_local_4gb.ipynb) | Fine-tune accuracy từ checkpoint V4 hiện có. |
| `local_4gb/evaluate/` | [conformer_bytehum_qbh_evaluation_local_4gb.ipynb](local_4gb/evaluate/conformer_bytehum_qbh_evaluation_local_4gb.ipynb) | Full-gallery evaluation. |
| `local_4gb/experiments/` | [conformer_bytehum_qbh_candidate_ablation_local_4gb.ipynb](local_4gb/experiments/conformer_bytehum_qbh_candidate_ablation_local_4gb.ipynb) | Candidate recall và reranking. |
| `local_4gb/experiments/` | [conformer_bytehum_qbh_retrieval_ablation_local_4gb.ipynb](local_4gb/experiments/conformer_bytehum_qbh_retrieval_ablation_local_4gb.ipynb) | So sánh scoring, chọn cấu hình trên validation. |

## Bản nguồn và phiên bản cũ

- [templates/conformer_bytehum_qbh_final_train.ipynb](templates/conformer_bytehum_qbh_final_train.ipynb): bản nguồn V5 dùng khi sinh notebook Kaggle. Cấu hình mặc định vẫn là Kaggle.
- [archive/conformer_bytehum_qbh.ipynb](archive/conformer_bytehum_qbh.ipynb): notebook QbH đời đầu, giữ cùng output lịch sử.
- [colab/README.md](colab/README.md): trạng thái hỗ trợ Colab.

Script sinh notebook đặt tại `../scripts/notebook_builders/`; Python runtime template nằm trong `templates/` của thư mục script đó. Bundle xuất Kaggle nằm tại `../dist/`; hình minh họa nằm tại `../reports/`.
