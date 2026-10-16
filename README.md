# PBL6: Nhận diện bài hát bằng tiếng ngân nga

Dự án gồm hai giai đoạn: học biểu diễn bài hát cover với CoverHunter (Stage 1), sau đó thích nghi và truy xuất tiếng humming bằng Conformer-ByteHum-inspired QbH (Stage 2). README này mô tả cây thư mục và các file đang có tại gốc `code`; dữ liệu âm thanh, cache và output được mô tả theo nhóm để tránh liệt kê hàng chục nghìn mẫu.

## Luồng hệ thống

```text
SHS100K / CoverHunter checkpoint
               |
               v
Stage 2 train: query humming + reference song -> CQT -> Conformer -> embedding
               |
               v
Full-gallery retrieval: local MaxMean + global score -> ranking / MRR / Recall@K
```

Stage 1 cung cấp checkpoint CoverHunter SHS100K. Stage 2 chứa pipeline QbH, notebook train/evaluation và kết quả nghiên cứu hiện tại. Các con số trong tài liệu cần được đối chiếu với artifacts trong `02_stage2_humming_qbh/runs/`; mục tiêu viết trong proposal không đồng nghĩa đã đạt được.

Mở nhanh: [Stage 1](01_stage1_cover_song_csi/README.md), [Stage 2](02_stage2_humming_qbh/README.md), [danh mục notebook theo môi trường](02_stage2_humming_qbh/notebooks/README.md), [hướng dẫn Kaggle V5](02_stage2_humming_qbh/reports/KAGGLE_TRAIN_READY.md).

## Cây thư mục

```text
code/
|-- README.md
|-- .gitignore                   Bỏ qua Python/Jupyter temporary files
|-- docs/
|   |-- project/                 Thuyết minh và báo cáo Word
|   |-- references/              Paper ByteHum
|   |-- history/                 Kế hoạch và review trước đây
|   |-- figures/                 Hình CMC/loss ở gốc trước đây
|   `-- maintenance/             Bảng chuyển đường dẫn và nhật ký sắp xếp
|-- 01_stage1_cover_song_csi/
|   |-- notebooks/local/         Notebook CSI chạy trong checkout local
|   |-- src/                     Mã nguồn CoverHunter và baseline
|   `-- configs/, datasets/, checkpoints/
`-- 02_stage2_humming_qbh/
    |-- notebooks/
    |   |-- kaggle/train/        Train Kaggle V5
    |   |-- kaggle/evaluate/     Đánh giá Kaggle V5
    |   |-- kaggle/archive/      Bản Kaggle V4 làm tham chiếu
    |   |-- local_4gb/train/     Train/fine-tune local
    |   |-- local_4gb/evaluate/  Đánh giá local
    |   |-- local_4gb/experiments/
    |   |-- colab/               Vị trí dành cho bản Colab khi bổ sung
    |   |-- templates/           Notebook nguồn V5
    |   `-- archive/             Notebook đời đầu
    |-- scripts/notebook_builders/
    |-- docs/, reports/, dist/   Tài liệu, kết quả và ZIP xuất Kaggle
    `-- configs/, experiments/, tests/, datasets/, datasets_unused/, checkpoints/, runs/, cache/
```

## Stage 1: `01_stage1_cover_song_csi/`

| Đường dẫn | Vai trò |
| --- | --- |
| `README.md` | Hướng dẫn Stage 1 và cấu trúc CoverHunter/Conformer hiện tại. |
| `test_stage1_pipeline.py` | Kiểm tra các thành phần pipeline Stage 1. |
| `configs/config_csi.yaml` | Cấu hình Stage 1 CSI. |
| `notebooks/local/coverhunter_conformer_csi.ipynb` | Notebook huấn luyện/thử nghiệm CoverHunter-Conformer cho cover-song identification. |
| `src/conformer_model.py` | Mô hình Conformer dùng trong Stage 1. |
| `src/coverhunter_model.py` | Thành phần/adapter mô hình CoverHunter. |
| `src/dataset_cover.py` | Dataset cho dữ liệu cover-song. |
| `src/losses.py` | Hàm mất mát metric learning cho CSI. |
| `src/metrics.py` | Metric đánh giá truy xuất/nhận diện. |
| `src/resnet_ibn.py` | Backbone ResNet-IBN cho baseline/thử nghiệm. |
| `src/transforms.py` | Tiền xử lý và biến đổi âm thanh/CQT. |
| `src/__init__.py` | Khởi tạo package `src`. |
| `src/coverhunter/Aligner.py` | Căn chỉnh đặc trưng trong implementation CoverHunter. |
| `src/coverhunter/basic_model.py` | Lớp cơ sở và thành phần dùng chung. |
| `src/coverhunter/cqt.py` | Trích xuất CQT. |
| `src/coverhunter/dataset.py` | Dataset/loader cho CoverHunter. |
| `src/coverhunter/eval_testset.py` | Đánh giá trên test set. |
| `src/coverhunter/hparams.yaml` | Siêu tham số CoverHunter. |
| `src/coverhunter/loss.py` | Loss của implementation CoverHunter. |
| `src/coverhunter/map.py` | Tiện ích ánh xạ/index dữ liệu. |
| `src/coverhunter/model.py` | Định nghĩa model CoverHunter. |
| `src/coverhunter/pytorch_utils.py` | Tiện ích PyTorch. |
| `src/coverhunter/scheduler.py` | Learning-rate scheduler. |
| `src/coverhunter/trainer.py` | Vòng lặp huấn luyện. |
| `src/coverhunter/utils.py` | Hàm tiện ích dùng chung. |
| `src/coverhunter/module/conformer.py` | Các khối Conformer. |
| `src/coverhunter/module/layers.py` | Layer nền tảng của backbone. |
| `src/coverhunter/__init__.py` | Khởi tạo package CoverHunter. |
| `datasets/download_covers80.py` | Tải dữ liệu Covers80. |
| `datasets/download_cover1000.py` | Tải dữ liệu/đặc trưng Cover1000 cho baseline. |
| `datasets/download_coverhunter_pretrained.py` | Tải checkpoint CoverHunter pretrained. |
| `datasets/covers80/`, `datasets/datacos/` | Dữ liệu nguồn, tùy trạng thái tải/chuẩn bị. |
| `datasets/cqt_cache/` | Cache CQT Stage 1. |
| `checkpoints/coverhunter_shs100k_pretrained.pth` | Checkpoint SHS100K được Stage 2 dùng khởi tạo backbone. |
| `checkpoints/coverhunter_shs100k.pt` | Bản checkpoint PyTorch khác của CoverHunter. |
| `checkpoints/pretrain_model.tar.gz` | Archive checkpoint pretrained. |
| `checkpoints/pretrain_model/config/hparams.yaml` | Hyperparameter đi kèm archive. |
| `checkpoints/pretrain_model/pt_model/do_00000040`, `g_00000040` | Trạng thái model/discriminator trong checkpoint đã giải nén. |

## Stage 2: `02_stage2_humming_qbh/`

| Đường dẫn | Vai trò |
| --- | --- |
| `README.md` | Chọn notebook theo môi trường và công việc. |
| `configs/bytehum_config.yaml` | Cấu hình âm thanh/CQT, augmentation, model và training QbH. |
| `experiments/retrieval_ablation.py` | Thí nghiệm ablation cho scoring/retrieval. |
| `datasets.zip` | Archive dữ liệu Stage 2, khoảng 3.7 GB tại lần kiểm kê. |
| `datasets/song_name_mapping.json` | Ánh xạ tên/ID bài hát. |
| `datasets/2_music_database/` | Bản thu gốc dùng làm gallery; thư mục có khoảng 301 file, protocol hiện tại dùng 300 bài hợp lệ. |
| `datasets/dataset_split/{train,val,test}/` | Query humming theo split; bên trong chia theo bài hát/fragment và file audio tương ứng. |
| `datasets_unused/cleaning_report.csv` | Báo cáo làm sạch của nhánh dữ liệu không hoạt động chính. |
| `datasets_unused/reorganize_mir_qbsh.py` | Tái tổ chức dữ liệu MIR-QBSH. |
| `datasets_unused/run_multikey_eval.py` | Đánh giá thử nghiệm multi-key. |
| `datasets_unused/synthesize_mir_qbsh_midi.py` | Tạo dữ liệu tổng hợp MIR-QBSH/MIDI. |
| `datasets_unused/3_test_queries/` | Query và ground truth thử nghiệm cũ; không thay test split chuẩn. |
| `datasets_unused/mir_qbsh/` | Dữ liệu MIR-QBSH dùng cho thí nghiệm/đối chiếu. |
| `datasets_unused/cqt_cache/` | Cache cho nhánh dữ liệu cũ. |
| `checkpoints/coverhunter_shs100k_pretrained.pth` | Bản Stage 1 checkpoint dùng trong Stage 2. |
| `checkpoints/.gitkeep` | Giữ thư mục checkpoint trong quản lý phiên bản. |
| `checkpoints/conformer_bytehum_full_d128_timegrid_lat_v4_best.pth` | Checkpoint Stage 2 v4 best, embedding 128-D. |
| `checkpoints/conformer_bytehum_finetuned_best.pth` | Checkpoint fine-tune đời trước. |
| `checkpoints/best_bytehum.pth`, `latest_bytehum.pth` | Checkpoint best/latest của pipeline ByteHum trước đó. |
| `checkpoints/conformer_song_map.pkl`, `full_db_song_map.pkl` | Ánh xạ song ID của gallery/index tương ứng. |
| `checkpoints/conformer_db_300_songs.index`, `full_db_300_songs.index` | FAISS index cho gallery 300 bài. |
| `notebooks/README.md` | Danh mục notebook theo môi trường và công việc. |
| `notebooks/kaggle/train/conformer_bytehum_qbh_final_train_kaggle.ipynb` | Notebook train V5 đóng gói để chạy Kaggle. |
| `notebooks/templates/conformer_bytehum_qbh_final_train.ipynb` | Notebook nguồn V5 để sinh bản Kaggle. |
| `notebooks/kaggle/evaluate/conformer_bytehum_qbh_final_evaluation.ipynb` | Evaluation V5 tách khỏi training. |
| `notebooks/kaggle/archive/conformer_bytehum_qbh_kaggle.ipynb` | Train Kaggle V4 và nguồn định nghĩa kiến trúc. |
| `notebooks/local_4gb/train/conformer_bytehum_qbh_local_4gb.ipynb` | Train Stage 2 trên GPU local 4 GB. |
| `notebooks/local_4gb/train/conformer_bytehum_qbh_accuracy_finetune_local_4gb.ipynb` | Fine-tune local; full-gallery validation và chọn theo fused MRR. |
| `notebooks/kaggle/archive/conformer_bytehum_qbh_evaluation_kaggle.ipynb` | Evaluator Kaggle V4. |
| `notebooks/local_4gb/evaluate/conformer_bytehum_qbh_evaluation_local_4gb.ipynb` | Evaluation local 4 GB, hỗ trợ full gallery. |
| `notebooks/local_4gb/experiments/conformer_bytehum_qbh_candidate_ablation_local_4gb.ipynb` | Ablation candidate retrieval/reranking local. |
| `notebooks/local_4gb/experiments/conformer_bytehum_qbh_retrieval_ablation_local_4gb.ipynb` | Ablation scoring/retrieval local. |
| `notebooks/archive/conformer_bytehum_qbh.ipynb` | Notebook QbH đời đầu, giữ output lịch sử. |
| `notebooks/colab/README.md` | Trạng thái Colab; chưa có notebook riêng cho nền tảng này. |
| `scripts/notebook_builders/build_final_solution_notebooks.py` | Sinh notebook final solution. |
| `scripts/notebook_builders/build_kaggle_ready_notebook.py` | Tạo wrapper Kaggle. |
| `scripts/notebook_builders/build_local_4gb_notebook.py` | Sinh notebook cấu hình local 4 GB. |
| `scripts/notebook_builders/build_local_evaluation_notebook.py` | Sinh notebook evaluation local. |
| `scripts/notebook_builders/build_accuracy_finetune_notebook.py` | Sinh notebook fine-tune accuracy local. |
| `scripts/notebook_builders/build_accuracy_finetune_kaggle_notebook.py` | Builder fine-tune Kaggle; cần sinh/kiểm chứng notebook đích. |
| `scripts/notebook_builders/build_retrieval_ablation_notebook.py` | Sinh notebook retrieval ablation. |
| `scripts/notebook_builders/build_retrieval_candidate_ablation_notebook.py` | Sinh notebook candidate ablation. |
| `scripts/notebook_builders/templates/final_solution_runtime.py` | Build-time template được nhúng vào notebook V5. |
| `scripts/README.md` | Danh mục builder, đầu ra và lệnh sinh notebook. |
| `docs/RESEARCH_EXPERIMENTS.md` | Ghi chú RUN_NAME và protocol ablation. |
| `dist/qbh_final_v5_kaggle_bundle.zip` | Bundle V5 để chuyển/upload Kaggle. |
| `dist/README.md` | Hướng dẫn sử dụng bundle và cấu trúc bên trong. |
| `reports/figures/cmc_curve.png` | Hình CMC trước đây nằm cạnh notebook. |
| `reports/FINAL_SOLUTION.md` | Pipeline V5 và nguyên tắc đánh giá test một lần. |
| `reports/IMPROVEMENT_PLAN.md` | Baseline và kế hoạch cải tiến/thí nghiệm. |
| `reports/KAGGLE_TRAIN_READY.md` | Hướng dẫn Kaggle và input cần gắn. |
| `reports/final_solution_validation.json` | Kết quả static/regression validation cho final evaluation. |
| `reports/kaggle_ready_validation.json` | Kết quả kiểm tra wrapper Kaggle bằng kiểm tra tĩnh/mocked GPU. |
| `reports/architecture/ARCHITECTURE_NOTES.md` | Nguồn và ghi chú hình kiến trúc. |
| `reports/architecture/bytehum_original_architecture.png` | Hình vẽ lại kiến trúc ByteHum gốc. |
| `reports/architecture/bytehum_paper_figure1_reference.png` | Hình tham chiếu từ paper. |
| `reports/architecture/conformer_bytehum_current_architecture.png` | Hình implementation hiện tại. |
| `tests/test_final_solution.py` | Regression cho protocol/logic final notebook. |
| `tests/test_kaggle_ready_notebook.py` | Kiểm tra wrapper Kaggle trên CPU/mocked GPU. |
| `tests/test_local_4gb_notebook.py` | Kiểm tra train local; CUDA smoke test là opt-in. |
| `tests/test_local_evaluation_notebook.py` | Kiểm tra evaluator local; CUDA execution là opt-in. |
| `tests/test_notebook_research.py` | Kiểm tra cell nghiên cứu, loss và retrieval. |
| `tests/test_retrieval_ablation.py` | Regression test scoring và protocol ablation. |
| `runs/local_4gb/` | Metrics JSON/CSV, query rankings, plots, index và checkpoint của các lần chạy local. Đây là output, không phải source. |
| `cache/` | Cache CQT/Numba phát sinh khi chạy. Không phải nguồn dữ liệu chuẩn. |

## Bắt đầu chạy

1. Đọc `02_stage2_humming_qbh/reports/FINAL_SOLUTION.md` và `KAGGLE_TRAIN_READY.md` trước khi chạy pipeline cuối.
2. Trên Kaggle, dùng `02_stage2_humming_qbh/notebooks/kaggle/train/conformer_bytehum_qbh_final_train_kaggle.ipynb`, gắn đúng dataset và Stage 1 checkpoint như report, bật GPU và lưu output/checkpoint.
3. Đánh giá bằng `02_stage2_humming_qbh/notebooks/kaggle/evaluate/conformer_bytehum_qbh_final_evaluation.ipynb`. Không dùng test set để chọn epoch hoặc tune hyperparameter.
4. Với thử nghiệm local 4 GB, dùng notebook train/evaluation local tương ứng và kiểm tra metrics trong `runs/local_4gb/`.

Các bài kiểm tra dùng `unittest`; ví dụ chạy từ thư mục Stage 2:

```powershell
cd 02_stage2_humming_qbh
python -X utf8 tests/test_final_solution.py
python -X utf8 tests/test_notebook_research.py
```

Static tests và syntax validation không thay thế train/evaluation thật trên GPU. `datasets/`, `datasets_unused/`, `cache/`, `runs/` và checkpoints có thể chiếm nhiều GB; không upload toàn bộ các thư mục này lên Kaggle nếu notebook chỉ cần một số input cụ thể.

## Tài liệu chung: `docs/`

| File | Mô tả |
| --- | --- |
| `docs/project/PBL6.md` | Tổng quan mục tiêu, kiến trúc và phương pháp. Đối chiếu report/artifacts trước khi trích dẫn mục tiêu hoặc claim như kết quả đã xác nhận. |
| `docs/history/Plan.md` | Kế hoạch cải tiến theo phase; có ghi chú lịch sử, cần đối chiếu implementation hiện tại. |
| `docs/history/Error.md` | Rủi ro/điểm cần sửa được ghi lại qua các vòng review; một phần có thể đã xử lý. |
| `docs/references/ICASSP2024_bytehum.pdf` | Paper ByteHum tham khảo cho Stage 2. |
| `docs/project/PBL6 - DOCUMEMT.docx` | Tài liệu báo cáo Word của dự án. |
| `docs/figures/cmc_curve.png` | Biểu đồ CMC trước đây nằm ở gốc workspace. |
| `docs/figures/loss_curves.png` | Biểu đồ loss trước đây nằm ở gốc workspace. |
| `docs/maintenance/FOLDER_LAYOUT.md` | Quy tắc bố trí thư mục và ghi chú dọn file. |
| `docs/maintenance/folder_migration_2026-10-16.json` | Đường dẫn cũ/mới, hash và kiểm kê dữ liệu trước/sau. |
| `docs/maintenance/layout_validation_2026-10-16.json` | Kiểm tra đường dẫn builder, build trong thư mục tạm và kết quả regression sau sắp xếp. |

## Provenance và phiên bản

- Có nhiều thế hệ notebook/checkpoint (baseline, v4 fine-tune, V5 final). Không chọn checkpoint chỉ theo tên `best`; kiểm tra `run_config`, split, gallery, epoch và metrics đi kèm.
- `reports/*validation.json` ghi nhận kiểm tra mã/protocol ở mức được mô tả trong JSON; không tự chứng minh GPU training đã chạy.
- `docs/project/PBL6.md`, `docs/history/Plan.md` và `docs/history/Error.md` được viết ở các thời điểm khác nhau. Dùng `02_stage2_humming_qbh/reports/FINAL_SOLUTION.md` cho quy trình Stage 2 hiện tại.
- Hai file `scratch_*`, Python bytecode cache và thư mục checkpoint rỗng ở gốc đã được dọn. Notebook cũ được giữ trong `archive/`; dữ liệu, trọng số và kết quả chạy giữ nguyên vị trí.
