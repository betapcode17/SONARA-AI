# Stage 1: Cover Song Identification

Thư mục này chứa code CoverHunter/Conformer, baseline ResNet-IBN và dữ liệu thử nghiệm CSI. Checkpoint `checkpoints/coverhunter_shs100k_pretrained.pth` cung cấp backbone pretrained SHS100K cho Stage 2.

Notebook hiện có: [notebooks/local/coverhunter_conformer_csi.ipynb](notebooks/local/coverhunter_conformer_csi.ipynb). Notebook tự tìm thư mục Stage 1 từ vị trí hiện tại trong checkout để truy cập `src/`, dữ liệu và checkpoint.

```text
01_stage1_cover_song_csi/
|-- notebooks/local/             Notebook CoverHunter-Conformer CSI
|-- configs/config_csi.yaml      Cấu hình CSI
|-- src/
|   |-- coverhunter_model.py     Adapter CoverHunter và CQT
|   |-- conformer_model.py       Mô hình Conformer
|   |-- coverhunter/             Implementation CoverHunter
|   |-- dataset_cover.py         Dataset và sampler
|   |-- transforms.py            Tiền xử lý/augmentation
|   |-- losses.py                Metric-learning losses
|   |-- metrics.py               Metric truy xuất
|   `-- resnet_ibn.py            Backbone baseline
|-- datasets/                    Download scripts, Covers80 và cache
|-- checkpoints/                 Trọng số và cấu hình pretrained
`-- test_stage1_pipeline.py      Kiểm tra thành phần pipeline
```

Các script tải dữ liệu/trọng số:

- `datasets/download_covers80.py`: tải Covers80.
- `datasets/download_cover1000.py`: tải dữ liệu/đặc trưng Cover1000.
- `datasets/download_coverhunter_pretrained.py`: tải model pretrained.

Các file checkpoint và dữ liệu đang có được giữ nguyên vị trí. Danh mục chi tiết từng file nằm trong [README ở gốc dự án](../README.md). Notebook QbH và hướng dẫn train Stage 2 nằm tại [02_stage2_humming_qbh/README.md](../02_stage2_humming_qbh/README.md).
