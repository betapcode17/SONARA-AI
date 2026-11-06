---
name: pbl6-qbh-workflow
description: >-
  Quy trình chuẩn hóa runbook toàn diện cho dự án PBL6: Huấn luyện, đánh giá mô hình Conformer-ByteHum QbH V6.2,
  đồng bộ kho nhạc Cloudinary CDN, trích xuất đặc trưng CQT, và triển khai dịch vụ Web API tìm kiếm bài hát qua tiếng ngân nga.
  Sử dụng khi người dùng yêu cầu thao tác với mô hình QBH, chạy so sánh V6.2 vs V6.1, cập nhật kho nhạc, hoặc xây dựng website.
---

# 🎵 PBL6 Conformer-ByteHum QbH Workflow & Runbook

Cẩm nang quy chuẩn các tác vụ trong dự án **Query-by-Humming (QbH)** dành cho Antigravity Agent và lập trình viên.

---

## 1. Cấu trúc Dự án
```
PBL6/code/
├── 01_stage1_cover_song_csi/       # Giai đoạn 1: Pre-training CoverHunter trên SHS100K
├── 02_stage2_humming_qbh/          # Giai đoạn 2: Thích ứng miền Conformer-ByteHum
│   ├── datasets/                   # Dữ liệu sạch V6.1 & MIR-QBSH
│   └── notebooks/colab/            # Notebooks Colab GPU T4:
│       ├── conformer_bytehum_qbh_final_v6_2_colab.ipynb             # Huấn luyện V6.2
│       ├── conformer_bytehum_qbh_v6_2_vs_v6_1_comparison_colab.ipynb # So sánh V6.2 vs V6.1
│       └── extract_gallery_vocals_demucs_colab.ipynb                 # Tách Vocal HTDemucs
├── 03_web_music_platform/          # Hệ thống Web & Triển khai
│   └── cloudinary_uploader/        # Đồng bộ 300 bài hát lên Cloudinary CDN
├── docs/                           # Tài liệu thiết kế & Báo cáo
│   └── figures/                    # Sơ đồ kiến trúc (.drawio, .png)
└── PBL6_QBH_PROJECT_REPORT.md      # Báo cáo tổng thể toàn diện của dự án
```

---

## 2. Quy trình Vận hành Trọng tâm

### A. Đánh giá đối đầu V6.2 vs V6.1 trên Google Colab
1. Mở notebook:
   `02_stage2_humming_qbh/notebooks/colab/conformer_bytehum_qbh_v6_2_vs_v6_1_comparison_colab.ipynb`
2. Kiểm tra các file đầu vào trên Google Drive:
   - `datasets_qbh_clean_mirqbsh_70_15_15_v6_1.zip`
   - `qbh_cqt_cache_v6_1.zip`
   - `qbh_final_v6_2_train_outputs.zip`
   - `qbh_final_v6_1_eval_outputs.zip` (hoặc `qbh_final_v6_1_train_outputs.zip`)
3. Chạy `Run all` trên GPU T4 để xuất bảng so sánh Delta ($\Delta$) và biểu đồ `v6_2_vs_v6_1_comparison.png`.

### B. Quản lý kho nhạc Cloudinary CDN
1. Kiểm tra kết nối và hạn mức:
   ```bash
   python 03_web_music_platform/cloudinary_uploader/test_connection.py
   ```
2. Upload kho bài hát (có hỗ trợ tự động resume):
   ```bash
   python 03_web_music_platform/cloudinary_uploader/upload_audio_gallery.py --workers 4
   ```
3. File kết quả ánh xạ link CDN HTTPS:
   `03_web_music_platform/cloudinary_uploader/cloudinary_songs_catalog.json`

### C. Sinh và Cập nhật Sơ đồ Kiến trúc (.drawio)
Sử dụng công cụ `diagramctl` từ `drawio-skill`:
```bash
python .agents/skills/drawio-skill/scripts/diagramctl.py build docs/figures/pbl6_qbh_architecture.ir.json --from ir -o docs/figures/pbl6_qbh_architecture.drawio
```
Mở file `docs/figures/pbl6_qbh_architecture.drawio` trực tiếp trên [app.diagrams.net](https://app.diagrams.net) để chỉnh sửa đồ họa hoặc nhúng vào báo cáo.

