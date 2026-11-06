# Dataset dựng lại theo split nguồn

**Trạng thái:** Đã dựng và kiểm tra cấu trúc/hash holdout; **chưa sẵn sàng cho train cuối** vì thiếu audio humming ở train/validation.

## Artifact

- ZIP: `02_stage2_humming_qbh\dist\qbh_v5_source_split_rebuilt_incomplete.zip`
- SHA256 ZIP: `9711e16b0777172b0904a2241a24bae570a0d7e2a246cee7f2f55ecb86ce5ebb`
- Mapping nguồn: `docs\project\qbh_stage_4_provenance\source_split_mapping.csv` (SHA256 `e0d2094858e44cbec76a9c87fb8d8909cdfb7350f905494a0cb6cda22c9ffb6c`)
- Holdout portable: `docs\project\qbh_stage_5_holdout_preflight\locked_final_test_manifest_portable.csv`
- Contract/checksum: `docs\project\qbh_stage_5_holdout_preflight\rebuilt_dataset_manifest.json`

## Đã xác minh

- Phân hoạch đúng theo song ID từ mapping: **210 train / 45 validation / 45 test**; tất cả 300 bài và full-song gallery 300/300.
- Các file humming hiện có được chuyển theo `song_id`/`audio_id`, không ghép theo tên gần đúng; SHA256 khớp manifest nguồn cho **5148 query**.
- Holdout sạch giữ nguyên **36 bài / 615 query**; cả 615 query nằm dưới split `test`, và SHA256 được kiểm tra lại trong ZIP.
- Không sửa `datasets.zip` gốc, không train hoặc đánh giá model.

## Thiếu dữ liệu trong nguồn hiện có

Archive nguồn không có 39 query WAV trong manifest nguồn: **29 train, 3 validation, 7 test**. Ba query thiếu thuộc clean holdout đã bị loại khỏi manifest khóa; cả 615 query khóa vẫn có mặt. Tuy nhiên 29 train và 3 validation làm dataset không đầy đủ để coi là bản train cuối. Cần khôi phục 32 file từ nguồn trước khi train.

```text
Train:  3583/3612 query files present
Val:    763/766 query files present
Test:   802/809 query files present
Locked: 615/615 query files present and hash verified
```

Training notebook chỉ dùng audio train/validation cho loss và chọn checkpoint; test query không dùng để model selection. Dù vậy, chưa chạy train cho tới khi phục hồi đủ train/validation và xác nhận log Kaggle bên ngoài chưa dùng 36 bài holdout.
