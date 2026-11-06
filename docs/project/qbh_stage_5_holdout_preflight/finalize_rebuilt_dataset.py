"""Validate and publish a fully written temporary rebuilt QbH ZIP."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[3]
STAGE2 = ROOT / "02_stage2_humming_qbh"
TEMP_ZIP = STAGE2 / "dist/qbh_v5_source_split_rebuilt_incomplete.zip.tmp"
OUTPUT_ZIP = STAGE2 / "dist/qbh_v5_source_split_rebuilt_incomplete.zip"
SOURCE_MANIFEST = Path.home() / ".codex/attachments/3f959db8-f012-4b77-b04c-ede89ec508bd/Văn bản đã dán.txt"
SOURCE_MAPPING = ROOT / "docs/project/qbh_stage_4_provenance/source_split_mapping.csv"
LOCKED_MANIFEST = ROOT / "docs/project/qbh_stage_5_holdout_preflight/locked_final_test_manifest.csv"
PORTABLE_MANIFEST = ROOT / "docs/project/qbh_stage_5_holdout_preflight/locked_final_test_manifest_portable.csv"
METADATA = ROOT / "docs/project/qbh_stage_5_holdout_preflight/rebuilt_dataset_manifest.json"
REPORT = ROOT / "docs/project/qbh_stage_5_holdout_preflight/REBUILT_DATASET_REPORT.md"
CHUNK = 1024 * 1024


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def sha256_zip_member(archive: zipfile.ZipFile, member: str) -> str:
    h = hashlib.sha256()
    with archive.open(member) as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    if not TEMP_ZIP.is_file():
        raise FileNotFoundError(TEMP_ZIP)
    mapping_rows = read_csv(SOURCE_MAPPING)
    mapping = {row["song_id"]: row for row in mapping_rows}
    source_rows = read_csv(SOURCE_MANIFEST)
    source_by_id = {row["audio_id"]: row for row in source_rows}
    locked_rows = read_csv(LOCKED_MANIFEST)
    locked_by_id = {row["audio_id"]: row for row in locked_rows}
    if len(locked_rows) != 615 or len(locked_by_id) != 615:
        raise ValueError("Locked manifest must have 615 unique query rows")

    with zipfile.ZipFile(TEMP_ZIP, "r") as archive:
        names = [name.replace("\\", "/") for name in archive.namelist()]
        name_set = set(names)
        query_paths: dict[str, str] = {}
        songs_by_split: dict[str, set[str]] = defaultdict(set)
        gallery_stems: set[str] = set()
        present_query_counts: Counter[str] = Counter()
        for name in names:
            path = PurePosixPath(name)
            if len(path.parts) >= 6 and path.parts[:2] == ("datasets", "dataset_split"):
                split, song = path.parts[2:4]
                if path.suffix.lower() == ".wav" and path.stem in source_by_id:
                    if path.stem in query_paths:
                        raise ValueError(f"Duplicate packaged audio_id: {path.stem}")
                    row = source_by_id[path.stem]
                    expected_split = mapping[row["song_id"]]["source_split"]
                    if split != expected_split:
                        raise ValueError(f"Wrong split for {path.stem}: {split} != {expected_split}")
                    query_paths[path.stem] = name
                    present_query_counts[split] += 1
                if split in {"train", "val", "test"}:
                    songs_by_split[split].add(song)
            if len(path.parts) == 3 and path.parts[:2] == ("datasets", "2_music_database") and path.suffix.lower() == ".mp3":
                gallery_stems.add(path.stem)

        if len(query_paths) != 5148:
            raise ValueError(f"Expected 5,148 present original queries, got {len(query_paths)}")
        if present_query_counts != Counter(train=3583, val=763, test=802):
            raise ValueError(f"Unexpected available query counts: {dict(present_query_counts)}")
        if {key: len(value) for key, value in songs_by_split.items()} != {"train": 210, "val": 45, "test": 45}:
            raise ValueError(f"Wrong song split counts: {dict((key, len(value)) for key, value in songs_by_split.items())}")
        if len(gallery_stems) != 300 or set().union(*songs_by_split.values()) != gallery_stems:
            raise ValueError("Gallery must contain exactly the 300 source-mapped song folders")

        portable_rows = []
        for row in locked_rows:
            source = source_by_id[row["audio_id"]]
            member = query_paths.get(row["audio_id"])
            if member is None:
                raise ValueError(f"Locked query missing in packaged archive: {row['audio_id']}")
            expected_path = "datasets/dataset_split/test/" + "/".join(PurePosixPath(member).parts[3:])
            if member != expected_path:
                raise ValueError(f"Locked query not held in test split: {member}")
            actual_hash = sha256_zip_member(archive, member)
            if actual_hash.lower() != row["expected_humming_sha256"].lower():
                raise ValueError(f"Locked query hash mismatch: {row['audio_id']}")
            portable_rows.append({**row, "dataset_relpath": member})

        required_members = {
            "datasets/dataset_split/split_manifest.json",
            "datasets/dataset_split/split_summary.csv",
            "datasets/holdout/locked_final_test_manifest.csv",
            "datasets/qbh_dataset_contract.json",
        }
        if not required_members.issubset(name_set):
            raise ValueError(f"Package metadata missing: {required_members - name_set}")
        embedded_locked = list(csv.DictReader(archive.read("datasets/holdout/locked_final_test_manifest.csv").decode("utf-8").splitlines()))
        if len(embedded_locked) != 615:
            raise ValueError("Embedded locked test manifest is not 615 rows")

    # Ensure the metadata manifest describes the omissions honestly.
    missing_rows = [row for row in source_rows if row["audio_id"] not in query_paths]
    missing_by_split = Counter(row["split"] for row in missing_rows)
    if len(missing_rows) != 39 or missing_by_split != Counter(train=29, val=3, test=7):
        raise ValueError(f"Unexpected missing source rows: {len(missing_rows)} / {dict(missing_by_split)}")
    if set(row["audio_id"] for row in locked_rows) & {row["audio_id"] for row in missing_rows}:
        raise ValueError("A locked query is among the missing source rows")

    # Publish only after archive contents and all locked query hashes validate.
    os.replace(TEMP_ZIP, OUTPUT_ZIP)
    portable_rows.sort(key=lambda row: row["audio_id"])
    fields = list(locked_rows[0]) + ["dataset_relpath"]
    PORTABLE_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with PORTABLE_MANIFEST.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(portable_rows)

    metadata = {
        "status": "built_with_missing_source_queries",
        "archive": str(OUTPUT_ZIP.relative_to(ROOT)),
        "archive_bytes": OUTPUT_ZIP.stat().st_size,
        "archive_sha256": sha256_file(OUTPUT_ZIP),
        "song_counts": {"train": 210, "val": 45, "test": 45},
        "available_queries_sha_verified_during_build": dict(present_query_counts),
        "missing_source_queries_by_split": dict(missing_by_split),
        "missing_source_audio_ids": sorted(row["audio_id"] for row in missing_rows),
        "locked_holdout_songs": 36,
        "locked_holdout_queries": 615,
        "locked_holdout_queries_hash_verified": 615,
        "locked_manifest_sha256": sha256_file(LOCKED_MANIFEST),
        "portable_locked_manifest": str(PORTABLE_MANIFEST.relative_to(ROOT)),
        "source_split_mapping_sha256": sha256_file(SOURCE_MAPPING),
        "source_query_manifest_sha256": sha256_file(SOURCE_MANIFEST),
        "gallery_songs": 300,
        "final_training_ready": False,
        "reason_not_training_ready": "29 original train and 3 original validation humming WAVs are absent from the available source ZIP; recover them before the final training run.",
        "external_kaggle_provenance_confirmed": False,
        "training_or_evaluation_run": False,
    }
    METADATA.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report = f"""# Dataset dựng lại theo split nguồn

**Trạng thái:** Đã dựng và kiểm tra cấu trúc/hash holdout; **chưa sẵn sàng cho train cuối** vì thiếu audio humming ở train/validation.

## Artifact

- ZIP: `{OUTPUT_ZIP.relative_to(ROOT)}`
- SHA256 ZIP: `{metadata['archive_sha256']}`
- Mapping nguồn: `{SOURCE_MAPPING.relative_to(ROOT)}` (SHA256 `{metadata['source_split_mapping_sha256']}`)
- Holdout portable: `{PORTABLE_MANIFEST.relative_to(ROOT)}`
- Contract/checksum: `{METADATA.relative_to(ROOT)}`

## Đã xác minh

- Phân hoạch đúng theo song ID từ mapping: **210 train / 45 validation / 45 test**; tất cả 300 bài và full-song gallery 300/300.
- Các file humming hiện có được chuyển theo `song_id`/`audio_id`, không ghép theo tên gần đúng; SHA256 khớp manifest nguồn cho **{sum(present_query_counts.values())} query**.
- Holdout sạch giữ nguyên **36 bài / 615 query**; cả 615 query nằm dưới split `test`, và SHA256 được kiểm tra lại trong ZIP.
- Không sửa `datasets.zip` gốc, không train hoặc đánh giá model.

## Thiếu dữ liệu trong nguồn hiện có

Archive nguồn không có 39 query WAV trong manifest nguồn: **29 train, 3 validation, 7 test**. Ba query thiếu thuộc clean holdout đã bị loại khỏi manifest khóa; cả 615 query khóa vẫn có mặt. Tuy nhiên 29 train và 3 validation làm dataset không đầy đủ để coi là bản train cuối. Cần khôi phục 32 file từ nguồn trước khi train.

```text
Train:  {present_query_counts['train']}/3612 query files present
Val:    {present_query_counts['val']}/766 query files present
Test:   {present_query_counts['test']}/809 query files present
Locked: 615/615 query files present and hash verified
```

Training notebook chỉ dùng audio train/validation cho loss và chọn checkpoint; test query không dùng để model selection. Dù vậy, chưa chạy train cho tới khi phục hồi đủ train/validation và xác nhận log Kaggle bên ngoài chưa dùng 36 bài holdout.
"""
    REPORT.write_text(report, encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
