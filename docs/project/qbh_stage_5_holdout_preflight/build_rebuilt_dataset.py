"""Repack the QbH dataset under the verified source song split.

The 615-query clean holdout stays under dataset_split/test and is recorded in a
portable manifest. The training notebook only scans train/val humming rows;
the evaluation notebook must explicitly consume the locked manifest.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[3]
STAGE2 = ROOT / "02_stage2_humming_qbh"
SOURCE_ZIP = STAGE2 / "datasets.zip"
SOURCE_MANIFEST = Path.home() / ".codex/attachments/3f959db8-f012-4b77-b04c-ede89ec508bd/Văn bản đã dán.txt"
SPLIT_MAPPING = ROOT / "docs/project/qbh_stage_4_provenance/source_split_mapping.csv"
LOCKED_MANIFEST = ROOT / "docs/project/qbh_stage_5_holdout_preflight/locked_final_test_manifest.csv"
OUTPUT_ZIP = STAGE2 / "dist/qbh_v5_source_split_rebuilt_incomplete.zip"
OUTPUT_METADATA = ROOT / "docs/project/qbh_stage_5_holdout_preflight/rebuilt_dataset_manifest.json"
OUTPUT_LOCKED_CSV = ROOT / "docs/project/qbh_stage_5_holdout_preflight/locked_final_test_manifest_portable.csv"

CHUNK = 1024 * 1024
SPLITS = {"train", "val", "test"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_stream(stream, out=None) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(CHUNK), b""):
        digest.update(block)
        if out is not None:
            out.write(block)
    return digest.hexdigest()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def canonical_member(name: str) -> str:
    return name.replace("\\", "/").lstrip("/")


def main() -> int:
    for required in (SOURCE_ZIP, SOURCE_MANIFEST, SPLIT_MAPPING, LOCKED_MANIFEST):
        if not required.is_file():
            raise FileNotFoundError(required)

    mapping_rows = load_csv(SPLIT_MAPPING)
    mapping: dict[str, dict[str, str]] = {}
    for row in mapping_rows:
        sid, split = row["song_id"], row["source_split"]
        if sid in mapping or split not in SPLITS:
            raise ValueError(f"Invalid/duplicate source split mapping row: {row}")
        mapping[sid] = row
    if Counter(row["source_split"] for row in mapping.values()) != Counter(train=210, val=45, test=45):
        raise ValueError("Source mapping must contain exactly 210/45/45 songs")

    manifest_rows = load_csv(SOURCE_MANIFEST)
    source_queries: dict[str, dict[str, str]] = {}
    source_song_query_counts: Counter[str] = Counter()
    for row in manifest_rows:
        aid, sid = row["audio_id"], row["song_id"]
        if aid in source_queries:
            raise ValueError(f"Duplicate source audio_id: {aid}")
        if sid not in mapping or row["split"] != mapping[sid]["source_split"]:
            raise ValueError(f"Source manifest split disagrees with mapping: {aid}")
        source_queries[aid] = row
        source_song_query_counts[sid] += 1
    if len(manifest_rows) != 5187:
        raise ValueError(f"Expected 5,187 source query rows, got {len(manifest_rows)}")

    locked_rows = load_csv(LOCKED_MANIFEST)
    if len(locked_rows) != 615 or len({row["audio_id"] for row in locked_rows}) != 615:
        raise ValueError("Locked holdout must contain 615 unique queries")
    locked_by_id = {row["audio_id"]: row for row in locked_rows}
    for aid, row in locked_by_id.items():
        source = source_queries.get(aid)
        if source is None or source["split"] != "test" or source["song_id"] != row["song_id"]:
            raise ValueError(f"Locked query is not from original test split: {aid}")
        if source["humming_sha256"].lower() != row["expected_humming_sha256"].lower():
            raise ValueError(f"Locked/source hash disagreement: {aid}")

    OUTPUT_ZIP.parent.mkdir(parents=True, exist_ok=True)
    temp_zip = OUTPUT_ZIP.with_suffix(OUTPUT_ZIP.suffix + ".tmp")
    if temp_zip.exists():
        temp_zip.unlink()

    folder_to_song: dict[str, str] = {}
    query_file_count: Counter[str] = Counter()
    verified_sha_count = Counter()
    missing_ids: set[str] = set()
    portable_locked: list[dict[str, str]] = []
    route_counts: Counter[str] = Counter()
    all_members_seen: set[str] = set()

    print(f"Opening source archive ({SOURCE_ZIP.stat().st_size / 1024**3:.2f} GiB)", flush=True)
    with zipfile.ZipFile(SOURCE_ZIP, "r") as source_zip:
        infos = [info for info in source_zip.infolist() if not info.is_dir()]
        normalized_names = [canonical_member(info.filename) for info in infos]
        if len(normalized_names) != len(set(normalized_names)):
            raise ValueError("Source ZIP contains duplicate normalized member paths")

        # Map every song directory to the authoritative song_id using query IDs,
        # never by fuzzy title matching.
        for name in normalized_names:
            path = PurePosixPath(name)
            parts = path.parts
            if len(parts) >= 6 and parts[0] == "datasets" and parts[1] == "dataset_split":
                split, folder = parts[2], parts[3]
                if split not in SPLITS:
                    raise ValueError(f"Unexpected split directory in source archive: {name}")
                if path.suffix.lower() == ".wav" and path.stem in source_queries:
                    sid = source_queries[path.stem]["song_id"]
                    old = folder_to_song.setdefault(folder, sid)
                    if old != sid:
                        raise ValueError(f"Song folder maps to multiple song_ids: {folder}")
        if len(folder_to_song) != 300 or set(folder_to_song.values()) != set(mapping):
            raise ValueError(
                f"Could not map all 300 dataset song folders via audio_id; mapped {len(folder_to_song)}"
            )

        # Confirm every present source query can be found exactly once. Missing
        # IDs are explicitly allowed only when listed as absent in preflight.
        present_query_paths: dict[str, str] = {}
        for name in normalized_names:
            path = PurePosixPath(name)
            if path.suffix.lower() == ".wav" and path.stem in source_queries:
                aid = path.stem
                if aid in present_query_paths:
                    raise ValueError(f"Query ID occurs more than once in source ZIP: {aid}")
                present_query_paths[aid] = name
        missing_ids = set(source_queries) - set(present_query_paths)
        locked_missing = set(locked_by_id) & missing_ids
        if locked_missing:
            raise ValueError(f"A locked holdout WAV is missing from source ZIP: {sorted(locked_missing)[:5]}")
        missing_by_split = Counter(source_queries[aid]["split"] for aid in missing_ids)
        present_by_song = Counter(source_queries[aid]["song_id"] for aid in present_query_paths)
        songs_without_queries = [
            sid for sid, row in mapping.items()
            if row["source_split"] in ("train", "val") and not present_by_song[sid]
        ]
        if songs_without_queries:
            raise ValueError(f"Train/validation songs with no available query: {songs_without_queries[:10]}")

        with zipfile.ZipFile(temp_zip, "w", allowZip64=True, compression=zipfile.ZIP_DEFLATED,
                             compresslevel=1) as output_zip:
            for index, (info, old_name) in enumerate(zip(infos, normalized_names), 1):
                path = PurePosixPath(old_name)
                if old_name in ("datasets/dataset_split/split_manifest.json",
                                "datasets/dataset_split/split_summary.csv"):
                    continue

                new_name = old_name
                if len(path.parts) >= 5 and path.parts[0:2] == ("datasets", "dataset_split"):
                    old_split, folder = path.parts[2], path.parts[3]
                    sid = folder_to_song.get(folder)
                    if sid is None:
                        raise ValueError(f"No song_id for dataset folder {folder}")
                    new_split = mapping[sid]["source_split"]
                    new_name = PurePosixPath("datasets", "dataset_split", new_split, *path.parts[3:]).as_posix()
                    if old_split != new_split:
                        route_counts[f"{old_split}->{new_split}"] += 1

                if new_name in all_members_seen:
                    raise ValueError(f"Path collision after source split routing: {new_name}")
                all_members_seen.add(new_name)

                qid = path.stem if path.suffix.lower() == ".wav" else None
                expected = source_queries.get(qid) if qid else None
                with source_zip.open(info, "r") as input_stream:
                    with output_zip.open(new_name, "w", force_zip64=True) as out_stream:
                        if expected:
                            actual_hash = sha256_stream(input_stream, out_stream)
                            if actual_hash.lower() != expected["humming_sha256"].lower():
                                raise ValueError(f"Source query SHA256 mismatch: {qid}")
                            verified_sha_count[expected["split"]] += 1
                            query_file_count[expected["split"]] += 1
                            if qid in locked_by_id:
                                target = PurePosixPath("datasets/dataset_split/test", *path.parts[3:]).as_posix()
                                portable_locked.append({**locked_by_id[qid], "dataset_relpath": target})
                        else:
                            shutil.copyfileobj(input_stream, out_stream, length=CHUNK)
                if index % 500 == 0:
                    print(f"Repacked {index}/{len(infos)} source members", flush=True)

            split_manifest = {}
            for folder, sid in sorted(folder_to_song.items()):
                row = mapping[sid]
                split_manifest[folder] = {
                    "split": row["source_split"],
                    "song_id": sid,
                    "source_title": row["source_title"],
                    "num_hummings": present_by_song[sid],
                    "source_manifest_num_hummings": source_song_query_counts[sid],
                }
            split_json = json.dumps(split_manifest, ensure_ascii=False, indent=2).encode("utf-8")
            output_zip.writestr("datasets/dataset_split/split_manifest.json", split_json)

            summary_rows = []
            for folder, sid in sorted(folder_to_song.items()):
                row = mapping[sid]
                summary_rows.append({
                    "split": row["source_split"], "song_name": folder,
                    "song_id": sid, "num_hummings": present_by_song[sid],
                    "source_manifest_num_hummings": source_song_query_counts[sid],
                })
            summary_fields = ["split", "song_name", "song_id", "num_hummings", "source_manifest_num_hummings"]
            import io
            summary_buffer = io.StringIO(newline="")
            summary_writer = csv.DictWriter(summary_buffer, fieldnames=summary_fields)
            summary_writer.writeheader()
            summary_writer.writerows(summary_rows)
            summary_text = summary_buffer.getvalue().encode("utf-8")
            output_zip.writestr("datasets/dataset_split/split_summary.csv", summary_text)

            portable_fields = list(locked_rows[0].keys()) + ["dataset_relpath"]
            if len(portable_locked) != 615 or {r["audio_id"] for r in portable_locked} != set(locked_by_id):
                raise ValueError(f"Portable locked manifest incomplete: {len(portable_locked)}")
            portable_locked.sort(key=lambda row: row["audio_id"])
            csv_buffer = io.StringIO(newline="")
            writer = csv.DictWriter(csv_buffer, fieldnames=portable_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(portable_locked)
            output_zip.writestr("datasets/holdout/locked_final_test_manifest.csv",
                                csv_buffer.getvalue().encode("utf-8"))

            package_contract = {
                "schema_version": 1,
                "source_split_counts_songs": {"train": 210, "val": 45, "test": 45},
                "source_query_counts": {"train": 3612, "val": 766, "test": 809},
                "query_files_present_and_sha_verified": dict(verified_sha_count),
                "missing_source_query_files": sorted(missing_ids),
                "missing_source_query_counts_by_split": dict(missing_by_split),
                "final_training_ready": not bool(missing_by_split.get("train", 0) or missing_by_split.get("val", 0)),
                "locked_final_test": {"songs": 36, "queries": 615,
                                       "manifest": "datasets/holdout/locked_final_test_manifest.csv",
                                       "policy": "Use only after model selection; never train or select on these queries."},
                "source_split_mapping_sha256": sha256_file(SPLIT_MAPPING),
                "source_humming_manifest_sha256": sha256_file(SOURCE_MANIFEST),
                "locked_manifest_sha256": sha256_file(LOCKED_MANIFEST),
                "note": "All song folders were routed by audio_id/song_id. Full gallery retained for V5 validation; test humming is not part of train/val rows.",
            }
            output_zip.writestr("datasets/qbh_dataset_contract.json",
                                json.dumps(package_contract, ensure_ascii=False, indent=2).encode("utf-8"))

    expected_present = Counter({split: sum(row["split"] == split for row in source_queries.values())
                                - missing_by_split[split] for split in SPLITS})
    if query_file_count != expected_present:
        raise ValueError(f"Repacked query counts differ: {dict(query_file_count)}")
    if verified_sha_count != query_file_count:
        raise ValueError("Not all repacked humming queries passed SHA256 verification")
    if temp_zip.stat().st_size < 1_000_000_000:
        raise ValueError("Output ZIP size unexpectedly small")

    # Re-open the result and check key structural invariants before publishing.
    with zipfile.ZipFile(temp_zip, "r") as check_zip:
        names = [canonical_member(info.filename) for info in check_zip.infolist()]
        top = set(names)
        split_dirs = defaultdict(set)
        gallery_stems = set()
        actual_locked = check_zip.read("datasets/holdout/locked_final_test_manifest.csv")
        if len(list(csv.DictReader(actual_locked.decode("utf-8").splitlines()))) != 615:
            raise ValueError("Packaged locked manifest does not contain exactly 615 rows")
        for name in names:
            path = PurePosixPath(name)
            if len(path.parts) >= 4 and path.parts[:2] == ("datasets", "dataset_split"):
                if path.parts[2] in SPLITS and path.parts[3] not in ("split_manifest.json", "split_summary.csv"):
                    split_dirs[path.parts[2]].add(path.parts[3])
            if len(path.parts) == 3 and path.parts[:2] == ("datasets", "2_music_database") and path.suffix.lower() in AUDIO_EXTS:
                gallery_stems.add(path.stem)
        expected_counts = {"train": 210, "val": 45, "test": 45}
        if {k: len(v) for k, v in split_dirs.items()} != expected_counts:
            raise ValueError(f"Packaged split directory counts mismatch: {dict((k,len(v)) for k,v in split_dirs.items())}")
        if set().union(*split_dirs.values()) != gallery_stems or len(gallery_stems) != 300:
            raise ValueError("Packaged gallery does not exactly match the 300 split song folders")
        if len(top) < 5000:
            raise ValueError("Output ZIP entry count unexpectedly low")

    os.replace(temp_zip, OUTPUT_ZIP)
    output_meta = {
        "status": "built_with_missing_source_queries" if missing_ids else "complete",
        "archive": str(OUTPUT_ZIP.relative_to(ROOT)),
        "archive_bytes": OUTPUT_ZIP.stat().st_size,
        "archive_sha256": sha256_file(OUTPUT_ZIP),
        "source_archive_sha256": sha256_file(SOURCE_ZIP),
        "song_counts": {"train": 210, "val": 45, "test": 45},
        "query_file_counts_sha_verified": dict(verified_sha_count),
        "source_queries_missing": sorted(missing_ids),
        "source_queries_missing_by_split": dict(missing_by_split),
        "final_training_ready": not bool(missing_by_split.get("train", 0) or missing_by_split.get("val", 0)),
        "locked_holdout_song_count": 36,
        "locked_holdout_query_count": 615,
        "locked_manifest_sha256": sha256_file(LOCKED_MANIFEST),
        "mapping_sha256": sha256_file(SPLIT_MAPPING),
        "gallery_song_count": 300,
        "query_ids_reassigned_by_audio_id": len(source_queries) - len(missing_ids),
        "query_sha256_matches": sum(verified_sha_count.values()),
        "split_file_routes": dict(route_counts),
        "portable_locked_manifest": str(OUTPUT_LOCKED_CSV.relative_to(ROOT)),
        "training_or_evaluation_run": False,
    }
    OUTPUT_METADATA.write_text(json.dumps(output_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    OUTPUT_LOCKED_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT_LOCKED_CSV, portable_locked, list(locked_rows[0].keys()) + ["dataset_relpath"])
    print(json.dumps(output_meta, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"BUILD FAILED: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise
