#!/usr/bin/env python3
"""Record whether source annotations or vocal references exist for MIR-QBSH."""
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_SOURCE = ROOT / "datasets_unused/mir_qbsh/MIR-QBSH.zip"
EXTENSION = ROOT / "dist/mir_qbsh_training_extension.zip"
PACKAGE = ROOT / "dist/datasets_qbh_clean_mirqbsh_70_15_15_v6_1.zip"
CONTRACT = ROOT / "dist/qbh_dataset_contract_v6_1.json"


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    with zipfile.ZipFile(RAW_SOURCE) as raw, zipfile.ZipFile(EXTENSION) as ext:
        raw_names = raw.namelist()
        ground_truth = list(csv.DictReader(io.StringIO(ext.read("mir_qbsh_training_extension/ground_truth.csv").decode("utf-8"))))
        columns = list(ground_truth[0]) if ground_truth else []
        timestamp_columns = [name for name in columns if any(token in name.lower() for token in ("start", "end", "time", "second", "offset"))]
        midi_count = sum(name.lower().endswith(".mid") or name.lower().endswith(".midi") for name in raw_names)
        raw_audio_count = sum(name.lower().endswith((".wav", ".mp3", ".flac", ".ogg")) for name in raw_names)
        extension_refs = [name for name in ext.namelist() if "/reference" in name.lower() and name.lower().endswith((".wav", ".flac"))]
        extension_vocals = [name for name in ext.namelist() if any(k in Path(name).name.lower() for k in ("vocal", "singing", "singer")) and name.lower().endswith((".wav", ".flac", ".mp3"))]
        audit = {
            "schema_version": 1,
            "source_archive": RAW_SOURCE.name,
            "source_archive_sha256": sha256_file(RAW_SOURCE),
            "source_midi_files": midi_count,
            "source_audio_files": raw_audio_count,
            "ground_truth_rows": len(ground_truth),
            "ground_truth_columns": columns,
            "ground_truth_timestamp_columns": timestamp_columns,
            "ground_truth_has_query_timestamps": bool(timestamp_columns),
            "extension_reference_files": len(extension_refs),
            "extension_vocal_audio_files": len(extension_vocals),
            "reference_type": "MIDI-synthesized full-song reference (22.05 kHz mono PCM16, per dataset manifest)",
            "vocal_replacement_status": "not_available_in_supplied_sources; no vocal references or SVS model outputs found",
            "timestamp_status": "not_available_in_ground_truth; query manifest contains blank start_second/end_second with explicit status",
        }

    audit_path = ROOT / "dist/qbh_reference_alignment_audit_v6_1.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    contract["reference_alignment_audit"] = "datasets/qbh_reference_alignment_audit_v6_1.json"
    contract["vocal_reference_available"] = False
    contract["vocal_reference_status"] = audit["vocal_replacement_status"]
    contract["timestamp_alignment_available"] = False
    contract["portable_cqt_cache_generated"] = False
    contract["portable_cqt_cache_status"] = "Not generated in the local CPU-only environment; scripts/precompute_cqt_cache.py requires CUDA and can produce an optional FP16 cache."
    contract["portable_cqt_cache_builder"] = "scripts/precompute_cqt_cache.py"
    CONTRACT.write_text(json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (ROOT / "dist/qbh_dataset_contract.json").write_bytes(CONTRACT.read_bytes())
    with zipfile.ZipFile(PACKAGE, "a", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as package:
        package.write(audit_path, "datasets/qbh_reference_alignment_audit_v6_1.json")
        package.write(CONTRACT, "datasets/qbh_dataset_contract_v6_1.json")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
