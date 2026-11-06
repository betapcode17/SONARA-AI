#!/usr/bin/env python3
"""Precompute portable FP16 CQT features for V6.1 train/validation inputs.

Run on Kaggle GPU after extracting the V6.1 dataset to DATASET_ROOT. It excludes
test/holdout audio, keys features by relative path plus SHA-256, and writes a
separate cache dataset. Online extraction remains available as a fallback.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import random
import shutil
import zipfile
from pathlib import Path

import librosa
import numpy as np
import scipy
import soundfile as sf
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/kaggle/train/conformer_bytehum_qbh_final_v6_1_kaggle.ipynb"
SAMPLE_RATE = 16000
CHUNK_SAMPLES = 10 * SAMPLE_RATE
HOP_SAMPLES = 3 * SAMPLE_RATE
PREPROCESS_CONFIG = dict(sr=16000, chunk_seconds=10, hop_seconds=3,
                          final_anchored_window=True, bins=96, frames=83,
                          cqt_hop=0.04, cqt_mean=3, floor_db=-80.0)


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def chunk_starts(length: int):
    if length <= CHUNK_SAMPLES:
        return [0]
    starts = list(range(0, length - CHUNK_SAMPLES + 1, HOP_SAMPLES))
    if starts[-1] + CHUNK_SAMPLES < length:
        starts.append(length - CHUNK_SAMPLES)
    return starts


def load_runtime_classes(notebook_path: Path):
    nb = json.loads(notebook_path.read_text(encoding="utf-8"))
    ns = globals()
    exec("".join(nb["cells"][7]["source"]), ns)
    exec("".join(nb["cells"][8]["source"]), ns)
    backend_sha = hashlib.sha256("".join(nb["cells"][8]["source"]).encode("utf-8")).hexdigest()
    return ns["AudioPreprocessor"](), ns["CQTExtractor"](), backend_sha


def low_band_8bit(audio: np.ndarray, sr=16000, seed=0):
    down = scipy.signal.resample_poly(audio, 1, 2).astype(np.float32)
    restored = scipy.signal.resample_poly(down, 2, 1).astype(np.float32)
    rng = np.random.default_rng(seed)
    step = 2.0 / 255.0
    dither = (rng.random(restored.shape) - rng.random(restored.shape)) * step
    return (np.round((np.clip(restored, -1.0, 1.0) + dither) / step) * step).clip(-1.0, 1.0).astype(np.float32)


def get_audio_inventory(dataset_root: Path):
    # All train/validation query and reference files are needed by triplet training.
    paths = []
    for rel_root in ("dataset_split/train", "dataset_split/val",
                     "mir_qbsh_extension/train", "mir_qbsh_extension/val"):
        folder = dataset_root / rel_root
        if folder.is_dir():
            paths.extend(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in {".wav", ".flac", ".mp3"})

    # The validation gallery only contains train/validation titles; omit locked test songs.
    mapping_path = dataset_root / "song_name_mapping.json"
    gallery_dir = dataset_root / "2_music_database"
    id_to_title = json.loads(mapping_path.read_text(encoding="utf-8")) if mapping_path.exists() else {}
    trainval_titles = set()
    for split in ("train", "val"):
        folder = dataset_root / "dataset_split" / split
        if folder.exists():
            trainval_titles.update(p.name for p in folder.iterdir() if p.is_dir())
    for path in gallery_dir.glob("*.mp3") if gallery_dir.exists() else []:
        if id_to_title.get(path.stem) in trainval_titles:
            paths.append(path)
    return sorted(set(paths))


def query_source_map(dataset_root: Path):
    path = dataset_root / "qbh_query_manifest_v6_1.csv"
    if not path.is_file():
        return {}
    import csv
    with path.open(encoding="utf-8", newline="") as f:
        result = {}
        for row in csv.DictReader(f):
            rel = row["path"].replace("\\", "/")
            if rel.startswith("datasets/"):
                rel = rel[len("datasets/"):]
            result[rel] = row["source"]
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True,
                        help="Extracted datasets/ directory from the V6.1 ZIP")
    parser.add_argument("--output", type=Path, default=ROOT / "dist/qbh_cqt_cache_v6_1")
    parser.add_argument("--notebook", type=Path, default=NOTEBOOK,
                        help="V6.1 notebook carrying the exact audio/CQT preprocessing classes")
    parser.add_argument("--limit-audio", type=int, default=0,
                        help="Smoke-test only this many audio files; 0 means all train/validation inputs")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("Precomputed CQT generation requires CUDA (for example Kaggle T4); no CPU-scale cache generation is attempted.")
    if not args.dataset_root.is_dir():
        raise FileNotFoundError(args.dataset_root)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    prep, extractor, backend_sha = load_runtime_classes(args.notebook)
    extractor.cqt_impl.py_cqt._device = device
    extractor.cqt_impl.py_cqt._kernel = extractor.cqt_impl.py_cqt._kernel.to(device)
    # The notebook constructor selects the active CUDA device; enforce the requested one.
    source_by_path = query_source_map(args.dataset_root)
    inventory = get_audio_inventory(args.dataset_root)
    if args.limit_audio:
        inventory = inventory[:args.limit_audio]
    if not inventory:
        raise ValueError("No train/validation audio found")
    args.output.mkdir(parents=True, exist_ok=True)
    feature_dir = args.output / "features"
    feature_dir.mkdir(exist_ok=True)
    index = []
    feature_count = 0

    for path in inventory:
        rel = path.relative_to(args.dataset_root).as_posix()
        source_hash = digest_file(path)
        audio = prep.load_audio(str(path))
        variants = [("base", audio)]
        is_query = "/1/" in ("/" + rel) and not path.name.lower().startswith("reference")
        if is_query and source_by_path.get(rel) == "original":
            seed = int(source_hash[:16], 16)
            variants.append(("degraded_8khz_8bit", low_band_8bit(audio, seed=seed)))
        for variant, waveform in variants:
            chunks = []
            for start in chunk_starts(len(waveform)):
                chunk = waveform[start:start + CHUNK_SAMPLES]
                if len(chunk) < CHUNK_SAMPLES:
                    chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
                feature = extractor.extract(chunk)
                if feature.shape != (1, 96, 83) or not torch.isfinite(feature).all():
                    raise ValueError(f"Invalid CQT shape/value for {rel} ({variant}): {tuple(feature.shape)}")
                chunks.append(feature.detach().cpu().to(torch.float16).contiguous())
            identity = dict(path=rel, source_sha256=source_hash, variant=variant,
                            preprocessing=PREPROCESS_CONFIG)
            key = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            feature_rel = f"features/{key}.pt"
            feature_path = args.output / feature_rel
            torch.save(chunks, feature_path)
            index.append(dict(relative_path=rel, source_sha256=source_hash, variant=variant,
                              feature_file=feature_rel, feature_sha256=digest_file(feature_path), chunks=len(chunks),
                              tensor_shape=[1, 96, 83], dtype="float16", cache_key=key))
            feature_count += len(chunks)
        if len(index) % 100 == 0:
            print(f"Processed {len(index)} variants / {feature_count} chunks; latest={rel}", flush=True)

    manifest = dict(format_version=1, pipeline="qbh_final_v6_1", device=str(device),
                    cqt_backend_sha256=backend_sha,
                    preprocessing=PREPROCESS_CONFIG, includes_test_or_holdout=False,
                    audio_files=len(inventory), feature_variants=len(index), feature_chunks=feature_count,
                    features=index)
    (args.output / "qbh_cqt_manifest_v6_1.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    zip_path = args.output.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for path in args.output.rglob("*"):
            if path.is_file():
                z.write(path, path.relative_to(args.output).as_posix())
    print(json.dumps(dict(output=str(args.output), zip=str(zip_path), audio_files=len(inventory),
                          feature_variants=len(index), feature_chunks=feature_count,
                          compressed_bytes=zip_path.stat().st_size), indent=2))


if __name__ == "__main__":
    main()
