#!/usr/bin/env python3
"""Build the portable QBH V6.1 package (IDs, FLAC, clean manifests).

The V6 source archive is never modified. Train/validation WAVs are encoded as
lossless FLAC; test/holdout WAVs remain byte-for-byte unchanged. Gallery MP3
bytes remain unchanged and receive stable ASCII song IDs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import unicodedata
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
SOURCE_ZIP = DIST / "datasets_qbh_clean_mirqbsh_70_15_15_v6.zip"
OUTPUT_ZIP = DIST / "datasets_qbh_clean_mirqbsh_70_15_15_v6_1.zip"
CONTRACT_IN = DIST / "qbh_dataset_contract.json"
OUTPUT_CONTRACT = DIST / "qbh_dataset_contract_v6_1.json"
OUTPUT_HASHES = DIST / "qbh_dataset_audio_sha256_v6_1.json"
OUTPUT_QUERY_MANIFEST = DIST / "qbh_query_manifest_v6_1.csv"
NOTEBOOK_IN = ROOT / "notebooks/kaggle/train/conformer_bytehum_qbh_final_v6_kaggle.ipynb"
NOTEBOOK_OUT = ROOT / "notebooks/kaggle/train/conformer_bytehum_qbh_final_v6_1_kaggle.ipynb"
EVAL_IN = ROOT / "notebooks/kaggle/evaluate/conformer_bytehum_qbh_final_evaluation.ipynb"
EVAL_OUT = ROOT / "notebooks/kaggle/evaluate/conformer_bytehum_qbh_final_v6_1_evaluation.ipynb"
COMPARE_KAGGLE = ROOT / "notebooks/kaggle/evaluate/conformer_bytehum_qbh_v3_vs_v5_300song.ipynb"
COMPARE_LOCAL = ROOT / "notebooks/local_4gb/evaluate/conformer_bytehum_qbh_v3_vs_v5_300song_local_4gb.ipynb"
CUDA_CQT_SOURCE = ROOT / "notebooks/kaggle/train/conformer_bytehum_qbh_final_train_kaggle_mir_qbsh_integrated.ipynb"


def jbytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def stable_title_key(title: str):
    return unicodedata.normalize("NFC", title).casefold(), title


def normalized_alias(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value)).casefold()
    return "".join(char for char in value if char.isalnum())


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_ids(title_to_stem: dict[str, str], extra_aliases: dict[str, set[str]] | None = None,
             stable_stem_order: list[str] | None = None):
    if len(set(title_to_stem.values())) != len(title_to_stem):
        raise ValueError("Gallery mapping is not one-to-one; refusing ID migration")
    if stable_stem_order is None:
        ordered = sorted(title_to_stem, key=stable_title_key)
    else:
        stem_order = {stem: index for index, stem in enumerate(stable_stem_order)}
        if set(stem_order) != set(title_to_stem.values()):
            raise ValueError("Stable gallery stem order does not cover the 300 mapped songs")
        ordered = sorted(title_to_stem, key=lambda title: stem_order[title_to_stem[title]])
    id_to_title = {f"song_{i:04d}": title for i, title in enumerate(ordered, 1)}
    title_to_id = {}
    id_to_legacy_filename = {}
    for song_id, title in id_to_title.items():
        stem = title_to_stem[title]
        for alias in {title, stem, *(extra_aliases or {}).get(title, set())}:
            prior = title_to_id.get(alias)
            if prior is not None and prior != song_id:
                raise ValueError(f"Ambiguous gallery alias {alias!r}")
            title_to_id[alias] = song_id
        id_to_legacy_filename[song_id] = stem + ".mp3"
    return id_to_title, title_to_id, id_to_legacy_filename


def map_split_titles_to_gallery(current_title_to_stem, split_manifest, combined_manifest, legacy_title_to_stem):
    """Use the exact query folder labels, while retaining the corresponding MP3s."""
    # V6 repaired the single comma-only filename. The pre-V6 title map preserves
    # the original relationship between query folder labels and gallery tracks.
    current_by_title = {normalized_alias(title): stem for title, stem in current_title_to_stem.items()}
    current_by_stem = {normalized_alias(stem): stem for stem in current_title_to_stem.values()}
    alias_to_current_stems = {}
    aliases_by_stem = {}
    for old_title, old_stem in legacy_title_to_stem.items():
        current_stem = current_title_to_stem.get(old_title)
        if current_stem is None:
            current_stem = current_by_stem.get(normalized_alias(old_stem))
        if current_stem is None:
            # V6 may have renamed the path while keeping the old title key.
            current_stem = current_by_title.get(normalized_alias(old_title))
        if current_stem is None:
            raise ValueError(f"Cannot resolve legacy gallery entry {old_title!r} -> {old_stem!r}")
        for alias in (old_title, old_stem, current_stem):
            alias_to_current_stems.setdefault(normalized_alias(alias), set()).add(current_stem)
            aliases_by_stem.setdefault(current_stem, set()).add(alias)

    songs_by_id = {row.get("song_id"): row for row in combined_manifest.get("songs", [])}
    labels_to_stems = {}
    extra_aliases = {}
    for folder_title, info in split_manifest.items():
        combined_song = songs_by_id.get(info.get("song_id"), {})
        candidates = {folder_title, info.get("source_title", ""),
                      combined_song.get("song_key", ""), combined_song.get("title", "")}
        matched_stems = set()
        for candidate in candidates:
            if candidate:
                matched_stems.update(alias_to_current_stems.get(normalized_alias(candidate), set()))
        if len(matched_stems) != 1:
            raise ValueError(f"Expected one gallery match for split title {folder_title!r}; got {sorted(matched_stems)}")
        stem = next(iter(matched_stems))
        labels_to_stems[folder_title] = stem
        extra_aliases[folder_title] = candidates | aliases_by_stem.get(stem, set())
    if len(labels_to_stems) != 300 or len(set(labels_to_stems.values())) != 300:
        raise ValueError(f"Expected one-to-one split/gallery mapping for 300 songs; got {len(labels_to_stems)} labels and {len(set(labels_to_stems.values()))} files")
    return labels_to_stems, extra_aliases


MIR_TRAINING_BLOCK = '''MIR_QBSH_ROOT = DATASET_SPLIT_DIR.parent / 'mir_qbsh_extension'
MIR_QBSH_MANIFEST_PATH = MIR_QBSH_ROOT / 'manifest.json'
MIR_QBSH_GT_PATH = MIR_QBSH_ROOT / 'ground_truth.csv'
if not MIR_QBSH_MANIFEST_PATH.is_file() or not MIR_QBSH_GT_PATH.is_file():
    raise FileNotFoundError('V6.1 requires the packaged MIR-QBSH manifest and ground_truth.csv')
MIR_QBSH_MANIFEST = json.loads(MIR_QBSH_MANIFEST_PATH.read_text(encoding='utf-8'))
_mir_gt = pd.read_csv(MIR_QBSH_GT_PATH, dtype={'Song ID': str})
_required_mir_columns = {'Filename', 'Song ID', 'split', 'song_key'}
if not _required_mir_columns.issubset(_mir_gt.columns):
    raise ValueError(f'MIR-QBSH ground truth lacks columns: {_required_mir_columns - set(_mir_gt.columns)}')
_mir_songs = {row['song_id']: row for row in MIR_QBSH_MANIFEST.get('songs', [])}
if len(_mir_songs) != 48 or len(_mir_gt) != 4431:
    raise ValueError('MIR-QBSH inventory must contain 48 songs and 4,431 queries')
MIR_QBSH_EXTRA_ROWS_BY_SPLIT = {'train': [], 'val': [], 'test': []}
MIR_QBSH_EXTRA_SONGS_BY_SPLIT = {'train': [], 'val': [], 'test': []}
for _sid, _song_meta in sorted(_mir_songs.items()):
    _song = _song_meta['song_key']
    _split = _song_meta['split']
    if _split not in MIR_QBSH_EXTRA_ROWS_BY_SPLIT or _song != f'MIR-QBSH-{_sid}':
        raise ValueError(f'Invalid MIR-QBSH song identity/split: {_sid}/{_split}')
    _song_dir = MIR_QBSH_ROOT / _split / _song / '1'
    _reference = MIR_QBSH_ROOT / _song_meta['reference_path']
    if not _reference.is_file():
        _reference = _reference.with_suffix('.flac')
    if not _reference.is_file():
        raise FileNotFoundError(f'Missing MIR-QBSH reference: {_reference}')
    MIR_QBSH_EXTRA_SONGS_BY_SPLIT[_split].append(_song)
    GALLERY_PATHS[_song] = str(_reference)
    _song_rows = _mir_gt[_mir_gt['Song ID'] == _sid]
    if len(_song_rows) != int(_song_meta['query_count']):
        raise ValueError(f'MIR-QBSH query count differs from manifest for {_sid}')
    for _record in _song_rows.to_dict('records'):
        _filename = Path(str(_record['Filename']))
        if _filename.name != str(_record['Filename']) or _filename.suffix.lower() not in {'.wav', '.flac'}:
            raise ValueError(f'Unsafe MIR-QBSH query filename: {_record["Filename"]}')
        _query = _song_dir / _filename
        if not _query.is_file():
            _query = _query.with_suffix('.flac')
        if not _query.is_file():
            raise FileNotFoundError(f'Missing MIR-QBSH query: {_query}')
        MIR_QBSH_EXTRA_ROWS_BY_SPLIT[_split].append(dict(
            song=_song, fragment='1', path=str(_query),
            key=f'mir_qbsh/{_split}/{_song}/1/{_query.name}',
            refs=[str(_reference)], vocals=[]))
for _split in ('train', 'val', 'test'):
    SPLIT_IDS[_split] = sorted(SPLIT_IDS[_split] + MIR_QBSH_EXTRA_SONGS_BY_SPLIT[_split])
if {k: len(v) for k, v in MIR_QBSH_EXTRA_SONGS_BY_SPLIT.items()} != {'train': 34, 'val': 7, 'test': 7}:
    raise ValueError('MIR-QBSH song split inventory is incomplete')
if {k: len(v) for k, v in MIR_QBSH_EXTRA_ROWS_BY_SPLIT.items()} != {'train': 3101, 'val': 665, 'test': 665}:
    raise ValueError('MIR-QBSH query split inventory is incomplete')
TRAIN_ROWS.extend(MIR_QBSH_EXTRA_ROWS_BY_SPLIT['train'])
VAL_ROWS.extend(MIR_QBSH_EXTRA_ROWS_BY_SPLIT['val'])
if {k: len(v) for k, v in SPLIT_IDS.items()} != {'train': 244, 'val': 52, 'test': 52}:
    raise ValueError(f'V6.1 song split inventory mismatch: { {k: len(v) for k, v in SPLIT_IDS.items()} }')
if {row['song'] for row in TRAIN_ROWS} != set(SPLIT_IDS['train']) or {row['song'] for row in VAL_ROWS} != set(SPLIT_IDS['val']):
    raise ValueError('Every V6.1 train/validation song must have resolved queries and positive references')
if len(TRAIN_ROWS) != 6684 or len(VAL_ROWS) != 1428:
    raise ValueError(f'V6.1 query inventory mismatch: train={len(TRAIN_ROWS)}, val={len(VAL_ROWS)}')
RUN_CONFIG['dataset_policy'] = 'v6_1_clean_original_plus_mir_qbsh_70_15_15'
qbh_success('Integrated MIR-QBSH: train=34/3101, val=7/665, test=7/665')
'''

TRAIN_SELECTION_BLOCK = '''split_candidates = list(INPUT_ROOT.rglob('dataset_split'))
music_candidates = list(INPUT_ROOT.rglob('2_music_database'))

def is_v61_dataset_root(split_dir):
    root = Path(split_dir).parent
    required = all((root / rel).is_file() for rel in (
        'song_name_mapping.json', 'song_title_to_id.json',
        'qbh_dataset_audio_sha256.json', 'mir_qbsh_extension/manifest.json',
        'qbh_dataset_contract.json'))
    if not required:
        return False
    contract = json.loads((root / 'qbh_dataset_contract.json').read_text(encoding='utf-8'))
    return contract.get('schema_version') == 3 and contract.get('final_training_ready') is True

split_candidates = [p for p in split_candidates if is_v61_dataset_root(p)]
music_candidates = [p for p in music_candidates if is_v61_dataset_root(p.parent / 'dataset_split')]
archive_candidates = []
for path in INPUT_ROOT.rglob('*'):
    if path.suffix.lower() not in ('.zip', '.bin', '.dat') or not path.is_file() or not zipfile.is_zipfile(path):
        continue
    try:
        with zipfile.ZipFile(path) as archive:
            members = set(archive.namelist())
            required = {'datasets/dataset_split/split_manifest.json',
                        'datasets/song_title_to_id.json',
                        'datasets/qbh_query_manifest_v6_1.csv',
                        'datasets/qbh_dataset_audio_sha256.json',
                        'datasets/mir_qbsh_extension/manifest.json',
                        'datasets/qbh_dataset_contract.json'}
            if required.issubset(members):
                archive_contract = json.loads(archive.read('datasets/qbh_dataset_contract.json'))
                if archive_contract.get('schema_version') == 3 and archive_contract.get('final_training_ready'):
                    archive_candidates.append(path)
    except Exception:
        continue

if DATASET_ARCHIVE_OVERRIDE is not None or (archive_candidates and DATASET_OVERRIDE is None and MUSIC_OVERRIDE is None):
    archive_path = unique_path(DATASET_ARCHIVE_OVERRIDE, archive_candidates, 'V6.1 DATASET_ARCHIVE_OVERRIDE')
    unpacked = extract_dataset_archive(archive_path, OUTPUT_ROOT / 'dataset_unpacked')
    split_candidates = list(unpacked.rglob('dataset_split'))
    music_candidates = list(unpacked.rglob('2_music_database'))
if DATASET_OVERRIDE is not None:
    split_candidates = [Path(DATASET_OVERRIDE).resolve()]
if MUSIC_OVERRIDE is not None:
    music_candidates = [Path(MUSIC_OVERRIDE).resolve()]
if not split_candidates or not music_candidates:
    raise FileNotFoundError('Attach/extract the V6.1 dataset ZIP or set its extracted V6.1 split/gallery paths.')
DATASET_SPLIT_DIR = unique_path(DATASET_OVERRIDE, split_candidates, 'DATASET_OVERRIDE')
MUSIC_DB_DIR = unique_path(MUSIC_OVERRIDE, music_candidates, 'MUSIC_OVERRIDE')
if not is_v61_dataset_root(DATASET_SPLIT_DIR):
    raise ValueError('Selected dataset is not a complete V6.1 bundle')
'''

TRAIN_EXTRACT_BLOCK = '''def extract_dataset_archive(archive, base):
    h = hashlib.sha256()
    with Path(archive).open('rb') as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(part)
    signature = h.hexdigest()
    destination = (Path(base) / signature).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / '.extracted.json'
    if marker.exists():
        if json.loads(marker.read_text(encoding='utf-8')) != {'sha256': signature}:
            raise ValueError('Dataset extraction marker mismatch')
        return destination
    with zipfile.ZipFile(archive) as zipped:
        planned, seen = [], set()
        for item in zipped.infolist():
            normalized = item.filename
            if normalized.startswith('/') or any(part in ('..', '') or ':' in part for part in normalized.rstrip('/').split('/')):
                raise ValueError('Unsafe archive member: ' + item.filename)
            target = (destination / normalized).resolve()
            if not target.is_relative_to(destination) or stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError('Archive traversal/symlink: ' + item.filename)
            if target in seen or target == marker:
                raise ValueError('Duplicate/reserved archive member: ' + item.filename)
            seen.add(target)
            planned.append((item, target))
        required = sum(item.file_size for item, _ in planned)
        if required > shutil.disk_usage(destination).free - 2 * 1024 ** 3:
            raise OSError('Not enough disk space to unpack dataset and retain training outputs.')
        for item, target in tqdm(planned, desc='Unpack V6.1 dataset'):
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with zipped.open(item) as source, target.open('wb') as output:
                    shutil.copyfileobj(source, output)
    temp = marker.with_suffix('.tmp')
    temp.write_text(json.dumps({'sha256': signature}), encoding='utf-8')
    os.replace(temp, marker)
    return destination
'''

V61_RUN_CONFIG = '''RUN_CONFIG = dict(pipeline='qbh_final_v6_1', seed=42, epochs=40, patience=8,
                  songs_per_batch=8, samples_per_song=2, accumulation=4,
                  freeze_blocks=6, gradual_unfreeze_epoch=6, embedding_dim=128, vocal_ratio=0.80,
                  global_triplet=1.0, lac=0.25, lat=1.0, grl=0.2,
                  prototype_diversity=0.05, margin=0.2, mining_top_k=5,
                  memory_capacity=4096, lr_backbone=1e-4, lr_heads=3e-4,
                  positive_selection='best fused chunk from same-fragment reference bag',
                  weight_decay=0.01, clip=1.0, amp='fp16',
                  low_band_degradation_probability=0.30, degradation_sample_rate=8000,
                  degradation_bit_depth=8, degradation_dither='TPDF',
                  lr_unfreeze_lower=1e-5, lr_unfreeze_upper=5e-5,
                  validation_gallery='train+val full recordings',
                  selection='fused_macro_song_mrr', score='0.5 local MaxMean + 0.5 global max',
                  test_gallery='all full recordings', dataset_policy='v6_1_clean_original_plus_mir_qbsh_70_15_15',
                  tie_rtol=1e-6, tie_atol=1e-7)'''

V61_DATA_CONTRACT_BLOCK = '''FILE_HASHES = {}
def file_sha256(path):
    path = Path(path).resolve()
    stat = path.stat()
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    if key not in FILE_HASHES:
        h = hashlib.sha256()
        with path.open('rb') as handle:
            for part in iter(lambda: handle.read(1024 * 1024), b''):
                h.update(part)
        FILE_HASHES[key] = h.hexdigest()
    try:
        relpath = path.relative_to(MUSIC_DB_DIR.parent.resolve()).as_posix()
    except ValueError:
        relpath = None
    if relpath is not None:
        expected = AUDIO_SHA_INDEX.get('datasets/' + relpath)
        if expected is None or FILE_HASHES[key] != expected:
            raise ValueError(f'Audio SHA-256 differs from the V6.1 inventory: {relpath}')
    return FILE_HASHES[key]

def query_records(rows):
    return [(row['key'], row['song'], file_sha256(row['path'])) for row in rows]

def positive_contract_path(path):
    path = Path(path).resolve()
    for base, prefix in ((DATASET_SPLIT_DIR, 'split/'), (MUSIC_DB_DIR, 'gallery/'),
                         (MIR_QBSH_ROOT, 'mir_qbsh/')):
        try:
            return prefix + path.relative_to(Path(base).resolve()).as_posix()
        except ValueError:
            continue
    raise ValueError(f'Positive path outside V6.1 dataset roots: {path}')

def data_contract():
    train_q, val_q = query_records(TRAIN_ROWS), query_records(VAL_ROWS)
    if {row[2] for row in train_q} & {row[2] for row in val_q}:
        raise ValueError('Duplicate humming audio across train/validation')
    positives = sorted({p for row in TRAIN_ROWS for p in row['refs'] + row['vocals']})
    return dict(split_ids=SPLIT_IDS, train_queries=train_q, val_queries=val_q,
                positive_reference_policy=RUN_CONFIG['positive_selection'],
                full_song_positive_fallbacks=sorted(set(REFERENCE_FALLBACKS)),
                positives=[(positive_contract_path(p), file_sha256(p)) for p in positives],
                val_gallery=[(song, file_sha256(path)) for song, path in VAL_GALLERY_PATHS.items()])

DATA_CONTRACT = json.loads(json.dumps(data_contract()))
DATA_DIGEST = digest(DATA_CONTRACT)'''


def query_path(parts: list[str], suffix: str) -> bool:
    if len(parts) < 6 or parts[0] != "datasets":
        return False
    if parts[1] == "dataset_split":
        return parts[2] in {"train", "val", "test"} and suffix.lower() in {".wav", ".flac"} and not parts[-1].lower().startswith("reference")
    if parts[1] == "mir_qbsh_extension":
        return parts[2] in {"train", "val", "test"} and suffix.lower() in {".wav", ".flac"} and not parts[-1].lower().startswith("reference")
    return False


def is_train_val_wav(name: str) -> bool:
    parts = name.replace("\\", "/").split("/")
    return (len(parts) >= 6 and parts[0] == "datasets" and parts[2] in {"train", "val"}
            and parts[1] in {"dataset_split", "mir_qbsh_extension"} and Path(name).suffix.lower() == ".wav")


def flac_encode(payload: bytes):
    info = sf.info(io.BytesIO(payload))
    data, sr = sf.read(io.BytesIO(payload), dtype="float32", always_2d=True)
    if info.subtype == "PCM_24":
        subtype = "PCM_24"
    elif info.subtype in {"PCM_16", "PCM_U8", "PCM_S8"}:
        subtype = "PCM_16"
    else:
        return payload, info, None
    target = io.BytesIO()
    sf.write(target, data, sr, format="FLAC", subtype=subtype)
    encoded = target.getvalue()
    decoded, decoded_sr = sf.read(io.BytesIO(encoded), dtype="float32", always_2d=True)
    if decoded_sr != sr or decoded.shape != data.shape:
        raise ValueError(f"FLAC round-trip shape/sample-rate mismatch: {decoded.shape} vs {data.shape}")
    error = float(np.max(np.abs(decoded - data))) if data.size else 0.0
    tolerance = 1.1 / (2 ** (24 if subtype == "PCM_24" else 16))
    if error > tolerance:
        raise ValueError(f"FLAC round-trip changed decoded PCM by {error}; tolerance {tolerance}")
    return encoded, info, dict(flac_subtype=subtype, max_decoded_pcm_error=error, lossless_pcm_verified=True)


def patch_notebooks(id_to_title: dict[str, str]):
    cuda_nb = json.loads(CUDA_CQT_SOURCE.read_text(encoding="utf-8"))
    cqt_backend_sha256 = hashlib.sha256("".join(cuda_nb["cells"][8]["source"]).encode("utf-8")).hexdigest()
    target_block = """GALLERY_PATHS = {}
for path in sorted(MUSIC_DB_DIR.iterdir()):
    if path.suffix.lower() not in AUDIO_EXTENSIONS:
        continue
    GALLERY_PATHS[path.stem] = str(path)
"""
    eval_target_block = """GALLERY_PATHS = {}
for path in sorted(MUSIC_DB_DIR.iterdir()):
    if path.suffix.lower() not in AUDIO_EXTENSIONS:
        continue
    if path.stem in GALLERY_PATHS:
        raise ValueError(f'Duplicate gallery song: {path.stem}')
    GALLERY_PATHS[path.stem] = str(path)
if set(GALLERY_PATHS) != set().union(*map(set, SPLIT_IDS.values())):
    raise ValueError('Full music gallery must exactly match the train/val/test song IDs.')
"""
    replacement = """GALLERY_MAP_PATH = MUSIC_DB_DIR.parent / 'song_name_mapping.json'
if not GALLERY_MAP_PATH.is_file():
    raise FileNotFoundError(f'Required ID-to-title gallery mapping missing: {GALLERY_MAP_PATH}')
GALLERY_ID_TO_TITLE = json.loads(GALLERY_MAP_PATH.read_text(encoding='utf-8'))
GALLERY_AUDIO_SHA_PATH = MUSIC_DB_DIR.parent / 'qbh_dataset_audio_sha256.json'
if not GALLERY_AUDIO_SHA_PATH.is_file():
    raise FileNotFoundError(f'Required gallery checksum map missing: {GALLERY_AUDIO_SHA_PATH}')
GALLERY_AUDIO_SHA = json.loads(GALLERY_AUDIO_SHA_PATH.read_text(encoding='utf-8'))['files']
GALLERY_PATHS = {}
seen_gallery_ids = set()
for path in sorted(MUSIC_DB_DIR.iterdir()):
    if path.suffix.lower() != '.mp3':
        continue
    song_id = path.stem
    if song_id not in GALLERY_ID_TO_TITLE:
        raise ValueError(f'Gallery ID missing from song_name_mapping.json: {song_id}')
    if f'datasets/2_music_database/{song_id}.mp3' not in GALLERY_AUDIO_SHA:
        raise ValueError(f'Gallery audio SHA-256 missing for {song_id}')
    title = GALLERY_ID_TO_TITLE[song_id]
    if title in GALLERY_PATHS:
        raise ValueError(f'Duplicate mapped gallery title: {title}')
    GALLERY_PATHS[title] = str(path)
    seen_gallery_ids.add(song_id)
if seen_gallery_ids != set(GALLERY_ID_TO_TITLE) or len(GALLERY_PATHS) != 300:
    raise ValueError(f'Gallery mapping/file inventory mismatch: {len(seen_gallery_ids)} IDs, {len(GALLERY_ID_TO_TITLE)} mappings')
"""
    for source_path, out_path, pipeline in (
        (NOTEBOOK_IN, NOTEBOOK_OUT, "qbh_final_v6_1"),
        (EVAL_IN, EVAL_OUT, "qbh_final_v6_1"),
    ):
        nb = json.loads(source_path.read_text(encoding="utf-8"))
        if source_path in (NOTEBOOK_IN, EVAL_IN):
            nb["cells"][8]["source"] = cuda_nb["cells"][8]["source"]
        replacements = 0
        for cell in nb["cells"]:
            text = "".join(cell.get("source", []))
            text = text.replace("qbh_final_v6", pipeline).replace("qbh_final_v5", pipeline)
            text = text.replace("Conformer-ByteHum Final V6", "Conformer-ByteHum Final V6.1")
            if source_path == EVAL_IN and "RUN_CONFIG = dict(" in text and "PREPROCESS_CONFIG = dict(" in text:
                start = text.index("RUN_CONFIG = dict(")
                end = text.index("PREPROCESS_CONFIG = dict(", start)
                text = text[:start] + V61_RUN_CONFIG + "\n" + text[end:]
            text = text.replace("refs = [p for p in files if p.name == 'reference.wav']",
                                "refs = [p for p in files if p.stem.casefold() == 'reference']")
            text = text.replace("hums = [p for p in files if not p.name.startswith('reference')]",
                                "hums = [p for p in files if not p.stem.casefold().startswith('reference')]")
            text = text.replace("def scan_queries(split):", "REFERENCE_FALLBACKS = []\ndef scan_queries(split):")
            text = text.replace(
                "                elif len(files) > 1:\n                    refs = [files[0]]\n                    hums = files[1:]",
                "                else:\n                    full_song = Path(GALLERY_PATHS.get(song, '')).resolve()\n"
                "                    if not full_song.is_file():\n"
                "                        raise FileNotFoundError(f'No fragment reference and no same-song gallery audio: {song}/{fragment.name}')\n"
                "                    refs = [full_song]\n"
                "                    REFERENCE_FALLBACKS.append((split, song, fragment.name))")
            if source_path == NOTEBOOK_IN:
                extract_start = text.find('def extract_dataset_archive(')
                selection_start = text.find("split_candidates = list(INPUT_ROOT.rglob('dataset_split'))")
                if extract_start >= 0 and selection_start >= 0:
                    text = text[:extract_start] + TRAIN_EXTRACT_BLOCK + "\n" + text[selection_start:]
                    selection_start = text.find("split_candidates = list(INPUT_ROOT.rglob('dataset_split'))")
                selection_end = text.find("qbh_success(f'Dataset split:", selection_start)
                if selection_start >= 0 and selection_end >= 0:
                    text = text[:selection_start] + TRAIN_SELECTION_BLOCK + "\n" + text[selection_end:]
            if "low_band_degradation_probability=0.30" in text:
                text = text.replace("low_band_degradation_probability=0.30, degradation_sample_rate=8000,",
                                    "low_band_degradation_probability=0.30, degradation_sample_rate=8000,\n                  degradation_bit_depth=8, degradation_dither='TPDF',")
            if source_path == NOTEBOOK_IN:
                text = text.replace("selection='fused_micro_mrr'", "selection='fused_macro_song_mrr'")
                text = text.replace("Best fused validation MRR", "Best macro-song validation MRR")
                text = text.replace("val_mrr = metrics['fused']['mrr']", "val_mrr = metrics['fused']['macro_song_mrr']")
                text = text.replace("val_fused_mrr=val_mrr, val_macro_mrr=metrics['fused']['macro_song_mrr'],",
                                    "val_fused_mrr=metrics['fused']['mrr'], val_macro_mrr=metrics['fused']['macro_song_mrr'],\n"
                                    "                            selection_mrr=val_mrr,")
                text = text.replace("Epoch {epoch}: fused val MRR={val_mrr:.4f}, best={best_mrr:.4f}",
                                    "Epoch {epoch}: micro MRR={metrics['fused']['mrr']:.4f}, macro-song MRR={val_mrr:.4f}, best macro={best_mrr:.4f}")
                text = text.replace("Epoch {epoch} finished: fused MRR={val_mrr:.4f}",
                                    "Epoch {epoch} finished: macro-song selection MRR={val_mrr:.4f}")
                text = text.replace("best_validation_fused_mrr=best_mrr", "best_validation_macro_song_mrr=best_mrr")
                text = text.replace("best fused MRR={best_mrr:.6f}", "best macro-song MRR={best_mrr:.6f}")
            if source_path == NOTEBOOK_IN:
                if "TRAIN_ROWS = scan_queries('train')\nVAL_ROWS = scan_queries('val')" in text and "MIR_QBSH_EXTRA_ROWS_BY_SPLIT" not in text:
                    text = text.replace("TRAIN_ROWS = scan_queries('train')\nVAL_ROWS = scan_queries('val')",
                                        "TRAIN_ROWS = scan_queries('train')\nVAL_ROWS = scan_queries('val')\n\n" + MIR_TRAINING_BLOCK.rstrip())
                if "print('Split sizes:'" in text and "def data_contract():" not in text:
                    text = text.replace("print('Split sizes:'", V61_DATA_CONTRACT_BLOCK + "\nprint('Split sizes:'", 1)
                text = text.replace("RUN_CONTRACT = dict(config=RUN_CONFIG, preprocessing=PREPROCESS_CONFIG,\n                    runtime=RUNTIME_VERSIONS, stage1=str(STAGE1_CKPT))",
                                    "RUN_CONTRACT = dict(config=dict(RUN_CONFIG), preprocessing=PREPROCESS_CONFIG,\n"
                                    "                    runtime=RUNTIME_VERSIONS, data=DATA_DIGEST, stage1=str(STAGE1_CKPT))")
                text = text.replace("train_songs=SPLIT_IDS['train'], checkpoint_schema=2)",
                                    "train_songs=SPLIT_IDS['train'], checkpoint_schema=2,\n"
                                    "                run_contract=RUN_CONTRACT, run_signature=digest(RUN_CONTRACT),\n"
                                    "                data_contract=DATA_CONTRACT)")
                text = text.replace(
                    "        best_snapshot = validated_best_snapshot(state)\n        model.load_state_dict(state['model_state_dict'], strict=False)",
                    "        if (state.get('checkpoint_schema') != 2\n"
                    "                or state.get('run_signature') != digest(RUN_CONTRACT)\n"
                    "                or state.get('run_contract') != RUN_CONTRACT\n"
                    "                or state.get('data_contract') != DATA_CONTRACT):\n"
                    "            raise ValueError('LAST checkpoint belongs to a different V6.1 code/config/data contract; use a fresh OUTPUT_ROOT.')\n"
                    "        for _required in ('model_state_dict', 'optimizer', 'scheduler', 'scaler', 'best_checkpoint'):\n"
                    "            if _required not in state: raise ValueError(f'Incompatible LAST checkpoint: missing {_required}')\n"
                    "        best_snapshot = validated_best_snapshot(state)\n        model.load_state_dict(state['model_state_dict'], strict=True)")
                text = text.replace(
                    "        if 'optimizer' in state:\n            try: optimizer.load_state_dict(state['optimizer'])\n            except Exception: pass\n        if 'scheduler' in state:\n            try: scheduler.load_state_dict(state['scheduler'])\n            except Exception: pass\n        if 'scaler' in state:\n            try: scaler.load_state_dict(state['scaler'])\n            except Exception: pass",
                    "        optimizer.load_state_dict(state['optimizer'])\n"
                    "        scheduler.load_state_dict(state['scheduler'])\n"
                    "        scaler.load_state_dict(state['scaler'])")
            if source_path == EVAL_IN:
                text = text.replace("FILE_HASHES = {}\ndef file_sha256(path):",
                                    "FILE_HASHES = {}\n"
                                    "AUDIO_SHA_INDEX_PATH = MUSIC_DB_DIR.parent / 'qbh_dataset_audio_sha256.json'\n"
                                    "if not AUDIO_SHA_INDEX_PATH.is_file(): raise FileNotFoundError(AUDIO_SHA_INDEX_PATH)\n"
                                    "AUDIO_SHA_INDEX = json.loads(AUDIO_SHA_INDEX_PATH.read_text(encoding='utf-8'))['files']\n"
                                    "def file_sha256(path):")
                text = text.replace("    return FILE_HASHES[key]\n\ndef atomic_json",
                                    "    try:\n"
                                    "        relpath = path.relative_to(MUSIC_DB_DIR.parent.resolve()).as_posix()\n"
                                    "    except ValueError:\n"
                                    "        relpath = None\n"
                                    "    if relpath is not None:\n"
                                    "        expected = AUDIO_SHA_INDEX.get('datasets/' + relpath)\n"
                                    "        if expected is None or FILE_HASHES[key] != expected:\n"
                                    "            raise ValueError(f'Audio SHA-256 differs from the V6.1 inventory: {relpath}')\n"
                                    "    return FILE_HASHES[key]\n\ndef atomic_json",1)
                text = text.replace(
                    "            if hums and not refs:\n                full_song = Path(GALLERY_PATHS.get(song, '')).resolve()\n",
                    "            if hums and not refs:\n                if vocals:\n                    refs = [vocals[0]]\n                else:\n                    full_song = Path(GALLERY_PATHS.get(song, '')).resolve()\n")
                text = text.replace(
                    "                if not full_song.is_file():\n                    raise FileNotFoundError(\n",
                    "                    if not full_song.is_file():\n                        raise FileNotFoundError(\n")
                text = text.replace(
                    "                # The package has two audited fragments without reference.wav.\n                # Use only that song's own gallery recording as the positive bag.\n                refs = [full_song]\n                REFERENCE_FALLBACKS.append((split, song, fragment.name))",
                    "                    # Fall back only when neither mixture nor vocal reference exists.\n"
                    "                    refs = [full_song]\n                    REFERENCE_FALLBACKS.append((split, song, fragment.name))")
                text = text.replace("_reference = _song_dir / 'reference.wav'",
                                    "_reference = MIR_QBSH_ROOT / _song_meta['reference_path']\n"
                                    "    if not _reference.is_file(): _reference = _reference.with_suffix('.flac')")
                text = text.replace("_query = _song_dir / _filename\n        if not _query.is_file():",
                                    "_query = _song_dir / _filename\n        if not _query.is_file():\n"
                                    "            _query = _query.with_suffix('.flac')\n        if not _query.is_file():")
                text = text.replace("key=f'mir_qbsh/{_split}/{_song}/1/{_filename}'",
                                    "key=f'mir_qbsh/{_split}/{_song}/1/{_query.name}'")
                text = text.replace("RUN_CONFIG['dataset_policy'] = str(RUN_CONFIG['dataset_policy']) + '+mir_qbsh_song_disjoint_70_15_15_v2'",
                                    "RUN_CONFIG['dataset_policy'] = 'v6_1_clean_original_plus_mir_qbsh_70_15_15'")
                text = text.replace(
                    "RUN_CONFIG['dataset_policy'] = 'v6_1_clean_original_plus_mir_qbsh_70_15_15'",
                    "RUN_CONFIG['dataset_policy'] = 'v6_1_clean_original_plus_mir_qbsh_70_15_15'\n"
                    "if {k: len(v) for k, v in SPLIT_IDS.items()} != {'train': 244, 'val': 52, 'test': 52}:\n"
                    "    raise ValueError('V6.1 song split inventory mismatch')\n"
                    "if len(TRAIN_ROWS) != 6684 or len(VAL_ROWS) != 1428:\n"
                    "    raise ValueError(f'V6.1 train/validation query inventory mismatch: {len(TRAIN_ROWS)}/{len(VAL_ROWS)}')\n"
                    "if {row['song'] for row in TRAIN_ROWS} != set(SPLIT_IDS['train']) or {row['song'] for row in VAL_ROWS} != set(SPLIT_IDS['val']):\n"
                    "    raise ValueError('V6.1 train/validation song coverage mismatch')")
                text = text.replace("TEST_ROWS.extend(MIR_QBSH_EXTRA_ROWS_BY_SPLIT['test'])",
                                    "TEST_ROWS.extend(MIR_QBSH_EXTRA_ROWS_BY_SPLIT['test'])\n"
                                    "if len(TEST_ROWS) != 1467 or {row['song'] for row in TEST_ROWS} != set(SPLIT_IDS['test']):\n"
                                    "    raise ValueError(f'V6.1 test inventory mismatch: {len(TEST_ROWS)} queries / {len({row[\"song\"] for row in TEST_ROWS})} songs')")
                text = text.replace("if contract['code'] != SOURCE_CODE_DIGEST or contract['config'] != RUN_CONFIG or contract['preprocessing'] != PREPROCESS_CONFIG:",
                                    "if contract['config'] != RUN_CONFIG or contract['preprocessing'] != PREPROCESS_CONFIG:")
                text = text.replace("if not checkpoint.get('finished'):\n    raise ValueError('This checkpoint was not finalized by the training notebook.')",
                                    "if not checkpoint.get('finished') or checkpoint.get('checkpoint_schema') != 2:\n"
                                    "    raise ValueError('Expected a finalized V6.1 schema-2 checkpoint.')")
            if target_block in text:
                text = text.replace(target_block, replacement)
                replacements += 1
            cell["source"] = text.splitlines(keepends=True)
        if source_path == EVAL_IN and replacements == 0:
            for cell in nb["cells"]:
                text = "".join(cell.get("source", []))
                if eval_target_block in text:
                    text = text.replace(eval_target_block, replacement)
                    replacements += 1
                    cell["source"] = text.splitlines(keepends=True)
        if replacements != 1:
            raise RuntimeError(f"Expected one gallery loader in {source_path.name}, replaced {replacements}")
        for cell in nb["cells"]:
            text = "".join(cell.get("source", []))
            if "if degrade_to_8khz:" in text and "RUN_CONFIG['degradation_bit_depth']" not in text:
                text = text.replace("            y = scipy.signal.resample_poly(y, 2, 1).astype(np.float32)\n",
                                    "            y = scipy.signal.resample_poly(y, 2, 1).astype(np.float32)\n"
                                    "            # Apply 8-bit uniform quantization with triangular dither to mimic low-grade capture hiss.\n"
                                    "            quant_step = 2.0 / (2 ** RUN_CONFIG['degradation_bit_depth'] - 1)\n"
                                    "            dither = (np.random.random(y.shape) - np.random.random(y.shape)) * quant_step\n"
                                    "            y = (np.round((np.clip(y, -1.0, 1.0) + dither) / quant_step) * quant_step).clip(-1.0, 1.0).astype(np.float32)\n")
                text = text.replace("    CQT_RAM[key] = chunks\n", "    CQT_RAM[key] = chunks\n")
            if "def raw_cqts(path, degrade_to_8khz=False):" in text:
                text = text.replace("CQT_DIR.mkdir(parents=True, exist_ok=True)\nFEATURE_DIR.mkdir(parents=True, exist_ok=True)\n",
                                    "CQT_DIR.mkdir(parents=True, exist_ok=True)\nFEATURE_DIR.mkdir(parents=True, exist_ok=True)\n"
                                    "AUDIO_SHA_INDEX_PATH = MUSIC_DB_DIR.parent / 'qbh_dataset_audio_sha256.json'\n"
                                    "if not AUDIO_SHA_INDEX_PATH.is_file():\n"
                                    "    raise FileNotFoundError(f'Required dataset audio checksum index missing: {AUDIO_SHA_INDEX_PATH}')\n"
                                    "AUDIO_SHA_INDEX = json.loads(AUDIO_SHA_INDEX_PATH.read_text(encoding='utf-8'))['files']\n"
                                    "CQT_PRECOMPUTED_ROOT = None\nCQT_PRECOMPUTED_INDEX = {}\n"
                                    f"CQT_BACKEND_SHA256 = '{cqt_backend_sha256}'\n"
                                    "_manifest_paths = list(INPUT_ROOT.rglob('qbh_cqt_manifest_v6_1.json'))\n"
                                    "if len(_manifest_paths) > 1:\n"
                                    "    raise ValueError(f'Ambiguous portable CQT cache manifests: {_manifest_paths}')\n"
                                    "for _manifest_path in _manifest_paths:\n"
                                    "    _manifest = json.loads(_manifest_path.read_text(encoding='utf-8'))\n"
                                    "    if _manifest.get('cqt_backend_sha256') != CQT_BACKEND_SHA256:\n"
                                    "        raise ValueError('Portable CQT backend hash mismatch; regenerate cache for this notebook.')\n"
                                    "    CQT_PRECOMPUTED_ROOT = _manifest_path.parent\n"
                                    "    CQT_PRECOMPUTED_INDEX = {(row['relative_path'], row['variant']): row for row in _manifest['features']}\n"
                                    "    print(f'[INFO] Found portable CQT cache: {len(CQT_PRECOMPUTED_INDEX)} feature entries')\n"
                                    "    break\n")
                text = text.replace("    stat = path.stat()\n    key = hashlib.md5(",
                                    "    stat = path.stat()\n"
                                    "    variant = 'degraded_8khz_8bit' if degrade_to_8khz else 'base'\n"
                                    "    try:\n"
                                    "        relpath = path.relative_to(MUSIC_DB_DIR.parent.resolve()).as_posix()\n"
                                    "    except ValueError:\n"
                                    "        relpath = None\n"
                                    "    ram_key = relpath + '|' + variant if relpath is not None else None\n"
                                    "    if ram_key is not None and ram_key in CQT_RAM:\n"
                                    "        return CQT_RAM[ram_key]\n"
                                    "    if relpath is not None and CQT_PRECOMPUTED_ROOT is not None:\n"
                                    "        _entry = CQT_PRECOMPUTED_INDEX.get((relpath, variant))\n"
                                    "        if _entry is not None:\n"
                                    "            _expected_sha = AUDIO_SHA_INDEX.get('datasets/' + relpath)\n"
                                    "            if _expected_sha != _entry['source_sha256']:\n"
                                    "                raise ValueError(f'Portable CQT source SHA mismatch: {relpath}')\n"
                                    "            _feature_path = CQT_PRECOMPUTED_ROOT / _entry['feature_file']\n"
                                    "            _feature_sha = hashlib.sha256(_feature_path.read_bytes()).hexdigest()\n"
                                    "            if _feature_sha != _entry['feature_sha256']:\n"
                                    "                raise ValueError(f'Portable CQT feature SHA mismatch: {_feature_path}')\n"
                                    "            _cached = torch.load(_feature_path, map_location='cpu', weights_only=True)\n"
                                    "            chunks = [tensor.float() for tensor in _cached]\n"
                                    "            if not chunks or any(t.shape != (1, 96, 83) or not torch.isfinite(t).all() for t in chunks):\n"
                                    "                raise ValueError(f'Invalid portable CQT cache entry for {relpath}')\n"
                                    "            CQT_RAM[ram_key] = chunks\n"
                                    "            return chunks\n"
                                    "    key = hashlib.md5(")
            cell["source"] = text.splitlines(keepends=True)
        nb.setdefault("metadata", {})["qbh_v6_1"] = {"pipeline": pipeline, "gallery_id_count": len(id_to_title)}
        out_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

    # Update comparison notebooks to resolve the new stable IDs while preserving
    # their title-normalization and paired-query logic.
    for source_path in (COMPARE_KAGGLE, COMPARE_LOCAL):
        nb = json.loads(source_path.read_text(encoding="utf-8"))
        changed = False
        for cell in nb["cells"]:
            text = "".join(cell.get("source", []))
            if "_mapping = json.loads(mapping_path.read_text(encoding='utf-8'))" in text:
                old = """_mapping = json.loads(mapping_path.read_text(encoding='utf-8'))
    for _source_name, _file_name in _mapping.items():
        for _alias in (_source_name, _file_name):
            _key = norm_name(_alias)
            if _key in song_alias and song_alias[_key] != _file_name: song_alias[_key] = None
            else: song_alias[_key] = _file_name"""
                new = """title_id_path = mapping_path.with_name('song_title_to_id.json')
    if not title_id_path.is_file():
        raise FileNotFoundError(title_id_path)
    _mapping = json.loads(title_id_path.read_text(encoding='utf-8'))
    for _source_name, _song_id in _mapping.items():
        _key = norm_name(_source_name)
        if _key in song_alias and song_alias[_key] != _song_id: song_alias[_key] = None
        else: song_alias[_key] = _song_id"""
                if old in text:
                    text = text.replace(old, new)
                    changed = True
            if "_mapping = json.loads(MAPPING_PATH.read_text(encoding='utf-8'))" in text:
                old = """_mapping = json.loads(MAPPING_PATH.read_text(encoding='utf-8'))
for _source_name, _file_name in _mapping.items():
    for _alias in (_source_name, _file_name):
        _key = norm_name(_alias)
        if _key in song_alias and song_alias[_key] != _file_name:
            song_alias[_key] = None
        else:
            song_alias[_key] = _file_name"""
                new = """_title_id_path = MAPPING_PATH.with_name('song_title_to_id.json')
if not _title_id_path.is_file():
    raise FileNotFoundError(_title_id_path)
_mapping = json.loads(_title_id_path.read_text(encoding='utf-8'))
for _source_name, _song_id in _mapping.items():
    _key = norm_name(_source_name)
    if _key in song_alias and song_alias[_key] != _song_id:
        song_alias[_key] = None
    else:
        song_alias[_key] = _song_id"""
                if old in text:
                    text = text.replace(old, new)
                    changed = True
            cell["source"] = text.splitlines(keepends=True)
        if not changed:
            raise RuntimeError(f"Comparison mapping block not found in {source_path.name}")
        out_path = source_path.with_name(source_path.stem + "_v6_1" + source_path.suffix)
        out_path.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE_ZIP)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        with zipfile.ZipFile(OUTPUT_ZIP) as z:
            print("bad member:", z.testzip())
        return

    with zipfile.ZipFile(args.source) as zin:
        title_to_stem = json.loads(zin.read("datasets/song_name_mapping.json"))
        original_song_manifest = json.loads(zin.read("datasets/dataset_split/split_manifest.json"))
        combined_song_manifest = json.loads(zin.read("datasets/combined_split_manifest.json"))
    legacy_mapping_path = ROOT / "datasets/song_name_mapping_before_v6.json"
    if not legacy_mapping_path.exists():
        raise FileNotFoundError(f"Pre-V6 mapping backup required to match gallery tracks: {legacy_mapping_path}")
    legacy_title_to_stem = json.loads(legacy_mapping_path.read_text(encoding="utf-8"))
    split_titles_to_stems, extra_aliases = map_split_titles_to_gallery(
        title_to_stem, original_song_manifest, combined_song_manifest, legacy_title_to_stem)
    stable_stem_order = [title_to_stem[title] for title in sorted(title_to_stem, key=stable_title_key)]
    id_to_title, title_to_id, id_to_legacy = make_ids(
        split_titles_to_stems, extra_aliases, stable_stem_order=stable_stem_order)
    id_by_legacy_stem = {Path(filename).stem: song_id for song_id, filename in id_to_legacy.items()}
    missing_contract = json.loads(CONTRACT_IN.read_text(encoding="utf-8"))
    excluded_ids = sorted(missing_contract.get("excluded_missing_source_query_files", []))
    if len(excluded_ids) != 39:
        raise RuntimeError("Expected exactly 39 audited missing source query IDs")
    old_summary = None
    for b in (ROOT / "backups/qbh_dataset_contract_before_v6.json",):
        if not b.exists():
            raise FileNotFoundError("Expected preserved pre-V6 contract backup")

    audio_hashes = {}
    query_rows = []
    query_counts = Counter()
    gallery_records = []
    conversion_rows = []
    id_to_title_bytes = jbytes(id_to_title)
    title_to_id_bytes = jbytes(title_to_id)
    temp_zip = OUTPUT_ZIP.with_suffix(".zip.tmp")

    with zipfile.ZipFile(args.source) as zin, zipfile.ZipFile(
        temp_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True
    ) as zout:
        for info in zin.infolist():
            name = info.filename.replace("\\", "/")
            dest = name
            if info.is_dir():
                zout.writestr(dest, b"")
                continue
            if name == "datasets/song_name_mapping.json":
                zout.writestr(dest, id_to_title_bytes)
                continue
            if name in {"datasets/song_title_to_id.json", "datasets/gallery_id_to_legacy_filename.json"}:
                continue
            if name == "datasets/qbh_dataset_audio_sha256.json" or name == "datasets/qbh_dataset_contract.json":
                continue
            if name == "datasets/combined_split_summary.csv":
                old_summary = list(csv.DictReader(io.StringIO(zin.read(info).decode("utf-8"))))
                continue
            if name == "datasets/combined_split_manifest.json":
                combined_manifest = json.loads(zin.read(info).decode("utf-8"))
                combined_manifest["missing_original_queries"] = 0
                combined_manifest["excluded_missing_original_queries"] = len(excluded_ids)
                combined_manifest["missing_original_query_ids_excluded"] = excluded_ids
                combined_manifest["actual_original_queries"] = dict(missing_contract["query_files_present_and_sha_verified"])
                zout.writestr(dest, jbytes(combined_manifest))
                continue
            if name == "datasets/mir_qbsh_extension/manifest.json":
                mir_manifest = json.loads(zin.read(info).decode("utf-8"))
                for song in mir_manifest.get("songs", []):
                    ref = song.get("reference_path")
                    if ref and ref.startswith(("train/", "val/")):
                        song["reference_path"] = str(Path(ref).with_suffix(".flac")).replace("\\", "/")
                mir_manifest["reference_encoding"] = "lossless FLAC for train/validation; original WAV preserved for test"
                zout.writestr(dest, jbytes(mir_manifest))
                continue

            if name.startswith("datasets/2_music_database/") and Path(name).suffix.lower() == ".mp3":
                old_stem = Path(name).stem
                song_id = id_by_legacy_stem.get(old_stem)
                if song_id is None:
                    raise RuntimeError(f"Gallery file has no stable ID mapping: {name}")
                dest = f"datasets/2_music_database/{song_id}.mp3"
                payload = zin.read(info)
                audio_hashes[dest] = sha256(payload)
                gallery_records.append({"song_id": song_id, "title": id_to_title[song_id],
                                        "legacy_filename": Path(name).name, "sha256": audio_hashes[dest]})
                zout.writestr(dest, payload)
                continue

            is_audio = Path(name).suffix.lower() in {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
            parts = name.split("/")
            is_query = query_path(parts, Path(name).suffix)
            if is_audio and (is_query or is_train_val_wav(name)):
                payload = zin.read(info)
                original_path = name
                meta = None
                conversion = None
                if is_train_val_wav(name):
                    payload, meta, conversion = flac_encode(payload)
                    if conversion is not None:
                        dest = str(Path(name).with_suffix(".flac")).replace("\\", "/")
                        conversion_rows.append({"source_path": original_path, "output_path": dest,
                                                "source_sha256": sha256(zin.read(info)),
                                                "output_sha256": sha256(payload),
                                                "source_sample_rate": meta.samplerate,
                                                "source_channels": meta.channels,
                                                "source_subtype": meta.subtype,
                                                "source_frames": meta.frames, **conversion})
                if is_audio:
                    audio_hashes[dest] = sha256(payload)
                zout.writestr(dest, payload)
                if is_query:
                    split = parts[2]
                    source = "MIR-QBSH" if parts[1] == "mir_qbsh_extension" else "original"
                    song = parts[3]
                    song_id = song[9:] if source == "MIR-QBSH" and song.startswith("MIR-QBSH-") else ""
                    if source == "original":
                        song_id = str(original_song_manifest.get(song, {}).get("song_id", ""))
                    sfinfo = sf.info(io.BytesIO(payload))
                    row = {"source": source, "split": split, "song": song, "song_id": song_id,
                           "fragment": parts[4], "query_id": Path(dest).stem, "path": dest,
                           "source_path": original_path,
                           "sha256": audio_hashes[dest], "sample_rate": sfinfo.samplerate,
                           "channels": sfinfo.channels, "subtype": sfinfo.subtype,
                           "source_sample_rate": meta.samplerate if meta else sfinfo.samplerate,
                           "source_channels": meta.channels if meta else sfinfo.channels,
                           "source_subtype": meta.subtype if meta else sfinfo.subtype,
                           "source_encoding": f"{Path(original_path).suffix[1:].upper()}_{meta.subtype if meta else sfinfo.subtype}",
                           "duration_seconds": sfinfo.frames / sfinfo.samplerate,
                           "start_second": "", "end_second": "", "alignment_status": "not_available_in_source"}
                    query_rows.append(row)
                    query_counts[(source, split)] += 1
                continue

            if is_audio:
                digest = hashlib.sha256()
                with zin.open(info) as src, zout.open(dest, "w") as dst:
                    while True:
                        chunk = src.read(1024 * 1024)
                        if not chunk:
                            break
                        digest.update(chunk); dst.write(chunk)
                audio_hashes[dest] = digest.hexdigest()
            else:
                with zin.open(info) as src, zout.open(dest, "w") as dst:
                    shutil.copyfileobj(src, dst, length=1024 * 1024)

        if old_summary is None:
            raise RuntimeError("combined_split_summary.csv is missing")
        zout.writestr("datasets/song_title_to_id.json", title_to_id_bytes)
        zout.writestr("datasets/gallery_id_to_legacy_filename.json", jbytes(id_to_legacy))
        zout.writestr("datasets/gallery_manifest_v6_1.json", jbytes(gallery_records))
        zout.writestr("datasets/audio_conversion_manifest_v6_1.json", jbytes(conversion_rows))
        zout.writestr("datasets/qbh_query_manifest_v6_1.csv", csv_bytes(query_rows))
        summary_rows = rebuild_summary(old_summary, query_counts)
        zout.writestr("datasets/combined_split_summary.csv", csv_bytes(summary_rows))
        checksum_manifest = {"algorithm": "sha256", "scope": "all audio files in V6.1 dataset ZIP", "files": audio_hashes}
        zout.writestr("datasets/qbh_dataset_audio_sha256.json", jbytes(checksum_manifest))
        contract = {
            "schema_version": 3, "dataset_archive": OUTPUT_ZIP.name,
            "source_split_counts_songs": missing_contract["source_split_counts_songs"],
            "source_query_counts": dict(missing_contract["query_files_present_and_sha_verified"]),
            "query_files_present_and_sha_verified": dict(missing_contract["query_files_present_and_sha_verified"]),
            "missing_source_query_files": [], "missing_source_query_counts_by_split": {"train": 0, "val": 0, "test": 0},
            "excluded_missing_source_query_files": excluded_ids,
            "excluded_missing_source_query_counts_by_split": {"train": 29, "val": 3, "test": 7},
            "audio_files_sha256_verified": len(audio_hashes), "audio_sha256_manifest": "datasets/qbh_dataset_audio_sha256.json",
            "gallery_songs": 300, "gallery_id_map": "datasets/song_name_mapping.json",
            "query_manifest": "datasets/qbh_query_manifest_v6_1.csv",
            "timestamp_alignment_available": False,
            "timestamp_alignment_status": "nullable fields included; source annotations were not found",
            "final_training_ready": True,
            "locked_final_test": missing_contract["locked_final_test"],
            "note": "Missing source query IDs are excluded from active counts and retained for audit. Train/validation PCM WAVs are losslessly transcoded to FLAC; test and locked-holdout WAVs remain unchanged. Gallery MP3 bytes are unchanged and renamed to stable ASCII IDs.",
        }
        zout.writestr("datasets/qbh_dataset_contract.json", jbytes(contract))

    os.replace(temp_zip, OUTPUT_ZIP)
    OUTPUT_CONTRACT.write_bytes(jbytes(contract))
    OUTPUT_HASHES.write_bytes(jbytes(checksum_manifest))
    OUTPUT_QUERY_MANIFEST.write_bytes(csv_bytes(query_rows))
    shutil.copy2(OUTPUT_CONTRACT, DIST / "qbh_dataset_contract.json")
    update_local_gallery(id_to_title, title_to_id, id_to_legacy)
    patch_notebooks(id_to_title)
    print("Archive:", OUTPUT_ZIP, OUTPUT_ZIP.stat().st_size)
    print("Audio SHA entries:", len(audio_hashes), "FLAC conversions:", len(conversion_rows))
    print("Queries:", dict(query_counts), "query manifest:", len(query_rows))


def csv_bytes(rows):
    if not rows:
        return b""
    out = io.StringIO(newline="")
    writer = csv.DictWriter(out, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    return out.getvalue().encode("utf-8")


def rebuild_summary(old_rows, counts):
    old = {(r["source"], r["split"]): r for r in old_rows if r["source"] in {"original", "MIR-QBSH"}}
    rows = []
    for source in ("original", "MIR-QBSH"):
        for split in ("train", "val", "test"):
            prior = old[(source, split)]
            active = counts[(source, split)]
            source_expected = int(prior["expected_queries"])
            excluded = source_expected - active
            rows.append({"source": source, "split": split, "songs": int(prior["songs"]),
                         "expected_queries": active, "available_queries": active, "missing_queries": 0,
                         "source_expected_queries": source_expected, "excluded_missing_source_queries": excluded})
    for split in ("train", "val", "test"):
        a = counts[("original", split)]; b = counts[("MIR-QBSH", split)]
        prior = next(r for r in old_rows if r["source"] == "combined" and r["split"] == split)
        rows.append({"source": "combined", "split": split, "songs": int(prior["songs"]),
                     "expected_queries": a + b, "available_queries": a + b, "missing_queries": 0,
                     "source_expected_queries": int(prior["expected_queries"]),
                     "excluded_missing_source_queries": int(prior["missing_queries"])})
    return rows


def update_local_gallery(id_to_title, title_to_id, id_to_legacy):
    directory = ROOT / "datasets/2_music_database"
    if not directory.is_dir():
        return
    backup = ROOT / "backups/song_name_mapping_before_v6_1.json"
    mapping_path = ROOT / "datasets/song_name_mapping.json"
    if mapping_path.exists() and not backup.exists():
        shutil.copy2(mapping_path, backup)
    old_to_id = {filename: song_id for song_id, filename in id_to_legacy.items()}
    for old_name, song_id in old_to_id.items():
        src = directory / old_name
        dst = directory / f"{song_id}.mp3"
        if src.exists() and not dst.exists():
            src.rename(dst)
        elif not dst.exists():
            raise FileNotFoundError(f"Local gallery file missing for {song_id}: {src}")
    mapping_path.write_bytes(jbytes(id_to_title))
    (ROOT / "datasets/song_title_to_id.json").write_bytes(jbytes(title_to_id))


if __name__ == "__main__":
    main()
