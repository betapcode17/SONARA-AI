"""Build-time template only. Its code is embedded in both final notebooks."""

# %% imports
from pathlib import Path
import os
import sys
import subprocess
import json
import hashlib
import random
import math
import time
import logging
import gc
import shutil
import zipfile
import stat
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from IPython.display import display
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import scipy.sparse
import soundfile as sf
import librosa
try:
    import typeguard
    if not hasattr(typeguard, 'check_argument_types'):
        raise ImportError('CoverHunter needs typeguard<3')
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'typeguard==2.13.3'])
    for name in list(sys.modules):
        if name == 'typeguard' or name.startswith('typeguard.'):
            del sys.modules[name]
    import typeguard

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type != 'cuda':
    raise RuntimeError('Select GPU T4 in Kaggle Settings before Run All.')
AMP_ENABLED = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
print('GPU:', torch.cuda.get_device_name(0), '| PyTorch:', torch.__version__)

# %% paths
INPUT_ROOT = Path(INPUT_ROOT)
OUTPUT_ROOT = Path(OUTPUT_ROOT)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
for env, folder in [('NUMBA_CACHE_DIR', 'numba'), ('MPLCONFIGDIR', 'matplotlib')]:
    (OUTPUT_ROOT / folder).mkdir(exist_ok=True)
    os.environ[env] = str(OUTPUT_ROOT / folder)

def unique_path(override, matches, description):
    if override is not None:
        result = Path(override).resolve()
        if not result.exists():
            raise FileNotFoundError(result)
        return result
    candidates = sorted({Path(p).resolve() for p in matches})
    if len(candidates) != 1:
        raise ValueError(f'Set {description} explicitly; found {len(candidates)}: {candidates}')
    return candidates[0]

def extract_dataset_archive(archive, base):
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
            normalized = item.filename.replace('\\', '/')
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
        for item, target in tqdm(planned, desc='Unpack dataset'):
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

split_candidates = list(INPUT_ROOT.rglob('dataset_split'))
music_candidates = list(INPUT_ROOT.rglob('2_music_database'))
if DATASET_ARCHIVE_OVERRIDE is not None or not (split_candidates or DATASET_OVERRIDE) or not (music_candidates or MUSIC_OVERRIDE):
    archive_candidates = []
    for path in INPUT_ROOT.rglob('*'):
        if path.suffix.lower() not in ('.zip', '.bin', '.dat') or not path.is_file() or not zipfile.is_zipfile(path):
            continue
        with zipfile.ZipFile(path) as archive:
            members = [name.replace('\\', '/') for name in archive.namelist()]
            if any('dataset_split/' in name for name in members) and any('2_music_database/' in name for name in members):
                archive_candidates.append(path)
    archive_path = unique_path(DATASET_ARCHIVE_OVERRIDE, archive_candidates, 'DATASET_ARCHIVE_OVERRIDE')
    unpacked = extract_dataset_archive(archive_path, OUTPUT_ROOT / 'dataset_unpacked')
    split_candidates.extend(unpacked.rglob('dataset_split'))
    music_candidates.extend(unpacked.rglob('2_music_database'))
DATASET_SPLIT_DIR = unique_path(DATASET_OVERRIDE, split_candidates, 'DATASET_OVERRIDE')
MUSIC_DB_DIR = unique_path(MUSIC_OVERRIDE, music_candidates, 'MUSIC_OVERRIDE')
EMBEDDING_DIM = 128
EVAL_BATCH_SIZE = 8
GALLERY_SONG_BLOCK = 8
REFERENCE_FRAME_BLOCK = 512
RUN_CONFIG = dict(pipeline='qbh_final_v5', seed=42, epochs=30, patience=5,
                  songs_per_batch=8, samples_per_song=2, accumulation=4,
                  freeze_blocks=3, embedding_dim=128, vocal_ratio=0.80,
                  global_triplet=1.0, lac=1.0, lat=1.0, grl=0.2,
                  prototype_diversity=0.05, margin=0.2, mining_top_k=5,
                  memory_capacity=4096, lr_backbone=1e-4, lr_heads=3e-4,
                  positive_selection='best fused chunk from same-fragment reference bag',
                  weight_decay=0.01, clip=1.0, amp='fp16',
                  validation_gallery='train+val full recordings',
                  selection='fused_micro_mrr', score='0.5 local MaxMean + 0.5 global max',
                  test_gallery='all full recordings', tie_rtol=1e-6, tie_atol=1e-7)
PREPROCESS_CONFIG = dict(sr=16000, chunk_seconds=10, hop_seconds=3,
                         final_anchored_window=True, bins=96, frames=83,
                         cqt_hop=0.04, cqt_mean=3, floor_db=-80.0)
RUNTIME_VERSIONS = dict(torch=str(torch.__version__), numpy=np.__version__,
                        librosa=librosa.__version__, scipy=scipy.__version__,
                        soundfile=sf.__version__, cuda=str(torch.version.cuda))
BEST_NAME = 'qbh_final_v5_best.pth'
LAST_NAME = 'qbh_final_v5_last.pth'
TRAIN_OUTPUT = OUTPUT_ROOT / 'train'
EVAL_OUTPUT = OUTPUT_ROOT / 'test'
TRAIN_OUTPUT.mkdir(exist_ok=True)
EVAL_OUTPUT.mkdir(exist_ok=True)

# %% common
def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
                                     separators=(',', ':')).encode()).hexdigest()

FILE_HASHES = {}
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
    return FILE_HASHES[key]

def atomic_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)

def atomic_torch(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    torch.save(data, temp)
    os.replace(temp, path)

def chunk_starts(length, window, hop):
    if min(length, window, hop) < 1 or hop > window:
        raise ValueError('Invalid audio length/window/hop')
    if length <= window:
        return [0]
    starts = list(range(0, length - window + 1, hop))
    if starts[-1] + window < length:
        starts.append(length - window)
    return starts

class FullAudioPreprocessor(AudioPreprocessor):
    def slice_into_chunks(self, y):
        starts = chunk_starts(len(y), self.chunk_samples, self.hop_samples)
        return [np.pad(y[start:start + self.chunk_samples],
                       (0, max(0, self.chunk_samples - len(y[start:start + self.chunk_samples]))))
                for start in starts]

prep = FullAudioPreprocessor()
cqt_ext = CQTExtractor()

class BoundedCache(OrderedDict):
    def __init__(self, limit_mb=512):
        super().__init__()
        self.limit = limit_mb * 1024 ** 2
        self.bytes = 0

    @staticmethod
    def size(value):
        return sum(t.numel() * t.element_size() for t in value)

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        if key in self:
            self.bytes -= self.size(super().pop(key))
        size = self.size(value)
        if size > self.limit:
            return
        while self and self.bytes + size > self.limit:
            self.bytes -= self.size(self.popitem(last=False)[1])
        super().__setitem__(key, value)
        self.bytes += size

CQT_RAM = BoundedCache()
CQT_DIR = OUTPUT_ROOT / 'cache' / 'cqt'
FEATURE_DIR = OUTPUT_ROOT / 'cache' / 'features'
CQT_DIR.mkdir(parents=True, exist_ok=True)
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

def raw_cqts(path):
    key = digest(dict(audio=file_sha256(path), prep=PREPROCESS_CONFIG,
                      code=SOURCE_CODE_DIGEST, versions=RUNTIME_VERSIONS))
    if key in CQT_RAM:
        return CQT_RAM[key]
    cache_path = CQT_DIR / (key + '.pt')
    if cache_path.is_file():
        chunks = torch.load(cache_path, map_location='cpu', weights_only=True)
    else:
        y = prep.load_audio(str(path))
        if np.max(np.abs(y)) <= 1e-6:
            raise ValueError(f'Silent audio is not a valid query/reference: {path}')
        chunks = [cqt_ext.extract(x) for x in prep.slice_into_chunks(y)]
        atomic_torch(cache_path, chunks)
    if not chunks or any(x.shape != (1, 96, 83) or not torch.isfinite(x).all() for x in chunks):
        raise ValueError(f'Invalid CQT cache: {cache_path}')
    CQT_RAM[key] = chunks
    return chunks

AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
def split_songs(split):
    directory = DATASET_SPLIT_DIR / split
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    songs = sorted(p.name for p in directory.iterdir() if p.is_dir())
    if not songs:
        raise ValueError(f'Empty split: {split}')
    return songs

def scan_queries(split):
    rows = []
    for song in split_songs(split):
        for fragment in sorted((DATASET_SPLIT_DIR / split / song).iterdir()):
            if not fragment.is_dir():
                continue
            files = sorted(p for p in fragment.iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS)
            refs = [p for p in files if p.name == 'reference.wav']
            vocals = [p for p in files if p.name.startswith('reference_vocal')]
            hums = [p for p in files if not p.name.startswith('reference')]
            if hums and not refs:
                raise ValueError(f'Missing mixture positive reference.wav: {fragment}')
            rows.extend(dict(song=song, fragment=fragment.name, path=str(h),
                             key=h.relative_to(DATASET_SPLIT_DIR).as_posix(),
                             refs=[str(p) for p in refs], vocals=[str(p) for p in vocals]) for h in hums)
    if not rows:
        raise ValueError(f'No humming queries: {split}')
    if {r['song'] for r in rows} != set(split_songs(split)):
        raise ValueError(f'Songs without humming queries in {split}')
    return rows

SPLIT_IDS = {name: split_songs(name) for name in ('train', 'val', 'test')}
for left, right in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
    if set(SPLIT_IDS[left]) & set(SPLIT_IDS[right]):
        raise ValueError(f'Song leakage: {left}/{right}')

GALLERY_PATHS = {}
for path in sorted(MUSIC_DB_DIR.iterdir()):
    if path.suffix.lower() not in AUDIO_EXTENSIONS:
        continue
    if path.stem in GALLERY_PATHS:
        raise ValueError(f'Duplicate gallery song: {path.stem}')
    GALLERY_PATHS[path.stem] = str(path)
if set(GALLERY_PATHS) != set().union(*map(set, SPLIT_IDS.values())):
    raise ValueError('Full music gallery must exactly match the train/val/test song IDs.')

TRAIN_ROWS, VAL_ROWS = scan_queries('train'), scan_queries('val')
VAL_GALLERY_PATHS = {s: p for s, p in GALLERY_PATHS.items()
                     if s in set(SPLIT_IDS['train'] + SPLIT_IDS['val'])}

def query_records(rows):
    return [(r['key'], r['song'], file_sha256(r['path'])) for r in rows]

def data_contract():
    # Training hashes train+val only. Test query audio is never opened here.
    train_q, val_q = query_records(TRAIN_ROWS), query_records(VAL_ROWS)
    if {r[2] for r in train_q} & {r[2] for r in val_q}:
        raise ValueError('Duplicate humming audio across train/val')
    positives = sorted({p for r in TRAIN_ROWS for p in r['refs'] + r['vocals']})
    return dict(split_ids=SPLIT_IDS, train_queries=train_q, val_queries=val_q,
                positives=[(Path(p).relative_to(DATASET_SPLIT_DIR).as_posix(), file_sha256(p)) for p in positives],
                val_gallery=[(s, file_sha256(p)) for s, p in VAL_GALLERY_PATHS.items()])

DATA_CONTRACT = json.loads(json.dumps(data_contract()))
DATA_DIGEST = digest(DATA_CONTRACT)
print('Split sizes:', {s: len(ids) for s, ids in SPLIT_IDS.items()})
print('Queries train/val:', len(TRAIN_ROWS), len(VAL_ROWS), '| Validation gallery:', len(VAL_GALLERY_PATHS))

class FinalModel(ConformerByteHumModel):
    def train(self, mode=True):
        super().train(mode)
        if mode:
            encoder = self.feature_extractor._encoder
            self.feature_extractor._global_cmvn.eval()
            # Frozen parameters alone do not freeze dropout or BN running buffers.
            encoder.embed.eval()
            for block in list(encoder.encoders)[:RUN_CONFIG['freeze_blocks']]:
                block.eval()
        return self

class FinalTripletDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows
        self.songs = SPLIT_IDS['train']
        self.song_to_idx = {s: i for i, s in enumerate(self.songs)}
        self.humming_samples = [(r['song'], r['fragment'], r['path']) for r in rows]
        self.reference_by_song = {s: sorted({p for r in rows if r['song'] == s for p in r['refs']}) for s in self.songs}
        self.warp = SpectrogramTempoAndMaskDistortion((0.7, 1.3), 8, 4, 0.6)
        self.pitch = CQTPitchShiftAugmentation(12, 0.8)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        a = random.choice(raw_cqts(row['path'])).clone()
        a = self.pitch(self.warp(a))
        use_vocal = row['vocals'] and random.random() < RUN_CONFIG['vocal_ratio']
        pos_path = random.choice(row['vocals'] if use_vocal else row['refs'])
        # Positive identity is always the same source fragment, never another song.
        n_song = random.choice([s for s in self.songs if s != row['song']])
        n = random.choice(raw_cqts(random.choice(self.reference_by_song[n_song]))).clone()
        return dict(anchor=a, positive_path=pos_path, negative=n,
                    song_idx=torch.tensor(self.song_to_idx[row['song']]),
                    negative_song_idx=torch.tensor(self.song_to_idx[n_song]))

@torch.inference_mode()
def encode_cqts(eval_model, cqts):
    gs, ls = [], []
    for start in range(0, len(cqts), EVAL_BATCH_SIZE):
        with torch.autocast('cuda', dtype=torch.float16):
            g, l, _, _ = eval_model(torch.stack(cqts[start:start + EVAL_BATCH_SIZE]).to(device),
                                    compute_lac=False, compute_domain=False)
        gs.append(F.normalize(g.float(), dim=-1).cpu())
        ls.append(F.normalize(l.float(), dim=-1).cpu())
    g, l = torch.cat(gs), torch.cat(ls).reshape(-1, EMBEDDING_DIM)
    if not torch.isfinite(g).all() or not torch.isfinite(l).all():
        raise ValueError('Non-finite embeddings')
    return g, l

@torch.no_grad()
def aligned_positives(eval_model, positive_paths, anchor_global, anchor_local):
    # Multiple-instance positive selection avoids forcing a random unrelated chunk
    # of the 30s source fragment to align with a short humming query.
    if not len(positive_paths) or len(positive_paths) != len(anchor_global) or len(positive_paths) != len(anchor_local):
        raise ValueError('Positive paths and anchor features must have equal nonzero batch size')
    was_training = eval_model.training
    eval_model.eval()
    selected = [None] * len(positive_paths)
    groups = {}
    for index, path in enumerate(positive_paths):
        groups.setdefault(str(path), []).append(index)
    try:
        anchor_global = F.normalize(anchor_global.detach().float(), dim=-1)
        anchor_local = F.normalize(anchor_local.detach().float(), dim=-1)
        # Features are shared only within this call, never across optimizer updates.
        for path, indices in groups.items():
            chunks = raw_cqts(path)
            best_scores = {i: float('-inf') for i in indices}
            best_indices = {i: 0 for i in indices}
            for start in range(0, len(chunks), EVAL_BATCH_SIZE):
                with torch.autocast('cuda', dtype=torch.float16):
                    g, l, _, _ = eval_model(torch.stack(chunks[start:start + EVAL_BATCH_SIZE]).to(device),
                                            compute_lac=False, compute_domain=False)
                g, l = F.normalize(g.float(), dim=-1), F.normalize(l.float(), dim=-1)
                if not torch.isfinite(g).all() or not torch.isfinite(l).all():
                    raise FloatingPointError(f'Non-finite positive selection features: {path}')
                for index in indices:
                    local = torch.einsum('td,nsd->nts', anchor_local[index], l).amax(-1).mean(-1)
                    scores = 0.5 * (g @ anchor_global[index] + local)
                    value, offset = scores.max(dim=0)
                    if float(value) > best_scores[index]:
                        best_scores[index], best_indices[index] = float(value), start + int(offset)
            for index in indices:
                selected[index] = chunks[best_indices[index]].clone()
    finally:
        eval_model.train(was_training)
    return torch.stack(selected).to(device)

def gallery_features(eval_model, paths, checkpoint_hash=None):
    key = digest(dict(checkpoint=checkpoint_hash, records=[(s, file_sha256(p)) for s, p in paths.items()],
                      code=SOURCE_CODE_DIGEST, versions=RUNTIME_VERSIONS, prep=PREPROCESS_CONFIG))
    cache_path = FEATURE_DIR / (key + '.pt')
    if checkpoint_hash is not None and cache_path.is_file():
        gallery = torch.load(cache_path, map_location='cpu', weights_only=True)
    else:
        gallery = {s: encode_cqts(eval_model, raw_cqts(p)) for s, p in tqdm(paths.items(), desc='Full-song features')}
        if checkpoint_hash is not None:
            atomic_torch(cache_path, gallery)
    if list(gallery) != list(paths):
        raise ValueError('Gallery feature cache IDs mismatch')
    for g, l in gallery.values():
        if g.ndim != 2 or l.ndim != 2 or g.shape[1] != 128 or l.shape[1] != 128 or not len(g) or not len(l):
            raise ValueError('Invalid gallery cache shape')
        if not torch.isfinite(g).all() or not torch.isfinite(l).all():
            raise ValueError('Non-finite gallery cache')
    return gallery

def rank_statistics(scores, target):
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError('All exact-gallery scores must be finite')
    tied = np.isclose(values, values[target], rtol=1e-6, atol=1e-7)
    greater = int(((values > values[target]) & ~tied).sum())
    ties = int(tied.sum())
    rank = 1 + greater + 0.5 * (ties - 1)
    return dict(rank=float(rank), rr=float(1 / rank), ties=ties, greater=greater,
                **{f'recall_at_{k}': float(np.clip((k - greater) / ties, 0, 1)) for k in (1, 5, 10, 20)})

def retrieval_metrics(rows):
    if not rows:
        raise ValueError('Empty evaluation')
    frame = pd.DataFrame(rows)
    result = dict(queries=len(frame), songs=int(frame.song.nunique()))
    for mode in ('local', 'global', 'fused'):
        fields = [mode + '_rr'] + [mode + f'_recall_at_{k}' for k in (1, 5, 10, 20)]
        micro = frame[fields].mean()
        macro = frame.groupby('song')[fields].mean().mean()
        result[mode] = dict(mrr=float(micro.iloc[0]), macro_song_mrr=float(macro.iloc[0]),
                            **{f'recall_at_{k}': float(micro.iloc[i + 1]) for i, k in enumerate((1, 5, 10, 20))})
    return result

@torch.inference_mode()
def evaluate_queries(eval_model, gallery, queries, progress_path=None):
    songs = list(gallery)
    targets = {s: i for i, s in enumerate(songs)}
    completed = json.loads(progress_path.read_text(encoding='utf-8')) if progress_path and progress_path.exists() else []
    keys = [r['key'] for r in queries]
    if [r['key'] for r in completed] != keys[:len(completed)]:
        raise ValueError('Evaluation progress does not match the locked query list')
    for query in tqdm(queries[len(completed):], desc='Exact full-gallery retrieval'):
        torch.cuda.synchronize()
        start = time.perf_counter()
        qg, ql = encode_cqts(eval_model, raw_cqts(query['path']))
        scores = score_gallery(qg, ql, gallery)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        row = dict(key=query['key'], song=query['song'], seconds=elapsed)
        for mode, values in scores.items():
            stats = rank_statistics(values.cpu().numpy(), targets[query['song']])
            row.update({mode + '_' + name: value for name, value in stats.items()})
        completed.append(row)
        if progress_path:
            atomic_json(progress_path, completed)
    if len(completed) != len(queries):
        raise ValueError('Incorrect query denominator')
    return completed

# %% train
def validated_best_snapshot(state, run_signature, data_contract):
    if state.get('checkpoint_schema') != 2 or 'best_checkpoint' not in state:
        raise ValueError('Legacy checkpoint: use its original notebook, or start a new run with this code.')
    best = state['best_checkpoint']
    for item in (state, best):
        if item.get('run_signature') != run_signature or item.get('data_contract') != data_contract:
            raise ValueError('Resume contract mismatch')
    metric = state['best_mrr']
    if not math.isfinite(metric) or not 0 <= metric <= 1 or best['best_mrr'] != metric:
        raise ValueError('Invalid or inconsistent best validation metric')
    if not 1 <= best['epoch'] <= state['epoch'] or len(state['history']) != state['epoch']:
        raise ValueError('Invalid best epoch/history in last checkpoint')
    if [row['epoch'] for row in state['history']] != list(range(1, state['epoch'] + 1)):
        raise ValueError('Noncontiguous training history')
    values = [row['val_fused_mrr'] for row in state['history']]
    if not all(math.isfinite(value) and 0 <= value <= 1 for value in values):
        raise ValueError('Non-finite validation history')
    if abs(max(values) - metric) > 1e-8 or values[best['epoch'] - 1] != metric:
        raise ValueError('Best snapshot does not match validation history')
    if best['validation_metrics']['fused']['mrr'] != metric:
        raise ValueError('Best validation report mismatch')
    return best


def export_best_checkpoint(path, snapshot, finished):
    expected = dict(snapshot, finished=bool(finished))
    if Path(path).exists():
        current = torch.load(path, map_location='cpu', weights_only=False)
        metadata = lambda item: {key: value for key, value in item.items() if key != 'model_state_dict'}
        left, right = current.get('model_state_dict', {}), expected['model_state_dict']
        if metadata(current) == metadata(expected) and left.keys() == right.keys() and all(
                left[key].dtype == right[key].dtype and left[key].shape == right[key].shape
                and torch.equal(left[key], right[key]) for key in right):
            return expected  # Preserve checkpoint bytes/hash on completed Run All.
    atomic_torch(path, expected)
    return expected


def commit_training_checkpoint(last_path, best_path, state):
    snapshot = validated_best_snapshot(state, RUN_SIGNATURE, DATA_CONTRACT)
    # LAST is the single atomic commit; BEST/reports can always be rebuilt from it.
    atomic_torch(last_path, state)
    export_best_checkpoint(best_path, snapshot, state['finished'])
    atomic_json(Path(best_path).parent / 'best_validation_metrics.json', snapshot['validation_metrics'])
    atomic_json(Path(last_path).parent / 'history.json', state['history'])


def training_objective(outputs, labels, negative_labels, epoch, memory_bank):
    ga, gp, gn = [F.normalize(out[0].float(), dim=-1) for out in outputs]
    la, lp, ln = [out[1].float() for out in outputs]
    curriculum = mining_hardest_ratio(epoch)
    zero = ga.sum() * 0
    triplet = in_batch_cosine_triplet_loss(ga, gp, gn, labels, negative_labels,
        margin=RUN_CONFIG['margin'], memory_bank=memory_bank, hardest_ratio=curriculum,
        top_k=RUN_CONFIG['mining_top_k']) if RUN_CONFIG['global_triplet'] > 0 else zero
    lat = local_alignment_triplet(la, lp, ln, labels, negative_labels,
        margin=RUN_CONFIG['margin'], hardest_ratio=curriculum,
        top_k=RUN_CONFIG['mining_top_k']) if RUN_CONFIG['lat'] > 0 else zero
    lac = torch.stack([F.cross_entropy(out[2].float(), target) for out, target in zip(outputs, (labels, labels, negative_labels))]).mean() if RUN_CONFIG['lac'] > 0 else zero
    domain = torch.stack([F.binary_cross_entropy_with_logits(out[3].float(), torch.full_like(out[3].float(), target)) for out, target in zip(outputs, (1., 0., 0.))]).mean() if RUN_CONFIG['grl'] > 0 else zero
    prototype = prototype_diversity_loss(model.lac.prototypes.float())
    total = RUN_CONFIG['global_triplet'] * triplet + RUN_CONFIG['lat'] * lat + RUN_CONFIG['lac'] * lac + domain + RUN_CONFIG['prototype_diversity'] * prototype
    return dict(global_triplet=triplet, lat=lat, lac=lac, domain=domain, prototype=prototype, total=total), (ga, gp, gn)


STAGE1_CKPT = unique_path(STAGE1_OVERRIDE, INPUT_ROOT.rglob('coverhunter_shs100k_pretrained.pth'), 'STAGE1_OVERRIDE')
RUN_CONTRACT = dict(config=RUN_CONFIG, preprocessing=PREPROCESS_CONFIG,
                    code=SOURCE_CODE_DIGEST, runtime=RUNTIME_VERSIONS,
                    data=DATA_DIGEST, stage1=file_sha256(STAGE1_CKPT))
RUN_SIGNATURE = digest(RUN_CONTRACT)
contract_path = TRAIN_OUTPUT / 'run_contract.json'
if contract_path.exists():
    if json.loads(contract_path.read_text(encoding='utf-8')) != RUN_CONTRACT:
        raise ValueError('Run configuration/data/code changed. Use a new OUTPUT_ROOT, never overwrite this run.')
else:
    atomic_json(contract_path, RUN_CONTRACT)

train_dataset = FinalTripletDataset(TRAIN_ROWS)
loader_generator = torch.Generator().manual_seed(42)
sampler = SongBalancedBatchSampler(train_dataset, RUN_CONFIG['songs_per_batch'], RUN_CONFIG['samples_per_song'])
train_loader = DataLoader(train_dataset, batch_sampler=sampler, num_workers=0,
                          pin_memory=True, generator=loader_generator)
model = FinalModel(ckpt_path=str(STAGE1_CKPT), num_classes=len(SPLIT_IDS['train']))
model.freeze_lower_layers(RUN_CONFIG['freeze_blocks'], freeze_frontend=True)
model.to(device)
optimizer = torch.optim.AdamW([
    dict(params=[p for name, p in model.named_parameters() if p.requires_grad and not any(k in name for k in ('lac', 'domain_classifier'))], lr=RUN_CONFIG['lr_backbone']),
    dict(params=[p for name, p in model.named_parameters() if p.requires_grad and any(k in name for k in ('lac', 'domain_classifier'))], lr=RUN_CONFIG['lr_heads'])
], weight_decay=RUN_CONFIG['weight_decay'])
updates_per_epoch = math.ceil(len(train_loader) / RUN_CONFIG['accumulation'])
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
    max_lr=[RUN_CONFIG['lr_backbone'], RUN_CONFIG['lr_heads']],
    total_steps=updates_per_epoch * RUN_CONFIG['epochs'], pct_start=0.1)
scaler = torch.amp.GradScaler('cuda', init_scale=1024)
memory = CrossBatchMemoryBank(RUN_CONFIG['memory_capacity'])
best_path, last_path = TRAIN_OUTPUT / BEST_NAME, TRAIN_OUTPUT / LAST_NAME
history, best_mrr, patience, start_epoch, finished = [], -1.0, 0, 1, False
best_snapshot = None

if RESUME:
    source = last_path if last_path.exists() else (Path(RESUME_LAST_OVERRIDE) if RESUME_LAST_OVERRIDE else None)
    if source is not None:
        # RNG tensors must remain on CPU even when the model/optimizer live on CUDA.
        state = torch.load(source, map_location='cpu', weights_only=False)
        best_snapshot = validated_best_snapshot(state, RUN_SIGNATURE, DATA_CONTRACT)
        model.load_state_dict(state['model_state_dict'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        history, best_mrr, patience = state['history'], state['best_mrr'], state['patience']
        start_epoch, finished = state['epoch'] + 1, state['finished']
        random.setstate(state['rng_python'])
        np.random.set_state(state['rng_numpy'])
        torch.set_rng_state(state['rng_torch'])
        torch.cuda.set_rng_state_all(state['rng_cuda'])
        loader_generator.set_state(state['rng_loader'])
        commit_training_checkpoint(last_path, best_path, state)
        del state
        print('Resuming epoch:', start_epoch, '| Best fused validation MRR:', best_mrr)
    elif best_path.exists() or RESUME_BEST_OVERRIDE is not None:
        raise FileNotFoundError('BEST alone cannot resume optimizer/RNG. Attach the matching LAST or use a fresh OUTPUT_ROOT.')
elif last_path.exists() or best_path.exists():
    raise FileExistsError('Existing run: enable RESUME or choose a new OUTPUT_ROOT.')

def model_cpu_state():
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

def base_checkpoint(epoch, complete=False):
    return dict(model_state_dict=model_cpu_state(), run_signature=RUN_SIGNATURE,
                run_contract=RUN_CONTRACT, data_contract=DATA_CONTRACT, epoch=epoch,
                best_mrr=best_mrr, finished=complete, train_songs=SPLIT_IDS['train'], checkpoint_schema=2)

def gpu_preflight():
    saved_rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
                 torch.cuda.get_rng_state_all(), loader_generator.get_state())
    saved_weights, was_training = model_cpu_state(), model.training
    saved_lambdas = [(module, module.lambda_val) for module in model.modules() if hasattr(module, 'lambda_val')]
    probe_optimizer = torch.optim.AdamW([dict(group, params=list(group['params'])) for group in optimizer.param_groups], **optimizer.defaults)
    probe_scaler = torch.amp.GradScaler('cuda', init_scale=1.0)
    probe_memory = CrossBatchMemoryBank(RUN_CONFIG['memory_capacity'])
    model.train()
    try:
        probe_optimizer.zero_grad(set_to_none=True)
        batches = iter(train_loader)
        group_size = min(RUN_CONFIG['accumulation'], len(train_loader))
        if group_size < 1:
            raise ValueError('Empty training loader')
        for step in range(group_size):
            batch = next(batches)
            progress = ((start_epoch - 1) * len(train_loader) + step) / (RUN_CONFIG['epochs'] * len(train_loader))
            model.set_lambda(RUN_CONFIG['grl'] * (2 / (1 + math.exp(-10 * progress)) - 1))
            a, n = [batch[key].to(device) for key in ('anchor', 'negative')]
            labels, negative_labels = batch['song_idx'].to(device), batch['negative_song_idx'].to(device)
            with torch.autocast('cuda', dtype=torch.float16):
                oa = model(a, compute_lac=RUN_CONFIG['lac'] > 0, compute_domain=RUN_CONFIG['grl'] > 0)
            p = aligned_positives(model, batch['positive_path'], oa[0], oa[1])
            with torch.autocast('cuda', dtype=torch.float16):
                outputs = [oa] + [model(x, compute_lac=RUN_CONFIG['lac'] > 0, compute_domain=RUN_CONFIG['grl'] > 0) for x in (p, n)]
            expected = (RUN_CONFIG['songs_per_batch'] * RUN_CONFIG['samples_per_song'], 128)
            if oa[0].shape != expected or oa[1].shape != (expected[0], 20, 128):
                raise ValueError('Model output shape mismatch')
            terms, globals_ = training_objective(outputs, labels, negative_labels, start_epoch, probe_memory)
            if not torch.isfinite(terms['total']):
                raise FloatingPointError('GPU preflight loss is not finite')
            probe_scaler.scale(terms['total'] / group_size).backward()
            probe_memory.enqueue(torch.cat(globals_), torch.cat((labels, labels, negative_labels)))
            del terms, globals_, outputs, oa, a, p, n
        probe_scaler.unscale_(probe_optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), RUN_CONFIG['clip'])
        if not torch.isfinite(norm):
            raise FloatingPointError('GPU preflight gradient is not finite')
        previous_scale = probe_scaler.get_scale()
        probe_scaler.step(probe_optimizer)
        probe_scaler.update()
        if probe_scaler.get_scale() < previous_scale:
            raise FloatingPointError('GPU preflight optimizer update was skipped')
        if any(not torch.isfinite(p).all() for p in model.parameters()):
            raise FloatingPointError('GPU preflight optimizer produced non-finite parameters')
        if not any(p.requires_grad and not torch.equal(p.detach().cpu(), saved_weights[name]) for name, p in model.named_parameters()):
            raise RuntimeError('GPU preflight optimizer did not change any trainable parameter')
        model.eval()
        g, l = encode_cqts(model, raw_cqts(TRAIN_ROWS[0]['path']))
        scores = score_gallery(g, l, {'self': (g, l), 'other': (F.normalize(torch.randn_like(g), dim=-1), F.normalize(torch.randn_like(l), dim=-1))})
        if float(scores['fused'][0]) < 0.999:
            raise ValueError('GPU exact scoring identity check failed')
        print('GPU preflight passed: train-mode AMP accumulation, all configured losses, AdamW update and exact scoring. Probe state is rolled back.')
    finally:
        model.load_state_dict(saved_weights, strict=True)
        for module, value in saved_lambdas:
            module.lambda_val = value
        optimizer.zero_grad(set_to_none=True)
        random.setstate(saved_rng[0])
        np.random.set_state(saved_rng[1])
        torch.set_rng_state(saved_rng[2])
        torch.cuda.set_rng_state_all(saved_rng[3])
        loader_generator.set_state(saved_rng[4])
        model.train(was_training)
        del probe_optimizer, probe_scaler, probe_memory, saved_weights
        gc.collect()
        torch.cuda.empty_cache()

if not finished:
    gpu_preflight()
    prewarm_paths = sorted({r['path'] for r in TRAIN_ROWS + VAL_ROWS}
                           | {p for r in TRAIN_ROWS for p in r['refs'] + r['vocals']}
                           | set(VAL_GALLERY_PATHS.values()))
    for path in tqdm(prewarm_paths, desc='Validate audio and cache complete CQT'):
        raw_cqts(path)

if not finished:
    for epoch in range(start_epoch, RUN_CONFIG['epochs'] + 1):
        model.train()
        memory.clear()
        optimizer.zero_grad(set_to_none=True)
        losses = dict(global_triplet=0., lat=0., lac=0., domain=0., prototype=0., total=0.)
        skipped_updates = 0
        for step, batch in enumerate(tqdm(train_loader, desc=f'Epoch {epoch}')):
            progress = ((epoch - 1) * len(train_loader) + step) / (RUN_CONFIG['epochs'] * len(train_loader))
            model.set_lambda(RUN_CONFIG['grl'] * (2 / (1 + math.exp(-10 * progress)) - 1))
            a, n = [batch[key].to(device, non_blocking=True) for key in ('anchor', 'negative')]
            labels, negative_labels = batch['song_idx'].to(device), batch['negative_song_idx'].to(device)
            with torch.autocast('cuda', dtype=torch.float16):
                oa = model(a, compute_lac=RUN_CONFIG['lac'] > 0, compute_domain=RUN_CONFIG['grl'] > 0)
            p = aligned_positives(model, batch['positive_path'], oa[0], oa[1])
            with torch.autocast('cuda', dtype=torch.float16):
                outputs = [oa] + [model(x, compute_lac=RUN_CONFIG['lac'] > 0, compute_domain=RUN_CONFIG['grl'] > 0) for x in (p, n)]
            terms, globals_ = training_objective(outputs, labels, negative_labels, epoch, memory)
            total = terms['total']
            if not bool(torch.isfinite(total)):
                raise FloatingPointError(f'Non-finite loss at epoch {epoch}, batch {step}')
            # The last accumulation group may have fewer microbatches.
            group_start = (step // RUN_CONFIG['accumulation']) * RUN_CONFIG['accumulation']
            group_size = min(RUN_CONFIG['accumulation'], len(train_loader) - group_start)
            scaler.scale(total / group_size).backward()
            memory.enqueue(torch.cat(globals_), torch.cat((labels, labels, negative_labels)))
            for key, value in terms.items():
                losses[key] += float(value.detach())
            if (step + 1) % RUN_CONFIG['accumulation'] == 0 or step + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), RUN_CONFIG['clip'])
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= previous_scale:
                    scheduler.step()
                else:
                    skipped_updates += 1
                optimizer.zero_grad(set_to_none=True)
            del outputs, oa, globals_, terms, total, a, p, n, value
        if skipped_updates == updates_per_epoch:
            raise FloatingPointError('All optimizer updates were skipped by AMP; do not accept this epoch.')
        model.eval()
        gallery = gallery_features(model, VAL_GALLERY_PATHS)
        val_rows = evaluate_queries(model, gallery, VAL_ROWS)
        metrics = retrieval_metrics(val_rows)
        del gallery
        val_mrr = metrics['fused']['mrr']
        history.append(dict(epoch=epoch, **{key: value / len(train_loader) for key, value in losses.items()},
                            val_fused_mrr=val_mrr, val_macro_mrr=metrics['fused']['macro_song_mrr'],
                            val_recall_at_1=metrics['fused']['recall_at_1'], skipped_updates=skipped_updates))
        if val_mrr > best_mrr + 1e-8:
            best_mrr, patience = val_mrr, 0
            best_snapshot = dict(base_checkpoint(epoch), validation_metrics=metrics)
        else:
            patience += 1
        finished = patience >= RUN_CONFIG['patience'] or epoch == RUN_CONFIG['epochs']
        state = base_checkpoint(epoch, finished)
        state.update(optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
                     history=history, patience=patience, rng_python=random.getstate(), rng_numpy=np.random.get_state(),
                     rng_torch=torch.get_rng_state(), rng_cuda=torch.cuda.get_rng_state_all(), rng_loader=loader_generator.get_state(),
                     best_checkpoint=best_snapshot)
        commit_training_checkpoint(last_path, best_path, state)
        del state
        print(f'Epoch {epoch}: fused val MRR={val_mrr:.4f}, best={best_mrr:.4f}, patience={patience}')
        if finished:
            break

if not finished or not best_path.exists():
    raise RuntimeError('Training is not complete. Resume it before running final evaluation.')
best = export_best_checkpoint(best_path, best_snapshot, finished=True)
receipt = dict(run_signature=RUN_SIGNATURE, checkpoint_sha256=file_sha256(best_path),
               best_epoch=best['epoch'], best_validation_fused_mrr=best['best_mrr'],
               completed_epochs=len(history), test_evaluated=False)
atomic_json(TRAIN_OUTPUT / 'final_manifest.json', receipt)
model.load_state_dict(best['model_state_dict'])
model.eval()
print('Training complete. Best checkpoint:', best_path)
display(pd.DataFrame([receipt]))

# %% plots
frame = pd.DataFrame(history)
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
frame.set_index('epoch')[['global_triplet', 'lat']].plot(ax=axes[0], title='Training retrieval losses')
frame.set_index('epoch')[['lac', 'domain', 'prototype']].plot(ax=axes[1], title='Training auxiliary losses')
frame.set_index('epoch')[['val_fused_mrr', 'val_macro_mrr']].plot(ax=axes[2], title='Validation exact fused retrieval')
fig.tight_layout()
fig.savefig(TRAIN_OUTPUT / 'training_curves.png', dpi=160)
plt.show()
print('Attach this training output to the separate final evaluation notebook. No test queries were evaluated here.')

# %% evaluation
CHECKPOINT_PATH = unique_path(CHECKPOINT_OVERRIDE,
    list(INPUT_ROOT.rglob(BEST_NAME)) + list(TRAIN_OUTPUT.glob(BEST_NAME)), 'CHECKPOINT_OVERRIDE')
checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
if not checkpoint.get('finished'):
    raise ValueError('This checkpoint was not finalized by the training notebook.')
contract = checkpoint['run_contract']
if checkpoint['run_signature'] != digest(contract):
    raise ValueError('Corrupt run signature')
if contract['code'] != SOURCE_CODE_DIGEST or contract['config'] != RUN_CONFIG or contract['preprocessing'] != PREPROCESS_CONFIG:
    raise ValueError('Checkpoint and evaluator code/config do not match')
if contract['data'] != DATA_DIGEST or checkpoint['data_contract'] != DATA_CONTRACT:
    raise ValueError('Dataset changed since training')
if contract['runtime'] != RUNTIME_VERSIONS:
    raise ValueError('Use the same PyTorch/audio runtime as training; exact resume/cache reproducibility requires it.')
if checkpoint['train_songs'] != SPLIT_IDS['train']:
    raise ValueError('Training song mapping changed')
TEST_ROWS = scan_queries('test')
test_records = query_records(TEST_ROWS)
old_hashes = {r[2] for r in DATA_CONTRACT['train_queries'] + DATA_CONTRACT['val_queries']}
if old_hashes & {r[2] for r in test_records}:
    raise ValueError('Duplicate humming audio across train/val/test')
gallery_records = [(s, file_sha256(p)) for s, p in GALLERY_PATHS.items()]
if {r[2] for r in test_records} & {r[1] for r in gallery_records}:
    raise ValueError('Humming queries must not be copies of gallery references')
test_contract = dict(run_signature=checkpoint['run_signature'], checkpoint=file_sha256(CHECKPOINT_PATH),
                     query_records=test_records, gallery_records=gallery_records,
                     scoring=RUN_CONFIG['score'], code=SOURCE_CODE_DIGEST, runtime=RUNTIME_VERSIONS)
test_contract = json.loads(json.dumps(test_contract))
test_signature = digest(test_contract)
lock_path = EVAL_OUTPUT / 'test_lock.json'
report_path = EVAL_OUTPUT / 'test_metrics.json'
progress_path = EVAL_OUTPUT / 'test_progress.json'
resume_folders = []
for candidate in INPUT_ROOT.rglob('test_lock.json'):
    saved_lock = json.loads(candidate.read_text(encoding='utf-8'))
    if saved_lock.get('code') == SOURCE_CODE_DIGEST:
        resume_folders.append(candidate.parent)
if EVAL_RESUME_DIR is not None or resume_folders:
    resume_folder = unique_path(EVAL_RESUME_DIR, resume_folders, 'EVAL_RESUME_DIR')
    saved_lock = json.loads((resume_folder / 'test_lock.json').read_text(encoding='utf-8'))
    if saved_lock != test_contract:
        raise ValueError('Attached evaluation output belongs to a different locked test.')
    if resume_folder.resolve() != EVAL_OUTPUT.resolve():
        for name in ('test_lock.json', 'test_progress.json', 'test_metrics.json', 'test_queries.csv'):
            source, target = resume_folder / name, EVAL_OUTPUT / name
            if source.exists() and not target.exists():
                temp = target.with_suffix(target.suffix + '.tmp')
                shutil.copyfile(source, temp)
                os.replace(temp, target)
        print('Restored attached evaluation state:', resume_folder)
if lock_path.exists():
    if json.loads(lock_path.read_text(encoding='utf-8')) != test_contract:
        raise ValueError('This test output is locked to a different checkpoint/data/protocol.')
else:
    if report_path.exists() or progress_path.exists():
        raise ValueError('Evaluation artifacts without a lock: use a fresh output directory.')
    atomic_json(lock_path, test_contract)

if report_path.exists():
    report = json.loads(report_path.read_text(encoding='utf-8'))
    if report['test_signature'] != test_signature:
        raise ValueError('Test report signature mismatch')
    print('Final test already completed. Displaying the saved report without evaluating again.')
else:
    eval_model = FinalModel(num_classes=len(SPLIT_IDS['train']), allow_random_init=True).to(device)
    eval_model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    eval_model.eval()
    started = time.perf_counter()
    gallery = gallery_features(eval_model, GALLERY_PATHS, file_sha256(CHECKPOINT_PATH))
    rows = evaluate_queries(eval_model, gallery, TEST_ROWS, progress_path)
    metrics = retrieval_metrics(rows)
    timings = [r['seconds'] for r in rows]
    # Cluster bootstrap by song, not by correlated humming recordings.
    song_rr = pd.DataFrame(rows).groupby('song').fused_rr.mean().to_numpy()
    rng = np.random.default_rng(42)
    bootstrap = rng.choice(song_rr, size=(2000, len(song_rr)), replace=True).mean(axis=1)
    report = dict(test_signature=test_signature, run_signature=checkpoint['run_signature'],
                  checkpoint_sha256=file_sha256(CHECKPOINT_PATH), best_epoch=checkpoint['epoch'],
                  protocol='closed-set exact full-recording gallery; no candidate gate, no DTW tuning',
                  gallery_songs=len(GALLERY_PATHS), queries=len(TEST_ROWS), metrics=metrics,
                  macro_song_mrr_ci95=np.quantile(bootstrap, [0.025, 0.975]).tolist(),
                  query_seconds_p50=float(np.median(timings)), query_seconds_p95=float(np.quantile(timings, .95)),
                  wall_seconds_this_session=time.perf_counter() - started,
                  open_set_calibrated=False, public_benchmark_evaluated=False,
                  historically_evaluated_test_split=True,
                  created_at_utc=datetime.now(timezone.utc).isoformat())
    pd.DataFrame(rows).to_csv(EVAL_OUTPUT / 'test_queries.csv', index=False)
    atomic_json(report_path, report)
display(pd.DataFrame(report['metrics']).T)
print('Gallery:', report['gallery_songs'], '| Queries:', report['queries'], '| Best epoch:', report['best_epoch'])
print('Fused macro-song MRR 95% bootstrap interval:', report['macro_song_mrr_ci95'])
print('Final report:', report_path)
