"""Build a standalone local evaluator from the Kaggle model definitions."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / 'notebooks'
TARGET = ROOT / 'local_4gb' / 'evaluate' / 'conformer_bytehum_qbh_evaluation_local_4gb.ipynb'

CONFIG = r'''from pathlib import Path
import os

# Edit before Run All. This notebook does not train or overwrite training artifacts.
PROJECT_ROOT_OVERRIDE = None
CHECKPOINT_OVERRIDE = None  # Optional absolute path to a compatible v4 best checkpoint.
GALLERY_MODE = 'full'  # 'test': split references; 'full': original music database.
GALLERY_MANIFEST = None  # Optional JSON mapping song IDs to audio path(s) for full mode.
EVAL_BATCH_SIZE = 4
GALLERY_SONG_BLOCK = 8
REFERENCE_FRAME_BLOCK = 512
LOCAL_RAM_CACHE_MB = 256
COMPARE_ZERO_SHOT = True
COMPARE_KAGGLE = True
RUN_FAISS_BENCHMARK = True  # Requires faiss-cpu; exact evaluation works without FAISS.
FAISS_CHUNK_K = 50  # Neighbors per query chunk, not unique songs.
MAX_EVAL_QUERIES = None  # None = all queries; subsets are explicitly labeled.
RUN_DTW_BENCHMARK = False  # CPU-intensive; never needed for the exact baseline.
DTW_MAX_QUERIES = 50  # None = all queries; optional benchmark has a separate denominator.
DTW_CANDIDATES = 10
DTW_WEIGHT = 0.40
DTW_MAX_FRAMES = 4000
QUERY_AUDIO_PATH = None  # Set a local humming WAV/MP3 path for the final search demo.

if PROJECT_ROOT_OVERRIDE is None:
    PROJECT_ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
                         if (p / '02_stage2_humming_qbh').is_dir()), None)
    if PROJECT_ROOT is None:
        raise FileNotFoundError('Set PROJECT_ROOT_OVERRIDE to your project root.')
else:
    PROJECT_ROOT = Path(PROJECT_ROOT_OVERRIDE).resolve()
STAGE2_ROOT = PROJECT_ROOT / '02_stage2_humming_qbh'
if GALLERY_MODE not in ('test', 'full'):
    raise ValueError("GALLERY_MODE must be 'test' or 'full'.")
if min(EVAL_BATCH_SIZE, GALLERY_SONG_BLOCK, REFERENCE_FRAME_BLOCK, FAISS_CHUNK_K, DTW_CANDIDATES, DTW_MAX_FRAMES) < 1:
    raise ValueError('Batch/candidate/frame limits must be positive.')
if not 0 <= DTW_WEIGHT <= 1:
    raise ValueError('DTW_WEIGHT must lie in [0,1].')
for limit in (MAX_EVAL_QUERIES, DTW_MAX_QUERIES):
    if limit is not None and limit < 1:
        raise ValueError('Query limits must be positive or None.')
OUTPUT_DIR = STAGE2_ROOT / 'runs' / 'local_4gb' / ('evaluation_' + GALLERY_MODE)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
for env, folder in [('NUMBA_CACHE_DIR', 'numba'), ('MPLCONFIGDIR', 'matplotlib')]:
    directory = STAGE2_ROOT / 'cache' / folder
    directory.mkdir(parents=True, exist_ok=True)
    os.environ[env] = str(directory)
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
print('Project:', PROJECT_ROOT)
print('Evaluation output:', OUTPUT_DIR)
'''

IMPORTS = '''import json
import hashlib
import gc
import time
import random
import sys
import logging
import math
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
from torch.utils.data import Dataset
import soundfile as sf
import librosa
import scipy.sparse
import typeguard
if not hasattr(typeguard, 'check_argument_types'):
    raise ImportError('CoverHunter requires typeguard<3 in this notebook kernel.')
try:
    import faiss
except ImportError:
    faiss = None
    if RUN_FAISS_BENCHMARK:
        print('FAISS benchmark skipped: install faiss-cpu in this kernel to enable it.')
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
AMP_ENABLED = device.type == 'cuda'
if AMP_ENABLED:
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)
print('PyTorch:', torch.__version__, '| Device:', device)
if AMP_ENABLED:
    print(torch.cuda.get_device_name(0), '| VRAM GiB:', round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
'''

PATHS = '''DATASET_SPLIT_DIR = STAGE2_ROOT / 'datasets' / 'dataset_split'
TRAIN_DIR, VAL_DIR, TEST_DIR = [DATASET_SPLIT_DIR / name for name in ('train', 'val', 'test')]
MUSIC_DB_DIR = STAGE2_ROOT / 'datasets' / '2_music_database'
for directory in (TRAIN_DIR, VAL_DIR, TEST_DIR):
    if not directory.is_dir():
        raise FileNotFoundError(directory)
stage1_candidates = [STAGE2_ROOT / 'checkpoints' / 'coverhunter_shs100k_pretrained.pth',
                     PROJECT_ROOT / '01_stage1_cover_song_csi' / 'checkpoints' / 'coverhunter_shs100k_pretrained.pth']
STAGE1_CKPT = next((str(p) for p in stage1_candidates if p.is_file()), None)
finetuned_ckpt_path = Path(CHECKPOINT_OVERRIDE) if CHECKPOINT_OVERRIDE else (
    STAGE2_ROOT / 'runs' / 'local_4gb' / 'checkpoints' / 'conformer_bytehum_full_d128_timegrid_lat_v4_local4gb_best.pth')
if not finetuned_ckpt_path.is_file():
    raise FileNotFoundError(finetuned_ckpt_path)
if COMPARE_ZERO_SHOT and STAGE1_CKPT is None:
    raise FileNotFoundError('Missing Stage-1 checkpoint for zero-shot comparison.')
print('Fine-tuned checkpoint:', finetuned_ckpt_path)
'''

SETUP = '''def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()

def load_compatible_checkpoint(path):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    cfg = payload.get('run_config', {})
    if cfg.get('pipeline_version') != 4 or cfg.get('embedding_dim') != 128:
        raise ValueError('Expected a pipeline-v4, 128-D checkpoint: ' + str(path))
    state = payload['model_state_dict']
    if 'lac.prototypes' not in state or state['lac.prototypes'].shape[-1] != 128:
        raise ValueError('Missing/incompatible LAC prototypes.')
    train_ids = sorted(p.name for p in TRAIN_DIR.iterdir() if p.is_dir())
    if payload.get('train_songs') != train_ids or len(train_ids) != state['lac.prototypes'].shape[0]:
        raise ValueError('Checkpoint train-song mapping does not match the dataset.')
    return payload

checkpoint_metadata = load_compatible_checkpoint(finetuned_ckpt_path)
EMBEDDING_DIM = 128
NUM_TRAIN_SONGS = checkpoint_metadata['model_state_dict']['lac.prototypes'].shape[0]
test_dataset = ByteHumCachedTripletDataset(str(TEST_DIR), prep, cqt_ext, is_training=False)
split_ids = {name: {p.name for p in directory.iterdir() if p.is_dir()}
             for name, directory in [('train', TRAIN_DIR), ('val', VAL_DIR), ('test', TEST_DIR)]}
for left, right in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
    if split_ids[left] & split_ids[right]:
        raise ValueError(f'Song leakage: {left}/{right}')

def resolve_gallery(dataset, mode=None):
    mode = GALLERY_MODE if mode is None else mode
    if mode == 'test':
        result = {}
        for song, fragments in dataset.song_data.items():
            result[song] = sorted({p for pool in fragments.values()
                                   for p in (pool['refs'] if pool['refs'] else pool['vocals'])})
    elif GALLERY_MANIFEST:
        manifest_path = Path(GALLERY_MANIFEST).resolve()
        mapping = json.loads(manifest_path.read_text(encoding='utf-8'))
        result = {}
        for song, paths in mapping.items():
            result[str(song)] = [str((manifest_path.parent / Path(p)).resolve())
                                for p in ([paths] if isinstance(paths, str) else paths)]
    else:
        if not MUSIC_DB_DIR.is_dir():
            raise FileNotFoundError(MUSIC_DB_DIR)
        result = {}
        for path in sorted(MUSIC_DB_DIR.iterdir()):
            if path.suffix.lower() in ('.wav', '.mp3', '.flac', '.ogg', '.m4a'):
                if path.stem in result:
                    raise ValueError('Duplicate gallery song ID; use GALLERY_MANIFEST: ' + path.stem)
                result[path.stem] = [str(path)]
    if not result or any(not paths for paths in result.values()):
        raise ValueError('Empty gallery or song without a reference.')
    for paths in result.values():
        for path in paths:
            if not Path(path).is_file():
                raise FileNotFoundError(path)
    missing = {song for song, _, _ in dataset.humming_samples} - result.keys()
    if missing:
        raise ValueError('Query ground truth missing from gallery: ' + str(sorted(missing)[:5]))
    query_paths = {str(Path(p).resolve()) for _, _, p in dataset.humming_samples}
    if query_paths & {str(Path(p).resolve()) for paths in result.values() for p in paths}:
        raise ValueError('Query/reference file overlap.')
    return dict(sorted(result.items()))

gallery_paths = resolve_gallery(test_dataset)
all_test_queries = list(test_dataset.humming_samples)
eval_queries = all_test_queries if MAX_EVAL_QUERIES is None else all_test_queries[:MAX_EVAL_QUERIES]
print('Best epoch:', checkpoint_metadata['epoch'], '| Val MRR:', checkpoint_metadata['best_val_mrr'])
print('Protocol:', GALLERY_MODE, '| Gallery:', len(gallery_paths), '| Queries:', len(eval_queries), '/', len(all_test_queries))
'''

ENGINE = '''@torch.inference_mode()
def encode_cqts(eval_model, cqts):
    if not cqts:
        raise ValueError('No CQT chunks to encode.')
    gs, ls = [], []
    for offset in range(0, len(cqts), EVAL_BATCH_SIZE):
        batch = torch.stack(cqts[offset:offset + EVAL_BATCH_SIZE]).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=AMP_ENABLED):
            g, l, _, _ = eval_model(batch, compute_lac=False, compute_domain=False)
        gs.append(F.normalize(g.float(), dim=-1).cpu())
        ls.append(F.normalize(l.float(), dim=-1).cpu())
    g, l = torch.cat(gs), torch.cat(ls).reshape(-1, EMBEDDING_DIM)
    if not torch.isfinite(g).all() or not torch.isfinite(l).all():
        raise ValueError('Non-finite model features.')
    return g, l

def synchronize():
    if AMP_ENABLED:
        torch.cuda.synchronize()

def gallery_signature(paths, checkpoint_hash):
    records = [(song, str(Path(path).resolve()), Path(path).stat().st_size, Path(path).stat().st_mtime_ns)
               for song, files in paths.items() for path in files]
    return hashlib.sha256(json.dumps({'checkpoint': checkpoint_hash, 'records': records,
        'pipeline': 'v4_16k_10s_hop3_96x83', 'amp': AMP_ENABLED,
        'torch': torch.__version__}, sort_keys=True).encode()).hexdigest()

@torch.inference_mode()
def build_gallery(eval_model, paths, checkpoint_hash):
    signature = gallery_signature(paths, checkpoint_hash)
    cache_dir = STAGE2_ROOT / 'cache' / 'evaluation_features'
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / (signature + '.pt')
    if cache_path.is_file():
        cached = torch.load(cache_path, map_location='cpu', weights_only=True)
        if cached['signature'] != signature or list(cached['features']) != list(paths):
            raise ValueError('Gallery feature-cache mismatch: ' + str(cache_path))
        for g, l in cached['features'].values():
            if g.ndim != 2 or l.ndim != 2 or g.shape[1] != EMBEDDING_DIM or l.shape[1] != EMBEDDING_DIM or not len(g) or not len(l) or not torch.isfinite(g).all() or not torch.isfinite(l).all():
                raise ValueError('Invalid gallery feature-cache: ' + str(cache_path))
        return cached['features'], signature
    features = {}
    for song, files in tqdm(paths.items(), desc='Encode gallery'):
        parts = [encode_cqts(eval_model, test_dataset._get_raw_cqts(path)) for path in files]
        features[song] = (torch.cat([g for g, _ in parts]), torch.cat([l for _, l in parts]))
    temp_path = cache_path.with_suffix('.tmp')
    torch.save({'signature': signature, 'features': features}, temp_path)
    os.replace(temp_path, cache_path)
    return features, signature

@torch.inference_mode()
def score_gallery(query_global, query_local, gallery):
    # Bound padding to one song group, and similarities to reference-frame blocks.
    qg, ql = query_global.to(device), query_local.to(device)
    songs = list(gallery)
    local_scores, global_scores = [], []
    for offset in range(0, len(songs), GALLERY_SONG_BLOCK):
        group = [gallery[s] for s in songs[offset:offset + GALLERY_SONG_BLOCK]]
        gp = nn.utils.rnn.pad_sequence([g for g, _ in group], batch_first=True).to(device)
        gm = torch.arange(gp.shape[1], device=device)[None, :] < torch.tensor([len(g) for g, _ in group], device=device)[:, None]
        similarities = torch.einsum('qd,sgd->sqg', qg, gp).masked_fill(~gm[:, None, :], float('-inf'))
        global_scores.append(similarities.amax(dim=(1, 2)).cpu())
        max_per_query = torch.full((len(group), len(ql)), float('-inf'), device=device)
        longest = max(len(l) for _, l in group)
        for frame_start in range(0, longest, REFERENCE_FRAME_BLOCK):
            slices = [l[frame_start:frame_start + REFERENCE_FRAME_BLOCK] for _, l in group]
            padded = nn.utils.rnn.pad_sequence(slices, batch_first=True).to(device)
            mask = torch.arange(padded.shape[1], device=device)[None, :] < torch.tensor([len(l) for l in slices], device=device)[:, None]
            sims = torch.einsum('qd,sgd->sqg', ql, padded).masked_fill(~mask[:, None, :], float('-inf'))
            max_per_query = torch.maximum(max_per_query, sims.amax(dim=-1))
        local_scores.append(max_per_query.mean(dim=-1).cpu())
    local, global_ = torch.cat(local_scores), torch.cat(global_scores)
    return {'local': local, 'global': global_, 'fused': 0.5 * (local + global_)}

def tie_statistics(scores, target):
    values = np.asarray(scores, dtype=np.float64)
    if not np.isfinite(values[target]):
        return float('inf'), 0, float('inf')
    tied = np.isclose(values, values[target], rtol=1e-6, atol=1e-7)
    # Treat close scores as one tie group, excluding them from greater counts.
    greater = int(((values > values[target]) & ~tied).sum())
    ties = int(tied.sum())
    return 1.0 + greater + 0.5 * (ties - 1), ties, greater

def summarize_rows(frame, mode):
    ranks = frame[mode + '_rank'].to_numpy()
    ties = frame[mode + '_ties'].to_numpy()
    greater = frame[mode + '_greater'].to_numpy()
    valid = np.isfinite(ranks)
    result = {'mrr': float(np.mean(np.where(valid, 1.0 / ranks, 0.0))),
              'mean_rank': float(ranks.mean()) if valid.all() else None,
              'retrieved_queries': int(valid.sum()), 'queries': len(frame)}
    cmc = []
    for k in range(1, 21):
        expected_hits = np.zeros(len(frame))
        expected_hits[valid] = np.clip((k - greater[valid]) / ties[valid], 0, 1)
        cmc.append(float(expected_hits.mean()))
    result.update({f'recall_at_{k}': cmc[k - 1] for k in (1, 5, 10, 20)})
    result['cmc'] = cmc
    return result

def build_faiss_gallery(gallery):
    if faiss is None:
        return None, None
    songs = list(gallery)
    matrix = np.ascontiguousarray(torch.cat([gallery[s][0] for s in songs]).numpy(), dtype=np.float32)
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(matrix)
    ids = np.concatenate([np.full(len(gallery[s][0]), i, dtype=np.int64) for i, s in enumerate(songs)])
    return index, ids

def candidate_ids(query_global, index, chunk_to_song, k):
    if k < 1:
        raise ValueError('FAISS k must be positive.')
    query = np.ascontiguousarray(query_global.numpy(), dtype=np.float32)
    _, nearest = index.search(query, min(k, index.ntotal))
    return set(int(chunk_to_song[i]) for i in nearest.ravel() if i >= 0)

@torch.inference_mode()
def evaluate_model(eval_model, gallery, queries, label):
    if not queries:
        raise ValueError('No queries; no result can be reported.')
    songs = list(gallery)
    song_to_index = {s: i for i, s in enumerate(songs)}
    index, chunk_map = build_faiss_gallery(gallery) if RUN_FAISS_BENCHMARK else (None, None)
    # Warm model/CUDA before timing; reported CQT/I/O remains cache-state-dependent.
    encode_cqts(eval_model, test_dataset._get_raw_cqts(queries[0][2])[:1])
    synchronize()
    rows, query_scores = [], {}
    for song, fragment, path in tqdm(queries, desc='Queries ' + label):
        if song not in song_to_index:
            raise ValueError('Missing ground truth: ' + song)
        start = time.perf_counter()
        cqts = test_dataset._get_raw_cqts(path)
        cqt_ms = (time.perf_counter() - start) * 1000
        synchronize()
        start_model = time.perf_counter()
        qg, ql = encode_cqts(eval_model, cqts)
        synchronize()
        model_ms = (time.perf_counter() - start_model) * 1000
        start_score = time.perf_counter()
        scores = score_gallery(qg, ql, gallery)
        synchronize()
        score_ms = (time.perf_counter() - start_score) * 1000
        row = {'song': song, 'fragment': fragment, 'query': path, 'cqt_io_ms': cqt_ms,
               'model_ms': model_ms, 'exact_scoring_ms': score_ms}
        if index is not None:
            gate_start = time.perf_counter()
            candidates = candidate_ids(qg, index, chunk_map, FAISS_CHUNK_K)
            row['faiss_ms'] = (time.perf_counter() - gate_start) * 1000
            row['faiss_candidate_hit'] = song_to_index[song] in candidates
            row['faiss_candidate_songs'] = len(candidates)
            gated = torch.full_like(scores['fused'], float('-inf'))
            kept = sorted(candidates)
            gated[kept] = scores['fused'][kept]
            scores['faiss_fused'] = gated
        for mode, values in scores.items():
            rank, ties, greater = tie_statistics(values.numpy(), song_to_index[song])
            row.update({mode + '_rank': rank, mode + '_ties': ties, mode + '_greater': greater,
                        mode + '_top1_song': songs[int(values.argmax())]})
        rows.append(row)
        query_scores[path] = scores['fused'].clone()
    frame = pd.DataFrame(rows)
    modes = ['local', 'global', 'fused'] + (['faiss_fused'] if index is not None else [])
    result = {'label': label, 'queries': len(frame), 'gallery_songs': len(songs),
              'modes': {mode: summarize_rows(frame, mode) for mode in modes},
              'latency': {column: {'p50_ms': float(frame[column].median()),
                                   'p95_ms': float(frame[column].quantile(.95))}
                          for column in ('cqt_io_ms', 'model_ms', 'exact_scoring_ms')}}
    if index is not None:
        result['faiss'] = {'kind': 'flat', 'chunk_k_per_query_chunk': FAISS_CHUNK_K,
                          'candidate_recall': float(frame.faiss_candidate_hit.mean()),
                          'mean_candidate_songs': float(frame.faiss_candidate_songs.mean())}
    frame.to_csv(OUTPUT_DIR / (label + '_queries.csv'), index=False)
    return result, frame, query_scores
'''

RUN = '''finetuned_eval_model = ConformerByteHumModel(num_classes=NUM_TRAIN_SONGS, embedding_dim=EMBEDDING_DIM, allow_random_init=True)
finetuned_eval_model.load_state_dict(checkpoint_metadata['model_state_dict'], strict=True)
finetuned_eval_model = finetuned_eval_model.to(device).eval()
if AMP_ENABLED:
    torch.cuda.reset_peak_memory_stats()
local_hash = file_sha256(finetuned_ckpt_path)
ft_gallery, ft_signature = build_gallery(finetuned_eval_model, gallery_paths, local_hash)
res_ft, ft_queries, ft_query_scores = evaluate_model(finetuned_eval_model, ft_gallery, eval_queries, 'local_4gb')
res_ft.update(checkpoint=str(finetuned_ckpt_path), checkpoint_sha256=local_hash,
              epoch=checkpoint_metadata['epoch'], gallery_signature=ft_signature)
results = [res_ft]

def compare_checkpoint(label, path, zero_shot=False):
    # A single model occupies the GPU at a time; cached gallery tensors stay on CPU.
    finetuned_eval_model.to('cpu')
    if AMP_ENABLED:
        torch.cuda.empty_cache()
    if zero_shot:
        comparison_model = ConformerByteHumModel(ckpt_path=str(path), num_classes=NUM_TRAIN_SONGS, embedding_dim=128)
        epoch = None
    else:
        payload = load_compatible_checkpoint(path)
        comparison_model = ConformerByteHumModel(num_classes=NUM_TRAIN_SONGS, embedding_dim=128, allow_random_init=True)
        comparison_model.load_state_dict(payload['model_state_dict'], strict=True)
        epoch = payload['epoch']
        del payload
    comparison_model = comparison_model.to(device).eval()
    digest = file_sha256(path)
    gallery, signature = build_gallery(comparison_model, gallery_paths, digest)
    result, _, _ = evaluate_model(comparison_model, gallery, eval_queries, label)
    result.update(checkpoint=str(path), checkpoint_sha256=digest, epoch=epoch, gallery_signature=signature)
    del comparison_model, gallery
    gc.collect()
    if AMP_ENABLED:
        torch.cuda.empty_cache()
    finetuned_eval_model.to(device).eval()
    return result

if COMPARE_ZERO_SHOT:
    results.append(compare_checkpoint('zero_shot', STAGE1_CKPT, zero_shot=True))
if COMPARE_KAGGLE:
    kaggle_path = STAGE2_ROOT / 'checkpoints' / 'conformer_bytehum_full_d128_timegrid_lat_v4_best.pth'
    if not kaggle_path.is_file():
        raise FileNotFoundError('Requested Kaggle comparison checkpoint not found: ' + str(kaggle_path))
    results.append(compare_checkpoint('kaggle_v4', kaggle_path))
'''

REPORT = '''report = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
    'protocol': {'gallery_mode': GALLERY_MODE, 'reference_source': 'split_reference' if GALLERY_MODE == 'test' else 'full_music_or_manifest',
                 'total_test_queries': len(all_test_queries), 'evaluated_queries': len(eval_queries),
                 'subset': len(eval_queries) != len(all_test_queries), 'all_audio_chunks': True,
                 'local_scoring': 'query-directed MaxMean over all local frames',
                 'global_scoring': 'maximum cosine over all query/reference chunks',
                 'fusion': '0.5 local + 0.5 global', 'ties': 'average rank; expected Recall@K'},
    'runtime': {'torch': torch.__version__, 'device': str(device), 'amp': AMP_ENABLED,
                'eval_batch_size': EVAL_BATCH_SIZE,
                'peak_tensor_allocated_mib': torch.cuda.max_memory_allocated() / 1024**2 if AMP_ENABLED else None},
    'training_run_config': checkpoint_metadata['run_config'], 'results': results}
with open(OUTPUT_DIR / 'evaluation_metrics.json', 'w', encoding='utf-8') as stream:
    json.dump(report, stream, indent=2, allow_nan=False)
summary_rows = []
for result in results:
    for mode, metrics in result['modes'].items():
        summary_rows.append({'model': result['label'], 'scoring': mode, 'queries': result['queries'],
                             'gallery': result['gallery_songs'], 'MRR': metrics['mrr'],
                             **{f'Recall@{k} (%)': 100 * metrics[f'recall_at_{k}'] for k in (1, 5, 10, 20)}})
summary = pd.DataFrame(summary_rows)
summary.to_csv(OUTPUT_DIR / 'evaluation_summary.csv', index=False)
display(summary.round(4))
fig, ax = plt.subplots(figsize=(10, 5))
for result in results:
    ax.plot(range(1, 21), np.array(result['modes']['fused']['cmc']) * 100, label=result['label'])
ax.set(xlabel='Rank K', ylabel='Recall@K (%)', title=f'Fused retrieval - {GALLERY_MODE} gallery ({len(gallery_paths)} songs)')
ax.grid(alpha=.25)
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'cmc_curve.png', dpi=180)
plt.show()
errors = ft_queries[ft_queries.fused_rank > 1].sort_values('fused_rank', ascending=False)
errors.to_csv(OUTPUT_DIR / 'local_4gb_errors.csv', index=False)
display(errors[['song', 'query', 'fused_rank', 'fused_top1_song']].head(10))
print('Saved:', OUTPUT_DIR)
print('Latency separates CQT/I/O from synchronized model/scoring; compare warm cache states separately.')
'''

DTW = '''DTW_RAM_CACHE = OrderedDict()

def variable_cqt(path):
    key = str(Path(path).resolve())
    if key in DTW_RAM_CACHE:
        DTW_RAM_CACHE.move_to_end(key)
        return DTW_RAM_CACHE[key]
    value = cqt_ext.extract(prep.load_audio(key), variable_length=True).squeeze(0).numpy()
    if value.shape[1] > DTW_MAX_FRAMES:
        raise ValueError(f'DTW audio exceeds DTW_MAX_FRAMES: {key}; raise the limit explicitly.')
    DTW_RAM_CACHE[key] = value
    while len(DTW_RAM_CACHE) > 16:
        DTW_RAM_CACHE.popitem(last=False)
    return value

def rerank_with_dtw(path, scores, candidates, top_candidates=DTW_CANDIDATES, weight=DTW_WEIGHT):
    if top_candidates < 1 or not 0 <= weight <= 1:
        raise ValueError('Invalid DTW candidate count or weight.')
    songs = list(ft_gallery)
    ordered = sorted(candidates, key=lambda i: (-float(scores[i]), songs[i]))
    top = ordered[:top_candidates]
    query_cqt = variable_cqt(path)
    adjusted = scores.clone()
    for i in top:
        distance = min(compute_cqt_dtw_distance(query_cqt, variable_cqt(ref))
                       for ref in gallery_paths[songs[i]])
        adjusted[i] = (1 - weight) * scores[i] + weight * max(0., 1 - distance)
    # Preserve the two-stage gate: only candidates in the rerank set receive a rank.
    gated = torch.full_like(scores, float('-inf'))
    gated[top] = adjusted[top]
    return gated, top

if RUN_DTW_BENCHMARK:
    dtw_queries = eval_queries if DTW_MAX_QUERIES is None else eval_queries[:DTW_MAX_QUERIES]
    songs = list(ft_gallery)
    rows = []
    for song, fragment, path in tqdm(dtw_queries, desc='Optional subsequence DTW'):
        adjusted, top = rerank_with_dtw(path, ft_query_scores[path], range(len(songs)))
        rank, ties, greater = tie_statistics(adjusted.numpy(), songs.index(song))
        rows.append({'query': path, 'song': song, 'dtw_rank': rank, 'dtw_ties': ties,
                     'dtw_greater': greater, 'candidate_hit': songs.index(song) in top})
    dtw_frame = pd.DataFrame(rows)
    dtw_result = dict(summarize_rows(dtw_frame, 'dtw'), candidate_recall=float(dtw_frame.candidate_hit.mean()),
                      candidates=DTW_CANDIDATES, weight=DTW_WEIGHT,
                      subset=len(dtw_queries) != len(all_test_queries),
                      algorithm='pitch-aligned subsequence DTW; no Sakoe-Chiba band')
    with open(OUTPUT_DIR / 'dtw_metrics.json', 'w', encoding='utf-8') as stream:
        json.dump(dtw_result, stream, indent=2, allow_nan=False)
    dtw_frame.to_csv(OUTPUT_DIR / 'dtw_queries.csv', index=False)
    display(pd.DataFrame([dtw_result]).drop(columns=['cmc']))
else:
    print('DTW benchmark disabled. Enable RUN_DTW_BENCHMARK before Run All to measure it.')
'''

SEARCH = '''db_index, db_chunk_map = build_faiss_gallery(ft_gallery)
if db_index is not None:
    faiss.write_index(db_index, str(OUTPUT_DIR / 'gallery.index'))
    with open(OUTPUT_DIR / 'gallery_index_metadata.json', 'w', encoding='utf-8') as stream:
        json.dump({'checkpoint_sha256': local_hash, 'gallery_signature': ft_signature,
                   'songs': list(ft_gallery), 'chunk_to_song': db_chunk_map.tolist(),
                   'gallery_mode': GALLERY_MODE, 'kind': 'flat'}, stream, indent=2)

@torch.inference_mode()
def search_humming(audio_path, top_k=5, use_faiss=False, faiss_k=FAISS_CHUNK_K,
                   enable_dtw=False, dtw_candidates=DTW_CANDIDATES, dtw_weight=DTW_WEIGHT):
    if top_k < 1:
        raise ValueError('top_k must be positive.')
    if use_faiss and db_index is None:
        raise ImportError('FAISS is unavailable; use exact search or install faiss-cpu.')
    path = str(Path(audio_path).resolve())
    if not Path(path).is_file():
        raise FileNotFoundError(path)
    synchronize()
    start = time.perf_counter()
    cqts = test_dataset._get_raw_cqts(path)
    qg, ql = encode_cqts(finetuned_eval_model, cqts)
    scores = score_gallery(qg, ql, ft_gallery)
    songs = list(ft_gallery)
    candidates = candidate_ids(qg, db_index, db_chunk_map, faiss_k) if use_faiss else set(range(len(songs)))
    rank_scores = scores['fused']
    if enable_dtw:
        rank_scores, selected = rerank_with_dtw(path, rank_scores, candidates, dtw_candidates, dtw_weight)
        candidates = set(selected)
    ordered = sorted(candidates, key=lambda i: (-float(rank_scores[i]), songs[i]))[:top_k]
    synchronize()
    return {'query': path, 'gallery_mode': GALLERY_MODE, 'candidate_songs': len(candidates),
            'faiss': use_faiss, 'dtw': enable_dtw, 'open_set_calibrated': False,
            'latency_ms_including_cqt_io': 1000 * (time.perf_counter() - start),
            'results': [{'rank': rank, 'song': songs[i], 'score': float(rank_scores[i]),
                         'local_score': float(scores['local'][i]), 'global_score': float(scores['global'][i])}
                        for rank, i in enumerate(ordered, 1)]}

demo_path = QUERY_AUDIO_PATH or eval_queries[0][2]
demo_result = search_humming(demo_path, top_k=5)
display(pd.DataFrame(demo_result['results']))
print('Demo query:', demo_path, '| End-to-end ms:', round(demo_result['latency_ms_including_cqt_io'], 2))
print('Closed-set ranking: no calibrated unknown-song rejection threshold is applied.')
'''


def cell(kind, text):
    result = {'cell_type': kind, 'metadata': {}, 'source': text.splitlines(True)}
    if kind == 'code':
        result.update(execution_count=None, outputs=[])
    return result


def build():
    kaggle = json.loads((ROOT / 'kaggle' / 'archive' / 'conformer_bytehum_qbh_evaluation_kaggle.ipynb').read_text(encoding='utf-8'))
    local = json.loads((ROOT / 'local_4gb' / 'train' / 'conformer_bytehum_qbh_local_4gb.ipynb').read_text(encoding='utf-8'))
    cells = [cell('markdown', '''# Conformer-ByteHum: Local Evaluation (GPU 4 GB)

Standalone evaluation derived from the Kaggle evaluator. No training is run.
Default: local best checkpoint, 774 test queries and all 300 original recordings,
with zero-shot and Kaggle-checkpoint comparisons. The 45-song split-reference
protocol remains available with GALLERY_MODE='test' in a fresh kernel.

Run All in your existing CUDA PyTorch kernel. Dependencies: torch, numpy, pandas,
matplotlib, tqdm, librosa, soundfile, scipy, IPython, typeguard<3; faiss-cpu is
optional for candidate gating, numba is optional for faster DTW. Packages are
not installed automatically. CPU evaluation is also supported.

Features and galleries remain on CPU; inference batch size is 4. CQT uses the
training notebook's bounded RAM/disk cache. Gallery features are cached by
checkpoint hash, audio paths/stat information, preprocessing and runtime.
Errors and missing ground truths fail explicitly. All chunks are included.

Reports separate exact local/global/fused metrics from FAISS-gated metrics.
Candidate recall counts all queries, including gate misses. Ties use average
rank and expected Recall@K. CQT/I/O latency is separated from model/scoring.
No open-set threshold or public-benchmark result is assumed. Choose settings
using validation, not test. Optional DTW measures a separately labeled subset.
'''), cell('code', CONFIG), cell('code', IMPORTS), cell('code', PATHS)]
    for i in (9, 11, 13, 15, 17, 19, 21):
        cells.append(cell('code', ''.join(kaggle['cells'][i]['source'])))
    dataset = next(''.join(c['source']) for c in local['cells']
                   if 'class BoundedCQTCache' in ''.join(c['source']))
    cells += [cell('markdown', '## Bounded audio/CQT cache\n'), cell('code', dataset)]
    for i in (25, 27, 29, 31):
        cells.append(cell('code', ''.join(kaggle['cells'][i]['source'])))
    cells += [cell('markdown', '## Checkpoint, splits and gallery protocol\n'), cell('code', SETUP),
              cell('markdown', '## Exact retrieval and FAISS candidate recall\n'), cell('code', ENGINE),
              cell('code', RUN), cell('markdown', '## Metrics, CMC and query errors\n'), cell('code', REPORT),
              cell('markdown', '## Optional pitch-aligned subsequence DTW\n'),
              cell('code', ''.join(kaggle['cells'][48]['source'])), cell('code', DTW),
              cell('markdown', '## Search a local humming recording\n'), cell('code', SEARCH)]
    for i, item in enumerate(cells):
        item['id'] = f'local-eval-{i:03d}'
        if item['cell_type'] == 'code':
            compile(''.join(item['source']), f'{TARGET.name}:cell{i}', 'exec')
    doc = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3 (CUDA PyTorch)', 'language': 'python', 'name': 'python3'},
                                     'language_info': {'name': 'python', 'version': '3.11'}}, 'nbformat': 4, 'nbformat_minor': 5}
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print(TARGET)


if __name__ == '__main__':
    build()
