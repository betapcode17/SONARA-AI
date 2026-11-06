from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_NOTEBOOK = ROOT / "notebooks/kaggle/evaluate/conformer_bytehum_qbh_final_evaluation.ipynb"
OUTPUT = ROOT / "notebooks/kaggle/evaluate/conformer_bytehum_qbh_v3_vs_v5_300song.ipynb"


def cell(kind: str, source: str) -> dict:
    return {
        "cell_type": kind,
        "metadata": {},
        "source": source.splitlines(keepends=True),
        **({"execution_count": None, "outputs": []} if kind == "code" else {}),
    }


source_nb = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
# These archived cells contain the exact Conformer backbone, audio preprocessing,
# CoverHunter CQT, ByteHum heads, and model forward code used by the final V5 evaluator.
model_cells = [source_nb["cells"][i] for i in (4, 5, 6, 7, 8)]
for model_cell in model_cells:
    if "class PyCqt" in "".join(model_cell.get("source", [])):
        _source = "".join(model_cell["source"])
        _source = _source.replace("        self._hop_size = hop_size\n", "        self._hop_size = hop_size\n        self.floor_db = -80.0\n")
        _source = _source.replace(
            "        cqt_gpu = (20.0 * torch.log10(magnitude) - 20.0 * torch.log10(ref_value)).clamp(-80.0, 0.0)",
            "        cqt_gpu = 20.0 * torch.log10(magnitude) - 20.0 * torch.log10(ref_value)\n"
            "        cqt_gpu = cqt_gpu.clamp(min=self.floor_db, max=0.0) if self.floor_db is not None else cqt_gpu.clamp(max=0.0)",
        )
        model_cell["source"] = _source.splitlines(keepends=True)

intro = """# So sánh checkpoint QbH V3 và V5 trên gallery 300 bài

Notebook này đánh giá **hai checkpoint trên cùng một giao thức**: gallery gồm đúng 300 bản nhạc gốc; query lấy từ test split V3, chỉ giữ các query có audio và ánh xạ nhãn chắc chắn; nhãn và thứ hạng được tính ở cấp bài hát. Mỗi checkpoint dùng cấu hình tiền xử lý được lưu trong contract của checkpoint. Scorer cố định: local MaxMean + global max, fused = 0.5 local + 0.5 global.

## Chuẩn bị Kaggle

1. Bật GPU (T4/P100 đều được).
2. Gắn dataset chứa `dataset_split/` và `2_music_database/` (300 file MP3).
3. Gắn `results.zip` của V3 và `qbh_final_v5_train_outputs.zip` của V5 làm Kaggle inputs.
4. Chạy Run All. Nếu Kaggle đặt tên file khác, sửa ba biến đường dẫn ở cell cấu hình.

Notebook chỉ đánh giá, không train. Nó dùng các query test V3 có sẵn trong dữ liệu; query thiếu audio hoặc chưa ánh xạ chắc chắn tới tên bài trong gallery sẽ bị loại và ghi số lượng vào JSON. Bài chưa map vẫn được giữ làm distractor. Nếu test song bị dùng trong train V5, notebook sẽ dừng. Đây là closed-set retrieval, không phải đánh giá open-set hay DTW.
"""

config = r'''# Chỉ sửa các biến đường dẫn này nếu Kaggle không tự tìm thấy input.
INPUT_ROOT = Path('/kaggle/input')
OUTPUT_ROOT = Path('/kaggle/working/qbh_v3_v5_300song')
V3_ZIP_OVERRIDE = None
V5_ZIP_OVERRIDE = None
DATASET_ROOT_OVERRIDE = None  # Thư mục cha có dataset_split/ và 2_music_database/.
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type != 'cuda':
    raise RuntimeError('Hãy bật GPU trong Kaggle Settings rồi chạy lại.')
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
print('GPU:', torch.cuda.get_device_name(0), '| Torch:', torch.__version__)
'''

setup = r'''import os, sys, json, csv, io, re, math, time, hashlib, zipfile, shutil, unicodedata, subprocess, gc
from pathlib import Path, PurePosixPath
from collections import defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse
import soundfile as sf
import librosa
from tqdm.auto import tqdm
from IPython.display import display
try:
    import typeguard
    if not hasattr(typeguard, 'check_argument_types'):
        raise ImportError
except ImportError:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'typeguard==2.13.3'])
    for _name in list(sys.modules):
        if _name == 'typeguard' or _name.startswith('typeguard.'):
            del sys.modules[_name]
'''

inputs = r'''def unique_file(override, pattern, description):
    if override:
        p = Path(override).resolve()
        if not p.is_file(): raise FileNotFoundError(p)
        return p
    found = sorted({p.resolve() for p in INPUT_ROOT.rglob(pattern) if p.is_file()})
    if len(found) != 1:
        raise RuntimeError(f'{description}: cần đúng một file, tìm thấy {len(found)}; đặt override. {found[:8]}')
    return found[0]

def locate_member(zf, suffix):
    hits = [n for n in zf.namelist() if n.replace('\\', '/').endswith(suffix)]
    if len(hits) != 1: raise RuntimeError(f'ZIP cần đúng một member {suffix}; thấy {len(hits)}')
    return hits[0]

def extract_member(zpath, suffix, dest):
    dest = Path(dest); dest.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath) as zf:
        member = locate_member(zf, suffix)
        with zf.open(member) as src, dest.open('wb') as out: shutil.copyfileobj(src, out)
    return dest

V3_ZIP = unique_file(V3_ZIP_OVERRIDE, 'results.zip', 'V3 results.zip')
V5_ZIP = unique_file(V5_ZIP_OVERRIDE, 'qbh_final_v5_train_outputs.zip', 'V5 train outputs ZIP')
DATASET_ROOTS = []
if DATASET_ROOT_OVERRIDE:
    DATASET_ROOTS = [Path(DATASET_ROOT_OVERRIDE).resolve()]
else:
    DATASET_ROOTS = sorted({p.parent.resolve() for p in INPUT_ROOT.rglob('dataset_split') if p.is_dir()})
    DATASET_ROOTS += [p.resolve() for p in INPUT_ROOT.rglob('*') if p.is_dir() and
                      (p / 'dataset_split').is_dir() and (p / '2_music_database').is_dir()]
    DATASET_ROOTS = list(dict.fromkeys(DATASET_ROOTS))
if len(DATASET_ROOTS) != 1:
    raise RuntimeError(f'Không xác định duy nhất dataset root; tìm thấy {len(DATASET_ROOTS)}. Đặt DATASET_ROOT_OVERRIDE.')
DATA_ROOT = DATASET_ROOTS[0]
SPLIT_ROOT = DATA_ROOT / 'dataset_split'
MUSIC_ROOT = DATA_ROOT / '2_music_database'
if not SPLIT_ROOT.is_dir() or not MUSIC_ROOT.is_dir(): raise FileNotFoundError('Dataset cần dataset_split/ và 2_music_database/.')
V3_DIR = OUTPUT_ROOT / 'inputs' / 'v3'; V5_DIR = OUTPUT_ROOT / 'inputs' / 'v5'
V3_CKPT = extract_member(V3_ZIP, 'checkpoints/best.pth', V3_DIR / 'best.pth')
V3_MANIFEST = extract_member(V3_ZIP, 'manifest.csv', V3_DIR / 'manifest.csv')
V3_CONTRACT = json.loads(extract_member(V3_ZIP, 'checkpoint_contract.json', V3_DIR / 'checkpoint_contract.json').read_text(encoding='utf-8'))
V5_CKPT = extract_member(V5_ZIP, 'qbh_final_v5_best.pth', V5_DIR / 'best.pth')
print('V3:', V3_ZIP.name, '| V5:', V5_ZIP.name, '| data:', DATA_ROOT)
print('Gallery MP3:', len(list(MUSIC_ROOT.glob('*.mp3'))))
'''

data_code = r'''def norm_name(s):
    s = unicodedata.normalize('NFKD', str(s)).casefold().strip()
    s = ''.join(c for c in s if not unicodedata.combining(c))
    s = s.replace('&', 'and').replace('+', 'plus')
    return ''.join(c for c in s if c.isalnum())

mp3_by_name = defaultdict(list)
for p in MUSIC_ROOT.glob('*.mp3'): mp3_by_name[norm_name(p.stem)].append(p)
if len(list(MUSIC_ROOT.glob('*.mp3'))) != 300: raise RuntimeError('Gallery phải gồm đúng 300 file MP3.')
mapping_path = DATA_ROOT / 'song_name_mapping.json'
if not mapping_path.is_file():
    _mapping_candidates = list(INPUT_ROOT.rglob('song_name_mapping.json'))
    if len(_mapping_candidates) == 1: mapping_path = _mapping_candidates[0]
song_alias = {}
if mapping_path.is_file():
    _mapping = json.loads(mapping_path.read_text(encoding='utf-8'))
    for _source_name, _file_name in _mapping.items():
        for _alias in (_source_name, _file_name):
            _key = norm_name(_alias)
            if _key in song_alias and song_alias[_key] != _file_name: song_alias[_key] = None
            else: song_alias[_key] = _file_name

def map_song_name(name):
    _alias = song_alias.get(norm_name(name))
    _keys = [norm_name(_alias), norm_name(name)] if _alias else [norm_name(name)]
    _hits = {p for key in _keys for p in mp3_by_name.get(key, [])}
    return next(iter(_hits)) if len(_hits) == 1 else None

v3_manifest = pd.read_csv(V3_MANIFEST, dtype=str, keep_default_na=False)
paired_test = v3_manifest[(v3_manifest['split'] == 'test') & (v3_manifest['has_reference'].str.lower() == 'true')].copy()
if paired_test.empty: raise RuntimeError('V3 manifest không có paired test queries.')
gallery_paths = {p.stem: p for p in sorted(MUSIC_ROOT.glob('*.mp3'))}

# Index humming files by immutable audio_id; only rows from the locked V3 test split are used.
audio_index = {}
for p in SPLIT_ROOT.rglob('*.wav'):
    if p.stem in audio_index: raise RuntimeError(f'ID humming bị trùng: {p.stem}')
    audio_index[p.stem] = p
queries, missing_ids, unmapped_target_ids = [], [], set()
for row in paired_test.to_dict('records'):
    audio_id, sid = row['audio_id'], row['song_id']
    p = audio_index.get(audio_id)
    if p is None:
        missing_ids.append(audio_id); continue
    # Ensure the file belongs to the held-out test folder.
    if 'test' not in [part.casefold() for part in p.parts]:
        raise RuntimeError(f'Query {audio_id} không nằm trong thư mục test: {p}')
    target_path = map_song_name(row['folder_name'])
    if target_path is None:
        unmapped_target_ids.add(sid); continue
    queries.append({'key': audio_id, 'song': target_path.stem, 'song_id': sid, 'path': p})
if len(queries) < 700: raise RuntimeError(f'Chỉ còn {len(queries)} query; yêu cầu tối thiểu 700 để comparison có ý nghĩa.')
if len({q['song'] for q in queries}) < 40: raise RuntimeError(f'Phải còn ít nhất 40 test target songs; thấy {len({q["song"] for q in queries})}')

v3_state = torch.load(V3_CKPT, map_location='cpu', weights_only=True)
v5_state = torch.load(V5_CKPT, map_location='cpu', weights_only=True)
if not v3_state.get('model_state_dict') or not v5_state.get('model_state_dict'):
    raise RuntimeError('Checkpoint thiếu model_state_dict.')
v3_train = set((V3_CONTRACT.get('train_song_to_idx') or {}).keys())
v5_train = set(map(str, v5_state.get('train_songs', [])))
if not v3_train or not v5_train: raise RuntimeError('Checkpoint thiếu train song IDs; không thể kiểm tra test leakage.')
query_song_ids = {q['song_id'] for q in queries}
overlap_v3 = sorted(query_song_ids & v3_train)
overlap_v5 = sorted(query_song_ids & v5_train)
if overlap_v3 or overlap_v5:
    raise RuntimeError(f'Leakage: query test song xuất hiện trong train songs. V3={len(overlap_v3)}, V5={len(overlap_v5)}. Không dùng tập này để kết luận tổng quát hóa.')
print(f'Locked test: {len(paired_test)} paired rows; {len(queries)} query dùng được; thiếu audio={len(missing_ids)}; target chưa map={len(unmapped_target_ids)}; target songs={len({q["song"] for q in queries})}.')
print('Gallery: đúng 300 bài | test query song overlap train: 0')
'''

evaluation = r'''# Các hàm inference/retrieval dùng chung scorer cho cả hai checkpoint.
audio_pre = AudioPreprocessor(sample_rate=16000, chunk_duration=10.0, hop_duration=3.0)
cqt = CQTExtractor(sample_rate=16000, n_bins=96, bins_per_octave=12,
                   hop_size=0.04, mean_size=3, target_frames=83)

def model_from_checkpoint(state, name):
    weights = state['model_state_dict']
    proto = weights.get('lac.prototypes')
    if proto is None or proto.ndim != 3:
        raise RuntimeError(f'{name}: không tìm thấy LAC prototypes [songs,4,128].')
    model = ConformerByteHumModel(ckpt_path=None, num_classes=int(proto.shape[0]),
                                  embedding_dim=128, allow_random_init=True)
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if missing or unexpected:
        raise RuntimeError(f'{name}: checkpoint/model không tương thích; missing={missing[:8]}, unexpected={unexpected[:8]}')
    model = model.to(DEVICE).eval()
    return model

def file_cqts(path, floor_db):
    cqt.cqt_impl.py_cqt.floor_db = floor_db
    y = audio_pre.load_audio(str(path))
    chunks = audio_pre.slice_into_chunks(y)
    out = []
    for chunk in chunks:
        x = cqt.extract(chunk)
        if floor_db is not None: x = torch.clamp(x, min=float(floor_db))
        out.append(x.squeeze(0))
    return torch.stack(out)

@torch.inference_mode()
def encode_file(model, path, floor_db, batch_size=32):
    spec = file_cqts(path, floor_db)
    gs, ls = [], []
    for start in range(0, len(spec), batch_size):
        x = spec[start:start+batch_size].unsqueeze(1).to(DEVICE)
        with torch.autocast('cuda', dtype=torch.float16):
            g, l, _, _ = model(x, compute_lac=False, compute_domain=False)
        gs.append(F.normalize(g.float(), dim=-1).cpu())
        ls.append(F.normalize(l.float(), dim=-1).reshape(-1, 128).cpu())
    g, l = torch.cat(gs), torch.cat(ls)
    if not torch.isfinite(g).all() or not torch.isfinite(l).all(): raise RuntimeError(f'Non-finite features: {path}')
    return g, l

@torch.inference_mode()
def encode_gallery(model, floor_db, tag):
    result = {}
    for sid, path in tqdm(gallery_paths.items(), desc=f'{tag}: encode 300-song gallery'):
        result[sid] = encode_file(model, path, floor_db)
    return result

@torch.inference_mode()
def scores_for_query(model, gallery, query, floor_db):
    qg, ql = encode_file(model, query['path'], floor_db)
    qg, ql = qg.to(DEVICE), ql.to(DEVICE)
    songs = list(gallery)
    local_values, global_values = [], []
    # Batched song blocks avoid hundreds of tiny CUDA launches per query.
    for start in range(0, len(songs), 16):
        group = songs[start:start+16]
        gseq = [gallery[s][0] for s in group]
        lseq = [gallery[s][1] for s in group]
        gp = nn.utils.rnn.pad_sequence(gseq, batch_first=True).to(DEVICE)
        gmask = torch.arange(gp.shape[1], device=DEVICE)[None, :] < torch.tensor([len(x) for x in gseq], device=DEVICE)[:, None]
        gsim = torch.einsum('qd,scd->sqc', qg, gp).masked_fill(~gmask[:, None, :], float('-inf'))
        global_values.append(gsim.amax(dim=(1,2)).cpu())
        lp = nn.utils.rnn.pad_sequence(lseq, batch_first=True).to(DEVICE)
        lmask = torch.arange(lp.shape[1], device=DEVICE)[None, :] < torch.tensor([len(x) for x in lseq], device=DEVICE)[:, None]
        lsim = torch.einsum('qd,sld->sql', ql, lp).masked_fill(~lmask[:, None, :], float('-inf'))
        local_values.append(lsim.amax(dim=-1).mean(dim=-1).cpu())
    local = dict(zip(songs, torch.cat(local_values).tolist()))
    global_ = dict(zip(songs, torch.cat(global_values).tolist()))
    fused = {sid: 0.5 * (local[sid] + global_[sid]) for sid in gallery}
    return {'local': local, 'global': global_, 'fused': fused}

def expected_rank_metrics(score_map, target):
    vals = np.asarray(list(score_map.values()), dtype=np.float64)
    ids = list(score_map)
    t = vals[ids.index(target)]
    tol = np.maximum(1e-7, 1e-6 * np.maximum(np.abs(vals), abs(t)))
    ahead = int(np.sum(vals > t + tol))
    tied = max(1, int(np.sum(np.abs(vals - t) <= tol)))
    rr = sum(1.0 / r for r in range(ahead + 1, ahead + tied + 1)) / tied
    return {'rr': float(rr), **{f'hit@{k}': float(min(max(k-ahead, 0), tied) / tied) for k in (1,5,10,20)}}

def evaluate_model(state, name, floor_db):
    model = model_from_checkpoint(state, name)
    gallery = encode_gallery(model, floor_db, name)
    rows = []
    for q in tqdm(queries, desc=f'{name}: score queries'):
        score_sets = scores_for_query(model, gallery, q, floor_db)
        row = {'audio_id': q['key'], 'song_id': q['song']}
        for mode, smap in score_sets.items():
            row.update({f'{mode}_{k}': v for k, v in expected_rank_metrics(smap, q['song']).items()})
        rows.append(row)
    del model, gallery
    gc.collect(); torch.cuda.empty_cache()
    return pd.DataFrame(rows)

# V3 contract has no dB floor; V5 run contract specifies -80 dB.
v5_contract = v5_state.get('run_contract', {})
if not v5_state.get('finished', False): raise RuntimeError('V5 BEST checkpoint is not finalized.')
if v5_state.get('checkpoint_schema') != 2: raise RuntimeError('V5 checkpoint schema is not the expected schema 2.')
v5_floor = v5_contract.get('preprocessing', {}).get('floor_db', -80.0)
v3_floor = V3_CONTRACT.get('preprocessing', {}).get('floor_db')
print('Preprocessing floor dB | V3:', v3_floor, '| V5:', v5_floor)
'''

run = r'''started = time.perf_counter()
v3_rows = evaluate_model(v3_state, 'V3', v3_floor)
v5_rows = evaluate_model(v5_state, 'V5', v5_floor)
if v3_rows['audio_id'].tolist() != v5_rows['audio_id'].tolist(): raise RuntimeError('Query lists differ between models.')

def summarize(frame):
    out = {}
    for mode in ('local', 'global', 'fused'):
        out[mode] = {
            'query_mrr': float(frame[f'{mode}_rr'].mean()),
            'macro_song_mrr': float(frame.groupby('song_id')[f'{mode}_rr'].mean().mean()),
            **{f'hit@{k}': float(frame[f'{mode}_hit@{k}'].mean()) for k in (1,5,10,20)},
        }
    return out

v3_metrics, v5_metrics = summarize(v3_rows), summarize(v5_rows)
comparison = v3_rows.merge(v5_rows, on=['audio_id','song_id'], suffixes=('_v3','_v5'), validate='one_to_one')
for metric in ['fused_rr','fused_hit@1','fused_hit@5','fused_hit@10','fused_hit@20']:
    comparison[f'delta_{metric}'] = comparison[f'{metric}_v5'] - comparison[f'{metric}_v3']

# Song-cluster bootstrap CI for the paired macro-song MRR delta.
song_scores = comparison.groupby('song_id')[['fused_rr_v3', 'fused_rr_v5']].mean()
song_delta = (song_scores['fused_rr_v5'] - song_scores['fused_rr_v3']).to_numpy()
rng = np.random.default_rng(42)
boot = np.asarray([rng.choice(song_delta, size=len(song_delta), replace=True).mean() for _ in range(2000)])
report = {
  'protocol': 'paired V3 test queries x exact 300 original-song gallery; local MaxMean + global max; fused 0.5/0.5',
  'gallery_songs': len(gallery_paths), 'manifest_paired_test_queries': len(paired_test),
  'evaluated_common_queries': len(queries), 'missing_query_files': len(missing_ids),
  'test_songs': len({q['song'] for q in queries}), 'unmapped_target_songs': len(unmapped_target_ids),
  'v3_checkpoint': str(V3_ZIP.name), 'v5_checkpoint_archive': str(V5_ZIP.name),
  'v3_preprocessing': V3_CONTRACT.get('preprocessing'), 'v5_preprocessing': v5_contract.get('preprocessing'),
  'metrics': {'V3': v3_metrics, 'V5': v5_metrics,
              'delta_v5_minus_v3': {m: v5_metrics['fused'][m]-v3_metrics['fused'][m] for m in v3_metrics['fused']}},
  'paired_macro_song_mrr_delta_v5_minus_v3_ci95': np.quantile(boot, [0.025,0.975]).tolist(),
  'overlap_test_with_v3_train_songs': len(overlap_v3), 'overlap_test_with_v5_train_songs': len(overlap_v5),
  'elapsed_seconds': time.perf_counter()-started,
  'caveat': 'Different checkpoint training runs; result compares complete checkpoints on this common test protocol, not a controlled one-module ablation.'
}
v3_rows.to_csv(OUTPUT_ROOT / 'v3_query_metrics.csv', index=False)
v5_rows.to_csv(OUTPUT_ROOT / 'v5_query_metrics.csv', index=False)
comparison.to_csv(OUTPUT_ROOT / 'paired_query_comparison.csv', index=False)
(OUTPUT_ROOT / 'comparison_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
display(pd.DataFrame({'V3 fused': v3_metrics['fused'], 'V5 fused': v5_metrics['fused'],
                      'Delta': report['metrics']['delta_v5_minus_v3']}))
print('Paired macro-song MRR delta 95% CI:', report['paired_macro_song_mrr_delta_v5_minus_v3_ci95'])
print('Saved artifacts to:', OUTPUT_ROOT)
'''

cells = [cell("markdown", intro), cell("code", setup + "\n\n" + config), cell("code", inputs), cell("code", data_code)]
for source_cell in model_cells:
    cells.append({**source_cell, "execution_count": None, "outputs": []} if source_cell["cell_type"] == "code" else source_cell)
cells.extend([cell("code", evaluation), cell("code", run)])

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(OUTPUT)
