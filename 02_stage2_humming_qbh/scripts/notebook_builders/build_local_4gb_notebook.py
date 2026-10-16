"""Derive the local 4 GB notebook from the self-contained Kaggle source."""
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / 'notebooks'
SOURCE = ROOT / 'kaggle' / 'archive' / "conformer_bytehum_qbh_kaggle.ipynb"
TARGET = ROOT / 'local_4gb' / 'train' / "conformer_bytehum_qbh_local_4gb.ipynb"


def cell(kind, source):
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


CONFIG = '''from pathlib import Path
import os

# Edit these settings before Run All. Resume is from a completed epoch.
PROJECT_ROOT_OVERRIDE = None  # Example: Path(r"E:\\HOCKY7\\PBL6\\code")
LOCAL_NUM_EPOCHS = 30
LOCAL_SONGS_PER_BATCH = 2
LOCAL_SAMPLES_PER_SONG = 2
LOCAL_ACCUMULATION_STEPS = 16
LOCAL_EVAL_BATCH_SIZE = 4
LOCAL_QUERY_BATCH_SIZE = 4
LOCAL_GALLERY_BLOCK_SIZE = 8
LOCAL_FREEZE_BLOCKS = 3
LOCAL_RAM_CACHE_MB = 256
LOCAL_RESUME = True
LOCAL_EVALUATE_ONLY = False
LOCAL_COMPARE_KAGGLE = True

if PROJECT_ROOT_OVERRIDE is None:
    roots = [Path.cwd(), *Path.cwd().parents]
    PROJECT_ROOT = next((root for root in roots
                         if (root / '02_stage2_humming_qbh').is_dir()), None)
    if PROJECT_ROOT is None:
        raise FileNotFoundError('Set PROJECT_ROOT_OVERRIDE to the project root.')
else:
    PROJECT_ROOT = Path(PROJECT_ROOT_OVERRIDE).resolve()
STAGE2_ROOT = PROJECT_ROOT / '02_stage2_humming_qbh'
NUMBA_CACHE_DIR = STAGE2_ROOT / 'cache' / 'numba'
NUMBA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ['NUMBA_CACHE_DIR'] = str(NUMBA_CACHE_DIR)
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ['QBH_PREWARM_CQT'] = '0'
os.environ['QBH_VALIDATION_GALLERY'] = 'val'
os.environ['QBH_RESUME_TRAINING'] = '1' if LOCAL_RESUME else '0'
if min(LOCAL_SONGS_PER_BATCH, LOCAL_SAMPLES_PER_SONG, LOCAL_ACCUMULATION_STEPS,
       LOCAL_EVAL_BATCH_SIZE, LOCAL_QUERY_BATCH_SIZE, LOCAL_GALLERY_BLOCK_SIZE) < 1:
    raise ValueError('Batch and accumulation sizes must be positive.')
if LOCAL_SONGS_PER_BATCH < 2:
    raise ValueError('Mining requires at least two songs per microbatch.')
print('Project:', PROJECT_ROOT)
print('Effective optimizer batch:', LOCAL_SONGS_PER_BATCH * LOCAL_SAMPLES_PER_SONG
      * LOCAL_ACCUMULATION_STEPS)
'''

PATHS = '''def find_dataset_and_checkpoint():
    dataset_root = STAGE2_ROOT / 'datasets'
    candidates = [STAGE2_ROOT / 'checkpoints' / 'coverhunter_shs100k_pretrained.pth',
                  PROJECT_ROOT / '01_stage1_cover_song_csi' / 'checkpoints'
                  / 'coverhunter_shs100k_pretrained.pth']
    checkpoint = next((path for path in candidates if path.is_file()), None)
    if checkpoint is None:
        raise FileNotFoundError('Missing Stage-1 checkpoint: ' + str(candidates))
    split = dataset_root / 'dataset_split'
    for name in ('train', 'val', 'test'):
        if not (split / name).is_dir():
            raise FileNotFoundError(str(split / name))
    music = dataset_root / '2_music_database'
    vocals = dataset_root / '4_vocal_separated'
    return str(split), str(music), str(vocals if vocals.is_dir() else music), str(checkpoint)
'''

CACHE = '''from collections import OrderedDict
import hashlib

class BoundedCQTCache(OrderedDict):
    def __init__(self, max_mb):
        super().__init__()
        self.limit = int(max_mb * 1024**2)
        self.bytes_used = 0

    @staticmethod
    def size(value):
        return sum(t.numel() * t.element_size() for t in value)

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        if key in self:
            self.bytes_used -= self.size(super().__getitem__(key))
            super().__delitem__(key)
        size = self.size(value)
        if size > self.limit:
            return
        while self and self.bytes_used + size > self.limit:
            _, old = self.popitem(last=False)
            self.bytes_used -= self.size(old)
        super().__setitem__(key, value)
        self.bytes_used += size

GLOBAL_CQT_RAM_CACHE = BoundedCQTCache(LOCAL_RAM_CACHE_MB)
CQT_CACHE_DIR = STAGE2_ROOT / 'cache' / 'cqt_timegrid_v4_16k_10s_hop3_96x83'
CQT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
'''

DISK_LOOKUP = '''        source = Path(path).resolve()
        stat = source.stat()
        fingerprint = f'{source}|{stat.st_size}|{stat.st_mtime_ns}|v4_16k_10s_hop3_96x83'
        cache_path = CQT_CACHE_DIR / (hashlib.sha256(fingerprint.encode()).hexdigest() + '.pt')
        if cache_path.is_file():
            cqts = torch.load(cache_path, map_location='cpu', weights_only=True)
            if not cqts or any(t.shape != (1, 96, 83) or not torch.isfinite(t).all() for t in cqts):
                raise RuntimeError(f'Invalid CQT cache; remove this file: {cache_path}')
            GLOBAL_CQT_RAM_CACHE[path] = cqts
            return cqts
'''

EVALUATION = '''# Test is evaluated only after model selection on validation.
# Protocol: test-only reference gallery, all chunks, exact local/global scoring;
# no FAISS gating or DTW. Selection MRR is the mean of local/global MRR,
# whereas fused ranking below averages their scores before ranking.
import json
import gc

@torch.no_grad()
def evaluate_test_retrieval(eval_model, dataset, label):
    eval_model.eval()
    gallery = validation_gallery_paths(dataset)
    song_ids = sorted(gallery)
    song_index = {song: i for i, song in enumerate(song_ids)}
    encoded = []
    for song in tqdm(song_ids, desc=label + ' gallery'):
        cqts = [chunk for path in gallery[song] for chunk in dataset._get_raw_cqts(path)]
        if not cqts:
            raise RuntimeError(f'Missing reference: {song}')
        encoded.append(_encode_cqt_list(eval_model, cqts))
    globals_list = [g for g, _ in encoded]
    locals_list = [l.reshape(-1, l.shape[-1]) for _, l in encoded]
    global_padded = nn.utils.rnn.pad_sequence(globals_list, batch_first=True)
    local_padded = nn.utils.rnn.pad_sequence(locals_list, batch_first=True)
    global_mask = torch.arange(global_padded.shape[1])[None, :] < torch.tensor([len(g) for g in globals_list])[:, None]
    local_mask = torch.arange(local_padded.shape[1])[None, :] < torch.tensor([len(l) for l in locals_list])[:, None]
    rows = []
    started = time.perf_counter()
    for song, fragment, path in tqdm(dataset.humming_samples, desc=label + ' queries'):
        g, l = _encode_cqt_list(eval_model, dataset._get_raw_cqts(path))
        local_scores = _vectorized_local_maxmean(l.to(device).reshape(-1, EMBEDDING_DIM), local_padded, local_mask)
        global_scores = _vectorized_global_chunk_scores(g.to(device), global_padded, global_mask)
        modes = {'local': local_scores, 'global': global_scores,
                 'fused': 0.5 * (local_scores + global_scores)}
        row = {'song': song, 'fragment': fragment, 'query': path}
        for mode, scores in modes.items():
            rank, ties = _average_tie_rank(scores, song_index[song])
            row[mode + '_rank'] = rank
            row[mode + '_ties'] = ties
            row[mode + '_top1_song'] = song_ids[int(scores.argmax())]
        rows.append(row)
    elapsed = time.perf_counter() - started
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError('No test queries.')
    result = {'label': label, 'queries': len(frame), 'gallery_songs': len(song_ids),
              'protocol': 'test_only_all_chunks_exact_maxmean_v4_average_ties',
              'query_wall_seconds': elapsed, 'ms_per_query_including_cqt_io': 1000 * elapsed / len(frame)}
    for mode in ('local', 'global', 'fused'):
        ranks = frame[mode + '_rank'].to_numpy()
        result[mode + '_mrr'] = float(np.mean(1.0 / ranks))
        # Expected Recall@K under uniform ordering within a tied score group.
        ties = frame[mode + '_ties'].to_numpy()
        greater = ranks - 1.0 - 0.5 * (ties - 1)
        for k in (1, 5, 10):
            result[f'{mode}_recall_at_{k}'] = float(np.clip((k - greater) / ties, 0, 1).mean())
    frame.to_csv(Path(OUTPUT_DIR) / (label + '_test_queries.csv'), index=False)
    return result

test_dataset = ByteHumCachedTripletDataset(TEST_DIR, prep, cqt_ext, is_training=False)
if set(test_dataset.songs) & (set(train_dataset.songs) | set(val_dataset.songs)):
    raise RuntimeError('Test/train/val song leakage.')
if not Path(finetuned_ckpt_path).is_file():
    raise FileNotFoundError(finetuned_ckpt_path)
best_data = torch.load(finetuned_ckpt_path, map_location='cpu', weights_only=False)
if best_data['run_config'] != RUN_CONFIG or best_data['train_songs'] != train_dataset.songs:
    raise ValueError('Best checkpoint does not match this local run.')
model.load_state_dict(best_data['model_state_dict'], strict=True)
results = [dict(evaluate_test_retrieval(model, test_dataset, 'local_4gb'),
                checkpoint=finetuned_ckpt_path, epoch=best_data['epoch'],
                validation_selection_mrr=best_data['best_val_mrr'])]

kaggle_checkpoint = STAGE2_ROOT / 'checkpoints' / 'conformer_bytehum_full_d128_timegrid_lat_v4_best.pth'
if LOCAL_COMPARE_KAGGLE and kaggle_checkpoint.is_file():
    comparison = torch.load(kaggle_checkpoint, map_location='cpu', weights_only=False)
    cfg = comparison.get('run_config', {})
    if cfg.get('pipeline_version') != 4 or cfg.get('embedding_dim') != EMBEDDING_DIM:
        raise ValueError('Kaggle checkpoint pipeline/embedding mismatch.')
    if comparison.get('train_songs') != train_dataset.songs:
        raise ValueError('Kaggle checkpoint train-song mapping mismatch.')
    model.load_state_dict(comparison['model_state_dict'], strict=True)
    results.append(dict(evaluate_test_retrieval(model, test_dataset, 'kaggle_v4'),
                        checkpoint=str(kaggle_checkpoint), epoch=comparison.get('epoch')))
    model.load_state_dict(best_data['model_state_dict'], strict=True)
elif LOCAL_COMPARE_KAGGLE:
    print('Comparison skipped: Kaggle checkpoint not found:', kaggle_checkpoint)

with open(Path(OUTPUT_DIR) / 'test_metrics.json', 'w', encoding='utf-8') as stream:
    json.dump({'run_config': RUN_CONFIG, 'results': results}, stream, indent=2)
display(pd.DataFrame(results).set_index('label'))
print('Results:', OUTPUT_DIR)
'''


def build():
    doc = json.loads(SOURCE.read_text(encoding="utf-8"))
    cells = copy.deepcopy(doc["cells"][:47])

    def source(index):
        return "".join(cells[index]["source"])

    def set_source(index, value):
        cells[index]["source"] = value.splitlines(True)

    def replace(index, old, new):
        text = source(index)
        if old not in text:
            raise ValueError(f"Source changed in cell {index}: {old[:70]}")
        set_source(index, text.replace(old, new))

    set_source(0, """# Conformer-ByteHum: Local GPU 4 GB

Derived from `conformer_bytehum_qbh_kaggle.ipynb`. Run cells in order in a CUDA
PyTorch kernel. Dependencies: torch, numpy, pandas, matplotlib, tqdm, librosa,
soundfile, scipy, IPython, and typeguard<3. This notebook does not install packages.

Default: 2 songs x 2 queries = 4 triplets/microbatch, FP16 AMP, accumulation 16
(64 triplets/optimizer update), freeze first 3 blocks. Mining and BatchNorm still
operate on microbatches of 4; accumulation does not reproduce Kaggle batch-16
mining or BatchNorm. The architecture, CQT, losses and data split are preserved.

Run All trains from Stage 1 or resumes the local last checkpoint, selects best
on validation, then reports test local/global/fused MRR and Recall@1/5/10.
Test gallery contains test songs only (typically 45), not the full music database.
The default comparison loads the existing Kaggle v4 checkpoint under the same
test protocol. Use LOCAL_EVALUATE_ONLY=True to evaluate an existing local best.

Outputs: `02_stage2_humming_qbh/runs/local_4gb/`. CQT cache lives on disk with a
bounded RAM LRU. Resume is epoch-level; an interrupted epoch is run again.
Changing training settings requires a fresh run directory or matching checkpoint.
""")
    set_source(6, PATHS)
    set_source(8, '''DATASET_SPLIT_DIR, MUSIC_DB_DIR, VOCAL_DIR, STAGE1_CKPT = find_dataset_and_checkpoint()
OUTPUT_DIR = str(STAGE2_ROOT / 'runs' / 'local_4gb')
CHECKPOINT_DIR = str(Path(OUTPUT_DIR) / 'checkpoints')
Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
TRAIN_DIR, VAL_DIR, TEST_DIR = [str(Path(DATASET_SPLIT_DIR) / name) for name in ('train', 'val', 'test')]
print('Dataset:', DATASET_SPLIT_DIR)
print('Stage 1:', STAGE1_CKPT)
print('Outputs:', OUTPUT_DIR)
''')
    replace(4, "print(f\"[+] Thiet bi tinh toan: {device}\")", "if device.type != 'cuda':\n    raise RuntimeError('Select a CUDA-enabled PyTorch kernel before training.')\nprint(f\"[+] Thiet bi tinh toan: {device}\")")
    replace(23, "GLOBAL_CQT_RAM_CACHE: Dict[str, List[torch.Tensor]] = {}", CACHE)
    replace(23, "        try:\n            y = self.audio_preprocessor.load_audio(path)", DISK_LOOKUP + "        try:\n            y = self.audio_preprocessor.load_audio(path)")
    replace(23, "        GLOBAL_CQT_RAM_CACHE[path] = cqts\n        return cqts", "        temp_path = cache_path.with_suffix('.tmp')\n        torch.save(cqts, temp_path)\n        os.replace(temp_path, cache_path)\n        GLOBAL_CQT_RAM_CACHE[path] = cqts\n        return cqts")
    replace(23, "self.song_to_idx.get(song_a, 0)", "self.song_to_idx[song_a]")
    replace(23, "self.song_to_idx.get(neg_song, 0)", "self.song_to_idx[neg_song]")
    replace(25, "SONGS_PER_BATCH = 8\nSAMPLES_PER_SONG = 2", "SONGS_PER_BATCH = LOCAL_SONGS_PER_BATCH\nSAMPLES_PER_SONG = LOCAL_SAMPLES_PER_SONG")
    replace(37, "RUN_TAG = f'{RUN_NAME}_d{EMBEDDING_DIM}_timegrid_lat_v4'", "RUN_CONFIG.update(local_profile='4gb', microbatch= BATCH_SIZE, accumulation=LOCAL_ACCUMULATION_STEPS, freeze_blocks=LOCAL_FREEZE_BLOCKS, num_epochs=LOCAL_NUM_EPOCHS, songs_per_batch=SONGS_PER_BATCH, samples_per_song=SAMPLES_PER_SONG)\nRUN_TAG = f'{RUN_NAME}_d{EMBEDDING_DIM}_timegrid_lat_v4_local4gb'")
    replace(38, "num_blocks_to_freeze=3, freeze_frontend=True", "num_blocks_to_freeze=LOCAL_FREEZE_BLOCKS, freeze_frontend=True")
    set_source(38, source(38) + "\ndel sample_cqt, g_emb, l_feat, lac_logits, dom_logits\ntorch.cuda.empty_cache()\n")
    replace(42, "NUM_EPOCHS = 30", "NUM_EPOCHS = LOCAL_NUM_EPOCHS")
    replace(42, "GRAD_ACCUMULATION_STEPS = 4", "GRAD_ACCUMULATION_STEPS = LOCAL_ACCUMULATION_STEPS")
    replace(42, "torch.amp.GradScaler('cuda', enabled=AMP_ENABLED)", "torch.amp.GradScaler('cuda', enabled=AMP_ENABLED, init_scale=1024.0)")
    replace(44, "song_batch_size=32", "song_batch_size=LOCAL_GALLERY_BLOCK_SIZE")
    replace(44, "song_batch_size=64", "song_batch_size=LOCAL_GALLERY_BLOCK_SIZE")
    replace(44, "query_batch_size=32", "query_batch_size=LOCAL_QUERY_BATCH_SIZE")
    replace(44, "batch_size=32", "batch_size=LOCAL_EVAL_BATCH_SIZE")
    replace(44, "map_location=device", "map_location='cpu'")
    replace(44, "if RESUME_TRAINING and not FORCE_RETRAIN", "if RESUME_TRAINING and not LOCAL_EVALUATE_ONLY and not FORCE_RETRAIN")
    replace(44, "start_epoch = 1", "start_epoch = 1\nepochs_without_improvement = 0")
    replace(44, "    torch.set_rng_state(resume_data['torch_rng_state'])", "    torch.set_rng_state(resume_data['torch_rng_state'])\n    loader_generator.set_state(resume_data['loader_rng_state'])\n    epochs_without_improvement = resume_data['epochs_without_improvement']\n    if resume_data['train_songs'] != train_dataset.songs:\n        raise ValueError('Resume train-song mapping mismatch')\n    del resume_data")
    # Restore CUDA RNG before dropping the payload.
    replace(44, "    del resume_data\n    if torch.cuda", "    if torch.cuda")
    replace(44, "    print(f\"[+] Resume training from epoch {start_epoch}\")", "    del resume_data\n    print(f\"[+] Resume training from epoch {start_epoch}\")")
    replace(44, "if not RESUME_TRAINING and not FORCE_RETRAIN and os.path.exists(finetuned_ckpt_path):", "if LOCAL_EVALUATE_ONLY and not os.path.exists(finetuned_ckpt_path):\n    raise FileNotFoundError(finetuned_ckpt_path)\nif LOCAL_EVALUATE_ONLY or (not RESUME_TRAINING and not FORCE_RETRAIN and os.path.exists(finetuned_ckpt_path)):")
    replace(44, "    epochs_without_improvement = 0\n    for epoch", "    for epoch")
    replace(44, "    for epoch in range(start_epoch, NUM_EPOCHS + 1):", "    epoch_range = range(start_epoch, NUM_EPOCHS + 1) if epochs_without_improvement < EARLY_STOPPING_PATIENCE else []\n    for epoch in epoch_range:")
    # Save last after selecting best, including updated patience and RNG state.
    loop = source(44)
    begin = loop.index("        resume_payload = {")
    end = loop.index("        # Chon checkpoint", begin)
    payload = loop[begin:end]
    loop = loop[:begin] + loop[end:]
    stop_begin = loop.index("            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:")
    stop_end = loop.index("\n    if os.path.exists(finetuned_ckpt_path):", stop_begin)
    payload = payload.replace("'run_config': RUN_CONFIG,", "'loader_rng_state': loader_generator.get_state(),\n            'epochs_without_improvement': epochs_without_improvement,\n            'train_songs': train_dataset.songs,\n            'run_config': RUN_CONFIG,")
    loop = loop[:stop_begin] + "\n" + payload + "        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:\n            print('[EARLY STOP] Validation did not improve.')\n            break\n" + loop[stop_end:]
    set_source(44, loop)
    set_source(22, "### Dataset and bounded CQT cache\nCQT is cached on disk; the RAM LRU is limited by LOCAL_RAM_CACHE_MB.\n")
    set_source(24, "### Song-balanced local DataLoaders\nDefault microbatch: 2 songs x 2 queries; Windows uses num_workers=0.\n")
    set_source(36, "### Initialize the pretrained model\nLOCAL_FREEZE_BLOCKS controls the frozen prefix (default: first 3 blocks).\n")
    set_source(41, "### Optimizer and local training schedule\nFP16 AMP and gradient accumulation reduce training VRAM.\n")
    cells.insert(1, cell("code", CONFIG))
    cells.extend([cell("markdown", "## Test retrieval and Kaggle checkpoint comparison\n"), cell("code", EVALUATION)])
    for index, item in enumerate(cells):
        item["id"] = f"local4gb-{index:03d}"
        item["metadata"] = {}
        if item["cell_type"] == "code":
            item.update(outputs=[], execution_count=None)
            compile("".join(item["source"]), f"{TARGET.name}:cell{index}", "exec")
    doc["cells"] = cells
    doc["metadata"] = {"kernelspec": {"display_name": "Python 3 (CUDA PyTorch)", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11"}}
    doc["nbformat_minor"] = 5
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(TARGET)


if __name__ == "__main__":
    build()
