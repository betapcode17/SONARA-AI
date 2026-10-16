"""Build a standalone full-gallery candidate-recall ablation notebook."""
import json
import copy
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[2] / 'notebooks'
SOURCE = ROOT / 'local_4gb' / 'evaluate' / "conformer_bytehum_qbh_evaluation_local_4gb.ipynb"
TARGET = ROOT / 'local_4gb' / 'experiments' / "conformer_bytehum_qbh_candidate_ablation_local_4gb.ipynb"

INTRO = """# Full-gallery candidate retrieval ablation (4 GB GPU)

This experiment reuses the local v4 checkpoint and cached gallery embeddings. It
compares the current global-chunk FAISS gate with a local-frame-mean-per-chunk
index and their union on the complete 300-song gallery. Set `QUERY_SPLIT` to
`val` for iteration or `test` for one final held-out report. Each split writes
to a separate output directory. Query embeddings are cached by checkpoint and
audio fingerprints; never choose production settings from test.
"""

QUERY_SPLIT_CELL = r'''# Keep validation for iteration; use test only for final reporting.
QUERY_SPLIT = os.environ.get('QBH_QUERY_SPLIT', 'val')  # 'val' or 'test'
if QUERY_SPLIT == 'val':
    query_dataset = ByteHumCachedTripletDataset(str(VAL_DIR), prep, cqt_ext, is_training=False)
elif QUERY_SPLIT == 'test':
    query_dataset = test_dataset
else:
    raise ValueError("QUERY_SPLIT must be 'val' or 'test'.")
eval_queries = list(query_dataset.humming_samples)
all_test_queries = eval_queries
if not eval_queries:
    raise ValueError(f'No {QUERY_SPLIT} queries found.')
OUTPUT_DIR = STAGE2_ROOT / 'runs' / 'local_4gb' / f'retrieval_ablation_full_{QUERY_SPLIT}'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f'Query split: {QUERY_SPLIT} | Queries: {len(eval_queries)} | Gallery songs: {len(gallery_paths)}')
'''

EXPERIMENT = r'''from collections import defaultdict

if GALLERY_MODE != 'full' or len(gallery_paths) != 300 or not eval_queries:
    raise ValueError('This ablation requires the complete 300-song gallery and a non-empty query split.')
if faiss is None:
    raise ImportError('Install faiss-cpu to run the candidate-recall ablation.')
faiss.omp_set_num_threads(max(1, min(4, os.cpu_count() or 1)))

local_hash = file_sha256(finetuned_ckpt_path)
model = ConformerByteHumModel(num_classes=NUM_TRAIN_SONGS,
                              embedding_dim=EMBEDDING_DIM,
                              allow_random_init=True)
model.load_state_dict(checkpoint_metadata['model_state_dict'], strict=True)
model = model.to(device).eval()
gallery, gallery_signature_value = build_gallery(model, gallery_paths, local_hash)

# Cache query embeddings separately from gallery features; input order and file
# fingerprints are part of the key so a changed query set cannot reuse stale data.
query_records = [(song, fragment, str(Path(path).resolve()),
                  Path(path).stat().st_size, Path(path).stat().st_mtime_ns)
                 for song, fragment, path in eval_queries]
query_signature = hashlib.sha256(json.dumps({
    'checkpoint': local_hash,
    'gallery': gallery_signature_value,
    'queries': query_records,
    'pipeline': 'v4_16k_10s_hop3_96x83',
    'torch': torch.__version__
}, sort_keys=True).encode()).hexdigest()
query_cache_path = STAGE2_ROOT / 'cache' / 'evaluation_features' / ('queries_' + query_signature + '.pt')
if query_cache_path.is_file():
    payload = torch.load(query_cache_path, map_location='cpu', weights_only=True)
    if payload.get('signature') != query_signature or len(payload.get('features', [])) != len(eval_queries):
        raise ValueError('Query feature cache does not match this checkpoint/query set.')
    query_features = payload['features']
    print('Reused cached query embeddings:', query_cache_path)
else:
    query_features = []
    for song, fragment, path in tqdm(eval_queries, desc='Encode queries'):
        cqts = query_dataset._get_raw_cqts(path)
        query_features.append(encode_cqts(model, cqts))
    temp_path = query_cache_path.with_suffix('.tmp')
    torch.save({'signature': query_signature, 'features': query_features}, temp_path)
    os.replace(temp_path, query_cache_path)
    print('Saved query embeddings:', query_cache_path)

song_ids = list(gallery)
song_to_index = {song: i for i, song in enumerate(song_ids)}
global_index, chunk_to_song = build_faiss_gallery(gallery)

# Pool local frames within each chunk for a compact complementary candidate
# index; reranking can still use the unpooled local sequence in a later step.
LOCAL_FRAMES_PER_CHUNK = 20
local_parts = [gallery[song][1].reshape(-1, LOCAL_FRAMES_PER_CHUNK, EMBEDDING_DIM)
               .mean(dim=1)
               for song in song_ids]
local_matrix = np.ascontiguousarray(torch.cat(local_parts).numpy(), dtype=np.float32)
local_to_song = np.concatenate([
    np.full(len(local_parts[i]), i, dtype=np.int64)
    for i, song in enumerate(song_ids)
])
faiss.normalize_L2(local_matrix)
local_index = faiss.IndexFlatIP(EMBEDDING_DIM)
local_index.add(local_matrix)
del local_matrix, local_parts
print('Global/local index vectors:', global_index.ntotal, '/', local_index.ntotal,
      '| local features are mean-pooled within each audio chunk.')

def batched_faiss_search(index, queries, k, batch_size=128):
    """Bound the exact-index distance matrix to one small query batch."""
    batches = [index.search(queries[start:start + batch_size], k)[1]
               for start in range(0, len(queries), batch_size)]
    return np.concatenate(batches, axis=0)

global_rows, global_slices = [], []
local_rows, local_slices = [], []
cursor = 0
for global_features, local_features in query_features:
    g = np.ascontiguousarray(global_features.numpy(), dtype=np.float32)
    l = np.ascontiguousarray(local_features.reshape(
        -1, LOCAL_FRAMES_PER_CHUNK, EMBEDDING_DIM).mean(dim=1).numpy(), dtype=np.float32)
    global_rows.append(g)
    global_slices.append(slice(cursor, cursor + len(g)))
    cursor += len(g)
    local_rows.append(l)
    local_slices.append(slice(sum(len(x) for x in local_rows[:-1]),
                              sum(len(x) for x in local_rows)))
global_matrix = np.ascontiguousarray(np.concatenate(global_rows), dtype=np.float32)
local_query_matrix = np.ascontiguousarray(np.concatenate(local_rows), dtype=np.float32)
faiss.normalize_L2(global_matrix)
faiss.normalize_L2(local_query_matrix)

K_VALUES = (10, 25, 50, 100, 200, 300, 500, 800, 1200)
MAX_K = max(K_VALUES)
global_neighbors = batched_faiss_search(global_index, global_matrix, MAX_K)
print('Global neighbor search complete.')
local_neighbors = batched_faiss_search(local_index, local_query_matrix, MAX_K)
print('Local neighbor search complete.')
rows = []
per_query = []
for query_i, ((song, fragment, path), g_slice, l_slice) in enumerate(
        zip(eval_queries, global_slices, local_slices)):
    target = song_to_index[song]
    g_hits = global_neighbors[g_slice]
    l_hits = local_neighbors[l_slice]
    for k in K_VALUES:
        global_candidates = set(int(chunk_to_song[j]) for j in g_hits[:, :k].ravel() if j >= 0)
        local_candidates = set(int(local_to_song[j]) for j in l_hits[:, :k].ravel() if j >= 0)
        strategies = {
            'global_only': global_candidates,
            'local_only': local_candidates,
            'global_plus_local': global_candidates | local_candidates,
        }
        for strategy, candidates in strategies.items():
            hit = target in candidates
            row = {'query_index': query_i, 'song': song, 'fragment': fragment,
                   'query': path, 'neighbors_per_query_vector': k,
                   'strategy': strategy, 'candidate_songs': len(candidates),
                   'ground_truth_hit': hit}
            rows.append(row)
            per_query.append(row)

frame = pd.DataFrame(rows)
summary = (frame.groupby(['neighbors_per_query_vector', 'strategy'], sort=False)
           .agg(queries=('ground_truth_hit', 'size'),
                candidate_recall=('ground_truth_hit', 'mean'),
                mean_candidate_songs=('candidate_songs', 'mean'),
                p95_candidate_songs=('candidate_songs', lambda x: float(x.quantile(.95))))
           .reset_index())
summary['candidate_recall_pct'] = 100 * summary['candidate_recall']
summary.to_csv(OUTPUT_DIR / 'candidate_recall_summary.csv', index=False)
frame.to_csv(OUTPUT_DIR / 'candidate_recall_queries.csv', index=False)

fig, ax = plt.subplots(figsize=(9, 5.5))
for strategy, group in summary.groupby('strategy', sort=False):
    ax.plot(group['mean_candidate_songs'], group['candidate_recall_pct'], marker='o', label=strategy)
ax.axhline(95, color='gray', linestyle='--', linewidth=1, label='95% target')
ax.set(xlabel='Mean unique candidate songs', ylabel='Ground-truth candidate recall (%)',
       title='Candidate recall vs. rerank workload (300-song gallery)')
ax.grid(True, alpha=.25)
ax.legend()
fig.tight_layout()
fig.savefig(OUTPUT_DIR / 'candidate_recall_curve.png', dpi=180)
plt.show()

metadata = {
    'created_at_utc': datetime.now(timezone.utc).isoformat(),
    'protocol': {'gallery_mode': GALLERY_MODE, 'query_split': QUERY_SPLIT,
                 'gallery_songs': len(song_ids), 'queries': len(eval_queries), 'subset': False,
                 'neighbors_per_query_vector': list(K_VALUES),
                 'strategies': ['global_only', 'local_only', 'global_plus_local']},
    'checkpoint': str(finetuned_ckpt_path), 'checkpoint_sha256': local_hash,
    'gallery_signature': gallery_signature_value, 'query_signature': query_signature,
    'local_index_vectors': int(len(local_to_song)),
    'method': 'FAISS IndexFlatIP over global chunks and mean-pooled local chunk embeddings; no model training.',
    'interpretation': 'Candidate recall measures whether the ground-truth song survives the gate; it is not final reranked Recall@K.'
}
(OUTPUT_DIR / 'candidate_recall_metadata.json').write_text(
    json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
display(summary[['neighbors_per_query_vector', 'strategy', 'candidate_recall_pct',
                 'mean_candidate_songs', 'p95_candidate_songs']])
print('Saved candidate-recall ablation:', OUTPUT_DIR)
'''

WINDOWED_EXPERIMENT = r'''@torch.inference_mode()
def score_windowed_local(query_local, gallery_features, window_scales=(0.75, 1.0, 1.25, 1.5)):
    """Align query chunks to one continuous reference window before MaxMean scoring."""
    q = query_local.to(device).reshape(-1, LOCAL_FRAMES_PER_CHUNK, EMBEDDING_DIM)
    query_chunks = q.shape[0]
    song_ids = list(gallery_features)
    output = []
    q_index = torch.arange(query_chunks, device=device)[:, None]
    for offset in range(0, len(song_ids), GALLERY_SONG_BLOCK):
        group_songs = song_ids[offset:offset + GALLERY_SONG_BLOCK]
        group = [gallery_features[song][1].reshape(-1, LOCAL_FRAMES_PER_CHUNK, EMBEDDING_DIM)
                 for song in group_songs]
        lengths = torch.tensor([len(x) for x in group], device=device)
        reference = nn.utils.rnn.pad_sequence(group, batch_first=True).to(device)
        chunk_mask = torch.arange(reference.shape[1], device=device)[None, :] < lengths[:, None]
        # [song, query_chunk, reference_chunk, query_frame, reference_frame]
        similarity = torch.einsum('atd,grud->gartu', q, reference)
        similarity = similarity.masked_fill(~chunk_mask[:, None, :, None, None], float('-inf'))
        pair_scores = similarity.amax(dim=-1).mean(dim=-1)  # [song, query_chunk, reference_chunk]
        for song_i, ref_length in enumerate(lengths.tolist()):
            best = torch.tensor(float('-inf'), device=device)
            for scale in window_scales:
                window_chunks = max(1, int(round(query_chunks * scale)))
                if window_chunks > ref_length:
                    continue
                starts = torch.arange(ref_length - window_chunks + 1, device=device)
                if query_chunks == 1:
                    positions = torch.zeros(1, dtype=torch.long, device=device)
                else:
                    positions = torch.round(torch.arange(query_chunks, device=device)
                        * (window_chunks - 1) / (query_chunks - 1)).long()
                reference_positions = starts[None, :] + positions[:, None]
                aligned = pair_scores[song_i, q_index, reference_positions]
                best = torch.maximum(best, aligned.mean(dim=0).amax())
            output.append(best)
    return torch.stack(output).cpu()


rows = []
window_scores = []
for query_i, ((song, fragment, path), (query_global, query_local)) in enumerate(
        zip(eval_queries, query_features)):
    baseline = score_gallery(query_global, query_local, gallery)
    local_window = score_windowed_local(query_local, gallery)
    fused_window = 0.5 * (local_window + baseline['global'])
    target = song_to_index[song]
    row = {'query_index': query_i, 'query': path, 'song': song, 'fragment': fragment}
    for name, value in [('baseline_fused', baseline['fused']),
                        ('windowed_local', local_window),
                        ('windowed_fused', fused_window)]:
        rank, ties, greater = tie_statistics(value.numpy(), target)
        row.update({name + '_rank': rank, name + '_ties': ties, name + '_greater': greater,
                    name + '_top1_song': song_ids[int(value.argmax())]})
    rows.append(row)
    window_scores.append({'baseline_global': baseline['global'],
                          'baseline_fused': baseline['fused'],
                          'windowed_local': local_window,
                          'windowed_fused': fused_window})
    if (query_i + 1) % 50 == 0:
        print(f'Windowed exact scoring: {query_i + 1}/{len(eval_queries)} queries', flush=True)

window_frame = pd.DataFrame(rows)
window_metrics = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
    'protocol': {'gallery_mode': GALLERY_MODE, 'query_split': QUERY_SPLIT,
                 'gallery_songs': len(song_ids),
                 'queries': len(window_frame), 'subset': len(window_frame) != len(all_test_queries),
                 'window_scales': [0.75, 1.0, 1.25, 1.5],
                 'local_scoring': 'query/reference chunk MaxMean aligned within one contiguous reference window',
                 'fusion': '0.5 windowed local + 0.5 existing global maximum cosine'},
    'checkpoint_sha256': local_hash, 'gallery_signature': gallery_signature_value,
    'results': {mode: summarize_rows(window_frame, mode)
                for mode in ('baseline_fused', 'windowed_local', 'windowed_fused')}}
window_frame.to_csv(OUTPUT_DIR / 'windowed_scoring_queries.csv', index=False)
(OUTPUT_DIR / 'windowed_scoring_metrics.json').write_text(
    json.dumps(window_metrics, indent=2, allow_nan=False), encoding='utf-8')
display(pd.DataFrame([{'mode': mode, 'MRR': result['mrr'],
                       'Top-1': result['recall_at_1'], 'Top-5': result['recall_at_5'],
                       'Top-10': result['recall_at_10'], 'Top-20': result['recall_at_20']}
                      for mode, result in window_metrics['results'].items()]))
print('Saved windowed-scoring comparison:', OUTPUT_DIR)
'''

CANDIDATE_RERANK_EXPERIMENT = r'''RERANK_K = 500
baseline_path = OUTPUT_DIR / 'windowed_scoring_queries.csv'
if QUERY_SPLIT == 'test' and not baseline_path.is_file():
    baseline_path = STAGE2_ROOT / 'runs' / 'local_4gb' / 'retrieval_ablation_full' / 'windowed_scoring_queries.csv'
if not baseline_path.is_file():
    raise FileNotFoundError('Run the temporal-scoring cell first: ' + str(baseline_path))
baseline_frame = pd.read_csv(baseline_path)
if len(baseline_frame) != len(eval_queries) or baseline_frame['query_index'].tolist() != list(range(len(eval_queries))):
    raise ValueError('Full-gallery baseline does not match the current query split/order.')

rerank_rows = []
for query_i, ((song, fragment, path), (query_global, query_local), g_slice, l_slice) in enumerate(
        zip(eval_queries, query_features, global_slices, local_slices)):
    global_candidates = {int(chunk_to_song[j]) for j in global_neighbors[g_slice, :RERANK_K].ravel() if j >= 0}
    local_candidates = {int(local_to_song[j]) for j in local_neighbors[l_slice, :RERANK_K].ravel() if j >= 0}
    candidate_indices = sorted(global_candidates | local_candidates)
    candidate_songs = [song_ids[i] for i in candidate_indices]
    candidate_gallery = {candidate_song: gallery[candidate_song] for candidate_song in candidate_songs}
    scores = score_gallery(query_global, query_local, candidate_gallery)['fused'].numpy()
    row = {'query_index': query_i, 'query': path, 'song': song,
           'candidate_count': len(candidate_songs), 'candidate_hit': song in candidate_gallery}
    if song in candidate_gallery:
        target = candidate_songs.index(song)
        rank, ties, greater = tie_statistics(scores, target)
    else:
        rank, ties, greater = float('inf'), 0, float('inf')
    row.update({'candidate_fused_rank': rank, 'candidate_fused_ties': ties,
                'candidate_fused_greater': greater,
                'baseline_fused_rank': baseline_frame.loc[query_i, 'baseline_fused_rank'],
                'baseline_fused_ties': baseline_frame.loc[query_i, 'baseline_fused_ties'],
                'baseline_fused_greater': baseline_frame.loc[query_i, 'baseline_fused_greater']})
    rerank_rows.append(row)
    if (query_i + 1) % 100 == 0:
        print(f'Candidate reranking: {query_i + 1}/{len(eval_queries)} queries', flush=True)

rerank_frame = pd.DataFrame(rerank_rows)
rerank_metrics = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
    'protocol': {'query_split': QUERY_SPLIT, 'gallery_mode': GALLERY_MODE,
                 'gallery_songs': len(song_ids), 'queries': len(eval_queries),
                 'candidate_strategy': 'global_plus_local', 'neighbors_per_query_vector': RERANK_K,
                 'reranker': 'exact existing MaxMean local + max-cosine global, fused 0.5/0.5'},
    'candidate_recall': float(rerank_frame['candidate_hit'].mean()),
    'mean_candidate_songs': float(rerank_frame['candidate_count'].mean()),
    'results': {'full_gallery': summarize_rows(baseline_frame, 'baseline_fused'),
                'candidate_reranked': summarize_rows(rerank_frame, 'candidate_fused')}}
comparison = []
for name, column, k in [('MRR', None, 0), ('Top-1', None, 1), ('Top-5', None, 5),
                        ('Top-10', None, 10), ('Top-20', None, 20)]:
    def expected_metric(frame, prefix, cutoff):
        rank = frame[prefix + '_rank'].to_numpy()
        if cutoff == 0:
            return np.where(np.isfinite(rank), 1.0 / rank, 0.0)
        ties = frame[prefix + '_ties'].to_numpy()
        greater = frame[prefix + '_greater'].to_numpy()
        valid = np.isfinite(rank)
        values = np.zeros(len(frame))
        values[valid] = np.clip((cutoff - greater[valid]) / ties[valid], 0, 1)
        return values
    base_values = expected_metric(rerank_frame, 'baseline_fused', k)
    candidate_values = expected_metric(rerank_frame, 'candidate_fused', k)
    delta = candidate_values - base_values
    rng = np.random.default_rng(20261003 + k)
    bootstrap = np.array([delta[rng.integers(0, len(delta), len(delta))].mean()
                          for _ in range(5000)])
    comparison.append({'metric': name, 'full_gallery': float(base_values.mean()),
        'candidate_reranked': float(candidate_values.mean()), 'delta': float(delta.mean()),
        'delta_ci95_low': float(np.quantile(bootstrap, 0.025)),
        'delta_ci95_high': float(np.quantile(bootstrap, 0.975))})
rerank_metrics['paired_comparison'] = comparison
rerank_frame.to_csv(OUTPUT_DIR / 'candidate_rerank_queries.csv', index=False)
(OUTPUT_DIR / 'candidate_rerank_metrics.json').write_text(
    json.dumps(rerank_metrics, indent=2, allow_nan=False), encoding='utf-8')
display(pd.DataFrame(comparison))
print('Saved candidate-limited reranking comparison:', OUTPUT_DIR)
'''


def build():
    notebook = nbformat.read(SOURCE, as_version=4)
    cells = [nbformat.v4.new_markdown_cell(INTRO)]
    # Reuse the standalone model, preprocessing, data, checkpoint, exact-scoring,
    # and gallery-cache definitions, but omit the old full evaluation and demo.
    for source_cell in notebook.cells[1:21]:
        copied = copy.deepcopy(source_cell)
        if copied.cell_type == 'code':
            copied.outputs = []
            copied.execution_count = None
        cells.append(copied)

    config = cells[1]
    config.source = config.source.replace(
        "('evaluation_' + GALLERY_MODE)",
        "('retrieval_ablation_' + GALLERY_MODE)",
    )
    cells.append(nbformat.v4.new_code_cell(QUERY_SPLIT_CELL))
    cells.append(nbformat.v4.new_markdown_cell(
        '## Candidate recall: global vs. local vs. union\n\n'
        'The experiment indexes all cached local frame features on CPU. Query '
        'embeddings are cached on disk; only nearest-neighbor candidate recall '
        'is measured here. Final reranking quality is evaluated in the next '
        'step using the candidate limit selected on validation.'
    ))
    cells.append(nbformat.v4.new_code_cell(EXPERIMENT))
    cells.append(nbformat.v4.new_markdown_cell(
        '## Temporal-windowed local scoring\n\n'
        'This exact 300-song comparison constrains each query chunk to align '
        'inside a contiguous reference window at a few duration ratios. It '
        'reports local-only and fused ranks alongside the existing exact fused '
        'baseline. Use validation to choose window scales; reserve test for '
        'one final report.'
    ))
    cells.append(nbformat.v4.new_code_cell(WINDOWED_EXPERIMENT))
    cells.append(nbformat.v4.new_markdown_cell(
        '## Candidate-limited reranking at K=500\n\n'
        'The global/local union is exact-reranked on a subset of the full '
        'gallery and compared per query with the saved full-gallery result. '
        'Paired bootstrap intervals show both ranking impact and candidate recall.'
    ))
    cells.append(nbformat.v4.new_code_cell(CANDIDATE_RERANK_EXPERIMENT))

    for index, item in enumerate(cells):
        item['id'] = f'candidate-ablation-{index:03d}'
        if item.cell_type == 'code':
            compile(item.source, f'{TARGET.name}:cell{index}', 'exec')
    result = nbformat.v4.new_notebook(
        cells=cells,
        metadata=notebook.metadata,
    )
    nbformat.validate(result)
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(result, TARGET)
    print(TARGET)


if __name__ == '__main__':
    build()
