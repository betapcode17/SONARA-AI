"""Read-only timing; run with core/.venv/Scripts/python.exe from any directory.

Reads core's existing cache and writes core/.cache/scalability-review-evidence.json.
No audio, training or humming accuracy evaluation.
"""
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2] / 'core'
index = json.loads((ROOT / '.cache/library_index.json').read_text(encoding='utf-8'))
arrays = []
for song in index['songs']:
    contract = dict(model=index['model']['checkpoint_sha256'], audio=song['sha256'],
                    preprocessing=index['model']['preprocessing'])
    key = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
    with np.load(ROOT / '.cache/embeddings' / (key + '.npz'), allow_pickle=False) as saved:
        value = saved['embeddings']
        assert value.shape == (song['chunks'], 128) and np.isfinite(value).all()
        arrays.append(value)
references = np.ascontiguousarray(np.concatenate(arrays))
offsets = np.cumsum([0] + [len(a) for a in arrays[:-1]])

def existing(query):
    return np.array([np.clip(query @ reference.T, -1., 1.).max(axis=1).mean()
                     for reference in arrays])

def blocked(query):
    similarities = np.clip(query @ references.T, -1., 1.)
    return np.maximum.reduceat(similarities, offsets, axis=1).mean(axis=0)

def measure(function, query):
    for _ in range(5):
        function(query)
    samples = []
    for _ in range(40):
        start = time.perf_counter()
        function(query)
        samples.append((time.perf_counter() - start) * 1000)
    return dict(p50_ms=float(np.median(samples)), p95_ms=float(np.percentile(samples, 95)),
                repeats=len(samples))

try:
    from threadpoolctl import threadpool_limits, threadpool_info
except ImportError:
    threadpool_limits = None
    threadpool_info = lambda: []

results = []
def run():
    for chunks in (1, 3, 8):
        # Vectors from real reference cache are used only to time the scorer.
        query = references[np.arange(chunks) * 29].copy()
        old, new = existing(query), blocked(query)
        error = float(np.max(np.abs(old - new)))
        assert np.allclose(old, new, rtol=1e-5, atol=1e-6)
        results.append(dict(query_chunks=chunks, existing=measure(existing, query),
                            concatenated_exact=measure(blocked, query), max_abs_score_error=error))

if threadpool_limits:
    with threadpool_limits(limits=4, user_api='blas'):
        pools = threadpool_info()
        run()
else:
    pools = threadpool_info()
    run()

value = dict(checkpoint=index['model']['checkpoint'],
             checkpoint_sha256=index['model']['checkpoint_sha256'],
             gallery_songs=len(arrays), gallery_vectors=len(references),
             dimension=references.shape[1], raw_vector_bytes=references.nbytes,
             audio_seconds=sum(s['duration'] for s in index['songs']),
             python=platform.python_version(), numpy=np.__version__, blas=pools,
             accuracy_evaluated=False, audio_encoding_evaluated=False,
             scope='Warm cached scoring only; existing scorer versus concatenated exact MaxMean. '
                   'Reference vectors serve as timing queries, not humming queries. '
                   'No throughput, ANN, cold cache or end-to-end benchmark.',
             results=results,
             raw_memory_projection=[dict(songs=n, vectors=round(n*len(references)/len(arrays)),
                                         gib=n*references.nbytes/len(arrays)/1024**3)
                                    for n in (90,1000,10000,100000,1000000)])
output = ROOT / '.cache/scalability-review-evidence.json'
output.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(value, ensure_ascii=False, indent=2))
