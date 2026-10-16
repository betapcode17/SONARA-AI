"""Focused evaluator tests and an opt-in real CUDA notebook execution."""
import ast
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / 'notebooks' / 'local_4gb' / 'evaluate' / 'conformer_bytehum_qbh_evaluation_local_4gb.ipynb'


def sources():
    return [''.join(c['source']) for c in json.loads(NOTEBOOK.read_text(encoding='utf-8'))['cells']
            if c['cell_type'] == 'code']


class LocalEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        import torch
        import pandas as pd
        import torch.nn as nn
        import torch.nn.functional as F
        cls.ns = dict(np=np, torch=torch, pd=pd, nn=nn, F=F, device=torch.device('cpu'),
                      GALLERY_SONG_BLOCK=2, REFERENCE_FRAME_BLOCK=3)
        engine = next(s for s in sources() if 'def score_gallery' in s)
        tree = ast.parse(engine)
        tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                     and n.name in ('score_gallery', 'tie_statistics', 'summarize_rows')]
        exec(compile(tree, str(NOTEBOOK), 'exec'), cls.ns)

    def test_valid_clean_notebook(self):
        import nbformat
        doc = nbformat.read(NOTEBOOK, as_version=4)
        nbformat.validate(doc)
        for c in doc.cells:
            if c.cell_type == 'code':
                compile(c.source, str(NOTEBOOK), 'exec')
                self.assertEqual(c.outputs, [])
                self.assertIsNone(c.execution_count)

    def test_reference_blocks_match_naive_scores(self):
        torch, F = self.ns['torch'], self.ns['F']
        torch.manual_seed(1)
        qg, ql = F.normalize(torch.randn(3, 8), dim=-1), F.normalize(torch.randn(7, 8), dim=-1)
        gallery = {str(i): (F.normalize(torch.randn(i + 1, 8), dim=-1),
                           F.normalize(torch.randn(length, 8), dim=-1))
                   for i, length in enumerate([2, 11, 5, 1, 8])}
        actual = self.ns['score_gallery'](qg, ql, gallery)
        expected_local = torch.stack([(ql @ l.T).max(dim=1).values.mean() for _, l in gallery.values()])
        expected_global = torch.stack([(qg @ g.T).max() for g, _ in gallery.values()])
        self.assertTrue(torch.allclose(actual['local'], expected_local, atol=1e-6))
        self.assertTrue(torch.allclose(actual['global'], expected_global, atol=1e-6))
        self.assertTrue(torch.allclose(actual['fused'], .5 * (expected_local + expected_global), atol=1e-6))

    def test_ties_do_not_double_count_close_scores(self):
        np = self.ns['np']
        rank, ties, greater = self.ns['tie_statistics'](np.array([.9, .9000001, .4]), 0)
        self.assertEqual((rank, ties, greater), (1.5, 2, 0))

    def test_candidate_miss_keeps_full_denominator(self):
        pd, np = self.ns['pd'], self.ns['np']
        stats = [self.ns['tie_statistics'](np.array([.5, .5]), 0),
                 self.ns['tie_statistics'](np.array([-np.inf, .5]), 0)]
        frame = pd.DataFrame([dict(f_rank=r, f_ties=t, f_greater=g) for r, t, g in stats])
        result = self.ns['summarize_rows'](frame, 'f')
        self.assertEqual(result['queries'], 2)
        self.assertAlmostEqual(result['mrr'], 1 / 3)
        self.assertAlmostEqual(result['recall_at_1'], .25)
        self.assertAlmostEqual(result['recall_at_5'], .5)


def run_notebook(full=False):
    import os
    os.environ['MPLBACKEND'] = 'Agg'
    ns = {'__name__': '__local_evaluation_validation__'}
    for source in sources():
        source = source.replace('plt.show()', 'plt.close(fig)')
        source = source.replace('from tqdm.auto import tqdm',
                                'from tqdm.auto import tqdm as _tqdm\ndef tqdm(*args, **kwargs):\n    kwargs[\"mininterval\"] = 30\n    kwargs[\"leave\"] = True\n    return _tqdm(*args, **kwargs)')
        if not full and 'MAX_EVAL_QUERIES = None' in source:
            source = source.replace('MAX_EVAL_QUERIES = None', 'MAX_EVAL_QUERIES = 4')
            source = source.replace("('evaluation_' + GALLERY_MODE)", "('evaluation_smoke_' + GALLERY_MODE)")
        print('Execute:', source.splitlines()[0], flush=True)
        exec(compile(source, str(NOTEBOOK), 'exec'), ns)
    report = ns['report']
    assert len(report['results']) == 3
    assert all(r['queries'] == (774 if full else 4) for r in report['results'])
    # Verify local/Kaggle exact results against the completed training evaluator.
    if full and ns['GALLERY_MODE'] == 'test':
        saved = json.loads((ROOT / 'runs' / 'local_4gb' / 'test_metrics.json').read_text(encoding='utf-8'))
        for previous in saved['results']:
            current = next(r for r in report['results'] if r['label'] == previous['label'])
            assert current['epoch'] == previous['epoch']
            for mode in ('local', 'global', 'fused'):
                assert abs(current['modes'][mode]['mrr'] - previous[mode + '_mrr']) < 1e-5
                for k in (1, 5, 10):
                    assert abs(current['modes'][mode][f'recall_at_{k}'] - previous[f'{mode}_recall_at_{k}']) < 1e-5
        print('Golden comparison with saved local/Kaggle test metrics: PASS', flush=True)
    if full and ns['GALLERY_MODE'] == 'full':
        assert all(r['gallery_songs'] == len(ns['gallery_paths']) == 300 for r in report['results'])
        assert not report['protocol']['subset']
        frames = [ns['pd'].read_csv(ns['OUTPUT_DIR'] / (r['label'] + '_queries.csv')) for r in report['results']]
        assert all(f['query'].nunique() == len(ns['all_test_queries']) for f in frames)
        assert all(f['query'].tolist() == frames[0]['query'].tolist() for f in frames)
        assert all(f['song'].tolist() == frames[0]['song'].tolist() for f in frames)
        print('Full-gallery comparison: identical 300 songs / 774 queries for all models: PASS', flush=True)
    # CPU DTW identity and optional reranking on a real query, outside saved exact report.
    np = ns['np']
    rng = np.random.default_rng(42)
    x = rng.normal(size=(96, 15)).astype('float32')
    assert ns['compute_cqt_dtw_distance'](x, x, max_pitch_search=0) < 1e-5
    query = ns['eval_queries'][0][2]
    demo = ns['search_humming'](query, top_k=2, use_faiss=True, enable_dtw=True, dtw_candidates=2)
    assert len(demo['results']) == 2 and demo['dtw'] and demo['faiss']
    print('Notebook execution, baselines, FAISS and real-query DTW search: PASS', flush=True)


if __name__ == '__main__':
    if '--cuda-full' in sys.argv or '--cuda-smoke' in sys.argv:
        run_notebook(full='--cuda-full' in sys.argv)
    else:
        unittest.main()
