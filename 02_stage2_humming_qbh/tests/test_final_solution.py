"""Final notebook protocol regressions. CUDA execution belongs on Kaggle."""
import ast
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = [ROOT / 'notebooks' / 'templates' / 'conformer_bytehum_qbh_final_train.ipynb',
             ROOT / 'notebooks' / 'kaggle' / 'evaluate' / 'conformer_bytehum_qbh_final_evaluation.ipynb']


def sources(path):
    return [''.join(c['source']) for c in json.loads(path.read_text(encoding='utf-8'))['cells']
            if c['cell_type'] == 'code']


def extract(names, ns):
    nodes = []
    for source in sources(NOTEBOOKS[0]):
        for node in ast.parse(source).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
                nodes.append(node)
    if {n.name for n in nodes} != set(names):
        raise AssertionError('Missing notebook definitions')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'notebook-functions', 'exec'), ns)
    return ns


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.ns = extract(['chunk_starts', 'rank_statistics', 'retrieval_metrics', 'digest',
                           'atomic_json', 'unique_path'], dict(np=np, pd=pd, Path=Path,
                                                           json=json, hashlib=__import__('hashlib'), os=__import__('os')))

    def test_notebook_structure_clean_and_compilable(self):
        for path in NOTEBOOKS:
            doc = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual((doc['nbformat'], doc['nbformat_minor']), (4, 5))
            self.assertEqual(len({c['id'] for c in doc['cells']}), len(doc['cells']))
            for cell in doc['cells']:
                self.assertIn(cell['cell_type'], ['code', 'markdown'])
                if cell['cell_type'] == 'code':
                    compile(''.join(cell['source']), str(path), 'exec')
                    self.assertEqual(cell['outputs'], [])
                    self.assertIsNone(cell['execution_count'])
            if importlib.util.find_spec('nbformat'):
                import nbformat
                nbformat.validate(nbformat.read(path, as_version=4))

    def test_self_contained_identical_common_code(self):
        train, evaluation = map(sources, NOTEBOOKS)
        self.assertEqual(train[:-2], evaluation[:-1])
        for source in train + evaluation:
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.ImportFrom):
                    self.assertFalse(node.level)
                    self.assertFalse((node.module or '').startswith(('build_', 'experiments', 'src')))
            self.assertNotIn('exec(', source)
            self.assertNotIn('runpy.', source)

    def test_every_sample_covered_without_duplicate_offsets(self):
        for length in range(1, 150):
            starts = self.ns['chunk_starts'](length, 10, 3)
            self.assertEqual(len(starts), len(set(starts)))
            self.assertEqual(starts[0], 0)
            self.assertGreaterEqual(starts[-1] + 10, length)
            self.assertTrue(all(right <= left + 10 for left, right in zip(starts, starts[1:])))
        self.assertEqual(self.ns['chunk_starts'](31, 10, 3), [0, 3, 6, 9, 12, 15, 18, 21])
        self.assertEqual(self.ns['chunk_starts'](32, 10, 3)[-1], 22)
        self.assertEqual(self.ns['chunk_starts'](11, 10, 3), [0, 1])

    def test_near_ties_excluded_from_greater_count(self):
        stats = self.ns['rank_statistics']([.9, .9000001, .4], 0)
        self.assertEqual((stats['rank'], stats['ties'], stats['greater']), (1.5, 2, 0))
        self.assertAlmostEqual(stats['recall_at_1'], .5)
        self.assertAlmostEqual(stats['rr'], 2 / 3)

    def test_invalid_scores_raise_not_drop_queries(self):
        for scores in ([np.nan, .2], [-np.inf, .2], [np.inf, .2]):
            with self.assertRaises(ValueError):
                self.ns['rank_statistics'](scores, 0)

    def test_micro_and_song_macro_different_denominators(self):
        rows = []
        for song, scores in [('a', [1., 0.]), ('a', [1., 0.]), ('b', [0., 1.])]:
            stats = self.ns['rank_statistics'](scores, 0)
            row = dict(song=song)
            for mode in ('local', 'global', 'fused'):
                row.update({mode + '_' + k: v for k, v in stats.items()})
            rows.append(row)
        metrics = self.ns['retrieval_metrics'](rows)
        self.assertEqual((metrics['queries'], metrics['songs']), (3, 2))
        self.assertAlmostEqual(metrics['fused']['mrr'], 5 / 6)
        self.assertAlmostEqual(metrics['fused']['macro_song_mrr'], .75)
        self.assertAlmostEqual(metrics['fused']['recall_at_1'], 2 / 3)

    def test_json_lock_normalization_and_content_hash(self):
        data = {'records': [('song', 'abc')], 'x': 1}
        canonical = json.loads(json.dumps(data))
        with tempfile.TemporaryDirectory(dir=ROOT / 'tests', prefix='final-test-') as folder:
            target = Path(folder) / 'lock.json'
            self.ns['atomic_json'](target, canonical)
            saved = json.loads(target.read_text(encoding='utf-8'))
            self.assertEqual(saved, canonical)
            self.assertEqual(self.ns['digest'](saved), self.ns['digest'](data))
            self.assertNotEqual(self.ns['digest'](saved), self.ns['digest']({'records': [('song', 'different')], 'x': 1}))
            self.assertFalse(target.with_suffix('.json.tmp').exists())

    def test_path_discovery_requires_unambiguous_input(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'tests', prefix='final-test-') as folder:
            a, b = Path(folder) / 'a', Path(folder) / 'b'
            a.mkdir()
            b.mkdir()
            with self.assertRaises(ValueError):
                self.ns['unique_path'](None, [a, b], 'override')
            self.assertEqual(self.ns['unique_path'](a, [a, b], 'override'), a.resolve())

    def test_zip_extraction_rejects_traversal_and_reuses_complete_archive(self):
        import zipfile
        import stat
        import shutil
        ns = extract(['extract_dataset_archive'], dict(self.ns, zipfile=zipfile, stat=stat,
                                                       shutil=shutil, tqdm=lambda items, **kwargs: items))
        with tempfile.TemporaryDirectory(dir=ROOT / 'tests', prefix='final-test-') as folder:
            directory = Path(folder)
            valid = directory / 'dataset.bin'
            with zipfile.ZipFile(valid, 'w') as handle:
                handle.writestr('datasets/dataset_split/train/song/fragment/reference.wav', b'audio')
                handle.writestr('datasets/2_music_database/song.mp3', b'music')
            result = ns['extract_dataset_archive'](valid, directory / 'unpacked')
            self.assertEqual((result / 'datasets/2_music_database/song.mp3').read_bytes(), b'music')
            self.assertEqual(result, ns['extract_dataset_archive'](valid, directory / 'unpacked'))
            for name in ['../outside.txt', '/absolute.txt', 'C:/drive.txt', 'nested/../../outside.txt']:
                invalid = directory / 'invalid.zip'
                with zipfile.ZipFile(invalid, 'w') as handle:
                    handle.writestr(name, b'bad')
                with self.assertRaises(ValueError):
                    ns['extract_dataset_archive'](invalid, directory / 'unpacked')
            self.assertFalse((directory / 'outside.txt').exists())

    def test_training_never_reads_or_evaluates_test_queries(self):
        train = '\n'.join(sources(NOTEBOOKS[0]))
        self.assertNotIn("scan_queries('test')", train)
        self.assertNotIn('TEST_ROWS', train)
        self.assertIn("val_mrr = metrics['fused']['mrr']", train)
        self.assertIn("model.train(was_training)", train)
        self.assertIn("if RUN_CONFIG['lat'] > 0 else zero", train)

    def test_resume_and_test_transaction_order(self):
        train = sources(NOTEBOOKS[0])[-2]
        self.assertIn("torch.load(source, map_location='cpu'", train)
        self.assertIn("loader_generator.set_state(state['rng_loader'])", train)
        self.assertLess(train.index('patience += 1'), train.rindex('commit_training_checkpoint(last_path, best_path, state)'))
        commit = next(n for n in ast.parse(train).body if isinstance(n, ast.FunctionDef) and n.name == 'commit_training_checkpoint')
        source = ast.get_source_segment(train, commit)
        self.assertLess(source.index('atomic_torch(last_path, state)'), source.index('export_best_checkpoint('))
        evaluation = sources(NOTEBOOKS[1])[-1]
        self.assertLess(evaluation.index('atomic_json(lock_path, test_contract)'), evaluation.index('rows = evaluate_queries'))
        self.assertLess(evaluation.index('if report_path.exists():'), evaluation.index('eval_model = FinalModel'))
        self.assertNotIn('MAX_EVAL_QUERIES', evaluation)

    def test_actual_dataset_and_full_gallery_ids(self):
        directory = ROOT / 'datasets' / 'dataset_split'
        ns = extract(['split_songs', 'scan_queries'], dict(Path=Path, DATASET_SPLIT_DIR=directory,
                                                          AUDIO_EXTENSIONS={'.wav', '.mp3', '.flac', '.ogg', '.m4a'}))
        expected = {'train': 210, 'val': 45, 'test': 45}
        ids = {s: ns['split_songs'](s) for s in expected}
        self.assertEqual({s: len(v) for s, v in ids.items()}, expected)
        self.assertFalse(set(ids['train']) & set(ids['val']))
        self.assertFalse(set(ids['val']) & set(ids['test']))
        self.assertEqual(len(ns['scan_queries']('val')), 773)
        self.assertEqual(len(ns['scan_queries']('test')), 774)
        gallery = {p.stem for p in (ROOT / 'datasets' / '2_music_database').iterdir()
                   if p.suffix.lower() in ns['AUDIO_EXTENSIONS']}
        self.assertEqual(gallery, set().union(*map(set, ids.values())))

    def test_interrupted_query_evaluation_resumes_exact_prefix(self):
        import types
        import time
        source = next(s for s in sources(NOTEBOOKS[0]) if 'def evaluate_queries' in s)
        node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'evaluate_queries')
        node.decorator_list = []
        calls = []

        class Scores:
            def cpu(self):
                return self

            def numpy(self):
                return np.array([1., .2])

        ns = dict(self.ns, torch=types.SimpleNamespace(cuda=types.SimpleNamespace(synchronize=lambda: None)),
                  time=time, tqdm=lambda values, **kwargs: values,
                  raw_cqts=lambda path: path,
                  encode_cqts=lambda model, path: (calls.append(path), None),
                  score_gallery=lambda *args: {mode: Scores() for mode in ('local', 'global', 'fused')})
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'query-resume', 'exec'), ns)
        queries = [dict(key=str(i), song='a', path=str(i)) for i in range(3)]
        with tempfile.TemporaryDirectory(dir=ROOT / 'tests', prefix='final-test-') as folder:
            progress = Path(folder) / 'progress.json'
            first = ns['evaluate_queries'](None, {'a': None, 'b': None}, queries[:1], progress)
            self.assertEqual(calls, ['0'])
            completed = ns['evaluate_queries'](None, {'a': None, 'b': None}, queries, progress)
            self.assertEqual(calls, ['0', '1', '2'])
            self.assertEqual(completed[:1], first)
            self.assertEqual(len(completed), 3)
            ns['evaluate_queries'](None, {'a': None, 'b': None}, queries, progress)
            self.assertEqual(calls, ['0', '1', '2'])
            with self.assertRaises(ValueError):
                ns['evaluate_queries'](None, {'a': None, 'b': None}, list(reversed(queries)), progress)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch not installed in this validation runtime')
class TorchTests(unittest.TestCase):
    def setUp(self):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        self.ns = extract(['score_gallery', 'CrossBatchMemoryBank', '_topk_masked_mean',
                           'in_batch_cosine_triplet_loss', 'local_alignment_triplet'],
                          dict(torch=torch, nn=nn, F=F, device=torch.device('cpu'),
                               GALLERY_SONG_BLOCK=2, REFERENCE_FRAME_BLOCK=3))

    def test_blocked_scores_match_naive_scores(self):
        torch, F = self.ns['torch'], self.ns['F']
        torch.manual_seed(42)
        qg, ql = F.normalize(torch.randn(3, 8), dim=-1), F.normalize(torch.randn(7, 8), dim=-1)
        gallery = {str(i): (F.normalize(torch.randn(i + 1, 8), dim=-1), F.normalize(torch.randn(length, 8), dim=-1))
                   for i, length in enumerate([1, 2, 11, 8, 5])}
        actual = self.ns['score_gallery'](qg, ql, gallery)
        expected_l = torch.stack([(ql @ l.T).max(1).values.mean() for _, l in gallery.values()])
        expected_g = torch.stack([(qg @ g.T).max() for g, _ in gallery.values()])
        self.assertTrue(torch.allclose(actual['fused'], .5 * (expected_l + expected_g), atol=1e-6))

    def test_triplet_losses_have_finite_gradients(self):
        torch, F = self.ns['torch'], self.ns['F']
        torch.manual_seed(42)
        raw = [torch.randn(4, 8, requires_grad=True) for _ in range(3)]
        a, p, n = [F.normalize(x, dim=-1) for x in raw]
        labels, negatives = torch.tensor([0, 0, 1, 1]), torch.tensor([1, 1, 0, 0])
        memory = self.ns['CrossBatchMemoryBank'](16)
        memory.enqueue(p, labels)
        loss = self.ns['in_batch_cosine_triplet_loss'](a, p, n, labels, negatives, memory_bank=memory)
        local = [torch.randn(4, 5, 8, requires_grad=True) for _ in range(3)]
        loss = loss + self.ns['local_alignment_triplet'](*local, labels, negatives)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(x.grad is not None and torch.isfinite(x.grad).all() for x in raw + local))


if __name__ == '__main__':
    unittest.main(verbosity=2)
