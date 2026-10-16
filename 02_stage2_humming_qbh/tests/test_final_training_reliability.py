"""Execute embedded notebook helpers; fake checkpoint IO is not a CUDA test."""
import ast
import copy
import gc
import importlib.util
import json
import math
import os
import pickle
import random
import tempfile
import types
import unittest
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from test_final_solution import ROOT, NOTEBOOKS, extract, sources


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(dir=ROOT / 'tests', prefix='reliability-')
        self.addCleanup(self.folder.cleanup)
        self.last = Path(self.folder.name) / 'last.pth'
        self.best = Path(self.folder.name) / 'best.pth'
        self.writes = []

        def save(data, path):
            self.writes.append(Path(path).name)
            Path(path).write_bytes(pickle.dumps(data))

        fake_torch = types.SimpleNamespace(save=save,
            load=lambda path, **kwargs: pickle.loads(Path(path).read_bytes()), equal=np.array_equal)
        self.ns = extract(['atomic_torch', 'atomic_json', 'validated_best_snapshot',
                           'export_best_checkpoint', 'commit_training_checkpoint'],
                          dict(torch=fake_torch, Path=Path, os=os, json=json, math=math,
                               RUN_SIGNATURE='run', DATA_CONTRACT={'data': 'same'}))

    def state(self, values=(.35, .40), finished=False):
        index = int(np.argmax(values))
        best = dict(run_signature='run', data_contract={'data': 'same'},
                    epoch=index + 1, best_mrr=values[index], finished=False,
                    model_state_dict={'weight': np.array([index], dtype=np.float32)},
                    validation_metrics={'fused': {'mrr': values[index]}}, checkpoint_schema=2)
        return dict(best, epoch=len(values), finished=finished,
                    history=[dict(epoch=i + 1, val_fused_mrr=value) for i, value in enumerate(values)],
                    best_checkpoint=best, optimizer={'step': len(values)})

    def commit(self, state):
        self.ns['commit_training_checkpoint'](self.last, self.best, state)

    def load(self, path):
        return self.ns['torch'].load(path)

    def test_last_commits_before_best_and_reports(self):
        self.commit(self.state())
        self.assertEqual(self.writes[:2], ['last.pth.tmp', 'best.pth.tmp'])
        self.assertEqual(self.load(self.best)['best_mrr'], .40)
        self.assertEqual(json.loads((self.best.parent / 'best_validation_metrics.json').read_text())['fused']['mrr'], .40)

    def test_crash_after_last_commit_recovers_best_without_regression(self):
        self.commit(self.state((.35,)))
        original = self.ns['atomic_torch']

        def interrupted(path, data):
            if path == self.best:
                raise OSError('Simulated interruption during BEST export')
            original(path, data)

        self.ns['atomic_torch'] = interrupted
        with self.assertRaises(OSError):
            self.commit(self.state())
        self.assertEqual(self.load(self.last)['best_mrr'], .40)
        self.assertEqual(self.load(self.best)['best_mrr'], .35)
        self.ns['atomic_torch'] = original
        self.commit(self.load(self.last))
        self.commit(self.state((.35, .40, .37)))
        self.assertEqual(self.load(self.best)['best_mrr'], .40)
        self.assertEqual(self.load(self.best)['epoch'], 2)

    def test_crash_before_last_commit_leaves_previous_best_untouched(self):
        self.commit(self.state((.35,)))
        before = self.best.read_bytes()
        self.ns['atomic_torch'] = lambda *args: (_ for _ in ()).throw(OSError('No disk space'))
        with self.assertRaises(OSError):
            self.commit(self.state())
        self.assertEqual(self.best.read_bytes(), before)
        self.assertEqual(self.load(self.last)['epoch'], 1)

    def test_missing_or_stale_best_is_repaired_from_last(self):
        state = self.state()
        self.commit(state)
        self.best.unlink()
        self.commit(self.load(self.last))
        self.assertEqual(self.load(self.best)['best_mrr'], .40)
        stale = self.load(self.best)
        stale['model_state_dict']['weight'][0] = -100
        self.ns['atomic_torch'](self.best, stale)
        self.commit(self.load(self.last))
        np.testing.assert_array_equal(self.load(self.best)['model_state_dict']['weight'], [1])

    def test_completed_export_is_byte_identical_on_repeat(self):
        state = self.state(finished=True)
        self.commit(state)
        before = self.best.read_bytes()
        self.writes.clear()
        self.commit(self.load(self.last))
        self.ns['export_best_checkpoint'](self.best, state['best_checkpoint'], True)
        self.assertEqual(before, self.best.read_bytes())
        self.assertNotIn('best.pth.tmp', self.writes)
        self.assertTrue(self.load(self.best)['finished'])

    def test_inconsistent_snapshots_are_rejected_before_writing(self):
        changes = [lambda s: s.update(checkpoint_schema=1),
                   lambda s: s['best_checkpoint'].update(run_signature='different'),
                   lambda s: s['best_checkpoint'].update(data_contract={'other': True}),
                   lambda s: s.update(best_mrr=float('nan')),
                   lambda s: s['best_checkpoint'].update(best_mrr=.5),
                   lambda s: s['best_checkpoint'].update(epoch=99),
                   lambda s: s['history'][0].update(epoch=9),
                   lambda s: s['history'][0].update(val_fused_mrr=.9),
                   lambda s: s['best_checkpoint']['validation_metrics']['fused'].update(mrr=.1)]
        for change in changes:
            with self.subTest(change=change):
                state = copy.deepcopy(self.state())
                change(state)
                with self.assertRaises(ValueError):
                    self.commit(state)
                self.assertFalse(self.last.exists())

    def test_best_without_last_cannot_silently_start_fresh(self):
        self.ns['atomic_torch'](self.best, self.state()['best_checkpoint'])
        source = sources(NOTEBOOKS[0])[-2]
        branch = next(node for node in ast.parse(source).body
                      if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == 'RESUME')
        ns = dict(self.ns, RESUME=True, best_path=self.best, last_path=self.last,
                  RESUME_LAST_OVERRIDE=None, RESUME_BEST_OVERRIDE=None)
        with self.assertRaisesRegex(FileNotFoundError, 'BEST alone'):
            exec(compile(ast.Module(body=[branch], type_ignores=[]), 'resume-branch', 'exec'), ns)


class Tensor(np.ndarray):
    """Minimal NumPy adapter for testing positive selection, not autograd."""
    def __new__(cls, values):
        return np.asarray(values, dtype=np.float32).view(cls)

    def float(self):
        return self

    def detach(self):
        return self

    def to(self, device):
        return self

    def clone(self):
        return self.copy()

    def amax(self, axis):
        return Tensor(np.asarray(self).max(axis))

    def max(self, dim):
        values = np.asarray(self)
        return Tensor(values.max(axis=dim)), values.argmax(axis=dim)


class PositiveSelectionTests(unittest.TestCase):
    def setUp(self):
        normalize = lambda x, dim: Tensor(np.asarray(x) / np.maximum(np.linalg.norm(x, axis=dim, keepdims=True), 1e-12))
        self.normalize = normalize
        fake_torch = types.SimpleNamespace(no_grad=lambda: lambda fn: fn,
            autocast=lambda *args, **kwargs: nullcontext(), float16=None,
            stack=lambda xs: Tensor(np.stack(xs)), isfinite=np.isfinite,
            einsum=lambda equation, *xs: Tensor(np.einsum(equation, *xs)))
        self.chunks = {'a': [Tensor([0, i]) for i in range(3)],
                       'b': [Tensor([1, i]) for i in range(3)]}
        self.ns = extract(['aligned_positives'], dict(torch=fake_torch,
            F=types.SimpleNamespace(normalize=normalize), raw_cqts=lambda path: self.chunks[path],
            EVAL_BATCH_SIZE=2, device='cpu'))
        self.calls = []
        owner = self

        class Model:
            training = True
            offset = 0

            def train(self, mode=True):
                self.training = mode

            def eval(self):
                self.train(False)

            def __call__(self, x, **kwargs):
                owner.calls.append(x.copy())
                g = Tensor([[row[0] + 1 + self.offset, row[1] + 1, 1] for row in x])
                l = Tensor(np.stack((g, g[:, ::-1]), axis=1))
                return g, l, None, None

        self.model = Model()
        self.ag = Tensor([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
        self.al = Tensor(np.stack((self.ag, self.ag[:, ::-1]), axis=1))

    def test_grouped_selection_matches_naive_scoring_and_query_order(self):
        paths = ['a', 'b', 'a']
        actual = self.ns['aligned_positives'](self.model, paths, self.ag, self.al)
        self.assertTrue(self.model.training)
        self.assertEqual(len(self.calls), 4)  # Two unique bags, two forwards per bag.
        expected = []
        for path, ag, al in zip(paths, self.ag, self.al):
            g, l, _, _ = self.model(Tensor(self.chunks[path]))
            g, l = self.normalize(g, -1), self.normalize(l, -1)
            local = np.einsum('td,nsd->nts', self.normalize(al, -1), l).max(-1).mean(-1)
            index = np.argmax(.5 * (g @ self.normalize(ag, -1) + local))
            expected.append(self.chunks[path][index])
        np.testing.assert_array_equal(actual, expected)

    def test_no_feature_cache_survives_the_call(self):
        self.ns['aligned_positives'](self.model, ['a'] * 3, self.ag, self.al)
        self.assertEqual(len(self.calls), 2)
        self.model.offset = 10
        self.ns['aligned_positives'](self.model, ['a'] * 3, self.ag, self.al)
        self.assertEqual(len(self.calls), 4)

    def test_mode_is_restored_if_reference_loading_fails(self):
        with self.assertRaises(KeyError):
            self.ns['aligned_positives'](self.model, ['missing'] * 3, self.ag, self.al)
        self.assertTrue(self.model.training)

    def test_mismatched_batch_is_rejected(self):
        with self.assertRaises(ValueError):
            self.ns['aligned_positives'](self.model, ['a'], self.ag, self.al)


class PreflightContractTests(unittest.TestCase):
    def test_probe_precedes_full_cache_and_restores_state_in_finally(self):
        source = sources(NOTEBOOKS[0])[-2]
        self.assertLess(source.index('    gpu_preflight()'), source.index('    prewarm_paths ='))
        probe = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'gpu_preflight')
        body = ast.get_source_segment(source, probe)
        self.assertIn('model.train()', body)
        self.assertIn('probe_scaler.step(probe_optimizer)', body)
        self.assertNotIn('scheduler.step()', body)
        self.assertNotIn('scaler.step(optimizer)', body)
        cleanup = ast.unparse(next(n for n in probe.body if isinstance(n, ast.Try)).finalbody)
        for fragment in ('model.load_state_dict', 'random.setstate', 'np.random.set_state',
                         'torch.set_rng_state', 'torch.cuda.set_rng_state_all',
                         'loader_generator.set_state', 'model.train(was_training)'):
            self.assertIn(fragment, cleanup)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch not installed in this validation runtime')
class PreflightTorchTests(unittest.TestCase):
    def test_cpu_probe_updates_then_restores_weights_buffers_rng_and_optimizer(self):
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from unittest.mock import patch

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(4, 128)
                self.bn = nn.BatchNorm1d(128)
                self.lambda_val = .2

            def set_lambda(self, value):
                self.lambda_val = value

            def forward(self, x, **kwargs):
                g = self.bn(self.fc(x))
                return g, g[:, None, :].expand(-1, 20, -1), None, None

        class Memory:
            def __init__(self, capacity):
                pass

            def enqueue(self, *args):
                pass

        model = Model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        generator = torch.Generator().manual_seed(42)
        batch = dict(anchor=torch.randn(4, 4), negative=torch.randn(4, 4),
                     song_idx=torch.tensor([0, 0, 1, 1]), negative_song_idx=torch.tensor([1, 1, 0, 0]),
                     positive_path=['a'] * 4)

        def objective(outputs, *args):
            # Consume RNG to exercise rollback beyond tensor weights and BN buffers.
            random.random()
            np.random.rand()
            torch.rand(1, generator=generator)
            g = [out[0] for out in outputs]
            return {'total': sum(value.square().mean() for value in g)}, g

        ns = extract(['model_cpu_state', 'gpu_preflight'], dict(torch=torch, F=F, model=model,
            optimizer=optimizer, random=random, np=np, math=math, gc=gc, device='cpu',
            loader_generator=generator, train_loader=[batch, batch], start_epoch=1,
            RUN_CONFIG=dict(accumulation=2, songs_per_batch=2, samples_per_song=2,
                            epochs=30, grl=.2, lac=0, memory_capacity=8, clip=1.),
            CrossBatchMemoryBank=Memory, training_objective=objective,
            aligned_positives=lambda *args: batch['anchor'], TRAIN_ROWS=[{'path': 'a'}],
            raw_cqts=lambda path: [], encode_cqts=lambda *args: (torch.ones(1, 128), torch.ones(20, 128)),
            score_gallery=lambda *args: {'fused': torch.ones(2)}))
        weights = copy.deepcopy(model.state_dict())
        rng = (random.getstate(), np.random.get_state(), torch.get_rng_state(), generator.get_state())
        scaler_class = torch.amp.GradScaler
        with patch.object(torch, 'autocast', lambda *args, **kwargs: nullcontext()), \
             patch.object(torch.amp, 'GradScaler', lambda *args, **kwargs: scaler_class('cpu', **kwargs)), \
             patch.object(torch.cuda, 'get_rng_state_all', lambda: []), \
             patch.object(torch.cuda, 'set_rng_state_all', lambda states: None), \
             patch.object(torch.cuda, 'empty_cache', lambda: None):
            ns['gpu_preflight']()
        self.assertTrue(all(torch.equal(value, model.state_dict()[key]) for key, value in weights.items()))
        self.assertEqual(model.lambda_val, .2)
        self.assertTrue(model.training)
        self.assertEqual(optimizer.state_dict()['state'], {})
        self.assertEqual(random.getstate(), rng[0])
        np.testing.assert_array_equal(np.random.get_state()[1], rng[1][1])
        self.assertTrue(torch.equal(torch.get_rng_state(), rng[2]))
        self.assertTrue(torch.equal(generator.get_state(), rng[3]))


if __name__ == '__main__':
    unittest.main(verbosity=2)
