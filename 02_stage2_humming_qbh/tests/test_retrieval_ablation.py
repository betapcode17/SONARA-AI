"""Scoring, protocol boundaries and notebook structure without GPU training."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("retrieval_ablation", ROOT / "experiments" / "retrieval_ablation.py")
engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)


class AblationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_fast_baseline_matches_exact(self):
        torch.manual_seed(2)
        gallery = {str(i): (F.normalize(torch.randn(n, 128), dim=-1),
                           F.normalize(torch.randn(n * 20, 128), dim=-1))
                   for i, n in enumerate((1, 3, 7))}
        qg = F.normalize(torch.randn(2, 128), dim=-1)
        ql = F.normalize(torch.randn(40, 128), dim=-1)
        scorer = engine.ChunkScorer(gallery, torch.device("cpu"), chunk_block=2)
        actual = scorer.score(qg, ql, engine.scoring_configs())
        expected = [float(.5 * (ql @ l.T).max(-1).values.mean() + .5 * (qg @ g.T).max())
                    for g, l in gallery.values()]
        np.testing.assert_allclose(actual["baseline"], expected, atol=1e-6)

    def test_window_matches_naive_and_same_window_fusion(self):
        torch.manual_seed(3)
        g, l = F.normalize(torch.randn(5, 128), dim=-1), F.normalize(torch.randn(100, 128), dim=-1)
        qg, ql = F.normalize(torch.randn(2, 128), dim=-1), F.normalize(torch.randn(40, 128), dim=-1)
        scorer = engine.ChunkScorer({"song": (g, l)}, torch.device("cpu"), chunk_block=2)
        configs = {"win": {"kind": "window", "chunks": 3, "alpha": .75},
                   "full": {"kind": "window", "chunks": 99, "alpha": .5},
                   "exact": {"kind": "exact", "alpha": .5}}
        actual = scorer.score(qg, ql, configs)
        naive = [float(.75 * (ql @ l[start * 20:(start + 3) * 20].T).max(-1).values.mean()
                       + .25 * (qg @ g[start:start + 3].T).max()) for start in range(3)]
        self.assertAlmostEqual(actual["win"][0], max(naive), places=6)
        self.assertAlmostEqual(actual["full"][0], actual["exact"][0], places=6)

    def test_window_rejects_disjoint_matches(self):
        e1, e2 = torch.eye(128)[:2]
        local = torch.stack([e1] * 20 + [e2] * 20)
        gallery = {"disjoint": (torch.stack([e1, e2]), local)}
        scorer = engine.ChunkScorer(gallery, torch.device("cpu"))
        configs = {"exact": {"kind": "exact", "alpha": 1},
                   "window": {"kind": "window", "chunks": 1, "alpha": 1}}
        scores = scorer.score(e1[None], torch.stack([e1, e2]), configs)
        self.assertEqual(scores["exact"][0], 1)
        self.assertEqual(scores["window"][0], .5)

    def test_chunk_starts_matches_existing_tail_policy(self):
        self.assertEqual(engine.chunk_starts(4, 1), [0])
        self.assertEqual(engine.chunk_starts(13, 2), [0, 3])
        self.assertEqual(engine.chunk_starts(14.9, 2), [0, 3])
        # The source sampler never appends a tail for remainder <= 3 sec.
        with self.assertRaises(ValueError):
            engine.chunk_starts(13, 3)

    def test_timeline_removes_overlap_and_padding(self):
        e = torch.eye(128)[0]
        l = e.repeat(40, 1)
        values, times = engine.timeline(l, [0, 3], 13)
        self.assertLess(len(values), 40)
        self.assertTrue(np.all(np.diff(times) > 0))
        self.assertTrue(np.all(times < 13 + .48))
        np.testing.assert_allclose(values[:, 0], 1)
        short, _ = engine.timeline(e.repeat(20, 1), [0], 2)
        self.assertLess(len(short), 20)

    def test_dtw_subsequence_and_order(self):
        import librosa
        x = np.eye(128, dtype=np.float32)[:4]
        ref = np.concatenate([x[3:4], x, x[0:1]])
        self.assertAlmostEqual(engine.dtw_similarity(x, ref, librosa), 1, places=6)
        self.assertLess(engine.dtw_similarity(x, x[::-1].copy(), librosa), 1)
        self.assertAlmostEqual(engine.dtw_similarity(x[:1], x[:1], librosa), 1, places=6)

    def test_ties_and_macro_metrics(self):
        stat = engine.tie_stats([.5, .5, .1], 0)
        self.assertEqual(stat, {"rank": 1.5, "ties": 2, "greater": 0})
        rows = [{"song": "a", "statistics": {"x": stat}},
                {"song": "a", "statistics": {"x": stat}},
                {"song": "b", "statistics": {"x": {"rank": 3, "ties": 1, "greater": 2}}}]
        metrics = engine.summarize(rows, "x")
        self.assertAlmostEqual(metrics["macro_mrr"], .5)
        self.assertAlmostEqual(metrics["recall_at_1"], 1 / 3)
        with self.assertRaises(ValueError):
            engine.tie_stats([float("nan")], 0)

    def test_selection_cannot_use_test_subset_or_test_gallery(self):
        report = {"protocol": {"split": "val", "subset": False, "test_songs_in_gallery": False},
                  "metrics": {"baseline": {"mrr": .4}, "window": {"mrr": .5}}}
        self.assertEqual(engine.select_validation(report), "window")
        for key, value in (("split", "test"), ("subset", True), ("test_songs_in_gallery", True)):
            modified = json.loads(json.dumps(report))
            modified["protocol"][key] = value
            with self.assertRaises(ValueError):
                engine.select_validation(modified)

    def test_atomic_report(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "report.json"
            engine.atomic_json(path, {"queries": 774})
            self.assertEqual(json.loads(path.read_text()), {"queries": 774})
            self.assertFalse(path.with_suffix(".json.tmp").exists())

    def test_notebook_clean_and_valid(self):
        import nbformat
        path = ROOT / "notebooks" / 'local_4gb' / 'experiments' / "conformer_bytehum_qbh_retrieval_ablation_local_4gb.ipynb"
        document = nbformat.read(path, as_version=4)
        nbformat.validate(document)
        for cell in document.cells:
            if cell.cell_type == "code":
                compile(cell.source, str(path), "exec")
                self.assertEqual(cell.outputs, [])
                self.assertIsNone(cell.execution_count)


if __name__ == "__main__":
    unittest.main()
