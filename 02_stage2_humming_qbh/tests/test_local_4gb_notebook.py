"""Static checks and opt-in real-data CUDA smoke test; never run a full epoch."""
import ast
import copy
import json
import sys
import unittest
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parents[1] / 'notebooks' / 'local_4gb' / 'train' / 'conformer_bytehum_qbh_local_4gb.ipynb'


def load_cells():
    return json.loads(NOTEBOOK.read_text(encoding='utf-8'))['cells']


class LocalNotebookTests(unittest.TestCase):
    def test_clean_valid_notebook(self):
        import nbformat
        nbformat.validate(nbformat.read(NOTEBOOK, as_version=4))
        for item in load_cells():
            if item['cell_type'] == 'code':
                compile(''.join(item['source']), str(NOTEBOOK), 'exec')
                self.assertEqual(item['outputs'], [])
                self.assertIsNone(item['execution_count'])

    def test_resume_payload_after_selection(self):
        source = next(''.join(c['source']) for c in load_cells()
                      if 'resume_payload = {' in ''.join(c['source']))
        self.assertLess(source.index('best_val_mrr = ep_va_mrr'), source.index('resume_payload = {'))
        self.assertIn("'epochs_without_improvement': epochs_without_improvement", source)
        self.assertIn("loader_generator.set_state(resume_data['loader_rng_state'])", source)
        self.assertNotIn('map_location=device', source)

    def test_cache_eviction(self):
        import torch
        source = next(''.join(c['source']) for c in load_cells()
                      if 'class BoundedCQTCache' in ''.join(c['source']))
        tree = ast.parse(source)
        tree.body = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'BoundedCQTCache']
        from collections import OrderedDict
        ns = {'OrderedDict': OrderedDict}
        exec(compile(tree, str(NOTEBOOK), 'exec'), ns)
        cache = ns['BoundedCQTCache'](max_mb=8 / 1024**2)
        cache['a'] = [torch.ones(2)]
        cache['b'] = [torch.ones(2)]
        self.assertNotIn('a', cache)
        self.assertEqual(cache.bytes_used, 8)
        cache['oversized'] = [torch.ones(3)]
        self.assertNotIn('oversized', cache)
        self.assertEqual(cache.bytes_used, 8)


def cuda_smoke():
    import faulthandler
    faulthandler.dump_traceback_later(60, repeat=True)
    namespace = {'__name__': '__local_smoke__'}
    cells = load_cells()
    # Execute setup, real CQT sampling and optimizer configuration, stopping before training.
    for item in cells:
        source = ''.join(item['source'])
        if item['cell_type'] != 'code':
            continue
        print('Smoke setup cell:', source.splitlines()[0], flush=True)
        if 'def _vectorized_local_maxmean' in source:
            tree = ast.parse(source)
            tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
            exec(compile(tree, str(NOTEBOOK), 'exec'), namespace)
            break
        exec(compile(source, str(NOTEBOOK), 'exec'), namespace)

    torch = namespace['torch']
    F = namespace['F']
    model = namespace['model']
    optimizer = namespace['optimizer']
    scaler = namespace['amp_scaler']
    device = namespace['device']
    batch = namespace['sample_batch']
    model.train()
    torch.cuda.reset_peak_memory_stats()
    # Two real microbatches exercise accumulation, all four losses and optimizer state.
    for step in range(2):
        tensors = [batch[key].to(device) for key in ('anchor', 'positive', 'negative')]
        ids, neg_ids = batch['song_idx'].to(device), batch['negative_song_idx'].to(device)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            outputs = [model(tensor) for tensor in tensors]
        g_a, l_a, lac_a, dom_a = [x.float() for x in outputs[0]]
        g_p, l_p, lac_p, dom_p = [x.float() for x in outputs[1]]
        g_n, l_n, lac_n, dom_n = [x.float() for x in outputs[2]]
        triplet = namespace['in_batch_cosine_triplet_loss'](
            g_a, g_p, g_n, song_ids=ids, neg_song_ids=neg_ids,
            memory_bank=namespace['negative_memory_bank'])
        local = namespace['local_alignment_triplet'](l_a, l_p, l_n, ids, neg_ids)
        lac = (F.cross_entropy(lac_a, ids) + F.cross_entropy(lac_p, ids)
               + F.cross_entropy(lac_n, neg_ids)) / 3
        targets = torch.cat([batch[k].to(device).squeeze(-1) for k in
                             ('domain_anchor', 'domain_positive', 'domain_negative')])
        domain = F.binary_cross_entropy_with_logits(torch.cat([dom_a, dom_p, dom_n]), targets)
        loss = triplet + local + lac + domain + 0.05 * namespace['prototype_diversity_loss'](model.lac.prototypes)
        if not torch.isfinite(loss):
            raise AssertionError('Non-finite training loss')
        scaler.scale(loss / 2).backward()
        namespace['negative_memory_bank'].enqueue(torch.cat([g_a, g_p, g_n]), torch.cat([ids, ids, neg_ids]))
    scaler.unscale_(optimizer)
    if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
        raise AssertionError('Non-finite gradients')
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    print('CUDA training peak allocated MiB:', round(torch.cuda.max_memory_allocated() / 1024**2, 1))
    print('CUDA training peak reserved MiB:', round(torch.cuda.max_memory_reserved() / 1024**2, 1))

    validation = copy.copy(namespace['val_dataset'])
    songs = validation.songs[:2]
    validation.song_data = {s: validation.song_data[s] for s in songs}
    validation.humming_samples = [q for q in validation.humming_samples if q[0] in songs][:4]
    metrics = namespace['compute_validation_retrieval_mrr'](model, validation)
    assert metrics['queries'] == 4 and metrics['gallery_songs'] == 2
    print('SMOKE ONLY (2-song gallery, 4 queries):', metrics)
    # Execute evaluator definitions, but do not select/write real test results.
    evaluation = ''.join(cells[-1]['source'])
    tree = ast.parse(evaluation)
    tree.body = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
    exec(compile(tree, str(NOTEBOOK), 'exec'), namespace)
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        namespace['OUTPUT_DIR'] = temp_dir
        result = namespace['evaluate_test_retrieval'](model, validation, 'smoke')
        assert result['queries'] == 4 and result['gallery_songs'] == 2
        assert all(0 <= result[key] <= 1 for key in result if '_mrr' in key or '_recall_' in key)
    print('Real-data CUDA forward/backward, optimizer update and evaluation: PASS')
    faulthandler.cancel_dump_traceback_later()


if __name__ == '__main__':
    if '--cuda-smoke' in sys.argv:
        cuda_smoke()
    else:
        unittest.main()
