"""Run scoped V5 checks, write measured reports, and refresh the Kaggle bundle."""
import builtins
import hashlib
import importlib.util
import json
import os
import symtable
import sys
import unittest
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / 'tests'
SOURCE = ROOT / 'notebooks/templates/conformer_bytehum_qbh_final_train.ipynb'
EVALUATION = ROOT / 'notebooks/kaggle/evaluate/conformer_bytehum_qbh_final_evaluation.ipynb'
KAGGLE = ROOT / 'notebooks/kaggle/train/conformer_bytehum_qbh_final_train_kaggle.ipynb'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_notebook(path):
    document = json.loads(path.read_text(encoding='utf-8'))
    cells = [cell for cell in document['cells'] if cell['cell_type'] == 'code']
    for cell in cells:
        compile(''.join(cell['source']), str(path), 'exec')
        if cell['outputs'] or cell['execution_count'] is not None:
            raise ValueError(f'Generated notebook is not clean: {path}')
    code = '\n'.join(''.join(cell['source']) for cell in cells)
    table = symtable.symtable(code, str(path), 'exec')
    defined = {symbol.get_name() for symbol in table.get_symbols()
               if symbol.is_assigned() or symbol.is_imported()}
    allowed = defined | set(dir(builtins)) | {'__name__', '__file__'}
    unresolved = set()

    def visit(scope):
        unresolved.update(symbol.get_name() for symbol in scope.get_symbols()
                          if symbol.is_global() and symbol.is_referenced() and symbol.get_name() not in allowed)
        for child in scope.get_children():
            visit(child)

    visit(table)
    if unresolved:
        raise ValueError(f'Unresolved globals in {path}: {sorted(unresolved)}')
    return dict(file=path.relative_to(ROOT).as_posix(), sha256=sha256(path),
                code_cells=len(cells), unresolved_globals=[])


def summarize(results):
    skipped = [dict(test=str(test), reason=reason) for result in results for test, reason in result.skipped]
    failures = sum(len(result.failures) for result in results)
    errors = sum(len(result.errors) for result in results)
    count = sum(result.testsRun for result in results)
    return dict(tests_run=count, passed=count - len(skipped) - failures - errors,
                skipped=skipped, failures=failures, errors=errors)


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    sys.path.insert(0, str(TESTS))
    loader = unittest.TestLoader()
    final = unittest.TextTestRunner(verbosity=2).run(loader.discover(str(TESTS), pattern='test_final*.py'))
    wrapper = unittest.TextTestRunner(verbosity=2).run(loader.discover(str(TESTS), pattern='test_kaggle_ready_notebook.py'))
    if not final.wasSuccessful() or not wrapper.wasSuccessful():
        return 1
    inspected = [inspect_notebook(path) for path in (SOURCE, EVALUATION, KAGGLE)]
    metadata = json.loads(KAGGLE.read_text(encoding='utf-8'))['metadata']['qbh_kaggle_ready']
    timestamp = datetime.now(timezone.utc).isoformat()
    final_report = dict(created_at_utc=timestamp, **summarize([final]), notebooks=inspected[:2],
        validation='Compile/symbol analysis, protocol/inventory, fake checkpoint IO, NumPy positive selection and preflight AST checks',
        gpu_training_executed=False, nbformat_schema_validator_available=bool(importlib.util.find_spec('nbformat')))
    wrapper_report = dict(created_at_utc=timestamp, **summarize([final, wrapper]),
        notebook=KAGGLE.relative_to(ROOT).as_posix(), sha256=sha256(KAGGLE), code_cells=inspected[2]['code_cells'],
        source_sha256=sha256(SOURCE), source_template_code_preserved=metadata['source_sha256'] == sha256(SOURCE),
        kaggle_bootstrap_tests_use_mock_gpu=True, gpu_training_executed=False, checkpoint_schema=2)
    write_json(ROOT / 'reports/final_solution_validation.json', final_report)
    write_json(ROOT / 'reports/kaggle_ready_validation.json', wrapper_report)
    files = [SOURCE, EVALUATION, KAGGLE] + [ROOT / 'reports' / name for name in (
        'FINAL_SOLUTION.md', 'KAGGLE_TRAIN_READY.md', 'V5_CODE_IMPROVEMENTS.md',
        'final_solution_validation.json', 'kaggle_ready_validation.json')]
    archive = ROOT / 'dist/qbh_final_v5_kaggle_bundle.zip'
    temp = archive.with_suffix('.zip.tmp')
    with zipfile.ZipFile(temp, 'w', compression=zipfile.ZIP_DEFLATED) as zipped:
        for path in files:
            zipped.write(path, path.relative_to(ROOT).as_posix())
        zipped.write(ROOT / 'dist/README.md', 'README.md')
    with zipfile.ZipFile(temp) as zipped:
        if zipped.testzip() is not None:
            raise ValueError('Bundle ZIP integrity failure')
        for path in files:
            if zipped.read(path.relative_to(ROOT).as_posix()) != path.read_bytes():
                raise ValueError('Bundle file content mismatch: ' + str(path))
    os.replace(temp, archive)
    print(json.dumps(dict(**summarize([final, wrapper]), bundle=str(archive),
                          bundle_sha256=sha256(archive), gpu_training_executed=False), indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
