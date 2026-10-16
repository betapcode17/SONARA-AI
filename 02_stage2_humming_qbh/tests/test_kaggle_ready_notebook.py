"""CPU-safe checks for the Kaggle wrapper, without claiming real GPU execution."""
import ast
import builtins
import hashlib
import io
import json
import os
import symtable
import tempfile
import types
import unittest
import zipfile
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'notebooks' / 'templates' / 'conformer_bytehum_qbh_final_train.ipynb'
TARGET = ROOT / 'notebooks' / 'kaggle' / 'train' / 'conformer_bytehum_qbh_final_train_kaggle.ipynb'


def sources(path):
    document = json.loads(path.read_text(encoding='utf-8'))
    return [''.join(c['source']) for c in document['cells'] if c['cell_type'] == 'code']


def extract(name):
    nodes = [n for source in sources(TARGET) for n in ast.parse(source).body
             if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(nodes) != 1:
        raise AssertionError('Expected one embedded function: ' + name)
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'kaggle-wrapper-function', 'exec'), namespace)
    return namespace[name]


class KaggleReadyTests(unittest.TestCase):
    def test_structure_clean_and_all_cells_compile(self):
        document = json.loads(TARGET.read_text(encoding='utf-8'))
        self.assertEqual((document['nbformat'], document['nbformat_minor']), (4, 5))
        self.assertEqual(len({c['id'] for c in document['cells']}), len(document['cells']))
        for cell in document['cells']:
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), str(TARGET), 'exec')
                self.assertEqual(cell['outputs'], [])
                self.assertIsNone(cell['execution_count'])
        code = '\n'.join(sources(TARGET))
        table = symtable.symtable(code, str(TARGET), 'exec')
        defined = {s.get_name() for s in table.get_symbols() if s.is_assigned() or s.is_imported()}
        allowed = defined | set(dir(builtins)) | {'__name__', '__file__'}

        def check_symbols(scope):
            for symbol in scope.get_symbols():
                if symbol.is_global() and symbol.is_referenced():
                    self.assertIn(symbol.get_name(), allowed)
            for child in scope.get_children():
                check_symbols(child)

        check_symbols(table)

    def test_original_training_code_and_digest_preserved(self):
        original = sources(SOURCE)
        ready = [s for s in sources(TARGET) if 'def prepare_kaggle_environment(' not in s
                 and 'def package_training_outputs(' not in s]
        self.assertEqual(original, ready)
        metadata = json.loads(TARGET.read_text(encoding='utf-8'))['metadata']['qbh_kaggle_ready']
        self.assertEqual(metadata['source_sha256'], hashlib.sha256(SOURCE.read_bytes()).hexdigest())
        self.assertFalse(metadata['gpu_training_verified_at_creation'])

    def test_no_project_imports_or_external_python_execution(self):
        for source in sources(TARGET):
            self.assertNotIn('runpy.', source)
            self.assertNotIn('exec(', source)
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.ImportFrom):
                    self.assertEqual(node.level, 0)
                    self.assertFalse((node.module or '').startswith(('build_', 'experiments', 'src')))
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertFalse(alias.name.startswith(('build_', 'experiments', 'src')))

    @staticmethod
    def fake_torch(cuda=True):
        return types.SimpleNamespace(
            amp=types.SimpleNamespace(GradScaler=object),
            cuda=types.SimpleNamespace(is_available=lambda: cuda,
                get_device_name=lambda index: 'Tesla T4 (mock)',
                get_device_properties=lambda index: types.SimpleNamespace(total_memory=16 * 1024 ** 3)))

    def test_bootstrap_requires_attached_inputs(self):
        bootstrap = extract('prepare_kaggle_environment')
        with tempfile.TemporaryDirectory(dir=ROOT / 'tests', prefix='kaggle-ready-') as folder:
            with self.assertRaisesRegex(RuntimeError, 'attach dataset/model Inputs'):
                bootstrap(Path(folder) / 'missing-input', Path(folder) / 'output')

    def test_bootstrap_fails_on_cpu_before_installation(self):
        bootstrap = extract('prepare_kaggle_environment')
        with tempfile.TemporaryDirectory(dir=ROOT / 'tests', prefix='kaggle-ready-') as folder:
            with patch.dict('sys.modules', {'torch': self.fake_torch(False)}), patch.dict(os.environ), \
                 patch('importlib.util.find_spec', return_value=object()), patch('subprocess.check_call') as install:
                with self.assertRaisesRegex(RuntimeError, 'CUDA is unavailable'):
                    bootstrap(folder, Path(folder) / 'output')
                install.assert_not_called()
                self.assertFalse((Path(folder) / 'output').exists())

    def test_bootstrap_sets_writable_caches_without_pip_if_present(self):
        bootstrap = extract('prepare_kaggle_environment')
        with tempfile.TemporaryDirectory(dir=ROOT / 'tests', prefix='kaggle-ready-') as folder:
            output = Path(folder) / 'output'
            with patch.dict('sys.modules', {'torch': self.fake_torch()}), patch.dict(os.environ), \
                 patch('importlib.util.find_spec', return_value=object()), patch('subprocess.check_call') as install, \
                 redirect_stdout(io.StringIO()):
                bootstrap(folder, output)
                install.assert_not_called()
                self.assertEqual(Path(os.environ['NUMBA_CACHE_DIR']), output / 'numba')
                self.assertTrue((output / 'matplotlib').is_dir())

    def test_bootstrap_installs_missing_audio_package_only(self):
        bootstrap = extract('prepare_kaggle_environment')
        with tempfile.TemporaryDirectory(dir=ROOT / 'tests', prefix='kaggle-ready-') as folder:
            with patch.dict('sys.modules', {'torch': self.fake_torch()}), patch.dict(os.environ), \
                 patch('importlib.util.find_spec', side_effect=lambda name: None if name == 'soundfile' else object()), \
                 patch('subprocess.check_call') as install, redirect_stdout(io.StringIO()):
                bootstrap(folder, Path(folder) / 'output')
                command = install.call_args.args[0]
                self.assertEqual(command[1:], ['-m', 'pip', 'install', '-q', 'soundfile'])

    def test_export_requires_completed_manifest_and_matches_checkpoint(self):
        export = extract('package_training_outputs')
        with tempfile.TemporaryDirectory(dir=ROOT / 'tests', prefix='kaggle-ready-') as folder:
            directory = Path(folder)
            train = directory / 'train'
            train.mkdir()
            with self.assertRaises(FileNotFoundError):
                export(train, directory)
            best = b'mock-best-checkpoint'
            (train / 'qbh_final_v5_best.pth').write_bytes(best)
            (train / 'qbh_final_v5_last.pth').write_bytes(b'mock-last-checkpoint')
            (train / 'final_manifest.json').write_text(json.dumps({'checkpoint_sha256': 'wrong'}), encoding='utf-8')
            with self.assertRaises(ValueError):
                export(train, directory)
            (train / 'final_manifest.json').write_text(json.dumps({'checkpoint_sha256': hashlib.sha256(best).hexdigest()}), encoding='utf-8')
            (train / 'history.json').write_text('[]', encoding='utf-8')
            (train / 'datasets.zip').write_bytes(b'dataset-must-not-be-exported')
            archive = export(train, directory)
            with zipfile.ZipFile(archive) as zipped:
                self.assertIsNone(zipped.testzip())
                self.assertEqual(zipped.read('train/qbh_final_v5_best.pth'), best)
                self.assertNotIn('train/datasets.zip', zipped.namelist())
                self.assertIn('train/history.json', zipped.namelist())
            self.assertFalse(archive.with_suffix('.zip.tmp').exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
