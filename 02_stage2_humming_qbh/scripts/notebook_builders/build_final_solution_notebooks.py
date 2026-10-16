"""Generate self-contained final notebooks; builders are never runtime imports."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / 'notebooks'


def selected_definitions(source, names):
    tree = ast.parse(source)
    found = {}
    lines = source.splitlines(True)
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
            start = min([node.lineno] + [d.lineno for d in node.decorator_list]) - 1
            found[node.name] = ''.join(lines[start:node.end_lineno])
    if set(found) != set(names):
        raise ValueError(f'Missing definitions: {set(names) - set(found)}')
    return '\n\n'.join(found[name] for name in names) + '\n'


def code_cell(source, index):
    compile(source, f'final-cell-{index}', 'exec')
    return dict(cell_type='code', id=f'final-{index:03d}', source=source.splitlines(True),
                metadata={}, execution_count=None, outputs=[])


def build():
    # The existing notebook, not an older extracted .py, owns the architecture.
    original = json.loads((ROOT / 'kaggle' / 'archive' / 'conformer_bytehum_qbh_kaggle.ipynb').read_text(encoding='utf-8'))
    sources = [''.join(c['source']) for c in original['cells']]

    def find(marker):
        matches = [s for s in sources if marker in s]
        if len(matches) != 1:
            raise ValueError(f'Ambiguous source cell: {marker}')
        return matches[0]

    backbone = find('class ConformerEncoder(')
    audio = selected_definitions(find('class AudioPreprocessor:'), ['AudioPreprocessor'])
    cqt = find('class PyCqt:') + '\n' + selected_definitions(find('class CQTExtractor:'), ['CQTExtractor'])
    augment = '\n'.join(find(s) for s in ('def apply_cqt_pitch_shift(', 'class CQTPitchShiftAugmentation:', 'class SpectrogramTempoAndMaskDistortion:'))
    model = '\n'.join(find(s) for s in ('class GradientReversalFunction(', 'class LocalDomainClassifier(', 'def build_coverhunter_model(', 'class LocalAlignmentClassifier('))
    losses = find('class CrossBatchMemoryBank:')
    sampler = selected_definitions(find('class SongBalancedBatchSampler:'), ['SongBalancedBatchSampler'])

    # Reuse the verified exact, blocked scorer; no project import appears in outputs.
    builder_source = (Path(__file__).resolve().parent / 'build_local_evaluation_notebook.py').read_text(encoding='utf-8')
    engine = next(ast.literal_eval(n.value) for n in ast.parse(builder_source).body
                  if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'ENGINE' for t in n.targets))
    scorer = selected_definitions(engine, ['score_gallery'])
    runtime = (Path(__file__).resolve().parent / 'templates' / 'final_solution_runtime.py').read_text(encoding='utf-8')
    parts = {}
    for segment in runtime.split('# %% ')[1:]:
        title, body = segment.split('\n', 1)
        parts[title.strip()] = body
    definitions = [backbone, audio, cqt, augment, model, losses, sampler, scorer]
    source_digest = hashlib.sha256(('\n'.join(definitions) + runtime).encode()).hexdigest()
    settings = '''# Edit paths only, before Run All. Learning/scoring settings are frozen below.
INPUT_ROOT = '/kaggle/input'
OUTPUT_ROOT = '/kaggle/working/qbh_final_v5'
DATASET_OVERRIDE = None  # Exact dataset_split directory if multiple inputs contain it.
MUSIC_OVERRIDE = None   # Exact 2_music_database directory if ambiguous.
DATASET_ARCHIVE_OVERRIDE = None  # Optional ZIP/.bin containing both dataset directories.
STAGE1_OVERRIDE = None  # coverhunter_shs100k_pretrained.pth (training only).
CHECKPOINT_OVERRIDE = None  # qbh_final_v5_best.pth (evaluation only).
RESUME = True
RESUME_LAST_OVERRIDE = None  # Matching schema-2 LAST; includes the best model snapshot.
RESUME_BEST_OVERRIDE = None  # Legacy setting: BEST alone cannot resume; leave None.
EVAL_RESUME_DIR = None  # Prior final-v5 test output folder; auto-detected when unique.
'''
    introductions = {
        'train': '# QbH Final V5: Train on Kaggle T4\n\nOne fixed run from the SHS100K Stage-1 checkpoint. '
                 'Global triplet + LAC + LAT + GRL, song-balanced mining, AMP and bounded disk/RAM caching. '
                 'Select by exact fused MRR on all validation queries against train+val full recordings. '
                 'No test humming is decoded for training or model selection. Code is entirely inside this notebook.\n\n'
                 'Attach dataset_split, 2_music_database (unpacked or a ZIP archive), and coverhunter_shs100k_pretrained.pth as Kaggle inputs. '
                 'Select GPU T4 and run all cells. If interrupted, attach the schema-2 LAST checkpoint '
                 'and set RESUME_LAST_OVERRIDE; BEST is recovered from it. Resume restarts an incomplete epoch. '
                 'Older notebook checkpoints require their original code, not this revision.\n',
        'evaluation': '# QbH Final V5: Locked Test Evaluation\n\nAttach the same dataset and the completed train output; select GPU T4. '
                      'Load only qbh_final_v5_best.pth, evaluate every held-out test query against all full songs, '
                      'and save micro/macro MRR, Recall@K, per-query ranks and a song-bootstrap interval. '
                      'No candidate gating or test-time tuning. All implementation is embedded here.\n\n'
                      'A locked report is reused on Run All. An interrupted evaluation resumes the same query list. '
                      'The guarantee applies to this output directory, not to independent copies on other Kaggle sessions. '
                      'Retain the lock, progress and report artifacts to preserve the one-test protocol. '
                      'This is closed-set evaluation, not a calibrated open-set recognizer.\n'
    }
    for phase in ('train', 'evaluation'):
        contents = [settings + f"SOURCE_CODE_DIGEST = '{source_digest}'\n", parts['imports'], parts['paths']]
        contents.extend(definitions)
        contents.append(parts['common'])
        contents.extend([parts['train'], parts['plots']] if phase == 'train' else [parts['evaluation']])
        cells = [dict(cell_type='markdown', id='final-intro', metadata={}, source=introductions[phase].splitlines(True))]
        cells.extend(code_cell(source, i + 1) for i, source in enumerate(contents))
        document = dict(cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                                                  language_info=dict(name='python', version='3.11')),
                        nbformat=4, nbformat_minor=5)
        target = (ROOT / 'templates' / 'conformer_bytehum_qbh_final_train.ipynb'
                  if phase == 'train' else ROOT / 'kaggle' / 'evaluate' / 'conformer_bytehum_qbh_final_evaluation.ipynb')
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(document, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
        print(target)


if __name__ == '__main__':
    build()
