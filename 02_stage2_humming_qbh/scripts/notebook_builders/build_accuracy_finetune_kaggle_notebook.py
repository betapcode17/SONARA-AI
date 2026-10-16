"""Build the Kaggle GPU variant of the accuracy-first fine-tuning notebook."""
import copy
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[2] / 'notebooks'
SOURCE = ROOT / 'local_4gb' / 'train' / "conformer_bytehum_qbh_accuracy_finetune_local_4gb.ipynb"
TARGET = ROOT / 'kaggle' / 'train' / "conformer_bytehum_qbh_accuracy_finetune_kaggle.ipynb"

INTRO = """# Accuracy-first QbH fine-tuning on Kaggle GPU

This notebook continues from an existing Stage-2 `timegrid_lat_v4` best checkpoint.
Attach the QbH dataset, the Stage-1 CoverHunter checkpoint, and the Stage-2 best
checkpoint as Kaggle inputs. Validation uses the full music gallery and selects
the checkpoint by fused MRR. The held-out test is evaluated only at the end.

The default positive vocal-separated ratio is 20%. Compare other ratios in
separate runs; each run has an isolated output/checkpoint directory. Kaggle T4
defaults use 8 examples per microbatch and gradient accumulation to keep the
effective optimizer batch comparable to the local experiment.
"""

SETTINGS = '''from pathlib import Path
import os

if not ('KAGGLE_KERNEL_RUN_TYPE' in os.environ or os.path.exists('/kaggle')):
    raise RuntimeError('This notebook is configured for Kaggle. Select a Kaggle GPU accelerator.')

# Attach dataset-qbh and the Stage-2 best checkpoint in the Kaggle notebook inputs.
KAGGLE_STAGE2_CHECKPOINT_OVERRIDE = None  # Optional absolute path under /kaggle/input.
LOCAL_NUM_EPOCHS = 12
LOCAL_POSITIVE_VOCAL_RATIO = 0.20  # Try 0.20, 0.50, and 0.80 in separate runs.
LOCAL_SONGS_PER_BATCH = 4
LOCAL_SAMPLES_PER_SONG = 2
LOCAL_ACCUMULATION_STEPS = 8
LOCAL_EVAL_BATCH_SIZE = 8
LOCAL_QUERY_BATCH_SIZE = 8
LOCAL_GALLERY_BLOCK_SIZE = 8
LOCAL_FREEZE_BLOCKS = 3
LOCAL_RAM_CACHE_MB = 512
LOCAL_RESUME = True
LOCAL_EVALUATE_ONLY = False
LOCAL_COMPARE_KAGGLE = False

STAGE2_ROOT = Path('/kaggle/working/02_stage2_humming_qbh')
NUMBA_CACHE_DIR = STAGE2_ROOT / 'cache' / 'numba'
NUMBA_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ['NUMBA_CACHE_DIR'] = str(NUMBA_CACHE_DIR)
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ['QBH_PREWARM_CQT'] = '0'
os.environ['QBH_VALIDATION_GALLERY'] = 'full'
os.environ['QBH_RESUME_TRAINING'] = '1' if LOCAL_RESUME else '0'

if min(LOCAL_SONGS_PER_BATCH, LOCAL_SAMPLES_PER_SONG, LOCAL_ACCUMULATION_STEPS,
       LOCAL_EVAL_BATCH_SIZE, LOCAL_QUERY_BATCH_SIZE, LOCAL_GALLERY_BLOCK_SIZE) < 1:
    raise ValueError('Batch and accumulation sizes must be positive.')
if LOCAL_SONGS_PER_BATCH < 2:
    raise ValueError('Mining requires at least two songs per microbatch.')
if not 0.0 <= LOCAL_POSITIVE_VOCAL_RATIO <= 1.0:
    raise ValueError('LOCAL_POSITIVE_VOCAL_RATIO must be between 0 and 1.')

print('Kaggle output root:', STAGE2_ROOT)
print('Effective optimizer batch:', LOCAL_SONGS_PER_BATCH * LOCAL_SAMPLES_PER_SONG
      * LOCAL_ACCUMULATION_STEPS)
'''

PATH_DISCOVERY = '''def find_dataset_and_checkpoint():
    """Locate read-only Kaggle inputs and require a compatible Stage-2 starter."""
    if not Path('/kaggle/input').is_dir():
        raise FileNotFoundError('/kaggle/input is unavailable; run this notebook on Kaggle.')

    dataset_split_dir = None
    music_db_dir = None
    vocal_dir = None
    stage1_matches = []
    stage2_matches = []
    override = KAGGLE_STAGE2_CHECKPOINT_OVERRIDE
    if override is not None and not Path(override).is_file():
        raise FileNotFoundError(f'Stage-2 checkpoint override does not exist: {override}')

    for root, dirs, files in os.walk('/kaggle/input'):
        if dataset_split_dir is None and 'dataset_split' in dirs:
            dataset_split_dir = os.path.join(root, 'dataset_split')
        if music_db_dir is None and '2_music_database' in dirs:
            music_db_dir = os.path.join(root, '2_music_database')
        if vocal_dir is None and '4_vocal_separated' in dirs:
            vocal_dir = os.path.join(root, '4_vocal_separated')
        for filename in files:
            lower = filename.lower()
            full_path = os.path.join(root, filename)
            if lower == 'coverhunter_shs100k_pretrained.pth':
                stage1_matches.append(full_path)
            if lower in {
                'conformer_bytehum_full_d128_timegrid_lat_v4_best.pth',
                'conformer_bytehum_full_d128_timegrid_lat_v4_local4gb_best.pth',
            }:
                stage2_matches.append(full_path)
        if dataset_split_dir and music_db_dir and stage1_matches and (override or stage2_matches):
            break

    if dataset_split_dir is None or music_db_dir is None:
        raise FileNotFoundError(
            'Could not find dataset_split and 2_music_database under /kaggle/input. '
            'Attach the QbH dataset as a Kaggle input.'
        )
    for split_name in ('train', 'val', 'test'):
        if not os.path.isdir(os.path.join(dataset_split_dir, split_name)):
            raise FileNotFoundError(f'Missing dataset split: {dataset_split_dir}/{split_name}')
    if not stage1_matches:
        raise FileNotFoundError(
            'Missing coverhunter_shs100k_pretrained.pth. Attach the Stage-1 model input.'
        )
    if override:
        stage2_checkpoint = str(Path(override).resolve())
    elif len(stage2_matches) == 1:
        stage2_checkpoint = stage2_matches[0]
    elif not stage2_matches:
        raise FileNotFoundError(
            'Missing Stage-2 best checkpoint. Attach the Kaggle v4 best checkpoint as an input, '
            'or set KAGGLE_STAGE2_CHECKPOINT_OVERRIDE to its exact path.'
        )
    else:
        raise RuntimeError(
            'Found multiple Stage-2 checkpoints. Set KAGGLE_STAGE2_CHECKPOINT_OVERRIDE '
            'to the exact best checkpoint to avoid loading the wrong run: '
            + repr(stage2_matches)
        )

    if not os.path.isdir(vocal_dir or ''):
        vocal_dir = music_db_dir
    return dataset_split_dir, music_db_dir, vocal_dir, stage1_matches[0], stage2_checkpoint
'''

DATA_SETUP = '''DATASET_SPLIT_DIR, MUSIC_DB_DIR, VOCAL_DIR, STAGE1_CKPT, INITIAL_LOCAL_CHECKPOINT = find_dataset_and_checkpoint()
OUTPUT_DIR = str(STAGE2_ROOT / 'runs' / 'accuracy_finetune')
CHECKPOINT_DIR = str(Path(OUTPUT_DIR) / 'checkpoints')
Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
TRAIN_DIR, VAL_DIR, TEST_DIR = [str(Path(DATASET_SPLIT_DIR) / name) for name in ('train', 'val', 'test')]
print('Dataset split:', DATASET_SPLIT_DIR)
print('Music gallery:', MUSIC_DB_DIR)
print('Stage-1 checkpoint:', STAGE1_CKPT)
print('Stage-2 initialization:', INITIAL_LOCAL_CHECKPOINT)
print('Outputs:', OUTPUT_DIR)
'''


def replace_cell(cells, predicate, source, label):
    matches = [cell for cell in cells if cell.cell_type == "code" and predicate(cell.source)]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {label} code cell; found {len(matches)}")
    matches[0].source = source
    return matches[0]


def replace_once(source, old, new, label):
    if old not in source:
        raise ValueError(f"Could not find {label} text: {old!r}")
    return source.replace(old, new, 1)


def build():
    notebook = nbformat.read(SOURCE, as_version=4)
    cells = copy.deepcopy(notebook.cells)
    cells.insert(0, nbformat.v4.new_markdown_cell(INTRO))

    replace_cell(cells, lambda source: "PROJECT_ROOT_OVERRIDE = None" in source,
                 SETTINGS, "Kaggle settings")
    replace_cell(cells, lambda source: "def find_dataset_and_checkpoint():" in source,
                 PATH_DISCOVERY, "Kaggle input discovery")
    replace_cell(cells, lambda source: "DATASET_SPLIT_DIR, MUSIC_DB_DIR, VOCAL_DIR, STAGE1_CKPT =" in source,
                 DATA_SETUP, "Kaggle data setup")

    device_cell = next(cell for cell in cells if cell.cell_type == "code" and
                       "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')" in cell.source)
    device_cell.source = replace_once(
        device_cell.source,
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
        "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n"
        "if device.type != 'cuda':\n"
        "    raise RuntimeError('Enable a Kaggle GPU accelerator before training.')",
        "CUDA enforcement",
    )

    run_cell = next(cell for cell in cells if cell.cell_type == "code" and "RUN_NAME = f'accuracy_posmix" in cell.source)
    run_cell.source = replace_once(run_cell.source, "local_profile='4gb'", "local_profile='kaggle_gpu'",
                                  "Kaggle run profile")

    for index, cell in enumerate(cells):
        cell["id"] = f"accuracy-kaggle-{index:03d}"
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
            cell.metadata.pop("execution", None)
            compile(cell.source, f"{TARGET.name}:cell{index}", "exec")

    result = nbformat.v4.new_notebook(cells=cells, metadata=notebook.metadata)
    nbformat.validate(result)
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(result, TARGET)
    print(f"Created {TARGET}")
    print(f"Validated {sum(cell.cell_type == 'code' for cell in cells)} code cells")


if __name__ == "__main__":
    build()
