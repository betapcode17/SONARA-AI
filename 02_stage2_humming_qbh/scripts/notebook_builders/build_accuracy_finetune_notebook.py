"""Build a controlled accuracy-first fine-tuning notebook from the local 4 GB pipeline."""
import copy
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[2] / 'notebooks'
SOURCE = ROOT / 'local_4gb' / 'train' / "conformer_bytehum_qbh_local_4gb.ipynb"
TARGET = ROOT / 'local_4gb' / 'train' / "conformer_bytehum_qbh_accuracy_finetune_local_4gb.ipynb"

INTRO = """# Accuracy-first local fine-tuning (4 GB GPU)

This controlled continuation starts from the current best local v4 checkpoint,
keeps the model, losses, mining, and optimizer schedule, and changes the
positive-reference sampling from 80% vocal-separated to 20% vocal-separated
(80% original mix when available). Checkpoint selection uses all 300 gallery
tracks against the 773 validation queries. The test set is reserved for the
notebook's final report and must not be used to tune settings.

The run has its own name and checkpoint files; the existing model is preserved.
It is capped at 12 epochs with early stopping. Full-gallery validation is more
expensive than split-only validation on a 4 GB laptop GPU.
"""


def replace_once(source, old, new, context):
    if old not in source:
        raise ValueError(f"Could not find expected {context} text: {old!r}")
    return source.replace(old, new, 1)


def build():
    notebook = nbformat.read(SOURCE, as_version=4)
    cells = [nbformat.v4.new_markdown_cell(INTRO)]
    cells.extend(copy.deepcopy(notebook.cells))

    for cell in cells[1:]:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None

    settings = next(cell for cell in cells if cell.cell_type == "code"
                    and "LOCAL_NUM_EPOCHS = 30" in cell.source)
    settings.source = replace_once(settings.source, "LOCAL_NUM_EPOCHS = 30",
                                    "LOCAL_NUM_EPOCHS = 12", "epoch limit")
    settings.source = replace_once(settings.source,
        "os.environ['QBH_VALIDATION_GALLERY'] = 'val'",
        "os.environ['QBH_VALIDATION_GALLERY'] = 'full'", "validation gallery")
    settings.source = replace_once(settings.source,
        "STAGE2_ROOT = PROJECT_ROOT / '02_stage2_humming_qbh'",
        "STAGE2_ROOT = PROJECT_ROOT / '02_stage2_humming_qbh'\n"
        "INITIAL_LOCAL_CHECKPOINT = (STAGE2_ROOT / 'runs' / 'local_4gb' / 'checkpoints' /\n"
        "                            'conformer_bytehum_full_d128_timegrid_lat_v4_local4gb_best.pth')",
        "initial checkpoint path")

    dataset = next(cell for cell in cells if cell.cell_type == "code"
                   and "train_dataset = ByteHumCachedTripletDataset" in cell.source)
    dataset.source = replace_once(dataset.source, "vocal_ratio=0.80", "vocal_ratio=0.20",
                                  "training positive mix ratio")

    run_config = next(cell for cell in cells if cell.cell_type == "code"
                      and "RUN_NAME = 'full'" in cell.source)
    run_config.source = replace_once(run_config.source,
        "RUN_NAME = 'full'", "RUN_NAME = 'accuracy_posmix20'", "run name")
    run_config.source = replace_once(run_config.source,
        "ABLATIONS = {'full': dict(global_triplet=1., lac=1., lat=1., grl=.2, memory=True)}",
        "ABLATIONS = {'full': dict(global_triplet=1., lac=1., lat=1., grl=.2, memory=True),\n"
        "             'accuracy_posmix20': dict(global_triplet=1., lac=1., lat=1., grl=.2, memory=True)}",
        "ablation config")
    run_config.source = replace_once(run_config.source,
        "RUN_CONFIG.update(local_profile='4gb', microbatch= BATCH_SIZE, accumulation=LOCAL_ACCUMULATION_STEPS, freeze_blocks=LOCAL_FREEZE_BLOCKS, num_epochs=LOCAL_NUM_EPOCHS, songs_per_batch=SONGS_PER_BATCH, samples_per_song=SAMPLES_PER_SONG)",
        "RUN_CONFIG.update(local_profile='4gb', microbatch=BATCH_SIZE, accumulation=LOCAL_ACCUMULATION_STEPS,\n"
        "                   freeze_blocks=LOCAL_FREEZE_BLOCKS, num_epochs=LOCAL_NUM_EPOCHS,\n"
        "                   songs_per_batch=SONGS_PER_BATCH, samples_per_song=SAMPLES_PER_SONG,\n"
        "                   positive_vocal_ratio=0.20, initialization_checkpoint=str(INITIAL_LOCAL_CHECKPOINT))",
        "run provenance")

    model_setup = next(cell for cell in cells if cell.cell_type == "code"
                       and "model.freeze_lower_layers(num_blocks_to_freeze=LOCAL_FREEZE_BLOCKS" in cell.source)
    model_setup.source = replace_once(model_setup.source,
        "model = model.to(device)",
        "model = model.to(device)\n\n"
        "if not INITIAL_LOCAL_CHECKPOINT.is_file():\n"
        "    raise FileNotFoundError(INITIAL_LOCAL_CHECKPOINT)\n"
        "initial_payload = torch.load(INITIAL_LOCAL_CHECKPOINT, map_location='cpu', weights_only=False)\n"
        "if initial_payload.get('run_config', {}).get('pipeline_version') != 4:\n"
        "    raise ValueError('Initial checkpoint is not a pipeline-v4 model.')\n"
        "if initial_payload.get('train_songs') != train_dataset.songs:\n"
        "    raise ValueError('Initial checkpoint train-song mapping does not match this dataset.')\n"
        "model.load_state_dict(initial_payload['model_state_dict'], strict=True)\n"
        "print('Initialized from existing local best:', INITIAL_LOCAL_CHECKPOINT)\n"
        "del initial_payload",
        "model initialization")

    for index, cell in enumerate(cells):
        cell["id"] = f"accuracy-finetune-{index:03d}"
        if cell.cell_type == "code":
            compile(cell.source, f"{TARGET.name}:cell{index}", "exec")

    result = nbformat.v4.new_notebook(cells=cells, metadata=notebook.metadata)
    nbformat.validate(result)
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(result, TARGET)
    print(TARGET)


if __name__ == "__main__":
    build()
