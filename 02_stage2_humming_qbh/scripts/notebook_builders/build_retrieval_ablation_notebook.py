"""Build the validation-first notebook; shared engine stays in experiments/."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / 'notebooks'


def cell(kind, source, index):
    result = {"cell_type": kind, "id": f"retrieval-ablation-{index:03d}", "metadata": {}, "source": source.splitlines(True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
        compile(source, f"ablation-cell-{index}", "exec")
    return result


def build():
    sources = [
        ("markdown", """# Conformer-ByteHum: Validation-First Retrieval Ablation

No training or checkpoint modification. Uses the original full-recording gallery
cache from the completed evaluator, plus resumable query embeddings and scores.
Validation uses all validation queries and only train/val gallery songs (255).
Test is evaluated only after a complete validation run locks a configuration.
The counterpart with 45 songs uses the SAME original recordings, not fragments.

The engine is `experiments/retrieval_ablation.py` (relative to Stage 2). CPU gallery tensors and
bounded similarity blocks keep this compatible with a 4 GB CUDA GPU. Embedding
DTW is an experimental temporal alignment, not a pitch-DTW baseline.
"""),
        ("code", """from pathlib import Path
import sys
import json
import importlib.util
import pandas as pd

PROJECT_ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
                     if (p / '02_stage2_humming_qbh').is_dir()), None)
if PROJECT_ROOT is None:
    raise FileNotFoundError('Open this notebook from the project directory.')
STAGE2_ROOT = PROJECT_ROOT / '02_stage2_humming_qbh'
MODEL_LABEL = 'kaggle_v4'  # 'local_4gb' is also supported.
MAX_VALIDATION_QUERIES = None  # A subset cannot select a model or unlock test.
RUN_EMBEDDING_DTW = False  # First run the faster window-scoring experiment.
RUN_LOCKED_TEST = True

engine_path = STAGE2_ROOT / 'experiments' / 'retrieval_ablation.py'
spec = importlib.util.spec_from_file_location('retrieval_ablation', engine_path)
engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)
"""),
        ("markdown", "## Validation: Full Original-Recording Gallery, No Test Songs\n"),
        ("code", """validation_report, selection = engine.run(
    model=MODEL_LABEL, split='val', max_queries=MAX_VALIDATION_QUERIES,
    dtw=RUN_EMBEDDING_DTW)
display(pd.DataFrame(validation_report['metrics']).T)
print('Same-audio 45-song control:', validation_report['same_audio_heldout_gallery'])
if selection is not None:
    print('Locked configuration:', selection['winner'], selection['configuration'])
else:
    print('Subset validation: no selection has been written.')
"""),
        ("markdown", "## Test: Locked Configuration, 300 Original Recordings\n"),
        ("code", """if RUN_LOCKED_TEST:
    if selection is None:
        raise RuntimeError('Complete all validation queries before running test.')
    test_report, _ = engine.run(model=MODEL_LABEL, split='test')
    display(pd.DataFrame(test_report['metrics']).T)
    print('Same-audio 45-song control:', test_report['same_audio_heldout_gallery'])
    print('Results:', test_report['output_directory'])
"""),
    ]
    doc = {"nbformat": 4, "nbformat_minor": 5,
           "metadata": {"kernelspec": {"display_name": "Python 3 (CUDA PyTorch)", "language": "python", "name": "python3"},
                        "language_info": {"name": "python", "version": "3.11"}},
           "cells": [cell(kind, source, i) for i, (kind, source) in enumerate(sources)]}
    target = ROOT / 'local_4gb' / 'experiments' / "conformer_bytehum_qbh_retrieval_ablation_local_4gb.ipynb"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    print(target)


if __name__ == "__main__":
    build()
