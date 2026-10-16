# QbH research experiments

V4 training/evaluation notebooks are in `../notebooks/kaggle/archive/`. Local
ablation notebooks are in `../notebooks/local_4gb/experiments/`. Build scripts
are in `../scripts/notebook_builders/`; dataset paths below are relative to Stage 2.

Both notebooks contain their runtime implementation. No project Python module is imported.

## Training

In `conformer_bytehum_qbh_kaggle.ipynb`, select `RUN_NAME` and `EMBEDDING_DIM` before model construction, then restart the kernel and run all cells. Supported dimensions are 128, 256 and 512. The 256/512 variants add learned projections over the pretrained 128-D backbone; they require training and do not create additional information simply by increasing dimension.

Run the same seed, splits, epoch budget and checkpoint-selection protocol for:

| RUN_NAME | Difference from full |
|---|---|
| full | Global triplet + LAC + LAT + GRL + memory |
| without_global_triplet | Global triplet weight zero |
| without_lac | LAC weight zero |
| without_lat | LAT weight zero |
| without_grl | Domain loss weight zero |
| without_memory | No cross-batch candidates |

Grid distortion is common to these runs. LAT uses query-directed frame MaxMean and different-song in-batch sequence mining. This is a Conformer frame-level variant, not a faithful reproduction of ByteHum chunk-level local features. Each run writes a separate checkpoint with its configuration and training song IDs. The first run is `conformer_bytehum_full_d128_grid_lat_v2_best.pth`.

The training memory bank stores global embeddings; LAT uses current-batch local sequences. Existing gradient accumulation remains four micro-batches, with correct scaling of an incomplete final group.

## Evaluation and full-song cache

Set `QBH_CHECKPOINT` to the chosen trained checkpoint. The evaluation notebook infers class count and projection dimension from `lac.prototypes`, then strictly loads the model. Full-song indexing batches encoder inputs and caches global embeddings, local features and complete CQT under `OUTPUT_DIR/full_song_cache`. Model tensors, source file identity and preprocessing settings determine cache identity. Rerunning indexing reuses these files.

Full-song DTW increases runtime. The current global mean-spectrum pitch-shift heuristic is not guaranteed to estimate the correct key for every subsequence; compare DTW on/off rather than assuming it improves accuracy.

## Public benchmark protocol

Set `QBH_BENCHMARK_MANIFEST` to a JSON file:

```json
{
  "dataset": "MIR-QBSH",
  "protocol": "original-audio",
  "gallery": [{"song_id": "00001", "path": "references/00001.wav"}],
  "queries": [{"song_id": "00001", "path": "queries/person01/00001.wav"}]
}
```

Paths are relative to the manifest or absolute. Supported dataset names: MIR-QBSH, IOACAS, MTG-QBH. Supply the complete dataset, including distractors, for a comparable benchmark. Missing files or ground-truth references cause an error, rather than reducing the denominator silently. The runner measures Top-1/5/10, MRR, candidate recall and p50/p95 latency using the FAISS gate and local reranking.

For MIR-QBSH, `prepare_mir_qbsh(extracted_root, output_dir)` can render reference MIDI using `pretty_midi` and produce a manifest. Install `pretty_midi` in the notebook environment before invoking it. The result is explicitly labeled `sine-synthesized-MIDI`; it must not be compared as the original-audio protocol used by ByteHum. The supplied archive is at `datasets_unused/mir_qbsh/MIR-QBSH.zip`. Its own index documents 48 MIDI references and 4,431 queries.

MTG-QBH metadata is described at https://www.upf.edu/web/mtg/mtg-qbh . Original reference recordings are not distributed with its queries. Map recordings using official IDs/class labels, not title guesses.

## Open-set calibration

Set `QBH_CALIBRATION_MANIFEST` to a JSON list of independent validation recordings:

```json
[
  {"path": "/data/validation/known.wav", "song_id": "gallery title", "split": "calibration"},
  {"path": "/data/validation/unknown.wav", "song_id": null, "split": "calibration"}
]
```

Unknown songs must actually be absent from the gallery. Exclude test recordings and preferably hold out singers as well. Choose the target false-accept rate before looking at test results. The default 1% is an empirical validation constraint, not a statistical guarantee with small calibration sets. The notebook reports sample counts, achieved false-accept rate and known acceptance, then writes `open_set_calibration.json`. Known acceptance does not by itself mean the returned song is correct.

`search_humming` returns `not_found` and no results below the calibrated threshold. Without calibration it explicitly reports `open_set_calibrated=False`. Changing checkpoint, index, gallery or search settings invalidates the calibration. The signature check reads/hashes the index and checkpoint; measure that overhead separately from the closed retrieval latency if used per request.

## Ablation and scale experiments

After independently training the six checkpoints, call `run_ablation_suite(checkpoints, manifest_path, output_path)` in the evaluation notebook, where `checkpoints` maps the six RUN_NAME values to their files. Each is evaluated with DTW off/on on the same manifest. Train separate full runs for dimensions 128/256/512, then run their benchmark and compare results.

`compare_index_backends(gallery_vectors, query_vectors)` measures flat, HNSW and IVF recall against exact neighbors on the same embeddings. IVF is skipped when there are fewer than `39*nlist` training vectors. This is neighbor recall; use the public benchmark runner for song-level recall. No automatic switch to an approximate index is made.

## Current execution status

- Unit checks cover LAT loss ordering/backward, nonlinear monotonic warping, strict 128/256/512 checkpoint round-trip, threshold ties and all three FAISS indexes.
- The existing `checkpoints/conformer_bytehum_finetuned_best.pth` is a legacy architecture: no `lac.prototypes`, different domain-classifier keys. It cannot supply valid results for these experiments.
- New training runs, full public benchmark, calibrated open-set evaluation and real ablation comparisons have not been measured yet. They require new checkpoints and independent calibration data. No performance improvement is claimed from synthetic tests.
