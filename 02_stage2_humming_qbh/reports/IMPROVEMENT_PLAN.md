# Conformer-ByteHum improvement plan

## Baseline

Full gallery: 300 original recordings, 774 test queries. Kaggle v4 fused MRR
0.263029, Recall@1 0.171835, Recall@10 0.431525. Local 4 GB fused MRR 0.235216.
These are observations, not targets to tune against. Existing checkpoints and
evaluation_full artifacts must remain unchanged.

## Step 1: Protocol control

- Compare 45 and 300 songs using the exact SAME original recordings and queries.
- Reuse the full-gallery feature cache, rather than encoding fragment references.
- Report micro and song-macro MRR, Recall@1/5/10/20, and query counts.
- Status: implementation in progress; do not interpret missing results as success.

## Step 2: Scoring before retraining

- Validation: all validation queries, train/val-only gallery (255 songs).
- Candidate configurations: exact baseline and windowed MaxMean, covering
  10/16/22/34 seconds; local/global fusion weights 0.5/0.75/1.0.
- Experimental embedding subsequence DTW uses nominal encoder sampling centers
  and overlap-averaged features, with no candidate gate. It is NOT pitch-DTW.
- Query feature caches and per-query JSON results are atomic/resumable; cache
  identity includes source stats, checkpoint SHA256, runtime/pipeline or code.
- Only a complete validation run may choose a configuration and unlock test.
- Accept a change only if validation improves; keep baseline if alternatives fail.
- Status: implementation in progress.

## Step 3: Reference representations (conditional)

- Audit wrong song/version labels and silent/missing vocal tracks first.
- Create source-separated vocals from FULL original recordings, never use only
  hand-selected reference fragments as a full-gallery replacement.
- Keep raw and vocal embeddings under the same song ID; retain instrumentals.
- Evaluate raw, vocal, raw+vocal independently on validation before selecting.
- Add melody/pitch alignment only with voiced confidence and octave/transposition
  handling; do not apply a monophonic pitch tracker directly to every mixture.
- Candidate retrieval must be evaluated with ground-truth candidate recall.
- Status: pending Step 2 evidence; no claimed result.

## Step 4: Fine-tuning (conditional)

- Retain batch 4 + AMP; accumulation does not enlarge local in-batch negatives.
- Global memory already exists: add bounded, age-controlled local negative mining
  only after a dedicated loss ablation and regression tests.
- Audit independently sampled positive chunks; use matching-fragment windows.
- Evaluate raw/vocal training ratios and GRL strength one change at a time.
- Select checkpoints using the chosen fused scoring on a large validation gallery.
- New run names/checkpoint directories, no overwriting v4 weights.
- Status: pending validation-approved representation/scoring; not trained yet.

## Reporting rules

- No test-query tuning, no test-song training, no skipped failures.
- Subset smoke checks cannot select a model and are explicitly labeled.
- Keep DTW baseline, embedding-DTW experiment and FAISS candidate recall separate.
- A full evaluation is not an open-set benchmark or a public-paper reproduction.
