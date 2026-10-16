# Architecture figures

Generated with the built-in image_gen tool and visually checked against local
sources. These are explanatory redrawings, not screenshots of the original paper.

## Sources

- Paper: `../../../docs/references/ICASSP2024_bytehum.pdf`, Fig. 1 and sections 2.1-2.3.
- Implementation: `../../notebooks/local_4gb/train/conformer_bytehum_qbh_local_4gb.ipynb`.
- The local notebook derives its model from `conformer_bytehum_qbh_kaggle.ipynb`.

## Deliverables

- `bytehum_original_architecture.png`: original ByteHum training architecture.
- `conformer_bytehum_current_architecture.png`: actual Conformer-ByteHum v4,
  including the local notebook's inference protocol and 4 GB training profile.
- `bytehum_paper_figure1_reference.png`: rendered source figure for verification.

## Interpretation

The original paper uses ResNet-IBN, GeM and PCA-FC (as shown in Fig. 1).
Section 2.1 describes 512-dimensional local embeddings, one per 10-second audio
chunk, with a 3-second hop and 22,050 Hz resampling. The weakly supervised training
domains are original music and Demucs-separated vocals. The domain discriminator
uses a MaxMean proxy, and the losses are LAC, LAT and domain BCE. Its frequency
and time grid distortion differs from the implementation's time-only warp.

The implementation uses a shared CoverHunter Conformer pretrained on SHS100K.
Its input is 16 kHz audio transformed to CQT [B,1,96,83]. The encoder returns
20 local vectors of 128 dimensions per chunk. Thus the meaning of a local vector
differs from the paper's chunk-level vector. At embedding_dim=128 the local/global
projections are identities; no additional PCA-FC reduction is applied.

The first three Conformer blocks and frontend are frozen. The remaining blocks,
pooling and task heads are trainable. CMVN statistics remain fixed. The global
head uses Attentive Statistics Pooling with FC and L2 normalization. The local
domain head uses two Conv1D/GroupNorm/ReLU/Dropout stages, a framewise linear
classifier and mean aggregation. Its GRL lambda ramps up toward 0.2. Domain 1 is
real humming and domain 0 is reference/vocal, rather than the paper's two domains.

Default full-run loss: L_global + L_lac + L_lat + L_domain + 0.05 L_div.
The last term regularizes LAC prototypes. A positive may fall back to another
humming recording when its fragment lacks references/vocals; its domain then
becomes 1. The diagram depicts the normal reference/vocal positive path.

The local notebook computes exact local MaxMean and global maximum cosine scores
and reports a fused score of 0.5 local + 0.5 global. It does not use FAISS gating
or DTW in its test evaluator. The separate Kaggle evaluation notebook has further
retrieval options; those are not represented in the local diagram. The training
selection MRR is the mean of local/global MRR, which differs from fused-score MRR.

Microbatch 4 with accumulation 16 gives 64 triplets per full optimizer update.
Mining and BatchNorm still operate within each microbatch. The final incomplete
accumulation window can contain fewer samples.

## Prompt Briefs

Paper figure: publication-ready landscape scientific diagram on a white
background, restrained feature/loss/domain colors, explicit preprocessing,
ResNet-IBN/GeM/PCA-FC, local 512-D features, LAC/LAT/GRL-domain branches,
domain labels directed to BCE, and a separate local-MaxMean retrieval band.
Preserve the original paper architecture and avoid Conformer/global-head additions.

Current figure: matching landscape scientific diagram with a shared pretrained
Conformer, frozen/trainable blocks, tensor shapes, four objective branches and
prototype regularization. Distinguish training-only losses from embeddings used
in inference. Show the exact local notebook's local/global/fused ranking and
the microbatch/AMP/accumulation profile. Final edits corrected global-head usage,
prototype-diversity wiring, augmentation wiring and reference/domain labels.
