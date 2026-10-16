PHASE 1 — Sửa Dataset trước

Đây là phase quan trọng nhất, phải làm trước Model.

1.1. Sửa lỗi chỉ lấy chunks[0]

Hiện tại:

chunks = self.audio_preprocessor.slice_into_chunks(y)
cqt = self.cqt_extractor.extract(chunks[0])

Notebook hiện tại thực sự chỉ cache CQT của chunk đầu tiên.

Sửa thành
Audio
 ↓
10s / hop 3s
 ↓
chunk_1
chunk_2
chunk_3
...
 ↓
Random chunk trong training

Không cần cache toàn bộ nếu RAM hạn chế.

Nên dùng:

chunk_idx = random.randrange(len(chunks))
cqt = self.cqt_extractor.extract(chunks[chunk_idx])
Sau khi sửa phải kiểm tra:
1 audio
→ bao nhiêu chunks?
→ training lấy chunk nào?
→ mỗi epoch có thay đổi chunk không?
PHASE 2 — Sửa CQT pipeline

Hiện tại CQTExtractor luôn ép:

[1, 96, 83]

Notebook ghi rõ việc pad/cắt về 83 frames.

Điều này phù hợp cho encoder 10s, nhưng không được dùng cho DTW dài.

Tách thành 2 extractor
Training/Embedding
CQTFixed10s

output:

[1, 96, 83]
DTW
CQTVariableLength

output:

[96, T]

Ví dụ:

10s  → T ≈ 83
30s  → T ≈ 250
120s → T ≈ 1000
PHASE 3 — Kiểm tra Dataset Leakage

Trước khi train model mới, phải kiểm tra:

TRAIN songs
    ∩
VAL songs
    ∩
TEST songs

Mục tiêu:

TRAIN ∩ TEST = ∅
VAL ∩ TEST   = ∅
TRAIN ∩ VAL  = ∅

Notebook hiện tại tạo riêng train, val, test, nhưng phần code chưa chứng minh đầy đủ rằng song ID hoàn toàn độc lập giữa các split.

Tạo bảng:
Split	Songs	Humming files
Train	?	?
Val	?	?
Test	?	?

Và bắt buộc:

Overlap = 0
PHASE 4 — Xây Model Local Feature

Đây là thay đổi kiến trúc lớn nhất.

Hiện tại model:

CQT
 ↓
CoverHunter
 ↓
ASP
 ↓
128-D
 ↓
GRL

Code hiện tại extract_embedding() trực tiếp gọi feature_extractor(x) và normalize vector 128-D.

Phải chuyển thành:
CQT
 ↓
CMVN
 ↓
Conv2D Subsampling
 ↓
Conformer × 6
 ↓
[T × 128]

Không ASP ở đây.

PHASE 5 — Tách Global Embedding và Local Embedding

Model mới nên có hai output:

{
    "local_features": [B, T, 128],
    "global_embedding": [B, 128]
}

Trong đó:

Local

Dùng cho:

LAC
Domain Adaptation
DTW / local matching
Global

Dùng cho:

FAISS
coarse retrieval

Kiến trúc:

                  Conformer
                     │
              [B, T, 128]
                     │
          ┌──────────┴──────────┐
          │                     │
          ▼                     ▼
    Local Features         ASP Pooling
       [T,128]                 │
                               ▼
                             [128]
          │                     │
          ▼                     ▼
       LAC/GRL                FAISS

Đây là thay đổi mình ưu tiên cao nhất.

PHASE 6 — Implement Local Alignment Classifier

Hiện tại notebook chưa có LAC.

Dataset có Triplet, nhưng model chỉ có:

GRL
DomainClassifier

Thêm:
LocalAlignmentClassifier

Với:

K = number of songs/classes
L = number of local proxies
C = 128

Ví dụ:

W:
[K, L, 128]

Nếu dataset hiện tại 300 bài:

W = [300, 4, 128]
PHASE 7 — Implement MaxMean đúng vai trò

Hiện tại bạn đang tính:

Query chunks
      ×
Reference chunks
      ↓
cosine
      ↓
Max
      ↓
Mean

Trong evaluation hiện tại, đoạn:

sub_sims = sim_mat[:, c_indices]
max_per_query = np.max(sub_sims, axis=1)
song_scores[sid] = np.mean(max_per_query)

đang thực hiện chunk-level MaxMean.

Sau khi sửa model:

phải có:

Local Query Features
        ↓
Local Proxy Alignment
        ↓
MaxMean
        ↓
Song score

Sau đó trong báo cáo mới gọi nó là:

ByteHum-style Local Alignment + MaxMean

PHASE 8 — Sửa Domain Adversarial Network

Hiện tại:

128-D global embedding
 ↓
Linear 128→64
 ↓
Linear 64→1

Notebook xác nhận Domain Classifier hiện tại là MLP trên embedding 128-D.

Sửa thành:
Local Features [B,T,128]
             ↓
             GRL
             ↓
      Local Domain Head
             ↓
        Local Proxy
             ↓
          MaxMean
             ↓
       Domain BCE

Domain:

0 = Studio
1 = Humming
PHASE 9 — Sửa Loss

Sau khi có LAC + Local Domain:

Loss tổng

Đề xuất:

$$ L = L_{LAC} + \alpha L_{Triplet} + \beta L_{Domain} $$

Ví dụ ban đầu:

α = 0.5
β = 0.2

Không nên chốt giá trị ngay.

Sau đó chạy ablation:

A: LAC
B: LAC + Triplet
C: LAC + Domain
D: LAC + Triplet + Domain
PHASE 10 — Giữ lại Hard Negative Mining

Phần này không cần bỏ.

Notebook hiện tại đã có:

3B candidate embeddings
 ↓
exclude same song
 ↓
highest cosine
 ↓
hardest negative

Đây là thành phần tốt.

Chỉ cần sau này quyết định:

global triplet

hay:

local triplet
PHASE 11 — Sửa Deep Freeze

Hiện tại đã có:

Freeze frontend
Freeze 3/6 Conformer blocks
Train upper blocks

Giữ lại trước.

Nhưng không khẳng định ngay:

59.4% là tối ưu.

Chạy ablation:

Freeze 0
Freeze 1
Freeze 2
Freeze 3
Freeze 4

Chọn theo validation retrieval.

PHASE 12 — Training mới

Training pipeline:

Humming
   │
   ├───────────────┐
   ▼               ▼
Local Encoder   Global Encoder
   │               │
   ▼               ▼
LAC             Triplet
   │               │
   └───────┬───────┘
           │
           ▼
      Domain GRL
           │
           ▼
      Total Loss
Giữ:
AMP
AdamW
Gradient clipping
Dynamic GRL
checkpoint

Các thành phần này đã có trong notebook.

PHASE 13 — Sửa Training Loop

Có một việc bắt buộc:

Không được:
except Exception:
    continue

Hiện tại training có cơ chế bỏ qua batch lỗi.

Thay bằng:

failed_batches = []

except Exception as e:
    failed_batches.append({
        "epoch": epoch,
        "step": step_idx,
        "error": str(e)
    })
    raise

Trong giai đoạn debug:

Lỗi → dừng ngay.

Sau khi production ổn định mới cân nhắc graceful handling.

PHASE 14 — Validation phải dùng Retrieval

Đây là thay đổi rất quan trọng.

Hiện tại checkpoint được chọn theo:

Val Triplet Loss
+
Val Domain Loss

Nhưng mục tiêu cuối cùng là:

Top-1
Top-5
MRR

Do đó nên chọn:

Best checkpoint theo Val MRR hoặc Recall@1

thay vì chỉ Val Loss.

Ví dụ:

Epoch 1 → MRR 0.42
Epoch 2 → MRR 0.55
Epoch 3 → MRR 0.61 ← BEST
Epoch 4 → MRR 0.58
Epoch 5 → MRR 0.57

→ save Epoch 3.

PHASE 15 — Xây Evaluation chuẩn

Chia thành 4 benchmark.

Benchmark A — Baseline
CoverHunter Zero-Shot
Benchmark B
Conformer + Triplet
Benchmark C
Conformer + Triplet + Domain
Benchmark D
Conformer + LAC + Triplet + Domain
Benchmark E
D + DTW

Bảng cuối:

Model	R@1	R@5	R@10	MRR
CoverHunter				
+ Triplet				
+ Domain				
+ LAC				
+ DTW				

Đây sẽ là bảng quan trọng nhất của PBL6.

PHASE 16 — Sửa Two-Stage DTW

Hiện tại code gọi:

compute_cqt_dtw_distance(...)

và comment gọi là Fast-DTW.

Nhưng implementation thực tế là band-constrained dynamic programming.

Sửa tên trước:
Pitch-Aligned Band-Constrained DTW

Nếu muốn gọi FastDTW, phải implement đúng FastDTW.

PHASE 17 — Sửa lỗi 30s/120s

Hiện tại bạn giới hạn gallery 120s:

y_s = y_s[:16000 * 120]

nhưng sau đó:

cqt_full = cqt_ext.extract(y_s[:16000 * 30])

Trong khi cqt_ext lại ép output về 83 frames.

Sửa thành:
Gallery embedding:
120s
 ↓
10s chunks
 ↓
embeddings

DTW:

Query
 ↓
Variable CQT
Reference
 ↓
Variable CQT

Không dùng fixed 83-frame extractor cho DTW.

PHASE 18 — Sửa Pitch Alignment

Hiện tại:

max_pitch_search=6

Trong khi project claim:

±12 semitones

Không nên để hai con số mâu thuẫn.

Đề xuất:

Không nhất thiết tăng DTW lên ±12 ngay.

Thử:

Embedding:
±12 augmentation

DTW:
±6

Sau đó benchmark:

DTW ±4
DTW ±6
DTW ±12

Chọn kết quả/latency tốt nhất.

PHASE 19 — FAISS sửa sau cùng

FAISS hiện tại:

IndexFlatIP

là đủ cho 300 bài.

Đừng mất thời gian tối ưu FAISS trước khi model đúng.

Sau khi model ổn:

300
1K
10K
100K

benchmark:

FlatIP
HNSW
IVF
PHASE 20 — Sửa latency benchmark

Hiện tại search_humming() đo toàn bộ thời gian từ đầu đến cuối.

Tách:

Audio loading
CQT
Model inference
FAISS
MaxMean
DTW
Total

Ví dụ output:

Audio preprocessing : 3.2 ms
CQT                  : 8.4 ms
Conformer            : 7.1 ms
FAISS                : 0.3 ms
MaxMean              : 1.7 ms
DTW                  : 14.2 ms
--------------------------------
Total                : 34.9 ms
PHASE 21 — Benchmark pitch/tempo

Đừng claim:

Pitch invariant ±12

chỉ vì augmentation.

Tạo test:

Pitch:
-12 -10 -8 -6 -4 -2 0 +2 +4 +6 +8 +10 +12

Tempo:

0.8
0.9
1.0
1.1
1.2

Xuất:

Pitch shift vs Recall@1
Tempo vs Recall@1
PHASE 22 — Benchmark Domain Adaptation

Cần chứng minh GRL thật sự có tác dụng.

Tạo:

Studio → Studio
Studio → Humming
Humming → Humming

và:

Before DA
After DA
PHASE 23 — t-SNE chỉ dùng để minh họa

Notebook đã có t-SNE trước/sau DA.

Giữ lại.

Nhưng không dùng t-SNE làm bằng chứng chính.

Bằng chứng chính:

Recall@K
MRR
Domain accuracy
Latency

t-SNE chỉ là:

qualitative visualization.

PHASE 24 — Sửa Benchmark Summary

Cuối notebook hiện có fallback:

0.40%
4.30%
0.0162
...

nếu biến chưa tồn tại.

Phải bỏ hoàn toàn.

Nếu chưa chạy:

N/A

Không được hard-code số kết quả.