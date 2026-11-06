# Review QbH: độ chính xác và tốc độ khi mở rộng thư viện

Ngày rà soát: 24/10/2026. Phạm vi: `core`, checkpoint thực tế và các báo cáo Stage 2 đang có trên máy. Đây là review và đo riêng bộ chấm điểm; không train lại, không thay model đang phục vụ, không chạy lại test humming.

## 1. Kết luận

Hệ thống có nền tảng hợp lý cho demo, nhưng chưa có bằng chứng đủ để triển khai tìm kiếm chính xác trên thư viện lớn. Hai vấn đề cần giải quyết đồng thời là **khả năng phân biệt giai điệu** và **chi phí lấy ứng viên/căn chỉnh**. ANN/FAISS tăng tốc tìm vector; tự nó không cải thiện biểu diễn giai điệu. Tăng số chiều hoặc số epoch cũng không bảo đảm cải thiện.

Lộ trình phù hợp: đo đúng trên gallery lớn → tối ưu exact scorer và indexing → xây bộ lấy ứng viên có recall cao → căn chỉnh theo thời gian trên ứng viên → hiệu chuẩn từ chối → fine-tune dựa trên lỗi đã đo.

## 2. Những gì thực sự đang chạy

Đối chiếu trực tiếp `GET /api/state`:

- Checkpoint `core/checkpoints/qbh_v3_inference.pt`, schema 3, epoch 26, CPU.
- SHA256: `85d76048ba92cd1353036851f88b8c056f560a281949c5e8aa157ddc75aa47a9`.
- 90 bài, 7.567 vector chunk; 265 tensor backbone được nạp.
- 10 giây/chunk, hop 3 giây, mỗi chunk một vector chuẩn hóa L2, 128 chiều.
- Đang khởi động ở `embedding`; vocal indexed 0/90 vì phiên này dùng stem-root thay thế để mở demo sớm.
- `best_val_mrr=0.3533221263510492` là metadata checkpoint, không phải độ chính xác humming trên 90 bài hiện tại.

Đọc metadata checkpoint bằng unpickler vô hiệu hóa dựng đối tượng Torch, không thực thi mã checkpoint:

- 210 song ID trong train mapping; backbone Conformer 6 block, attention dim 256, freeze 3 block.
- `use_memory=false`, dù có khai báo `memory_capacity=4096`.
- Có trọng số global triplet 0,25; contrastive 0,25; LAT 1; LAC 1; GRL 0,2; prototype diversity 0,01.
- Microbatch theo 8 bài × 2 mẫu, accumulation 4; accumulation không làm các batch riêng tự chia sẻ negatives.
- Local unit là `chunk_10s`, scorer `maxmean`, không phải chuỗi frame local của notebook V4/V5.
- CQT dùng official log với epsilon 1e-9, không có floor cố định -80 dB.

Các thông tin huấn luyện trên là cấu hình lưu trong checkpoint; không thay thế log/benchmark độc lập của checkpoint đó.

## 3. Review bốn chế độ

| Chế độ | Cách chạy thực tế | Điểm mạnh | Hạn chế khi tăng số bài |
| --- | --- | --- | --- |
| Model v3 / `embedding` | Encode query; MaxMean cosine trên toàn bộ chunk của từng bài | Nhẹ nhất trong các nhánh xếp hạng; toàn bài đã có cache | Quét toàn gallery; không ràng buộc thứ tự; dễ khớp ngẫu nhiên một đoạn |
| Giai điệu / `melody` | Harmonic CQT/chroma; pYIN cho query; 12 pitch-class shifts; subsequence DTW toàn gallery | Có thứ tự và cho phép khác tempo/tông | Chroma gộp octave, chưa tách được lead melody khỏi hòa âm; DTW toàn bài/toàn gallery tốn chi phí |
| Kết hợp / `hybrid` | RRF: embedding 1 + melody 1 + vocal 0,5 khi đủ gallery; k=10 | Kết hợp bằng chứng khác nhau | Trọng số cố định; chưa chứng minh tốt hơn từng nhánh; DTW vẫn chạy toàn gallery |
| Model → DTW / `cascade` | Embedding chọn top 20 bài; chỉ DTW trong shortlist; xếp cuối bằng melody score | Giảm số bài cần DTW | Mất bài đúng ngoài top 20; vẫn quét embedding toàn gallery; một nhánh DTW yếu có thể đảo kết quả model tốt |

Đây là **một model học sâu và bốn chiến lược retrieval**, không phải bốn checkpoint được train độc lập. pYIN/DTW không phải model QbH đã fine-tune.

Chi tiết cần lưu ý:

- Mọi chế độ hiện đều chạy `self.extract(y)` và chấm embedding toàn gallery, kể cả `melody`.
- Khi có vocal embeddings, vòng lặp còn tính vocal score cả ở chế độ `embedding`, dù điểm đó không quyết định thứ hạng embedding.
- Cascade chỉ fallback về embedding khi **không có bất kỳ** melody score nào; chưa có chính sách dựa trên độ tin cậy từng query/ứng viên.
- Trong phiên hiện tại không có nhánh vocal indexed, nên không nên diễn giải hybrid như đang chạy đủ ba nhánh.

Nguồn: [engine.py](../../core/engine.py), [melody.py](../../core/melody.py).

## 4. Những checkpoint đã có và bằng chứng chất lượng

| Artifact | Bằng chứng trên máy | Diễn giải |
| --- | --- | --- |
| CoverHunter SHS100K pretrained | Có trọng số; có báo cáo zero-shot | Baseline pretrained, chưa phải model chuyên humming của demo |
| Model v3 inference | Có trọng số, contract và metadata epoch 26 | Đang phục vụ; chưa tìm thấy benchmark humming gắn với gallery 90 bài của demo |
| V4 Kaggle | Có checkpoint epoch 28 và báo cáo 774 test query | Có kết quả thật trong artifact, nhưng khác kiến trúc local/preprocessing/protocol so với demo v3 |
| V4 local 4 GB | Có BEST epoch 21 và báo cáo cùng bộ query | Có thể đối chiếu V4 Kaggle trong từng protocol; không phải checkpoint demo hiện tại |
| Legacy `best_bytehum` / `conformer_bytehum_finetuned_best` | Có file trọng số và metadata cũ | Không đủ căn cứ chọn để thay model triển khai bằng tên file hoặc val loss |
| V5 final Kaggle | Có notebook và kiểm tra cấu trúc; validation JSON ghi `gpu_training_executed=false` | Trong các vị trí đã rà soát chưa có checkpoint/metric V5 từ GPU để kết luận V5 tốt hơn |

Số liệu **lịch sử**, fused scoring của notebook V4; không phải RRF của chế độ hybrid trong demo:

| Checkpoint | Gallery / tham chiếu | Query | MRR | Recall@1 | Recall@10 |
| --- | --- | --- | --- | --- | --- |
| V4 Kaggle | 45 bài, split reference | 774 | 0,6173 | 50,90% | 84,37% |
| V4 Kaggle | 300 bài, full recording | 774 | 0,2630 | 17,18% | 43,15% |
| V4 local 4 GB | 45 bài, split reference | 774 | 0,5756 | 45,35% | 81,65% |
| V4 local 4 GB | 300 bài, full recording | 774 | 0,2352 | 15,25% | 40,83% |

SHA checkpoint và thứ tự/ID query khớp giữa hai protocol. Tuy nhiên, gallery 45 bài chỉ có **131 chunk tham chiếu**, còn gallery 300 bài có **22.935 chunk**, và trường `reference_source` thay đổi từ `split_reference` sang `full_music_or_manifest`. Vì vậy đây **không phải ablation chỉ thay số bài**. Có đồng thời thêm distractors, kéo dài references và thay nguồn tham chiếu. Cần chạy lại ablation trên validation với cùng bản ghi gốc/cache cho mọi kích thước.

Nguồn:

- [evaluation_test/evaluation_metrics.json](../../02_stage2_humming_qbh/runs/local_4gb/evaluation_test/evaluation_metrics.json)
- [evaluation_full/evaluation_metrics.json](../../02_stage2_humming_qbh/runs/local_4gb/evaluation_full/evaluation_metrics.json)
- [kaggle_ready_validation.json](../../02_stage2_humming_qbh/reports/kaggle_ready_validation.json)

Không đưa trực tiếp checkpoint V4 lịch sử vào `core`: artifact đó thiếu contract mà engine yêu cầu, preprocessing có floor -80 dB và local representation khác. Cần adapter, kiểm tra parity, cache/index mới và benchmark đúng protocol; đổi tên file không giải quyết tương thích.

## 5. Vì sao thư viện lớn làm khó hệ thống

### 5.1. MaxMean bỏ qua liên tục và thứ tự

Hiện tại `score(q,s) = mean_i max_j cosine(q_i, r_sj)`. Mỗi query chunk được tự chọn một reference chunk bất kỳ, có thể ở các đoạn xa nhau hoặc đảo thứ tự.

Nếu query dài không quá 10 giây, chỉ có một vector: phép tính thành `max_j cosine(q,r_sj)`. Bài dài có nhiều cơ hội xuất hiện đoạn giống ngẫu nhiên. Với cùng vector/query, thêm reference chunk chỉ có thể giữ hoặc tăng maximum; đó là cơ chế gây length bias cần đo. Thêm bài cũng tăng số đối thủ có điểm cao, dù weights model không đổi.

Khắc phục: ứng viên theo chunk có timestamp → gom các hit thành cửa sổ liên tục → rerank có thứ tự/thời lượng; kiểm tra score theo số chunk và duration trước khi áp dụng hiệu chỉnh độ dài.

### 5.2. Shortlist 20 bài có trần recall

Luôn có `Recall@1 cuối ≤ Candidate Recall@20` nếu kết quả cuối chỉ chọn trong top 20. DTW mạnh cũng không cứu được bài ngoài shortlist.

Báo cáo V4 full-gallery từng có local Recall@20 khoảng 51,55% (Kaggle); đây chỉ là ví dụ về mức rủi ro, **không phải Candidate Recall@20 đã đo của v3**. Trước khi chọn top 20 cho production, phải đo Candidate Recall@K của đúng checkpoint v3 và gallery mục tiêu.

FAISS Flat của thí nghiệm V4, 50 neighbors/query vector, candidate recall chỉ 57,62% trong protocol full-gallery. Thí nghiệm union global+local trên validation, 500 neighbors/vector, đạt 95,99% nhưng giữ trung bình 187,77/300 bài; 1.200 neighbors đạt 98,97% nhưng giữ 248,83/300 bài. Các số K này là **số neighbor vector**, không phải số bài cuối.

Điều đó cho thấy tăng K cứu recall bằng cách xét gần hết gallery; bottleneck không chỉ nằm ở cấu trúc index. Candidate recall thấp ngay với Flat cũng không thể quy cho sai số ANN.

Nguồn: [candidate_recall_summary.csv](../../02_stage2_humming_qbh/runs/local_4gb/retrieval_ablation_full_val/candidate_recall_summary.csv). Đây là protocol lịch sử với gallery 300 bài. Protocol chọn cấu hình mới cần giới hạn gallery validation ở train+val và distractors độc lập được phép dùng; không dùng humming test để chọn K.

### 5.3. Melody branch chưa phải lead-melody estimator

Harmonic mask giữ nhiều nhạc cụ có cao độ; chroma 12 chiều gộp octave. Hợp âm/đoạn nhạc đệm giống nhau có thể được chấm cao dù người dùng humming phần hát khác.

Khắc phục thử nghiệm: reference lead/vocal F0 có confidence, pitch interval/relative contour, giữ biểu diễn chroma làm nhánh bổ sung. RMVPE là ứng viên vocal-pitch estimator cần benchmark; không bảo đảm giải quyết bài instrumental, bè hát, lỗi octave hoặc lead melody ngoài vocal.

### 5.4. Huấn luyện chưa chắc chịu được nhiều distractor

Metadata v3 ghi memory bank tắt. Small microbatch + accumulation không tương đương mining trên một batch lớn. Cần negatives là các giai điệu gần nhau, không chỉ bài khác có âm sắc khác.

Khắc phục: mining từ toàn train gallery/queue có tuổi giới hạn, semi-hard → hard, loại cùng song/work/version, multi-positive contrastive và positive đúng đoạn. LAT hiện dùng MaxMean cũng chưa tương đương loss căn chỉnh có thứ tự; thử thêm sequence loss là một thí nghiệm riêng, không đổi tên LAT rồi coi như đã có temporal alignment.

Không kết luận chỉ bật memory sẽ tăng accuracy. Queue có embedding cũ và false negatives nên phải có ablation trên validation; không mining từ query/test đã khóa.

### 5.5. Chưa có cơ chế từ chối đã hiệu chuẩn

Runtime trả `rejection_threshold_calibrated=False`, diagnostics cũng `calibrated=False`. Có thể vẫn trả top 1 khi bài không ở gallery hoặc humming thiếu bằng chứng.

Khắc phục: calibration có query known/unknown độc lập, theo mode/checkpoint/gallery/index; dùng score, margin, pitch confidence và consistency. Báo cáo false accept rate, tỷ lệ chấp nhận và accuracy trên các query được chấp nhận. Từ chối giúp giảm trả sai; không tự nâng Recall@1 trên toàn bộ known queries.

### 5.6. Thêm một bài có thể làm mất cả nhánh vocal

Code chỉ cho vocal tham gia xếp hạng khi indexed vocal đủ toàn bộ gallery. Thêm bài chưa có stem có thể làm nhánh này ngừng tham gia cho tất cả bài. Hiện 0/90 là lựa chọn chủ động của phiên demo, không phải bằng chứng file stems bị mất.

Khắc phục: staging → chuẩn bị đủ features cần thiết → publish snapshot gallery/index atomically. Nếu cho phép branch thiếu dữ liệu, cần cơ chế fusion/calibration đã kiểm chứng với missing branches; không cộng bonus tùy tiện cho bài đã có stem.

## 6. Điểm nghẽn tốc độ và vận hành

| Vị trí | Hiện trạng | Hướng nâng cấp |
| --- | --- | --- |
| `engine.py:244` | Python loop, nhiều phép nhân ma trận nhỏ cho từng bài | Gộp matrix theo checkpoint, giữ offsets/song ID, exact scoring theo block và segmented max |
| `engine.py:267` | Melody/Hybrid DTW toàn bài cho toàn bộ candidates | DTW chỉ shortlist và cửa sổ thời gian được retrieval đề xuất; coarse-to-fine, confidence fallback |
| `engine.py:237` | Melody-only vẫn encode backbone/chấm embedding | Bỏ nhánh không dùng sau khi có regression so sánh kết quả |
| `engine.py:155`, `add_song` | Mỗi thêm bài gọi rebuild: hash/đọc lại gallery; stem còn có thể decode lại | Job ingest riêng, content hash/checksum manifest, chỉ index phần mới, publish index version |
| `run_demo.py:109` | Một lock chung; request khác có thể nhận 409 | Queue có giới hạn, inference worker/batching; tách ingest, search và phục vụ clip; snapshot immutable |
| `engine.py:223`, `static/app.js:37` | API/DOM và dropdown feedback chứa mọi bài | Phân trang, server-side search và chọn bài đúng qua autocomplete |

Cache toàn bài đã có trong core, không còn giới hạn chỉ 30 giây đầu. Tuy nhiên cache tránh **trích xuất lại**; không tránh quét mọi vector/DTW khi query mới.

### Phép đo mới, chỉ scorer

Chạy trên cache thật của 90 bài/7.567 vector, CPU/OpenBLAS 4 threads, 5 warmup + 40 lần/config. Query timing lấy vector có sẵn trong reference cache; không dùng làm test accuracy humming. Không bao gồm CQT, model, pYIN, DTW, HTTP hay đồng thời nhiều người dùng.

| Query chunks | Python loop hiện tại p50 | Matrix gộp exact p50 | Sai khác điểm lớn nhất |
| --- | --- | --- | --- |
| 1 | 2,852 ms | 0,242 ms | 0 |
| 3 | 3,555 ms | 0,989 ms | 0 |
| 8 | 3,900 ms | 0,884 ms | 5,96e-8 |

Đây là bằng chứng có thể tối ưu scorer ở cùng biểu diễn. Không chứng minh tốc độ end-to-end tăng tương ứng, cũng không phải kiểm tra parity cho mọi query/tie. Chưa thay scorer production trong lần review này. Phép đo lặp lại có thể biến động theo tải CPU.

Chi tiết máy/phép đo: [QBH_SCALABILITY_TIMINGS_2026-10-24.json](QBH_SCALABILITY_TIMINGS_2026-10-24.json). Script lặp lại: [qbh_scalability_timing_2026_10_24.py](qbh_scalability_timing_2026_10_24.py), chạy bằng Python trong `core/.venv`; script đọc cache và ghi kết quả mới vào `core/.cache/scalability-review-evidence.json`.

### Dung lượng khi mở rộng

Hiện trung bình 84,08 vector/bài. Ngoại suy cùng độ dài/hop, chỉ raw FP32 embedding 128-D:

| Bài | Vector xấp xỉ | Raw embedding |
| --- | --- | --- |
| 90 | 7.567 | 3,70 MiB |
| 1.000 | 84.078 | 0,040 GiB |
| 10.000 | 840.778 | 0,401 GiB |
| 100.000 | 8.407.778 | 4,009 GiB |
| 1.000.000 | 84.077.778 | 40,091 GiB |

Chưa tính vocal, melody/local sequences, timestamps, metadata, copies và ANN graph. Nếu thêm local 20 frame/chunk, riêng phần local có thể thêm khoảng 20 lần số byte vector chunk cùng dimension. Chỉ nên lưu dense local trên disk/mmap và nạp cho ứng viên, hoặc dùng temporal representation thưa hơn đã benchmark.

HNSW không phải index tiết kiệm RAM: có graph overhead. Chọn HNSW/IVF/PQ theo số **vector**, RAM, tốc độ và recall đã đo, không chỉ theo số bài.

## 7. Kiến trúc đề xuất

```text
Offline ingest:
Audio gốc → kiểm tra/nhận diện song-work-version → chunk embeddings + melody cần dùng
          → feature store + index có version → publish snapshot

Online:
Humming → kiểm tra chất lượng → query embedding / pitch confidence
        → tìm neighbor chunk → gom candidate song + timestamp windows
        → exact local/sequence rerank → DTW quanh cửa sổ ứng viên
        → fusion theo chất lượng → calibrated accept/reject → top bài + đoạn khớp
```

1. Giữ exact baseline gộp matrix cho quy mô vừa và để kiểm tra recall. `IndexFlatIP` trên vector L2-normalized là baseline cosine; lấy top vector từ Flat vẫn là candidate gate, không đồng nghĩa exact MaxMean full-song nếu cắt neighbor quá sớm.
2. Khi số vector/latency vượt ngân sách, thử HNSWFlat trên CPU hoặc IVF trên GPU/CPU, rồi IVF-PQ nếu RAM yêu cầu. Tune `efSearch`/`nprobe` trên validation.
3. Lấy nhiều neighbor qua các query chunk; gom theo song, chống một bài chiếm hết slots; vote theo thời gian để tìm cửa sổ nhất quán. Có thể union thêm melody retrieval nếu nhánh đó đủ tốt.
4. Đo K={20,50,100,200,500} **bài** sau gom. Có thể thử mở rộng K khi bằng chứng yếu; phải báo p95 latency của đường mở rộng. Không chốt một K trước khi đo candidate recall.
5. Rerank bằng exact/chunk-window + sequence matching. Chỉ dùng local features pre-pooling nếu exporter/model mới được huấn luyện và kiểm tra contract tương ứng; backbone frame output tự nó chưa chắc là melody descriptor tốt.
6. So sánh RRF hiện tại, fusion chuẩn hóa có trọng số và confidence-aware fallback. Không ép DTW yếu ghi đè model chỉ vì có một alignment hợp lệ về mặt kỹ thuật.
7. Thử 256-D sau khi xử lý retrieval/data; 512-D chỉ nếu gain đủ bù RAM/latency. Cần train/projection hợp lệ và rebuild toàn index. Không pad vector hoặc đổi config của checkpoint 128-D.

## 8. Thứ tự triển khai và tiêu chí chấp nhận

| Ưu tiên | Công việc | Cần train lại? | Tiêu chí |
| --- | --- | --- | --- |
| P0 | Benchmark gallery tăng dần, cùng bản ghi gốc và query, song/work/speaker split rõ | Không | Exact baseline; Recall@1/5/10, micro/song-macro MRR; p50/p95 riêng encode/retrieve/rerank |
| P0 | Exact matrix scoring + index incremental; không đưa cache miss/ingest vào online | Không | Score/rank parity với tie policy; cold/warm đo riêng |
| P1 | ANN/chunk candidates + temporal windows + adaptive K | Không với v3 embedding | Mục tiêu candidate recall ≥99% trên validation, chi phí đáp ứng ngân sách; đây là mục tiêu, chưa đạt |
| P1 | Rerank/fusion + calibrated reject | Không cho bản DSP thử nghiệm | Tăng metric validation so baseline, false accepts/coverage có báo cáo; không chọn bằng test |
| P2 | Dataset humming thực tế + negatives gần + queue/multi-positive/aligned positives | Có | Ablation từng nhóm thay đổi, validation gallery có đủ distractors, holdout người thu/bài hát |
| P2 | Lead melody extractor hoặc local sequence retriever mới | Có thể | Tăng candidate recall/accuracy trên vocal lẫn instrumental; không chỉ raw pitch accuracy |
| P3 | 256/512-D, IVF-PQ, batching/quantization | Tùy thay đổi | Đo trade-off accuracy/RAM/p95; index parity và rollback |

Benchmark cần có:

- Fixed query set; gallery lồng nhau, bài đúng luôn giữ nguyên, distractors khác work với query holdout.
- Validation 90/300/1k/10k bài khi **có dữ liệu thật**; không nhân bản vector để công bố accuracy ở quy mô lớn.
- Report Candidate Recall@K bài riêng với ANN neighbor recall và Recall@K cuối.
- Reference theo cùng full recordings; thêm các cửa sổ full song không đổi protocol giữa lần so sánh.
- Mốc ngắn 3–5s, 8–15s, dài hơn; humming, singing, whistle; noise/mic/tempo/key; known và unknown queries.
- Dedupe file, recordings, song/work/cover family; đánh giá theo work và theo exact recording nếu sản phẩm cần phân biệt phiên bản. Chỉ dùng MD5/SHA file không phát hiện được hai bản encode của cùng bài.
- Chọn model, K, fusion và threshold bằng validation/calibration. Test cuối khóa sau khi chọn; dữ liệu test lịch sử đã xem không được gọi là test mới hoàn toàn mù.
- Đo p50/p95 end-to-end và throughput dưới tải trên đúng CPU/GPU deployment; tách thời gian ingest/index, cold cache và warm query.
- Không dự đoán một con số accuracy ở 10k/100k bài từ 90 bài; mức đạt được phải đo thực tế.

V5 đã chuẩn bị một phần đúng hướng: validation trên full recordings train+val, positive đúng fragment, cache kiểm soát và memory global. Nhưng V5 hiện chưa có kết quả GPU trong artifact đã tìm thấy, vẫn cần benchmark và chưa tự giải quyết indexing/ANN/calibration của core.

## 9. Tài liệu gốc đối chiếu

- [Google Research: The Machine Learning Behind Hum to Search](https://research.google/blog/the-machine-learning-behind-hum-to-search/): học embedding cùng melody qua humming và bản ghi nhiều nhạc cụ; dữ liệu humming/augmentation là phần quan trọng. Không dùng kết quả Google làm metric cho dự án này.
- [FAISS: Guidelines to choose an index](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index): Flat baseline, trade-off HNSW/IVF/PQ, RAM và search parameters.
- [FAISS: MetricType and distances](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances): cosine tương ứng inner product sau chuẩn hóa cả query và gallery.
- [FAISS: Faiss indexes](https://github.com/facebookresearch/faiss/wiki/Faiss-indexes): bytes/vector, graph/ID overhead và approximate versus exhaustive.
- [Cross-Batch Memory for Embedding Learning](https://arxiv.org/abs/1912.06798): cơ sở thử memory mining, không bảo đảm mức tăng cho QbH.
- [Supervised Contrastive Learning](https://arxiv.org/abs/2004.11362): cơ sở loss multi-positive; cần kiểm chứng trên dữ liệu audio của dự án.
- [RMVPE](https://arxiv.org/abs/2306.15412): vocal pitch trong polyphonic music, không phải mô hình retrieval QbH hoàn chỉnh.

Mọi hướng nâng cấp ở đây là đề xuất có tiêu chí kiểm chứng; không phải tuyên bố đã đạt accuracy/tốc độ ở quy mô lớn.
