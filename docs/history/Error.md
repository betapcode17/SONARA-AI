Chỉ lấy chunks[0], khiến các đoạn còn lại của audio không được sử dụng khi train.
CQT training đang cố định về [1, 96, 83], không phù hợp khi cần xử lý chuỗi có độ dài thay đổi cho DTW.
CQT dùng cho embedding và CQT dùng cho DTW chưa được tách riêng.
Chưa kiểm tra đầy đủ data leakage giữa Train/Validation/Test theo song ID.
Backbone hiện tại sau ASP chỉ tạo global embedding 128-D, chưa giữ lại chuỗi local feature [T, 128].
Chưa có Local Alignment Classifier (LAC) đúng theo kiến trúc ByteHum.
MaxMean hiện tại là chunk-level MaxMean, chưa phải Local Alignment + MaxMean của ByteHum.
Domain Classifier hiện tại nhận global 128-D embedding, chưa hoạt động trên local feature sequence.
Domain Adaptation hiện tại chưa hoàn toàn tương ứng với Local Domain Classifier + GRL của ByteHum.
Kiến trúc hiện tại chưa thể gọi là ByteHum reproduction, nên nên mô tả là ByteHum-inspired nếu chưa sửa LAC/Local Domain.
Chưa có đầy đủ pipeline Local Feature → LAC → Local Domain Adaptation → MaxMean.
Pitch Shift ±12 semitones mới là data augmentation, chưa đủ bằng chứng để khẳng định model đạt true pitch invariance ±12.
Tempo augmentation 0.8–1.2× mới là augmentation, chưa chứng minh tempo invariance ±20%.
Chưa có benchmark riêng cho pitch shift từ -12 → +12 semitones.
Chưa có benchmark riêng cho tempo từ 0.8× → 1.2×.
Grid Distortion của ByteHum chưa được reproduce đầy đủ; hiện đang dùng Pitch Shift + Tempo + SpecAugment thay thế.
DTW hiện tại thực chất là band-constrained DTW, chưa phải FastDTW chuẩn.
Code đang gọi là Fast-DTW nhưng implementation chưa tương ứng với thuật toán FastDTW.
DTW đang sử dụng CQT bị ép về 83 frames, làm mất thông tin temporal length thực tế.
Gallery 30/120 giây chưa được chuyển thành chuỗi CQT variable-length đúng nghĩa cho DTW.
Pitch search của DTW chỉ [-6,+6], không thống nhất với mục tiêu pitch invariance ±12.
FAISS IndexFlatIP hiện chưa thực sự cần thiết để chứng minh scalability với database chỉ vài trăm bài hát.
Chưa benchmark scalability của retrieval ở các quy mô 1K/10K/100K bài hát.
Chưa tách riêng latency của Audio Loading / CQT / Model / FAISS / MaxMean / DTW.
Mục tiêu <50 ms/query hiện chưa có benchmark end-to-end đủ chặt chẽ để chứng minh.
Evaluation hiện giới hạn một số thí nghiệm ở 150 queries, chưa phù hợp làm final benchmark nếu toàn bộ test set lớn hơn.
Checkpoint hiện chủ yếu được lựa chọn dựa trên Validation Loss, chưa lựa chọn theo metric retrieval như Recall@1/MRR.
Chưa có ablation để chứng minh hiệu quả của Triplet Loss.
Chưa có ablation để chứng minh hiệu quả của Domain Adversarial Training.
Chưa có ablation để chứng minh hiệu quả của LAC.
Chưa có ablation cho chiến lược Deep Freeze 0/1/2/3/4 blocks.
Chưa có ablation đầy đủ cho pipeline Baseline → Triplet → Domain → LAC → DTW.
except Exception: continue có thể âm thầm bỏ qua query/batch lỗi và làm sai số liệu evaluation/training.
Cần thống kê rõ Total / Successful / Failed queries khi evaluation.
Benchmark Summary đang có các fallback số liệu hard-code, không nên để xuất hiện trong kết quả nghiên cứu thật.
Nếu experiment chưa chạy, kết quả nên hiển thị N/A thay vì số giả định.
Chưa phân biệt rõ benchmark Seen Song và Unseen Song.
Chưa phân biệt rõ benchmark Seen Singer và Unseen Singer.
Chưa có benchmark khó nhất: Unseen Song + Unseen Singer.
Chưa chứng minh mục tiêu Top-1 >85%, Top-5 >95%, Top-10 >98%, MRR >0.88 trên một benchmark đủ lớn và chuẩn hóa.
Không nên gọi các mục tiêu trên là SOTA, mà nên gọi là target performance cho benchmark của project nếu chưa so sánh trực tiếp với paper.
Chưa có benchmark so sánh rõ ràng với ByteHum Origin / ResNet-IBN.
Chưa có benchmark rõ ràng giữa CoverHunter Zero-Shot và model Conformer-ByteHum sau adaptation.
Chưa có baseline đầy đủ để chứng minh Domain Adaptation thực sự làm giảm domain gap.
Chưa có phân tích riêng Studio → Humming trước và sau Domain Adaptation.
T-SNE chỉ nên dùng làm visualization, không nên dùng làm bằng chứng chính cho hiệu quả Domain Adaptation.
Cần kiểm tra lại tỷ lệ tham số thực sự được freeze thay vì mặc định khẳng định 59.4%.
Cần kiểm tra lại claim In-RAM Caching 20.000× nhanh hơn bằng benchmark thực tế thay vì chỉ suy luận từ thời gian đọc dữ liệu.
Cần benchmark thời gian Epoch 1 và Epoch 2+ để chứng minh hiệu quả caching.
Cần kiểm tra lại toàn bộ preprocessing CQT để bảo đảm đúng cấu hình CoverHunter thay vì dùng fallback khác.
Không nên trộn implementation fallback với implementation CoverHunter chính thức nếu mục tiêu là reproduce kết quả paper.
Cần lưu riêng Global Embedding [128] cho FAISS và Local Features [T,128] cho LAC/Domain/Alignment.
FAISS hiện reconstruct vector lại trong quá trình query; nên lưu sẵn gallery embeddings để truy xuất nhanh hơn.
Candidate retrieval và fine-grained reranking cần được tách rõ thành Stage 1 FAISS → Stage 2 DTW/Local Alignment.
Cần đánh giá riêng hiệu quả của Top-30 candidate retrieval trước khi chạy DTW.
Cần đo Recall@30 để biết FAISS có đưa đúng bài hát vào candidate set hay không.
Cần kiểm tra Top-30 có phải tối ưu hay không bằng thử nghiệm Top-10/20/30/50.
Cần chuẩn hóa cách gọi thuật toán: nếu không implement FastDTW thì gọi là Band-Constrained DTW/Sakoe-Chiba DTW.
Cần bổ sung benchmark noise để đánh giá humming trong môi trường thực tế.
Cần bổ sung confusion/error analysis để biết model sai ở loại humming, singer, pitch hoặc bài hát nào.
Cần giữ lại notebook hiện tại làm Baseline v1 trước khi sửa kiến trúc.
Phiên bản cuối nên được mô tả chính xác là Conformer-based ByteHum-inspired QbH Domain Adaptation nếu không reproduce toàn bộ ByteHum.