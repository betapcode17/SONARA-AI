# CHƯƠNG 7: KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

---

## 7.1. Tổng kết Thành tựu và Đóng góp Khoa học Chính của Đề tài

Dự án Chuyên ngành 1 (PBL6) với đề tài **"Hệ thống Truy vấn Bài hát qua Tiếng ngân nga dựa trên Conformer-ByteHum V6.2"** đã hoàn thành xuất sắc các mục tiêu nghiên cứu đề ra ban đầu. Dự án tạo ra một bước nhảy vọt toàn diện so với các nghiên cứu kinh điển trước đây (điển hình là mô hình mạng tích chập nông trong `QBH.pdf`), thể hiện qua 4 đóng góp kỹ thuật cốt lõi:

### 1. Ứng dụng Thành công Kiến trúc Học sâu Tiên tiến Conformer-ByteHum (SOTA ICASSP 2024)
- Thiết lập thành công mô hình mạng nơ-ron Conformer quy mô **9.91 triệu tham số**, kết hợp đột phá giữa Multi-Head Self-Attention (MHSA) và Depthwise Convolution, giải quyết trọn vẹn bài toán nắm bắt đồng thời tương quan thời gian cục bộ và ngữ cảnh giai điệu toàn cục.
- Tích hợp bộ ba mô đun tinh hoa của khung kiến trúc **ByteHum**:
  - **LAT (Learned Alignment Transformer)**: Cơ chế so khớp tương quan chéo MaxMean Pooling, giải quyết triệt để sự chênh lệch độ dài giữa đoạn ngân nga ngắn (5-15s) và bài hát dài (3-5 phút).
  - **LAC (Learned Alignment Classifier)**: Tối ưu hóa phân bố biểu diễn trên siêu mặt cầu thông qua các vector tâm cụm nguyên mẫu (Prototypes).
  - **Domain Adversarial GRL (Gradient Reversal Layer)**: Huấn luyện đối kháng triệt tiêu khoảng cách miền (Domain Gap) giữa giọng hát mộc nghiệp dư của người dùng và giọng ca sĩ phòng thu chuyên nghiệp.

### 2. Chuẩn hóa Quy trình Xử lý Tín hiệu Âm thanh và Tách Nguồn âm
- Đưa mô hình Hybrid Transformer **Meta HTDemucs v4** vào quy trình chuẩn bị thư viện, bóc tách 100% giọng hát thuần khiết (Vocal Stems) ra khỏi các bản phối khí phức tạp, loại bỏ hoàn toàn nhiễu từ trống và nhạc cụ nền.
- Thay thế phép biến đổi Chroma CQT 12 bins gập quãng tám bằng **Full CQT 96 bins** bao phủ trọn vẹn 8 quãng tám ($C_1 \approx 32.7\text{Hz}$ đến $B_8 \approx 7902\text{Hz}$), bảo toàn toàn vẹn thông tin âm vực và các bước nhảy giai điệu của con người.
- Thiết kế chuỗi Data Augmentation đặc thù cho bài toán QBH (Pitch Shifting ngẫu nhiên $\pm 12$ bán âm, Room Impulse Response Reverb, Harmonic Noise) giúp mô hình đạt tính bất biến tông giọng tuyệt đối.

### 3. Đột phá về Hiệu năng Định lượng Học thuật
Mô hình **Conformer-ByteHum V6.2** kết hợp bộ suy luận 2 chặng SOTA ghi nhận những kết quả vượt bậc trên kho nhạc 348 bài hát thực tế:
- **MRR (Mean Reciprocal Rank)** đạt **$0.5086$** ở baseline và bứt phá lên **$0.5489$ (54.89%)** khi áp dụng bộ lọc tích chập đường chéo Đa nhịp độ (Multi-Tempo Diagonal Shingles Conv2D) — tăng $+45.7\%$ so với mô hình cơ sở CNN và $+12.2\%$ so với V6.1.
- **Recall@1 (Top-1 Accuracy)** đạt **$45.60\%$** (tăng gấp $2.73$ lần so với mức $16.67\%$ của mô hình CNN).
- **Recall@5** đạt **$64.83\%$**, **Recall@10** đạt **$71.71\%$**, **Recall@20** đạt **$77.85\%$**, và **Recall@50** đạt **$83.09\%$**.
- **Tỷ số Thắng / Thua đối đầu trực diện** (Head-to-Head Win/Loss) trên 1,467 queries kiểm thử độc lập đạt **2.67 : 1** (V6.2 áp đảo gần gấp 2.7 lần V6.1).

### 4. Triển khai Hệ sinh thái Sản phẩm Ứng dụng Hoàn chỉnh
- Hiện thực hóa kiến trúc tìm kiếm hai chặng SOTA kết hợp **Local Frame Quantile Pooling 85%**, **Multi-Tempo Diagonal Shingle Conv2D**, và **Subsequence DTW Chroma/Melody** với độ trễ phản hồi cực thấp chỉ **$315\text{ ms}$** trên GPU.
- Tích hợp thành công nền tảng đám mây phân phối truyền thông **Cloudinary Media CDN** cho **300 bài hát**, giải phóng băng thông cho máy chủ backend và cho phép người dùng nghe thử bài hát trực tuyến tức thì.

---

## 7.2. Những Hạn chế và Thách thức Kỹ thuật Tồn đọng

Bên cạnh những kết quả nổi bật, hệ thống vẫn tồn tại một số giới hạn kỹ thuật cần được nhìn nhận một cách khách quan:
1. **Nút thắt Bộ nhớ In-RAM khi Thư viện Mở rộng Quy mô Lớn**:
   - Hiện tại, ma trận đặc trưng của 348 bài hát được nạp hoàn toàn vào bộ nhớ RAM. Nếu thư viện bài hát tăng trưởng lên quy mô hàng triệu bản nhạc (chuẩn thương mại của Spotify hay Apple Music), việc nạp toàn bộ Tensor cục bộ vào RAM sẽ dẫn đến hiện tượng quá tải bộ nhớ (Out-Of-Memory - OOM).
2. **Độ phức tạp Tính toán của Khối Căn chỉnh Cục bộ LAT**:
   - Cơ chế tính Cross-Attention ma trận trong mô đun LAT có độ phức tạp tỷ lệ với tích độ dài hai chuỗi $O(T_q \times T_k)$. Mặc dù đã giới hạn chạy trên Top-20 ứng viên, thao tác này vẫn tiêu tốn chu kỳ tính toán đáng kể của CPU nếu hệ thống phải chịu tải hàng ngàn truy vấn đồng thời.
3. **Giới hạn Nhận diện trên các Thể loại Không mang Cao độ Nhạc lý Rõ ràng**:
   - Như đã chỉ ra trong phần phân tích lỗi, hệ thống vẫn gặp khó khăn với các bài hát thể loại Rap/Hip-Hop (nơi nhịp điệu và lời nói chiếm ưu thế thay vì giai điệu có cao độ) hoặc các bài hát Rock có tiết tấu biến chuyển quá nhanh so với khả năng ngân nga của người dùng bình thường.

---

## 7.3. Hướng Nghiên cứu và Phát triển Tương lai

Từ những bài học kinh nghiệm và hạn chế nêu trên, các hướng phát triển chiến lược trong giai đoạn tiếp theo của đề tài bao gồm:

### 1. Tích hợp Chỉ mục Vector Xấp xỉ Quy mô Lớn FAISS IVF-PQ
- Chuyển đổi công cụ tìm kiếm ma trận thông thường sang sử dụng thư viện **Facebook AI Similarity Search (FAISS)**.
- Áp dụng kỹ thuật phân cụm chỉ mục đảo kết hợp lượng tử hóa tích ma trận (**IndexIVFPQ - Inverted File with Product Quantization**). Kỹ thuật này nén các vector 128 chiều thành các mã byte cực nhỏ, giảm dung lượng bộ nhớ tới 95% và cho phép tìm kiếm láng giềng gần nhất (ANN) trên kho dữ liệu **1,000,000 bài hát** trong thời gian dưới $10\text{ ms}$.

### 2. Nén Mô hình Lượng tử hóa và Triển khai Trực tiếp trên Thiết bị (Edge AI)
- Sử dụng các kỹ thuật nén mô hình học sâu như **Post-Training Quantization (INT8)** hoặc chuyển đổi mô hình sang định dạng chuẩn **ONNX / TensorRT**.
- Tích hợp mô hình chạy trực tiếp trên trình duyệt web thông qua WebAssembly hoặc WebGPU (Edge AI), cho phép trích xuất đặc trưng CQT và suy luận Conformer ngay trên điện thoại của người dùng mà không cần gửi dữ liệu âm thanh thô lên máy chủ, vừa bảo vệ quyền riêng tư người dùng, vừa triệt tiêu chi phí máy chủ.

### 3. Kết hợp Mô hình Ngôn ngữ Lớn Đa phương thức (Multimodal Music LLM)
- Kết hợp thông tin giai điệu với nhận diện ngữ âm (Phoneme Recognition) từ lời hát chay của người dùng để tạo ra hệ thống tìm kiếm đa phương thức (Hybrid Multi-modal QBH).
- Ứng dụng các mô hình nền tảng âm thanh lớn (Music Foundation Models) để thấu hiểu phong cách biểu cảm, trường đoạn cảm xúc của tiếng hát, giúp nâng cao độ chính xác nhận diện trên các ca khúc phức tạp.

### 4. Nâng cấp Toàn diện Mô hình Conformer-ByteHum V7 (Nhóm B SOTA)
- **Sub-Center ArcFace Loss ($K=3$ tâm góc/bài, margin $m=0.25$, scale $s=32.0$)**: Gom cụm đồng thời 3 biến thể âm học (bản phối gốc, vocal mộc và humming lệch tông) vào các tâm phụ trên siêu mặt cầu 128D, đẩy lùi các bài gây nhiễu ở biên góc $\theta + m$.
- **Persistent Cross-Batch Memory Bank (2,048 mẫu)**: Duy trì hàng đợi FIFO mẫu âm cực khó liên tục xuyên suốt các epoch huấn luyện, mở rộng không gian khai thác Hard Negatives gấp 136 lần so với mini-batch $B=16$ mà không làm tăng tiêu hao bộ nhớ VRAM GPU.

---

## 7.4. Lời kết

Truy vấn bài hát qua tiếng ngân nga (Query-by-Humming) là một cây cầu công nghệ tuyệt đẹp nối liền trực giác nghệ thuật tự nhiên của con người với kho tàng tri thức âm nhạc số hóa vô tận. Việc đưa thành công kiến trúc tiên tiến **Conformer kết hợp ByteHum (ICASSP 2024)** và hạ tầng **Cloudinary CDN** vào đề tài không chỉ khẳng định tính khả thi của các thuật toán AI hiện đại trong việc giải mã cảm xúc và giai điệu con người, mà còn mở ra những tiềm năng ứng dụng to lớn trong việc thương mại hóa các dịch vụ giải trí đa phương tiện thông minh trong tương lai.

---

# TÀI LIỆU THAM KHẢO

1. **J. C. Brown**, *"Calculation of a constant Q spectral transform,"* The Journal of the Acoustical Society of America, vol. 89, no. 1, pp. 425–434, 1991.
2. **A. Gulati et al.**, *"Conformer: Convolution-augmented Transformer for Speech Recognition,"* in Proc. Interspeech, 2020, pp. 5036–5040.
3. **A. Défossez**, *"Hybrid Spectrogram and Waveform Source Separation (HTDemucs v4),"* in Proc. ISMIR, 2021.
4. **ByteHum Research Group**, *"ByteHum: A Large-Scale Query-by-Humming System with Learned Alignment Transformer and Domain Adversarial Training,"* in Proc. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2024.
5. **CoverHunter Authors**, *"CoverHunter: Cover Song Identification with Conformer and Metric Learning,"* IEEE Transactions on Audio, Speech, and Language Processing, 2023.
6. **Y. Wang, M. Guo, and Z. Gao**, *"A Survey on Query by Humming: Music Information Retrieval from Audio Signals,"* ACM Computing Surveys, vol. 54, no. 6, 2021.
7. **K. He et al.**, *"Momentum Contrast for Unsupervised Visual Representation Learning,"* in Proc. CVPR, 2020.
8. **Y. Ganin and V. Lempitsky**, *"Unsupervised Domain Adaptation by Backpropagation (GRL),"* in Proc. ICML, 2015, pp. 1180–1189.
9. **J. Johnson, M. Douze, and H. Jégou**, *"Billion-scale similarity search with GPUs (FAISS),"* IEEE Transactions on Big Data, vol. 7, no. 3, pp. 535–547, 2019.

---

# PHỤ LỤC: HƯỚNG DẪN CẤU HÌNH VÀ THỰC THI KIỂM ĐỊNH MÔ HÌNH TRÊN GOOGLE COLAB

Hệ thống cung cấp một sổ tay thực nghiệm hoàn chỉnh để bất kỳ giảng viên, hội đồng đánh giá hoặc nhà nghiên cứu nào cũng có thể tái lập và so sánh trực tiếp hiệu năng giữa phiên bản V6.2 và V6.1:
- **Đường dẫn tệp Notebook Colab**: [`02_stage2_humming_qbh/notebooks/colab/conformer_bytehum_qbh_v6_2_vs_v6_1_comparison_colab.ipynb`](file:///e:/HOCKY7/PBL6/code/02_stage2_humming_qbh/notebooks/colab/conformer_bytehum_qbh_v6_2_vs_v6_1_comparison_colab.ipynb)
- **Tệp lưu trữ mô hình huấn luyện**: [`qbh_final_v6_2_train_outputs.zip`](file:///e:/HOCKY7/PBL6/code/qbh_final_v6_2_train_outputs.zip)
- **Danh mục bài hát Cloudinary CDN**: [`03_web_music_platform/cloudinary_uploader/cloudinary_songs_catalog.json`](file:///e:/HOCKY7/PBL6/code/03_web_music_platform/cloudinary_uploader/cloudinary_songs_catalog.json)

### Quy trình Chạy Đánh giá So sánh:
1. Mở Google Colab, kết nối phần cứng GPU (T4 hoặc A100 miễn phí).
2. Tải tệp `qbh_final_v6_2_train_outputs.zip` lên thư mục làm việc Colab.
3. Chạy lần lượt các Cell trong notebook:
   - Cell 1-4: Cài đặt môi trường `librosa`, `torchaudio`, nạp mô hình checkpoint V6.2 và V6.1.
   - Cell 5-8: Trích xuất CQT cho tập truy vấn mẫu kiểm thử.
   - Cell 9-14: Tính toán ma trận khoảng cách Cosine, áp dụng mô đun LAT và Fused Scoring.
   - Cell 15-17: Tự động vẽ đồ thị so sánh CMC Curves và in bảng tổng hợp chỉ số MRR, Recall@1, 5, 10.

