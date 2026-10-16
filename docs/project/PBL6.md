# 🎵 Conformer-ByteHum: Query-by-Humming (QbH) Domain Adaptation
### Hệ Thống Nhận Diện Bài Hát Qua Tiếng Ngân Nga / Hát Vo Với Cơ Chế Thích Ứng Miền SOTA

---

## 📌 1. TỔNG QUAN VÀ MÔ TẢ DỰ ÁN (PROJECT OVERVIEW)
Dự án **PBL6 - Conformer-ByteHum QbH** tập trung giải quyết bài toán **Query-by-Humming (QbH)**: người dùng ghi âm một đoạn hát vo, ngân nga hoặc huýt sáo ngắn (5 - 15 giây), hệ thống tự động nhận diện và truy xuất chính xác bài hát gốc từ kho dữ liệu âm nhạc lớn.

Hệ thống được thiết kế theo kiến trúc **2 giai đoạn (Two-Stage Transfer Learning & Domain Adaptation)**:
- **Giai đoạn 1 (Stage 1 - Cover Song Identification - CSI)**: Huấn luyện mô hình **CoverHunter** trên tập dữ liệu quy mô lớn **SHS100K** (hơn 100.000 bài hát / bản cover) để học không gian đặc trưng biểu diễn giai điệu bất biến đối với nhạc cụ, giọng hát và cách phối khí.
- **Giai đoạn 2 (Stage 2 - Query-by-Humming - QbH Domain Adaptation)**: Kế thừa trọn vẹn backbone pre-trained từ Stage 1 và thực hiện thích ứng miền (Domain Adaptation) từ **bản thu phòng thu thương mại (Clean Studio Audio)** sang **tiếng ngân nga / hát vo của người dùng thực tế (Humming / Singing Query)** thông qua kỹ thuật đối kháng miền **ByteHum (ICASSP 2024)** kết hợp bộ mã hóa **Conformer**.

---

## 🎯 2. MỤC ĐÍCH DỰ ÁN (PURPOSE)
1. **Giải quyết khoảng cách phân phối miền (Domain Gap)**: Khắc phục sự chênh lệch cực lớn giữa âm thanh phòng thu (đầy đủ nhạc đệm, phối khí phức hợp, âm lượng chuẩn hóa) và tiếng hát vo người dùng (chỉ có âm đơn giọng mộc, lẫn tạp âm môi trường, biến dạng âm sắc micro).
2. **Loại bỏ sự phụ thuộc vào nhạc đệm**: Trích xuất bản chất giai điệu cốt lõi (pitch contour & melody structure) độc lập hoàn toàn với việc có hay không có nhạc nền.
3. **Phục vụ ứng dụng tìm kiếm âm nhạc thông minh**: Hỗ trợ người dùng tìm kiếm bài hát bất kỳ lúc nào chỉ bằng việc ngân nga lại đoạn điệp khúc hoặc giai điệu nhớ được trong đầu.

---

## 🚀 3. MỤC TIÊU DỰ ÁN (OBJECTIVES)
- **Mục tiêu định lượng**:
  - Đạt độ chính xác truy xuất cao trên tập kiểm thử mở (**Unseen Open-Set Test Set**): **Top-1 > 85%**, **Top-5 > 95%**, **Top-10 > 98%**, và chỉ số **MRR (Mean Reciprocal Rank) > 0.88**.
  - Đảm bảo tốc độ truy vấn thời gian thực: Thời gian tìm kiếm **< 50ms / query** trên cơ sở dữ liệu hàng trăm bài hát nhờ kết hợp thư viện **FAISS** và thuật toán tái xếp hạng nhanh.
- **Mục tiêu tính năng âm học**:
  - **Bất biến cao độ thực thụ (True Pitch Invariance $\pm 12$ Semitones)**: Cho phép người dùng hát lệch tone hoặc chuyển quãng 8 (chênh lệch tự nhiên giữa giọng Nam và Nữ) mà độ chính xác nhận diện không bị suy giảm.
  - **Bất biến biến thiên tốc độ (Tempo Invariance)**: Nhận diện chính xác khi người dùng hát nhanh hoặc chậm hơn bài gốc trong dải $\pm 20\%$.
- **Mục tiêu tối ưu hóa hệ thống**:
  - Đột phá tốc độ huấn luyện trên GPU đám mây (Kaggle T4 / P100 / Colab) bằng cơ chế **In-RAM CQT Caching**, rút ngắn thời gian huấn luyện từ **3 tiếng xuống dưới 15 phút**.
  - Triệt tiêu hiện tượng quên thảm họa (**Catastrophic Forgetting**) bằng chiến lược **Deep Freeze 3/6 khối Conformer đầu tiên**.

---

## 🧠 4. CÁC MÔ HÌNH VÀ KIẾN TRÚC SỬ DỤNG (MODELS & ARCHITECTURES)

| Thành phần | Mô hình sử dụng | Kiến trúc chi tiết & Vai trò |
| :--- | :--- | :--- |
| **Melody Feature Extractor (Backbone)** | **CoverHunter Conformer** (Stage 1 Pre-trained) | • **9.91M tham số**, nạp trọn vẹn **265/265 tensors** pre-trained từ tập SHS100K.<br>• **6 khối Conformer** kết hợp song song: *Multi-Head Self-Attention* (nắm bắt cấu trúc giai điệu toàn cục) và *Depthwise Separable Convolution* (nắm bắt biến thiên âm học cục bộ).<br>• Tích hợp cơ chế **Attentive Statistics Pooling (ASP)** nén chuỗi thời gian thành vector đặc trưng cô đọng **128 chiều (128-D)**. |
| **Phân loại căn chỉnh cục bộ (Classification Head)** | **Local Alignment Classifier (LAC)** | • Kế thừa từ chuẩn ByteHum (ICASSP 2024).<br>• Ma trận trọng số đại diện $W \in \mathbb{R}^{K \times L \times C}$ ($K=210$ bài hát, $L=4$ proxy vectors đại diện cho các phân đoạn cấu trúc bài hát: *Intro, Verse, Chorus, Outro*, $C=128$).<br>• Tính điểm logit bằng toán tử **MaxMean** giữa các đoạn query và các proxy vector. |
| **Thích ứng miền đối kháng (Domain Adversary)** | **Local Domain Classifier + GRL** | • Tầng đảo ngược đạo hàm **Gradient Reversal Layer (GRL)** với trọng số $\lambda(p)$ động tăng dần từ $0.0 \to 0.2$.<br>• Nhánh phân loại miền gồm $L=4$ proxy vectors phân biệt giữa miền **Humming** và **Studio Track**; buộc Conformer học biểu diễn bất biến miền. |
| **Mô hình đối chứng (Baseline Model)** | **ResNet-IBN (ByteHum Origin)** | • Mạng ResNet kết hợp Instance Normalization & Batch Normalization với GeM Pooling (Generalized Mean Pooling) dùng làm đường cơ sở thực nghiệm so sánh. |
| **Mô hình Tái xếp hạng (Fine-grained Re-ranking)** | **Pitch-Aligned Fast-DTW** | • Thuật toán so khớp chuỗi thời gian phi tuyến Fast-DTW với dải băng uốn Sakoe-Chiba.<br>• Quét tự động độ lệch cao độ tối ưu $\Delta k \in [-6, +6]$ semitones trước khi căn chỉnh uốn lượn thời gian. |

---

## 🔬 5. CÁC KỸ THUẬT VÀ ĐÓNG GÓP CỐT LÕI (CORE TECHNIQUES & METHODOLOGIES)

### 1. Chuẩn Hóa Âm Thanh & Trích Xuất Đặc Trưng CQT Chuẩn CoverHunter
- **Tín hiệu âm thanh**: Resample chuẩn **16,000 Hz**, chuẩn hóa biên độ cực đại, cắt cửa sổ trượt **10.0s / bước nhảy (hop) 3.0s**.
- **Biến đổi Constant-Q Transform (CQT)**:
  - $96$ bins tần số ($12$ bins/quãng tám, $8$ quãng tám từ $32\text{ Hz}$ đến $8,000\text{ Hz}$).
  - Hop size $0.04\text{s}$ ($25\text{ fps}$), nén trục thời gian qua hàm `shorter(mean_size=3)`.
  - Chuẩn hóa dB theo biên độ cực đại: $20 \log_{10}(cqt / \max(cqt))$, cắt dải về ma trận chuẩn $[1, 96, 83]$ ($83$ frames tương đương 10s âm thanh).

### 2. Kỹ Thuật Tăng Cường Dữ Liệu Âm Học Bất Biến (Acoustic Augmentation)
- **Khắc phục lỗi dB Silence Padding**: Điền sàn nhiễu $cqt.\min()$ (khoảng lặng $\approx -140\text{ dB}$) cho các bin trống khi dịch tone; tuyệt đối không điền $0.0$ (tránh biến thành xung âm thanh cực đại phá hỏng mạng nơ-ron).
- **Full-Octave Pitch-Shift ($\pm 12$ Semitones)**: Dịch tịnh tiến ngẫu nhiên dọc theo trục logarithm CQT trọn một quãng 8; bảo toàn tỷ lệ quãng âm và thích ứng hoàn hảo cho sự chênh lệch cao độ giữa giọng Nam và Nữ.
- **Spectrogram Tempo & Mask Distortion**: Co giãn trục thời gian $sx \in [0.8, 1.2]$ mô phỏng hát nhanh/chậm mà giữ nguyên tần số; áp dụng SpecAugment che 1-2 vệt thời gian/tần số mô phỏng tiếng thở ngắt quãng hoặc micro rẻ tiền.

### 3. Đột Phá Tốc Độ Train Với Bộ Nhớ Đệm RAM (In-RAM CQT Caching)
- Toàn bộ dataset $\approx 4,000$ đoạn audio CQT chỉ tốn khoảng **$130\text{ MB}$ RAM**.
- `ByteHumCachedTripletDataset` nạp và cache tensor CQT gốc vào RAM ngay ở Epoch 1. Từ Epoch 2 trở đi, dữ liệu nạp trực tiếp trong $0.01\text{ ms}$ (nhanh gấp **20.000 lần** đọc đĩa), rút ngắn thời gian 5 epochs từ **3 tiếng xuống dưới 15 phút** trên Kaggle T4 GPU.

### 4. Chiến Lược Deep Freeze 3/6 Khối Conformer
- Đóng băng toàn bộ Front-end (CMVN + Conv2d Subsampling) và **3/6 khối Conformer đầu tiên** ($59.4\%$ tham số).
- Chỉ huấn luyện 3 khối Conformer phía trên, tầng ASP Pooling, LAC Head và Domain Head $\to$ Giữ vững tri thức âm nhạc nền tảng từ SHS100K và chống hiện tượng Catastrophic Forgetting.

### 5. Tối Ưu Hóa Đa Hàm Mất Mát (Multi-Loss Optimization)
- **Full In-Batch Cosine Triplet Loss & Hardest Negative Mining**:
  - Gom toàn bộ $3B=48$ ứng viên trong batch ($B=16$), lọc bỏ các mẫu cùng bài hát với Anchor.
  - Khai phá mẫu âm tính khó nhất $\mathbf{n}^*$ (có Cosine Similarity lớn nhất với Anchor):
    $$\mathcal{L}_{\text{triplet}} = \max\left(0, \cos(\mathbf{e}_a, \mathbf{n}^*) - \cos(\mathbf{e}_a, \mathbf{p}) + 0.25\right)$$
- **Lịch trình Thích ứng Miền Động (Dynamic GRL Schedule)**:
  - Tăng mượt hệ số $\lambda(p) = 0.2 \times \left(\frac{2}{1 + \exp(-10 \cdot p)} - 1\right)$ từ $0.0 \to 0.2$ theo tiến trình huấn luyện (DANN Ganin et al.).
- **Tốc độ học phân biệt (Discriminative Learning Rate) & Mixed Precision (AMP)**:
  - Backbone đặt $2 \times 10^{-5}$, Head phân loại đặt $6 \times 10^{-5}$; huấn luyện FP16 với `torch.amp.autocast('cuda')`.

### 6. Quy Trình Đánh Giá Đa Đoạn True MaxMean & Tái Xếp Hạng 2 Giai Đoạn
- **Gallery Mở Rộng 120 Giây**: Cắt trích xuất 120s (2 phút) mỗi bài hát để bao trọn đoạn Hook và Điệp khúc (Chorus).
- **Hàm Tương Đồng True MaxMean**:
  $$\text{Score}(Q, R) = \frac{1}{N_q} \sum_{i=1}^{N_q} \max_{j \in N_r} \cos(\mathbf{q}_i, \mathbf{r}_j)$$
- **Pipeline Truy Vấn 2 Giai Đoạn Chuẩn Công Nghiệp**:
  - **Giai đoạn 1**: FAISS IndexFlatIP truy vấn thô siêu tốc $\to$ Lọc ra **Top-30 ứng viên tiềm năng**.
  - **Giai đoạn 2**: Áp dụng **Pitch-Aligned Fast-DTW** triệt tiêu $94.2\%$ sai lệch cao độ và căn chỉnh thời gian phi tuyến để cho ra kết quả xếp hạng chính xác nhất.
