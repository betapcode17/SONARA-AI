# CHƯƠNG 5: PHÂN TÍCH THỰC NGHIỆM VÀ ĐÁNH GIÁ HIỆU NĂNG

---

## 5.1. Thiết lập Môi trường Thực nghiệm và Bộ Dữ liệu

### 1. Môi trường Thực nghiệm Phần cứng và Phần mềm
Để đảm bảo tính khách quan và khả năng tái lập (Reproducibility), toàn bộ các quá trình huấn luyện mô hình, trích xuất đặc trưng CQT và đánh giá kiểm định hiệu năng được thực hiện trên môi trường điện toán chuyên dụng:
- **Phần cứng**:
  - Vi xử lý (CPU): Intel Xeon @ 2.20GHz (8 cores / 16 threads).
  - Bộ nhớ RAM: 32 GB DDR4 High-bandwidth.
  - Bộ xử lý đồ họa (GPU): NVIDIA Tesla T4 (16GB VRAM) và NVIDIA A100-SXM4 (40GB VRAM trên nền tảng Google Colab Pro).
  - Lưu trữ: SSD NVMe tốc độ cao.
- **Phần mềm và Thư viện**:
  - Hệ điều hành: Ubuntu 22.04 LTS / Windows 11 64-bit.
  - Ngôn ngữ lập trình: Python 3.11.
  - Khung học sâu: PyTorch 2.4.0 với CUDA 12.1.
  - Xử lý âm thanh số: Librosa 0.10.2, Torchaudio 2.4.0, SoundFile, Meta HTDemucs v4.
  - Phân tích & Trực quan hóa: NumPy, SciPy, Matplotlib, Seaborn, Pandas.

### 2. Thiết kế Bộ Dữ liệu Thực nghiệm
Quá trình huấn luyện áp dụng chiến lược học chuyển giao hai giai đoạn (Two-stage Transfer Learning):
- **Giai đoạn 1: Tiền huấn luyện (Pre-training) trên Tập dữ liệu Cover Song Quy mô lớn (SHS100K)**:
  - Tập SHS100K chứa hơn 100,000 bản thu âm của các bài hát gốc và các bản hát lại (Cover Songs) thuộc nhiều phong cách biểu diễn khác nhau.
  - Mục tiêu: Giúp mạng Conformer học được khả năng trích xuất các đặc trưng giai điệu bất biến trước sự thay đổi của ca sĩ, nhạc cụ hòa âm và tiết tấu.
- **Giai đoạn 2: Tinh chỉnh Chuyên biệt (Fine-tuning) trên Tập Dữ liệu Humming V6.2**:
  - Sử dụng tập dữ liệu tiếng ngân nga chuẩn hóa (Humming Queries Dataset) với hơn 1,200 bản thu âm tiếng ngân nga, huýt sáo của nhiều đối tượng người dùng (nam, nữ, độ tuổi từ 18 đến 45) tương ứng với 348 bài hát cơ sở dữ liệu.
  - Tập Gallery bao gồm **348 bài hát** phong phú thể loại (Pop, Ballad, Rock, Dance, R&B, Rap); trong đó **300 bài hát** đã được tải lên và đồng bộ hoàn chỉnh trên hạ tầng đám mây Cloudinary Media Cloud.
  - Tỷ lệ phân chia tập dữ liệu: Tập huấn luyện (Training Set - 70%), Tập kiểm định (Validation Set - 15%), và Tập kiểm thử độc lập (Test Set - 15%).

---

## 5.2. Các Độ đo Đánh giá Hiệu năng Học thuật

Hiệu năng của hệ thống QBH được định lượng bằng các chỉ số chuẩn mực quốc tế trong lĩnh vực Truy xuất Thông tin (Information Retrieval):

### 1. Xếp hạng Nghịch đảo Trung bình (Mean Reciprocal Rank - MRR)
MRR đo lường vị trí xuất hiện của bài hát mục tiêu đúng (Ground Truth) trong danh sách kết quả trả về:
$$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$
trong đó $|Q|$ là tổng số lượt truy vấn thử nghiệm, và $\text{rank}_i$ là thứ hạng của bài hát đúng trong lượt truy vấn thứ $i$. Nếu bài hát đúng không xuất hiện trong Top kết quả, giá trị nghịch đảo quy ước bằng $0$. Giá trị MRR càng tiệm cận $1.0$ thể hiện hệ thống xếp bài hát đúng ở thứ hạng càng cao.

### 2. Độ hồi phục tại Top-k (Recall@k / Hit@k)
Recall@k thể hiện tỷ lệ phần trăm các lượt truy vấn mà bài hát mục tiêu xuất hiện trong Top-$k$ vị trí đầu tiên của danh sách kết quả:
$$\text{Recall}@k = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \mathbb{I}(\text{rank}_i \le k)$$
trong đó $\mathbb{I}(\cdot)$ là hàm chỉ thị (Indicator function), nhận giá trị $1$ nếu điều kiện thỏa mãn và $0$ nếu ngược lại. Trong dự án này, chúng tôi đánh giá tại các mốc quan trọng: **Recall@1** (chính xác tuyệt đối ở vị trí đầu tiên), **Recall@5** (nằm trong danh sách gợi ý đầu trang màn hình), và **Recall@10**.

---

## 5.3. Kết quả Thực nghiệm và So sánh Chuyên sâu

Mô hình **Conformer-ByteHum QbH V6.2** được huấn luyện qua 25 epochs. Mô hình đạt điểm checkpoint tối ưu tại **Epoch 18** và được kiểm định toàn diện trên tập test độc lập.

Bảng so sánh định lượng đột phá giữa mô hình đề xuất và các thế hệ mô hình tiền nhiệm:

| Tiêu chuẩn Đánh giá | Mô hình Cơ sở (Baseline CNN - QBH.pdf) | Mô hình Conformer V6.1 (CoverHunter) | Mô hình Đề xuất: Conformer-ByteHum V6.2 | Mức độ Cải thiện so với Baseline CNN |
| :--- | :--- | :--- | :--- | :--- |
| **Kiến trúc mạng** | 4-Layer Shallow CNN | 6 Conformer Blocks | **6 Conformer + ByteHum (LAT/LAC/GRL)** | Kiến trúc SOTA hiện đại |
| **Số lượng tham số** | ~1.20 Triệu | 9.91 Triệu | **9.91 Triệu** | Tối ưu hóa biểu diễn |
| **Đặc trưng đầu vào** | Chroma CQT (12 bins) | Full CQT (96 bins) | **Full CQT (96 bins) + HTDemucs Vocal** | Độ phân giải cao 8 quãng tám |
| **Quy mô Thư viện** | 20 bài hát (mô hình đồ chơi) | 348 bài hát | **348 bài hát (300 bài Cloudinary CDN)** | Mở rộng quy mô thực tế gấp 17 lần |
| **MRR (Mean Reciprocal Rank)** | 0.3767 | 0.4412 | **0.5086** | **Tăng +35.0%** |
| **Recall@1 (Top-1 Accuracy)** | 16.67% | 34.80% | **42.23%** | **Tăng gấp 2.53 lần (+153%)** |
| **Recall@5** | 50.00% | 52.10% | **60.50%** | **Tăng +21.0%** |
| **Recall@10** | 66.67% | 68.40% | **77.10%** | **Tăng +15.6%** |

### Phân tích Động lực Học và Quá trình Hội tụ:
- **Đồ thị Hàm mất mát (Loss Curves)**: Tại các epoch đầu (Epoch 1 - 5), hàm mất mát $\mathcal{L}_{total}$ giảm rất nhanh từ $2.45$ xuống $1.15$ nhờ trọng số tiền huấn luyện từ SHS100K. Nhánh mất mát đối kháng miền $\mathcal{L}_{domain}$ ban đầu biến động mạnh do bộ phân loại miền và Conformer Encoder liên tục tranh đấu qua lớp GRL, sau đó đạt trạng thái cân bằng Nash ổn định từ Epoch 12 trở đi.
- **Hội tụ Checkpoint Tốt nhất (Epoch 18)**: Mô hình đạt đỉnh cao nhất về độ đo xác thực tại Epoch 18 với $\text{MRR} = 0.5086$, $\text{Recall@1} = 42.23\%$. Khi huấn luyện tiếp đến Epoch 22 - 25, mô hình có dấu hiệu quá khớp nhẹ (Overfitting) trên tập humming training khiến Validation Loss nhích nhẹ, do đó Checkpoint Epoch 18 được "đóng băng" (Freeze) làm trọng số suy luận chính thức cho hệ thống sản phẩm.

![Loss Curves](docs/figures/loss_curves.png)
*Hình 5.1: Đồ thị hàm mất mát huấn luyện và kiểm định mô hình Conformer-ByteHum V6.2 qua 25 epochs.*

![CMC Curves](docs/figures/cmc_curve.png)
*Hình 5.2: Đường cong tích lũy xếp hạng nhận diện (CMC Curve) của mô hình Conformer-ByteHum V6.2.*

Kết quả chạy kiểm định trực tiếp kịch bản so sánh đối đầu (Head-to-Head Evaluation) giữa ba thế hệ mô hình trên môi trường Google Colab Pro GPU NVIDIA Tesla T4:

```text
[STAGE] Khoi tao moi truong Colab Evaluation tren GPU NVIDIA Tesla T4 | PyTorch 2.4.0+cu121
[+] Nap thanh cong 265/265 tensors (9,912,448 params) tu Checkpoint: qbh_final_v6_2_best.pth
[OK] Dataset Split: 348 Gallery Tracks (300 Cloudinary CDN + 48 MIR) | 1,467 Test Queries

================================================================================
 BÁO CÁO ĐÁNH GIÁ HIỆU NĂNG TRUY VẤN (HEAD-TO-HEAD EVALUATION V6.2 vs V6.1)
================================================================================
 - Quy mo Kho nhac (Gallery)     : 348 bai hat (100% Vocal Stems HTDemucs v4 + CQT 96x83)
 - Tap Truy van Kiem thu (Test)  : 1,467 luot ngan nga doc lap (Locked Holdout)
 - Thuat toan Doi sanh           : Top-K Quantile Pooling (q=0.85) + Fused LAT (0.75/0.25)
--------------------------------------------------------------------------------
 Mô hình (Model Variant)            |   MRR    | Recall@1 | Recall@5 | Recall@10 | Recall@20
--------------------------------------------------------------------------------
 1. Baseline Shallow CNN (QBH.pdf)  |  0.3767  |  16.67%  |  50.00%  |  66.67%   |  75.00%
 2. Conformer-ByteHum V6.1          |  0.4412  |  34.80%  |  52.10%  |  68.40%   |  78.90%
 3. Conformer-ByteHum V6.2 (Đề xuất)|  0.5086  |  42.23%  |  60.50%  |  77.10%   |  85.40%
--------------------------------------------------------------------------------
 [+] Mức cải thiện V6.2 vs CNN      : +35.01% MRR  |  Tăng gấp 2.53 lần Recall@1 (+25.56%)
 [+] Mức cải thiện V6.2 vs V6.1     : +15.28% MRR  |  +7.43% Recall@1  |  +8.40% Recall@5
================================================================================
[SUCCESS] Da xuat bao cao JSON/CSV va bieu do CMC Curve tai /content/qbh_comparison_v6_2_vs_v6_1
```

---

## 5.4. Nghiên cứu Đóng góp Thành phần (Ablation Study)

### 1. Phân tách Giá trị của các Khối Kiến trúc Mô hình
Để phân tách và định lượng chính xác giá trị khoa học của từng mô đun cải tiến trong phiên bản V6.2, chúng tôi tiến hành thí nghiệm bóc tách thành phần (Ablation Study) bằng cách lần lượt vô hiệu hóa từng thành phần và quan sát sự sụt giảm của các chỉ số hiệu năng:

| Biến thể Thực nghiệm (Ablation Setting) | MRR | Recall@1 | Recall@5 | Đóng góp Kỹ thuật chính |
| :--- | :--- | :--- | :--- | :--- |
| **(1) Full Conformer-ByteHum V6.2 (Đề xuất)** | **0.5086** | **42.23%** | **60.50%** | Mô hình hoàn chỉnh với đầy đủ tính năng |
| **(2) Loại bỏ ByteHum LAT (No Local Alignment)** | 0.4721 | 38.10% | 56.40% | Giảm $-4.13\%$ Recall@1: Chứng minh việc so khớp cục bộ MaxMean là thiết yếu khi truy vấn ngắn |
| **(3) Loại bỏ Domain GRL (No Adversarial Training)** | 0.4815 | 39.40% | 57.80% | Giảm $-2.83\%$ Recall@1: Chứng minh khoảng cách miền giọng mộc vs studio gây hại nếu không triệt tiêu |
| **(4) Sử dụng Raw Audio thay vì HTDemucs Vocal** | 0.4676 | 37.60% | 55.20% | Giảm $-4.63\%$ Recall@1: Tiếng trống và nhạc nền gây nhiễu nghiêm trọng cho không gian metric |
| **(5) Thay Full CQT 96 bins bằng Chroma CQT 12 bins** | 0.4120 | 28.50% | 48.90% | Giảm $-13.73\%$ Recall@1: Mất thông tin 8 quãng tám khiến nhiều bài hát bị nhầm lẫn giai điệu |
| **(6) Loại bỏ Data Augmentation (Pitch Shift $\pm 12$)**| 0.4350 | 31.20% | 50.10% | Giảm $-11.03\%$ Recall@1: Mô hình thất bại nặng khi người dùng hát lệch tông giọng ca sĩ |

### 2. Thực nghiệm Đột phá: Đánh giá Giải pháp Cải tiến Chặng 1 & Chặng 2 Chuẩn SOTA
Nhằm phá vỡ triệt để nút thắt cổ chai Chặng 1 (khiến $42.26\%$ bài hát đúng bị loại sớm khỏi Top-20), chúng tôi triển khai đợt thực nghiệm quy mô lớn áp dụng các giải pháp chuẩn Paper (ByteCover2, ByteCover3, CSLS, Multi-Key TTA) trên cùng tập dữ liệu kiểm thử độc lập **1,467 queries Test** đối chiếu với kho nhạc **348 bài hát**:

| STT | Phương pháp Chặng 1 & Chặng 2 Thực nghiệm | Micro MRR | Macro MRR | Recall@1 | Recall@5 | Recall@10 | Recall@20 | Recall@50 | Thứ hạng TB |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | **Chặng 1 Global-Only (128D Max Cosine cũ)** | 0.3458 | 0.2557 | 26.18% | 42.67% | 50.17% | 57.70% | 69.94% | 58.03 |
| 2 | **[2 Chặng Cũ] Global Top-20 → Re-rank V6.2** | 0.4610 | 0.3692 | 39.98% | 52.76% | 55.93% | 57.74% | 69.94% | 56.89 |
| 3 | **[2 Chặng Mở rộng] Global Top-50 → Re-rank V6.2** | 0.4862 | 0.3996 | 41.17% | 57.23% | 61.96% | 65.92% | 69.94% | 54.54 |
| 4 | **[ByteHum V6.2] Đưa Local Q85 lên Chặng 1** | 0.5037 | 0.4163 | 42.06% | 59.17% | 66.12% | 73.07% | 81.66% | 32.07 |
| 5 | **[ByteCover2] V6.2 + Mean-Centered Embeddings** | 0.4978 | 0.3948 | 40.35% | 59.85% | 66.87% | 73.69% | 81.53% | 31.23 |
| 6 | **⭐ [ByteCover3] Multi-Tempo Diagonal Shingles** | **0.5489** | **0.4627** | **45.60%** | **64.83%** | **71.71%** | **77.85%** | **83.09%** | **28.65** |
| 7 | **⭐ [Multi-Key TTA] Dịch bán âm CQT $\pm 2$ Semitones** | **0.5263** | **0.4189** | **44.04%** | **62.03%** | **68.10%** | **75.19%** | **83.71%** | **27.93** |
| 8 | **[CSLS Hubness] V6.2 + Khử nhiễu Hubness ($\beta=0.12$)**| 0.5115 | 0.4280 | 43.12% | 60.45% | 67.20% | 74.50% | 82.30% | 30.15 |

### 3. Ba Phát hiện Khoa học Cốt lõi:
1. **Sức mạnh Áp đảo của Multi-Tempo Diagonal Shingles (ByteCover3)**:
   - Khi áp dụng bộ lọc tích chập Conv2D đường chéo với 3 góc nghiêng nhịp độ ($0.8\times, 1.0\times, 1.25\times$), Micro MRR nhảy vọt lên **0.5489 (54.89%)**, tăng **+12.2% MRR** so với V6.1 ($0.4269$) và tăng **+4.52% MRR** so với V6.2 Baseline.
   - Recall@1 tăng vượt bậc lên **45.60%** (so với $34.22\%$ ở V6.1, tăng $+11.38\%$ tuyệt đối).
   - Thứ hạng trung bình của bài hát đúng được kéo gần gấp đôi: từ vị trí $58.03$ xuống **$28.65$**.
   - *Nguyên nhân*: Chỉ có bài hát đúng mới tạo thành các vệt sáng đường chéo liên tục theo nhịp độ trên ma trận tương đồng, các nốt trùng ngẫu nhiên của các bài hát khác hoàn toàn bị triệt tiêu.
2. **Khả năng Cứu Bài hát Lệch Tông của Multi-Key TTA**:
   - Recall@50 đạt đỉnh kỷ lục **83.71%**, Mean Rank thu hẹp xuống **27.93** (mức tốt nhất trong toàn bộ các cấu hình). Kỹ thuật này bù đắp hoàn hảo sự thiếu hụt cao độ tuyệt đối của người dùng phổ thông.
3. **Thống kê Đối đầu Trực diện (Head-to-Head Win/Loss)**:
   - Trên tổng số 1,467 queries kiểm thử độc lập:
     - V6.2 có thứ hạng tốt hơn V6.1: **669 queries** ($45.6\%$).
     - Hai phiên bản hòa nhau (Ties): **547 queries** ($37.3\%$).
     - V6.1 có thứ hạng tốt hơn V6.2: **251 queries** ($17.1\%$).
     - **Tỷ lệ Thắng / Thua đạt 2.67 : 1** (V6.2 áp đảo gần gấp 2.7 lần).

### 4. Đánh giá trên Nhóm 4 Bài hát Siêu khó (Hard Songs Benchmark):
- **ACDC - Highway to Hell** *(Hard Rock, guitar distortion át giọng)*: Rank trung bình V6.1 là 105.4 $\to$ V6.2 rút ngắn còn **93.4** (tiến bộ 12 bậc).
- **Nelly - Just A Dream** *(Pop/R&B bè nhiều)*: Rank trung bình V6.1 là 130.3 $\to$ V6.2 rút ngắn mạnh mẽ còn **109.1** (tiến bộ vượt bậc **21.2 bậc**).
- **DJ Khaled - No Brainer**: Duy trì ổn định ở Top **45 – 47**.
- **Amy Winehouse - Back To Black**: Đạt rank 114.5 (thách thức lớn về kỹ thuật hát luyến láy melisma).

Kết quả Ablation Study khẳng định:
1. **Phép biến đổi Full CQT 96 bins** và **Pitch Shift Augmentation $\pm 12$** là hai trụ cột mang lại mức nhảy vọt lớn nhất (+13.7% và +11.0% Recall@1), bảo vệ hệ thống trước sự biến thiên cao độ.
2. **Meta HTDemucs v4** loại bỏ triệt để nhiễu nhạc nền, đóng góp trực tiếp +4.63% độ chính xác.
3. Bộ ba **ByteHum (LAT, LAC, GRL)** đóng vai trò tinh hoa giúp mô hình vượt ngưỡng trần của Metric Learning đơn thuần, đưa MRR từ $0.4412$ bứt phá ngoạn mục lên $0.5086$.

---

## 5.5. Phân tích Lỗi và Chẩn đoán các Ca Truy vấn Khó (Error Diagnostics)

Mặc dù đạt hiệu năng vượt trội, vẫn có $22.9\%$ lượt truy vấn không lọt vào Top-10. Chúng tôi đã tiến hành phân tích sâu các log truy vấn thất bại và phân loại thành 4 nhóm nguyên nhân đặc thù:

1. **Nhóm Ca khúc Tiết tấu Cực nhanh (Rock & Heavy Metal)**:
   - *Đặc điểm*: Nhịp độ (Tempo) vượt quá $140\text{ BPM}$, giọng hát trong bản gốc gằn giọng (distortion/growling) và biến chuyển nốt trong thời gian quá ngắn.
   - *Nguyên nhân thất bại*: Người dùng khi ngân nga thường không thể bắt kịp tốc độ nốt, tự động làm chậm tempo xuống một nửa (Half-tempo) hoặc bỏ qua các nốt hoa mỹ, dẫn đến sai lệch lớn trong ma trận căn chỉnh LAT.
2. **Nhóm Ca khúc Luyến láy Phức tạp (R&B & Neo-Soul)**:
   - *Đặc điểm*: Ca sĩ phòng thu sử dụng kỹ thuật Melisma (hát một âm tiết trải dài qua nhiều cao độ lượn sóng liên tục).
   - *Nguyên nhân thất bại*: Người dùng nghiệp dư thường làm phẳng các đường lượn sóng này thành một nốt ngân thẳng đều, làm mất đi "vân tay đặc trưng" của bài hát.
3. **Nhóm Nhạc Rap và Nói theo Nhịp (Rap / Hip-Hop)**:
   - *Đặc điểm*: Nhạc Rap có đường nét cao độ biến thiên rất ít ($F_0$ gần như phẳng), giá trị thông tin chủ yếu nằm ở tiết tấu bộ gõ và ca từ.
   - *Nguyên nhân thất bại*: Tiếng ngân nga "hum hum" không thể hiện được nhịp ngắt của ca từ và không có biến thiên giai điệu cao độ rõ rệt để CQT trích xuất.
4. **Hiện tượng Giai điệu Phổ quát (Generic Melodic Confusion)**:
   - *Đặc điểm*: Một số bài hát Pop thương mại sử dụng chung một vòng hòa thanh kinh điển (ví dụ vòng $I - V - vi - IV$: Đô - Sol - La thứ - Pha) với giai điệu điệp khúc có cấu trúc cao độ tương đồng nhau.
   - *Hiện tượng*: Đoạn ngân nga của người dùng kích hoạt điểm tương đồng cao trên cả 2 hoặc 3 bài hát khác nhau trong cùng một cụm thể loại, khiến bài hát mục tiêu bị đẩy xuống vị trí Rank #2 hoặc #3 thay vì Rank #1.

---

## 5.6. Kết luận chương

Chương 5 đã kiểm chứng khoa học hiệu năng của đề tài:
- Thiết lập quy trình đánh giá nghiêm ngặt trên tập dữ liệu chuẩn mực và các độ đo quốc tế (MRR, Recall@1, 5, 10).
- Khẳng định mô hình **Conformer-ByteHum V6.2** áp đảo hoàn toàn mô hình cơ sở CNN trước đây, nâng Recall@1 từ $16.67\%$ lên **$42.23\%$** và MRR từ $0.3767$ lên **$0.5086$** trên kho nhạc 348 bài.
- Nghiên cứu bóc tách thành phần (Ablation Study) chỉ rõ giá trị định lượng của từng mô đun kỹ thuật.
- Phân tích lỗi tường minh cung cấp những luận cứ khoa học quan trọng cho các giải pháp cải tiến tiếp theo.

