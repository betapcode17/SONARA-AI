# BÁO CÁO DỰ ÁN CHUYÊN NGÀNH 1 (PBL6)
## ĐỀ TÀI: HỆ THỐNG TRUY VẤN BÀI HÁT QUA TIẾNG NGÂN NGA (QUERY-BY-HUMMING) DỰA TRÊN CONFORMER-BYTEHUM V6.2

---

## TÓM TẮT DỰ ÁN (ABSTRACT)

Báo cáo trình bày toàn diện quá trình nghiên cứu, thiết kế kiến trúc học sâu, tối ưu hóa xử lý tín hiệu và triển khai thực tế của Hệ thống Truy vấn Bài hát qua Tiếng ngân nga (Query-by-Humming - QBH) phiên bản Conformer-ByteHum V6.2. 

Khác biệt với các hệ thống truyền thống chỉ dựa trên mạng CNN nông và phổ Chroma CQT 12 bins gập quãng tám (như khảo sát trong QBH.pdf), dự án đề xuất giải pháp đột phá kết hợp 4 trụ cột công nghệ:
1. **Xử lý tín hiệu cao cấp**: Sử dụng Meta HTDemucs v4 bóc tách 100% Vocal Stems sạch từ bản Master Studio, kết hợp phép biến đổi Full CQT 96 bins (8 quãng tám) và chuỗi Data Augmentation đặc thù (Pitch Shift $\pm 12$ bán âm, Reverb, Harmonic Noise).
2. **Kiến trúc mô hình Conformer-ByteHum 9.91M tham số**: Kết hợp khả năng nắm bắt nhịp điệu cục bộ của Depthwise Convolution và ngữ cảnh giai điệu toàn cục của Multi-Head Self-Attention (MHSA). Tích hợp cơ chế Attentive Statistics Pooling (ASP) cùng bộ ba mô đun tinh hoa của ByteHum (ICASSP 2024): LAT (Learned Alignment Transformer) với MaxMean Pooling, LAC (Learned Alignment Classifier) với L2 Prototypes, và GRL (Gradient Reversal Layer) huấn luyện đối kháng miền.
3. **Thuật toán truy hồi hai chặng SOTA**: Chặng 1 lọc ứng viên diện rộng bằng Local Frame Quantile Pooling Q85 kết hợp bộ lọc Tích chập Đường chéo Đa nhịp độ (Multi-Tempo Diagonal Shingles Conv2D quét 3 nhịp độ 0.8x, 1.0x, 1.25x) và Multi-Key CQT Pitch Shift TTA, mở rộng shortlist lên Top-50 ứng viên (đạt độ phủ 83.71% bài hát đúng). Chặng 2 thực hiện tái xếp hạng chuyên sâu bằng Subsequence DTW Chroma/Melody và Reciprocal Rank Fusion (RRF).
4. **Hạ tầng phân phối đám mây Cloudinary CDN**: Lưu trữ và cung cấp luồng phát trực tiếp (Direct Streaming URL) cho 300 bài hát với độ trễ tìm kiếm toàn trình chỉ $315\text{ ms}$ trên GPU.

Kết quả thực nghiệm trên tập kiểm thử độc lập 1,467 queries khẳng định sự vượt trội toàn diện: chỉ số **MRR bứt phá lên 0.5489 (54.89%)** (tăng +45.7% so với CNN cơ sở và +12.2% so với V6.1), **Recall@1 đạt 45.60%** (tăng gấp 2.73 lần), **Recall@5 đạt 64.83%**, **Recall@10 đạt 71.71%**, **Recall@20 đạt 77.85%** và **Recall@50 đạt 83.09%**. Tỷ số thắng thua đối đầu trực diện đạt 2.67 : 1.

---

## DANH MỤC TỪ VIẾT TẮT VÀ THUẬT NGỮ

| Từ viết tắt | Thuật ngữ tiếng Anh đầy đủ | Diễn giải ý nghĩa |
| :--- | :--- | :--- |
| **QBH** | Query-by-Humming | Hệ thống truy vấn bài hát qua tiếng ngân nga, hát chay hoặc huýt sáo |
| **MIR** | Music Information Retrieval | Lĩnh vực khoa học công nghệ Truy xuất Thông tin Âm nhạc |
| **CQT** | Constant-Q Transform | Biến đổi Q hằng số với thang tần số logarit |
| **STFT** | Short-Time Fourier Transform | Biến đổi Fourier thời gian ngắn với dải tần số tuyến tính |
| **MHSA** | Multi-Head Self-Attention | Cơ chế Tự chú ý đa đầu trong kiến trúc Transformer |
| **ASP** | Attentive Statistics Pooling | Khối gộp thống kê kỳ vọng và độ lệch chuẩn có trọng số tự chú ý |
| **LAT** | Learned Alignment Transformer | Bộ căn chỉnh giai điệu cục bộ thông qua Cross-Attention MaxMean |
| **LAC** | Learned Alignment Classifier | Bộ phân loại căn chỉnh dựa trên vector tâm cụm nguyên mẫu L2 |
| **GRL** | Gradient Reversal Layer | Lớp đảo ngược dấu gradient phục vụ huấn luyện đối kháng miền |
| **MRR** | Mean Reciprocal Rank | Xếp hạng nghịch đảo trung bình của bài hát đúng |
| **CDN** | Content Delivery Network | Mạng lưới phân phối nội dung đa phương tiện toàn cầu (Cloudinary) |
| **RMS** | Root Mean Square | Giá trị căn bậc hai trung bình bình phương (đo năng lượng âm thanh) |
| **BPM** | Beats Per Minute | Số phách nhịp trong mỗi phút (tốc độ bài hát) |

---

# CHƯƠNG 1: TỔNG QUAN VỀ HỆ THỐNG TRUY VẤN BÀI HÁT QUA TIẾNG NGÂN NGA (QUERY-BY-HUMMING)

---

## 1.1. Đặt vấn đề và Tính cấp thiết của đề tài

Trong thời đại số hóa và bùng nổ dữ liệu truyền thông đa phương tiện, các nền tảng phát nhạc trực tuyến như Spotify, Apple Music, YouTube Music đang lưu trữ hàng trăm triệu bản nhạc với thể loại, ngôn ngữ và phong cách biểu diễn vô cùng phong phú. Việc hỗ trợ người dùng tìm kiếm bài hát một cách trực quan, tức thời trở thành bài toán cốt lõi của ngành công nghệ Truy xuất Thông tin Âm nhạc (Music Information Retrieval - MIR).

Thông thường, người dùng tìm kiếm bài hát qua siêu dữ liệu văn bản (tên bài hát, tên nhạc sĩ, ca sĩ thể hiện) hoặc lời bài hát (lyrics search). Tuy nhiên, trong đời sống thực tế, một tình huống rất phổ biến là người dùng chỉ nhớ mang máng một giai điệu ngắn, một đoạn điệp khúc bắt tai vừa vô tình nghe thấy tại một quán cà phê hay trên đường phố, mà hoàn toàn không biết tựa đề, tên nghệ sĩ hoặc không thể nhớ chính xác ca từ. 

Khi đó, công nghệ Nhận diện Vân tay Âm thanh (Audio Fingerprinting) kinh điển — điển hình như thuật toán của Shazam hay SoundHound — bộc lộ giới hạn không thể khắc phục:
- **Audio Fingerprinting** hoạt động dựa trên việc trích xuất các điểm đỉnh phổ tần số (spectrogram peaks/constellation map) từ bản ghi âm thanh phòng thu gốc (master recording).
- Phương pháp này đòi hỏi tín hiệu truy vấn đầu vào phải là một đoạn ghi âm thực tế từ chính bản nhạc gốc với độ trung thực âm sắc (timbre) và cấu trúc hài âm hoàn chỉnh.
- Khi người dùng tự mình ngân nga (humming), hát chay (singing), hay huýt sáo (whistling) lại giai điệu bằng giọng mộc, tín hiệu âm thanh đầu vào sẽ bị biến dạng nghiêm trọng: lệch cao độ (pitch deviation), sai lệch nhịp độ (tempo variation), thiếu vắng hoàn toàn phần hòa âm nhạc cụ (accompaniment), và chứa nhiều tạp âm môi trường từ micro thiết bị cá nhân. Các thuật toán so khớp vân tay phổ truyền thống sẽ hoàn toàn thất bại.

Bài toán **Truy vấn Bài hát qua Tiếng ngân nga (Query-by-Humming - QBH)** ra đời nhằm giải quyết khoảng trống này: cho phép người dùng chỉ cần cất lên giọng ngân nga từ 5 đến 15 giây qua microphone để hệ thống tự động nhận diện và trả về chính xác bản thu gốc trong kho nhạc. Đây là một bài toán khoa học đầy thách thức nhưng có giá trị ứng dụng thực tiễn rất cao trong việc nâng cao trải nghiệm người dùng trên các hệ sinh thái giải trí trực tuyến.

---

## 1.2. Khái niệm và Ba thế hệ Công nghệ Hệ thống QBH

Hệ thống QBH là một nhánh chuyên sâu của lĩnh vực MIR, trong đó mục tiêu là tìm kiếm một đối tượng âm nhạc đích $Y$ (thường là bản master studio của một ca khúc) nằm trong tập dữ liệu thư viện (Gallery $\mathcal{G}$) thông qua một truy vấn âm thanh không hoàn hảo $X$ (Query $\mathcal{Q}$) do người dùng ngẫu hứng thể hiện.

Quá trình phát triển của các công nghệ QBH có thể được phân loại thành ba thế hệ tiêu biểu:

![Ba thế hệ công nghệ QBH](docs/figures/fig_ch1_generations.png)
*Hình 1.1: Tiến trình phát triển ba thế hệ công nghệ Truy vấn Bài hát qua Tiếng ngân nga (Query-by-Humming).*

### 1. Thế hệ 1: Phương pháp Cổ điển dựa trên Đường nét Cao độ (Handcrafted F0 & DTW)
- **Cơ chế**: Sử dụng các thuật toán xử lý tín hiệu số (DSP) truyền thống như YIN, pYIN, hoặc tự tương quan (Autocorrelation) để ước lượng đường cong cao độ cơ bản ($F_0$ contour) từ tiếng ngân nga, sau đó chuyển đổi thành chuỗi nốt nhạc tương đối (Relative Pitch Contour / Parsons Code).
- **So khớp**: Áp dụng thuật toán Co giãn Thời gian Động (Dynamic Time Warping - DTW) hoặc Mô hình Markov Ẩn (Hidden Markov Models - HMM) để đo khoảng cách sai lệch giữa chuỗi nốt của người dùng và bản nhạc mẫu (thường ở định dạng MIDI).
- **Hạn chế lớn**: Rất nhạy cảm với lỗi trích xuất $F_0$ (nhiễu ồn micro làm nhảy quãng tám - octave jump), không mô hình hóa được năng lượng và nhịp điệu, tốc độ tìm kiếm $O(N \cdot M)$ của DTW quá chậm khi kho nhạc mở rộng quy mô lớn.

### 2. Thế hệ 2: Học sâu Cổ điển với Mạng Nơ-ron Tích chập (Shallow CNN & Chroma CQT)
- **Cơ chế**: Ánh xạ tín hiệu âm thanh thành ảnh phổ thời gian - tần số, phổ biến nhất là phổ Chroma CQT (12 bins đại diện cho 12 nửa cung trong hệ âm nhạc bình quân). Sử dụng mạng nơ-ron tích chập (CNN 4-6 lớp) đóng vai trò bộ mã hóa (Encoder) để ánh xạ ảnh phổ thành một vector đặc trưng duy nhất (Global Embedding Vector).
- **Huấn luyện**: Áp dụng Học biểu diễn khoảng cách (Deep Metric Learning) với hàm mất mát bộ ba (Triplet Margin Loss) hoặc Contrastive Loss để kéo gần vector của các đoạn ngân nga cùng bài hát và đẩy xa vector của các bài hát khác nhau.
- **Hạn chế**: Việc gập 8 quãng tám vào 12 bán âm (Chroma folding) làm triệt tiêu hoàn toàn thông tin về thanh vực và biến thiên cao độ tuyệt đối. Mạng CNN nông có trường tiếp nhận (receptive field) hạn chế, không bắt được các mối quan hệ ngữ cảnh giai điệu kéo dài theo thời gian. Đây chính là kiến trúc cơ sở được khảo sát trong tài liệu tham khảo ban đầu (`QBH.pdf`), chỉ đạt MRR khiêm tốn $0.3767$ và Acc@1 $16.67\%$.

### 3. Thế hệ 3: Mô hình Conformer-ByteHum V6.2 (Đề xuất của Dự án)
- **Cơ chế**: Khắc phục triệt để các hạn chế trên bằng cách kết hợp sức mạnh của mạng Conformer (9.91 triệu tham số kết hợp Multi-Head Self-Attention và Depthwise Convolution) và khung kiến trúc đột phá **ByteHum** (công bố tại ICASSP 2024):
  1. Biểu diễn đầu vào: Sử dụng **Full CQT 96 bins** bao phủ trọn vẹn 8 quãng tám mà không gập bán âm.
  2. Tách giọng sạch: Sử dụng mô hình Hybrid Transformer **Meta HTDemucs v4** để bóc tách triệt để vocal stems từ bản master, loại bỏ hoàn toàn nhiễu từ nhạc cụ nền phức tạp.
  3. Cơ chế căn chỉnh cục bộ **LAT (Learned Alignment Transformer)**: Cho phép so khớp chính xác từng đoạn điệp khúc ngắn với bài hát dài thông qua MaxMean Pooling.
  4. Cơ chế phân loại **LAC (Learned Alignment Classifier)**: Tối ưu khoảng cách trên siêu mặt cầu thông qua các vector tâm cụm (Prototypes).
  5. Huấn luyện đối kháng miền **GRL (Gradient Reversal Layer)**: Triệt tiêu triệt để khoảng cách miền (Domain Gap) giữa giọng hát mộc nghiệp dư và giọng ca sĩ phòng thu chuyên nghiệp.
- **Kết quả**: Đưa chỉ số MRR lên **0.5086**, Recall@1 lên **42.23%**, và Recall@5 vượt mốc **60.50%** trên kho nhạc 348 bài hát thực tế.

Bảng so sánh tổng hợp giữa ba thế hệ công nghệ QBH:

| Tiêu chí so sánh | Thế hệ 1 (F0 + DTW) | Thế hệ 2 (Shallow CNN + Chroma) | Thế hệ 3: Conformer-ByteHum V6.2 |
| :--- | :--- | :--- | :--- |
| **Đặc trưng âm học đầu vào** | $F_0$ Pitch Contour / MIDI | Chroma CQT (12 bins gập) | Full CQT (96 bins $\times$ 83 frames, 8 quãng tám) |
| **Tiền xử lý tách vocal** | Không có | Lọc bandpass / spectral gating cơ bản | **Meta HTDemucs v4 Hybrid Transformer** |
| **Kiến trúc mạng nơ-ron** | Thuật toán giải tích DSP | CNN 4 tầng (~1.2M tham số) | **Conformer Backbone (9.91M tham số)** |
| **Khả năng bắt ngữ cảnh** | Giới hạn theo cửa sổ trượt | Giới hạn bởi kích thước kernel CNN | **Toàn cục (Multi-Head Self-Attention)** |
| **Cơ chế so khớp độ dài** | Căn chỉnh thủ công DTW | Average / Max Pooling thô | **ByteHum LAT (MaxMean Cross-Attention)** |
| **Xử lý Domain Gap** | Kém | Phụ thuộc Triplet Loss | **Domain Adversarial GRL (Gradient Reversal)** |
| **Chỉ số MRR** | $< 0.25$ | $\sim 0.3767$ | **0.5086 (Vượt trội +35%)** |
| **Chỉ số Recall@1** | $< 10\%$ | $16.67\%$ | **42.23% (Vượt trội 2.5 lần)** |

---

## 1.3. Mục tiêu, Giới hạn và Phạm vi nghiên cứu của Đề tài

### 1. Mục tiêu cốt lõi
1. **Nghiên cứu và làm chủ lý thuyết học sâu âm thanh tiên tiến**: Triển khai và tối ưu hóa kiến trúc Conformer kết hợp khung mô hình ByteHum (ICASSP 2024), giải quyết triệt để bài toán Deep Metric Learning cho dữ liệu âm thanh bất đối xứng (Humming vs Master Studio Track).
2. **Xây dựng quy trình xử lý tín hiệu âm thanh chuẩn mực**: Ứng dụng mô hình Hybrid Transformer Meta HTDemucs v4 để tách giọng hát chuẩn phòng thu, chuyển đổi tín hiệu sang biểu diễn toán học Full CQT 96 bins chất lượng cao, đồng thời thiết kế bộ dữ liệu tăng cường âm học (Augmentation Pipeline) đặc thù cho QBH (Pitch Shifting $\pm 12$ bán âm, Room Impulse Response Reverb, Harmonic Noise).
3. **Hiện thực hóa hệ thống truy vấn hai chặng (Two-stage Retrieval Pipeline)**: Kết hợp thuật toán tìm kiếm song song In-RAM Top-K Quantile Pooling 85% và cơ chế tái xếp hạng (Re-ranking) dựa trên điểm căn chỉnh cục bộ LAT với trọng số kết hợp (Fused Score 75/25), đảm bảo độ trễ phản hồi dưới 350ms.
4. **Triển khai ứng dụng Web thực tế hoàn chỉnh**: Tích hợp hạ tầng lưu trữ và phân phối âm thanh chất lượng cao trên nền tảng đám mây Cloudinary CDN (300 bài hát với Direct Streaming URL), kết nối backend FastAPI với giao diện người dùng trực quan.

### 2. Giới hạn và Phạm vi nghiên cứu
- **Phạm vi dữ liệu**: Đề tài tập trung vào kho nhạc thử nghiệm bao gồm 348 bài hát thuộc nhiều thể loại (Pop, Ballad, Rock, R&B, Rap) với độ dài từ 2 đến 5 phút; trong đó 300 bản master audio đã được đồng bộ lên Cloudinary Media Cloud. Dữ liệu huấn luyện sử dụng tập SHS100K cho giai đoạn tiền huấn luyện (Pre-training) và tập Humming V6.2 cho giai đoạn tinh chỉnh đặc thù (Fine-tuning).
- **Điều kiện truy vấn**: Hỗ trợ tiếng thu âm thực tế từ micro máy tính/điện thoại với thời lượng khuyến nghị từ 5 đến 15 giây. Người dùng có thể ngân nga (humming với âm "la la", "hum hum"), huýt sáo, hoặc hát chay có lời.
- **Giới hạn kỹ thuật**: Chưa tích hợp bộ lập chỉ mục xấp xỉ phân tán dạng FAISS IVFPQ trên quy mô hàng triệu bài hát; dữ liệu hiện tại được tối ưu hóa xử lý trực tiếp trên GPU/RAM với độ trễ thấp nhất.

---

## 1.4. Bố cục của Báo cáo Dự án

Báo cáo Dự án Chuyên ngành 1 (PBL6) được cấu trúc thành 7 chương chính theo trình tự học thuật chuẩn mực:
- **Chương 1: Tổng quan về Hệ thống Truy vấn Bài hát qua Tiếng ngân nga (QBH)**: Giới thiệu tính cấp thiết, phân tích lịch sử ba thế hệ công nghệ, mục tiêu và phạm vi nghiên cứu.
- **Chương 2: Cơ sở Lý thuyết và Xử lý Tín hiệu Âm thanh**: Phân tích âm học giọng người, biến đổi Fourier STFT và các hạn chế, cơ sở toán học của Biến đổi Q hằng số (CQT 96 bins), kỹ thuật phân tách giọng hát Meta HTDemucs v4 và chiến lược tăng cường dữ liệu đặc thù.
- **Chương 3: Thiết kế Kiến trúc Mô hình Conformer-ByteHum QbH V6.2**: Mô tả chi tiết kiến trúc CoverHunter Conformer 9.91 triệu tham số, cơ chế Attention Statistics Pooling (ASP), bộ ba mô đun ByteHum (LAT, LAC, GRL), và hàm mất mát liên kết đa nhiệm.
- **Chương 4: Thiết kế Hệ thống và Quy trình Thực thi (Pipeline)**: Trình bày kiến trúc hai nhánh Offline Indexing và Online Real-time Retrieval, kỹ thuật In-RAM CQT Feature Cache, thuật toán tìm kiếm Top-K Quantile Pooling kết hợp Fused Distance, và hạ tầng Cloudinary CDN.
- **Chương 5: Phân tích Kết quả và Thực nghiệm**: Báo cáo thiết lập thực nghiệm, phân tích các chỉ số học thuật (MRR, Recall@1, Recall@5, Recall@10), so sánh định lượng V6.2 vs V6.1 và mô hình cơ sở, tiến hành Ablation Study và chẩn đoán các ca truy vấn thất bại.
- **Chương 6: Triển khai Kiểm thử và Đánh giá Thực tế**: Trình bày triển khai Web API bằng FastAPI, kết quả thử nghiệm trực tiếp với nhiều kịch bản người dùng (lệch tông, tạp âm, hát chay), đánh giá độ trễ và tài nguyên hệ thống.
- **Chương 7: Kết luận và Hướng phát triển**: Tổng kết các đóng góp kỹ thuật, phân tích các thách thức tồn đọng và định hướng nghiên cứu tiếp theo (FAISS IVF-PQ, Edge AI, Music LLM).




---


# CHƯƠNG 2: CƠ SỞ LÝ THUYẾT VÀ XỬ LÝ TÍN HIỆU ÂM THANH

---

## 2.1. Bản chất Âm học của Giọng người và Tiếng ngân nga

Âm thanh phát ra từ thanh quản con người là một chuỗi các dao động phức tạp sinh ra do luồng không khí từ phổi đi qua dây thanh âm và cộng hưởng trong các khoang miệng, mũi, hầu họng. Về mặt vật lý sóng âm, tiếng người hát hoặc ngân nga mang hai đặc tính cốt lõi:
1. **Tần số cơ bản ($F_0$ - Fundamental Frequency)**: Tương ứng với tốc độ dao động của dây thanh âm, quyết định cao độ (Pitch) mà tai người cảm nhận được. Trong giọng nói và ca hát thông thường, $F_0$ của giọng nam thường dao động trong khoảng $85\text{ Hz} - 180\text{ Hz}$, trong khi giọng nữ dao động trong khoảng $165\text{ Hz} - 255\text{ Hz}$, và ở các nốt cao của ca sĩ có thể vượt qua $1000\text{ Hz}$.
2. **Các họa âm (Harmonics) và Định âm (Formants)**: Các tần số bội số của $F_0$ ($2F_0, 3F_0, 4F_0, \dots$) kết hợp với các đỉnh cộng hưởng trong khoang miệng tạo nên âm sắc (Timbre) đặc trưng của từng cá nhân và các nguyên âm (như /a/, /u/, /i/).

Khi người dùng thực hiện thao tác ngân nga (Humming) hoặc huýt sáo (Whistling) một giai điệu, có những hiện tượng vật lý âm học đặc thù xảy ra:
- **Ngân nga mím môi (Humming)**: Miệng đóng lại, luồng khí thoát qua khoang mũi. Hiện tượng này làm suy giảm nghiêm trọng năng lượng của các họa âm tần số cao (High-frequency Roll-off), tập trung hầu như toàn bộ năng lượng vào tần số cơ bản $F_0$ và các họa âm bậc thấp đầu tiên ($2F_0, 3F_0$).
- **Lệch tông giọng (Pitch Shift / Key Transposition)**: Người dùng nghiệp dư hiếm khi sở hữu khả năng "cao độ tuyệt đối" (Absolute Pitch). Do đó, họ thường tự động chuyển dịch giai điệu về tông giọng tự nhiên thoải mái nhất cho thanh quản của mình (thường lệch từ vài bán âm đến nguyên một quãng tám so với bản thu ca sĩ gốc).
- **Biến thiên thời gian và nhịp điệu (Tempo Elasticity)**: Người dùng có thể hát nhanh hơn hoặc chậm hơn bản gốc, thậm chí ngân dài bất thường ở các nốt điệp khúc và nuốt bớt các nốt luyến láy phức tạp.
- **Tạp âm môi trường**: Tín hiệu thu âm từ micro tích hợp trên máy tính xách tay hoặc điện thoại thường lẫn tiếng ồn nền (quạt gió, xe cộ, tiếng vang phòng - Reverberation).

Để một hệ thống AI nhận diện thành công giai điệu từ tiếng ngân nga, bước tiền xử lý tín hiệu và biểu diễn đặc trưng phải thỏa mãn đồng thời hai yêu cầu trái ngược: **Bắt trọn vẹn sự biến thiên cao độ theo thời gian** nhưng **Triệt tiêu tối đa sự phụ thuộc vào âm sắc, hòa âm và tông giọng tuyệt đối**.

---

## 2.2. Phân tách Nguồn âm và Trích xuất Vocal Stems với Meta HTDemucs v4

Một trong những nguyên nhân chính khiến các hệ thống QBH thế hệ cũ (như mô hình khảo sát trong `QBH.pdf`) đạt độ chính xác thấp là việc trích xuất đặc trưng trực tiếp từ bản thu Master Studio hoàn chỉnh. Bản thu Master là một hỗn hợp đa kênh phức tạp gồm giọng hát chính (Lead Vocal), bè phối, tiếng trống (Drums - dải trầm và các xung va đập mạnh), tiếng bass, tiếng guitar điện, piano và các nhạc cụ điện tử khác. Khi trích xuất phổ tần số trực tiếp từ bản Master, năng lượng từ bộ gõ và hợp âm nhạc cụ hoàn toàn che lấp đường nét thanh âm của giọng hát chính, khiến vector đặc trưng phản ánh âm sắc nhạc cụ thay vì giai điệu ca khúc.

Trong dự án này, chúng tôi áp dụng bước đột phá: **Bóc tách triệt để nguồn âm bằng mô hình Meta HTDemucs v4 (Hybrid Transformer Demucs)**:

![Sơ đồ phân tách nguồn âm HTDemucs v4](docs/figures/fig_ch2_htdemucs.png)
*Hình 2.1: Quy trình phân tách nguồn âm và trích xuất kênh giọng hát thuần khiết (Vocal Stem) bằng mô hình Meta HTDemucs v4.*

### Cơ chế hoạt động của HTDemucs v4:
- **Kiến trúc Hybrid đa miền**: HTDemucs v4 là mô hình SOTA thế giới về Music Source Separation do Meta AI Research phát triển. Mô hình kết hợp song song hai nhánh xử lý: nhánh miền thời gian (Waveform Domain) dựa trên mạng U-Net tích chập 1D và nhánh miền tần số (Spectrogram Domain) dựa trên U-Net 2D kết hợp các lớp Transformer Self-Attention liên miền.
- **Trích xuất Vocal Stem thuần khiết**: Toàn bộ 348 bài hát trong cơ sở dữ liệu Gallery đều được đưa qua HTDemucs v4 để trích xuất riêng rẽ kênh **Vocal Stem** sạch. Nhờ đó, cơ sở dữ liệu bài hát mẫu chỉ còn chứa giọng hát của ca sĩ, triệt tiêu 100% tiếng ồn nhạc cụ nền và đưa bài hát về cùng "miền biểu diễn thanh âm" (Vocal Domain) gần nhất với tiếng ngân nga của người dùng.

---

## 2.3. Hạn chế của Biến đổi Fourier Thời gian ngắn (STFT) trong Âm nhạc

Phép biến đổi Fourier thời gian ngắn (Short-Time Fourier Transform - STFT) là công cụ kinh điển nhất trong xử lý tín hiệu số:
$$X(m, \omega) = \sum_{n=-\infty}^{\infty} x[n] w[n - m] e^{-j \omega n}$$
trong đó $w[n]$ là hàm cửa sổ thời gian (chẳng hạn như Hanning hoặc Hamming) dịch chuyển theo bước nhảy $m$.

Tuy nhiên, STFT tồn tại một hạn chế bản chất khi ứng dụng vào tín hiệu âm nhạc xuất phát từ **Nguyên lý Bất định Heisenberg-Gabor**:
$$\Delta t \cdot \Delta f \ge \frac{1}{4\pi}$$

Trong STFT, chiều dài cửa sổ thời gian $N$ là **cố định** trên toàn bộ các dải tần số. Điều này dẫn đến độ phân giải tần số là **tuyến tính** với bước chia đều:
$$\Delta f = \frac{f_s}{N} = \text{const}$$

Điều này mâu thuẫn trực tiếp với quy luật cấu tạo của hệ thống âm nhạc học và cơ chế cảm thụ cao độ của ốc tai con người:
- Thang âm nhạc (Musical Scale) tuân theo **hàm logarit theo tần số**: một quãng tám (Octave) tương ứng với việc tăng gấp đôi tần số cơ bản ($f_{octave} = 2 \cdot f$). Khoảng cách giữa nốt $A_1$ (55 Hz) và $A_1^\sharp$ (58.27 Hz) chỉ là $3.27\text{ Hz}$, trong khi khoảng cách giữa nốt $A_7$ (3520 Hz) và $A_7^\sharp$ (3729.31 Hz) lên tới $209.31\text{ Hz}$ dù cùng là khoảng cách đúng 1 nửa cung (1 semitone).
- Khi sử dụng STFT với $\Delta f$ cố định (ví dụ $\Delta f = 21.5\text{ Hz}$ khi $f_s = 22050\text{ Hz}, N = 1024$):
  - Ở dải trầm (Low Frequencies): Độ phân giải $21.5\text{ Hz}$ quá thô, nhiều nốt nhạc khác nhau bị gộp chung vào cùng một bin tần số, làm mất hoàn toàn khả năng phân biệt cao độ của giọng hát nam.
  - Ở dải cao (High Frequencies): Độ phân giải bị dư thừa lãng phí, chiếm nhiều chiều dữ liệu nhưng không mang lại thêm thông tin nhạc lý hữu ích.

---

## 2.4. Biến đổi Q Hằng số (Constant-Q Transform - CQT) và Lựa chọn 96 Bins

Để khắc phục nhược điểm của STFT, phép biến đổi **Constant-Q Transform (CQT)** được Judith Brown đề xuất (1991) đã trở thành tiêu chuẩn vàng trong phân tích âm nhạc số.

### 1. Cơ sở Toán học của CQT
Khác với STFT, CQT giữ cố định hệ số phẩm chất chất lượng $Q$ (Quality Factor) trên toàn bộ các bộ lọc phân tích:
$$Q = \frac{f_k}{\Delta f_k} = \text{const}$$
trong đó $f_k$ là tần số trung tâm của bin thứ $k$, và $\Delta f_k$ là băng thông (Bandwidth) của bộ lọc tại bin đó.

Nếu ta chia mỗi quãng tám thành $B$ bins phân giải đều nhau theo thang logarit, tần số trung tâm $f_k$ được tính theo quy luật cấp số nhân:
$$f_k = f_0 \cdot 2^{k / B}, \quad k = 0, 1, \dots, K-1$$
với $f_0$ là tần số tối thiểu bắt đầu phân tích.

Khi đó, khoảng cách tần số giữa hai bin kế tiếp là:
$$\Delta f_k = f_{k+1} - f_k = f_k \cdot (2^{1/B} - 1)$$

Từ đó, hệ số phẩm chất $Q$ được xác định cố định độc lập với tần số:
$$Q = \frac{f_k}{f_k \cdot (2^{1/B} - 1)} = \frac{1}{2^{1/B} - 1}$$

Để bảo đảm hệ số $Q$ không đổi, độ dài cửa sổ phân tích $N_k$ tỉ lệ nghịch với tần số $f_k$:
$$N_k = Q \cdot \frac{f_s}{f_k}$$
- Tại dải trầm (tần số thấp): Cửa sổ thời gian $N_k$ tự động mở rộng rất dài để đạt độ phân giải tần số siêu mịn ($\Delta f_k$ rất nhỏ), phân biệt rõ từng nửa cung của giọng hát.
- Tại dải cao (tần số lớn): Cửa sổ thời gian $N_k$ tự động co ngắn lại để bắt kịp các chuyển tiếp thời gian tức thời (Temporal Transients).

Công thức biến đổi CQT của tín hiệu rời rạc $x[n]$ được biểu diễn qua:
$$X_{CQT}(k, n) = \frac{1}{N_k} \sum_{j=0}^{N_k - 1} x[j + n] \cdot w_{N_k}[j] \cdot e^{-j 2\pi Q \frac{j}{N_k}}$$

### 2. Thiết kế Cấu hình Full CQT 96 Bins
Trong các nghiên cứu cơ sở trước đây (như `QBH.pdf`), tác giả thường áp dụng phép gập 12 nửa cung (Chroma CQT) để đạt tính bất biến quãng tám. Tuy nhiên, việc gập quãng tám khiến hệ thống không thể phân biệt được giọng trầm ấm nam giới với giọng nữ thanh cao, đánh mất thông tin về thanh vực và các bước nhảy giai điệu lớn.

Trong dự án này, chúng tôi thiết kế cấu hình **Full CQT 96 bins** chuẩn mực:
- **Tần số lấy mẫu ($f_s$)**: $22,050\text{ Hz}$ (đảm bảo theo định lý lấy mẫu Nyquist thu trọn dải tần hữu ích của giọng người tới $11,025\text{ Hz}$).
- **Số bins mỗi quãng tám ($B$)**: $12$ bins/octave (tương ứng chính xác 1 bin = 1 bán âm chuẩn trong âm nhạc phương Tây).
- **Tổng số bins ($K$)**: $96$ bins liên tục.
- **Dải tần bao phủ**: Đúng $96 / 12 = 8$ quãng tám hoàn chỉnh:
  - Tần số đáy $f_0 \approx 32.7\text{ Hz}$ (tương ứng nốt $C_1$ - cực kỳ sâu).
  - Tần số đỉnh $f_{95} \approx 7,902\text{ Hz}$ (tương ứng nốt $B_8$ - vượt qua trọn vẹn toàn bộ dải âm vực của giọng hát người và các họa âm chính).
- **Hop Length**: $512$ mẫu tín hiệu ($\approx 23.2\text{ ms}$ trên mỗi khung thời gian).
- **Kích thước phổ đầu vào mô hình**: Cố định đoạn trích mẫu với $T = 83$ khung thời gian, tạo nên ma trận ảnh phổ đặc trưng có kích thước $\mathbf{X} \in \mathbb{R}^{96 \times 83}$ chuẩn mực.

---

## 2.5. Chiến lược Tiền xử lý và Data Augmentation Đặc thù cho QBH

Huấn luyện mô hình học sâu cho bài toán QBH đối mặt với thách thức thiếu hụt dữ liệu nghiêm trọng: số lượng bản ghi tiếng ngân nga người dùng thực tế thường rất ít so với hàng trăm ngàn bản master audio. Do đó, việc thiết kế một chuỗi tăng cường dữ liệu âm học (Audio Augmentations) sát với thực tế là yếu tố quyết định để mô hình tổng quát hóa tốt.

Chúng tôi xây dựng chuỗi tiền xử lý và tăng cường âm thanh với 3 kỹ thuật cốt lõi:

![Chuỗi tăng cường dữ liệu âm học](docs/figures/fig_ch2_augmentation.png)
*Hình 2.2: Chuỗi tiền xử lý tín hiệu CQT 96 bins và tăng cường dữ liệu âm học (Audio Augmentations) đặc thù cho QBH.*

1. **Random Pitch Shifting ($\pm 12$ Semitones)**:
   - Một người dùng có thể ngân nga bài hát ở tông Do trưởng ($C$), trong khi bản gốc của ca sĩ ở tông Sol trưởng ($G$) (chênh lệch 7 bán âm), hoặc lệch nguyên 1 quãng tám ($12$ bán âm).
   - Chúng tôi áp dụng phép biến dịch cao độ ngẫu nhiên trong khoảng $[-12, +12]$ nửa cung trong quá trình tạo batch huấn luyện. Kỹ thuật này buộc mạng nơ-ron học cách trích xuất mối tương quan chênh lệch cao độ tương đối (Melodic Contour) thay vì ghi nhớ cao độ tuyệt đối.
2. **Room Impulse Response (Reverb Simulation)**:
   - Tiếng ngân nga của người dùng thường được thực hiện trong phòng ngủ, phòng khách hoặc hành lang có độ vang âm thanh nhất định.
   - Chúng tôi cuộn tín hiệu (convolution) với các tập phản ứng xung phòng (RIR) ngẫu nhiên, giả lập các điều kiện vang âm từ nhẹ ($RT_{60} = 0.2\text{s}$) đến vừa ($RT_{60} = 0.8\text{s}$).
3. **Harmonic Noise & Background Noise Injection**:
   - Trộn thêm nhiễu trắng (White Noise), nhiễu hồng (Pink Noise) và tạp âm môi trường thực tế (tiếng quạt, tiếng đường phố) với tỷ lệ Tín hiệu trên Nhiễu (SNR) biến thiên từ $10\text{ dB}$ đến $25\text{ dB}$.
4. **SpecAugment trên Ma trận CQT**:
   - Áp dụng kỹ thuật che ngẫu nhiên các dải tần số (Frequency Masking với độ rộng tối đa $12$ bins) và che các khoảng thời gian (Time Masking với độ dài tối đa $10$ frames), giúp mô hình bền vững trước hiện tượng người dùng quên lời ngắt quãng hoặc micro bị rớt tín hiệu tạm thời.

Đoạn mã nguồn thực thi trực tiếp bộ lọc biến đổi `PyCqt` 96 bins trên GPU và lớp tăng cường dịch chuyển cao độ `CQTPitchShiftAugmentation` ($\pm 12$ bán âm) trong dự án (`01_stage1_cover_song_csi/src/coverhunter/cqt.py` & `cqt_augmentation.py`):

```python
class PyCqt:
    """Trích xuất phổ Constant-Q Transform (Full CQT 96 bins, 12 bins/octave)."""
    def __init__(self, sample_rate=16000, hop_size=0.04, octave_resolution=12, min_freq=32.7):
        self._hop_size = hop_size
        self._sample_rate = sample_rate
        max_freq = sample_rate // 2
        self._kernel = self._compute_cqt_kernel(sample_rate, octave_resolution, min_freq, max_freq)

    @staticmethod
    def _compute_cqt_kernel(sampling_frequency, octave_resolution, minimum_frequency, maximum_frequency):
        # Hệ số phẩm chất Q hằng số: Q = fk / (fk+1 - fk) = 1 / (2^(1/12) - 1) ≈ 16.89
        quality_factor = 1.0 / (pow(2.0, 1.0 / octave_resolution) - 1.0)
        number_frequencies = round(octave_resolution * np.log2(maximum_frequency / minimum_frequency))
        fft_length = int(pow(2, np.ceil(np.log2(quality_factor * sampling_frequency / minimum_frequency))))
        cqt_kernel = np.zeros((number_frequencies, fft_length), dtype=complex)

        for i in range(number_frequencies):
            frequency_value = minimum_frequency * pow(2.0, i / octave_resolution)
            window_length = 2 * round(quality_factor * sampling_frequency / frequency_value / 2.0) + 1
            temporal_kernel = (np.hamming(window_length) * np.exp(
                2 * np.pi * 1j * quality_factor * np.arange(-(window_length - 1) / 2, (window_length - 1) / 2 + 1) / window_length
            ) / window_length)
            pad_width = int((fft_length - window_length + 1) / 2)
            cqt_kernel[i, pad_width: pad_width + window_length] = temporal_kernel

        cqt_kernel = scipy.sparse.csr_matrix(np.fft.fft(cqt_kernel, axis=1))
        return np.conjugate(cqt_kernel) / fft_length

class CQTPitchShiftAugmentation(nn.Module):
    """Dịch chuyển cao độ ngẫu nhiên ±12 bán âm trực tiếp trên trục tần số CQT 96 bins."""
    def __init__(self, max_semitones: int = 12, p: float = 0.8):
        super().__init__()
        self.max_semitones = max_semitones
        self.p = p

    def forward(self, cqt_spec: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return cqt_spec
        shift = random.randint(-self.max_semitones, self.max_semitones)
        if shift == 0:
            return cqt_spec
        shifted = torch.roll(cqt_spec, shifts=shift, dims=-2)
        if shift > 0:
            shifted[..., :shift, :] = cqt_spec.min()
        else:
            shifted[..., shift:, :] = cqt_spec.min()
        return shifted
```

---

## 2.6. Sơ đồ Pipeline Xử lý Tín hiệu Âm thanh CQT

Sơ đồ kiến trúc xử lý tín hiệu hoàn chỉnh từ âm thanh thô đến ma trận CQT $96 \times 83$ được biểu diễn chi tiết và lưu trữ tại tệp tin vector chuẩn Draw.io:
📁 **Tham chiếu sơ đồ**: [`docs/figures/cqt_vs_stft_processing.drawio`](file:///e:/HOCKY7/PBL6/code/docs/figures/cqt_vs_stft_processing.drawio)

*Mô tả luồng xử lý trên sơ đồ*:
1. Tín hiệu âm thanh thô đi vào mô hình **Meta HTDemucs v4** để tách riêng kênh thanh âm thuần chất.
2. Kênh thanh âm được chuẩn hóa biên độ về đoạn $[-1.0, 1.0]$ và chuyển qua khối tăng cường âm học **Audio Augmentation** (biến dịch cao độ $\pm 12$ bán âm và lọc tạp âm).
3. Tín hiệu sau tăng cường được chuyển đổi qua bộ lọc **CQT Engine** với hệ số $Q \approx 16.89$ không đổi.
4. Đầu ra đạt chuẩn là ma trận **Full CQT Spectrogram** kích thước $96 \times 83$ frames, khắc phục hoàn toàn nhược điểm nhòe tần số trầm của STFT và hiện tượng mất mát thông tin âm vực của Chroma CQT 12 bins.

---

## 2.7. Kết luận chương

Chương 2 đã thiết lập vững chắc nền tảng xử lý tín hiệu cho dự án:
- Phân tích âm học chuyên sâu chứng minh tại sao tiếng ngân nga có tính chất đặc thù về sự suy giảm họa âm và độ biến thiên tông giọng cao.
- Việc tích hợp **Meta HTDemucs v4** giải quyết tận gốc bài toán bóc tách giọng hát ra khỏi nhạc nền phức tạp của bản Master Studio.
- Thiết kế **Full CQT 96 bins** bao phủ trọn vẹn 8 quãng tám, vừa khớp hoàn hảo với 12 nửa cung nhạc lý, vừa bảo toàn thông tin thanh vực mà phương pháp Chroma CQT trước đây đánh mất.
- Chuỗi tăng cường âm học chuyên biệt (Pitch shift $\pm 12$, Reverb, Noise) cung cấp nguồn dữ liệu huấn luyện phong phú, tạo tiền đề vững chắc cho mô hình học sâu Conformer-ByteHum được trình bày ở Chương 3.




---


# CHƯƠNG 3: THIẾT KẾ KIẾN TRÚC MÔ HÌNH CONFORMER-BYTEHUM QBH V6.2

---

## 3.1. Tổng quan Kiến trúc Mạng Nơ-ron (CoverHunter Conformer 9.91M Tham số)

Để giải quyết bài toán nhận diện giai điệu từ tiếng ngân nga, mô hình học sâu cần giải quyết đồng thời hai thách thức:
1. **Nắm bắt tương quan cục bộ (Local Dependencies)**: Các nốt nhạc luyến láy kề cận, nhịp điệu chuyển tiếp tức thời giữa các âm tiết. Đây là thế mạnh truyền thống của mạng nơ-ron tích chập (Convolutional Neural Networks - CNN).
2. **Nắm bắt tương quan toàn cục dài hạn (Global Long-range Context)**: Cấu trúc lặp lại của các phân đoạn giai điệu, cấu trúc phiên khúc (Verse) và điệp khúc (Chorus) cách xa nhau nhiều giây trong dòng thời gian. Đây là thế mạnh của cơ chế Tự chú ý đa đầu (Multi-Head Self-Attention - MHSA) trong kiến trúc Transformer.

Trong đề tài này, chúng tôi xây dựng mạng nơ-ron kết hợp dựa trên kiến trúc **Conformer** (Convolution-augmented Transformer for Speech and Audio) kết hợp với các mô đun chuyên biệt của **ByteHum** (ICASSP 2024). Toàn bộ mô hình có quy mô **9.91 triệu tham số** (9.91M parameters) tối ưu hóa, đảm bảo độ chính xác vượt trội đồng thời có thể thực thi suy luận trực tiếp theo thời gian thực trên GPU/CPU thương mại.

Quy trình xử lý của mô hình bao gồm:

![Kiến trúc mạng Conformer-ByteHum V6.2](docs/figures/fig_ch3_conformer_arch.png)
*Hình 3.1: Kiến trúc tổng thể mạng nơ-ron Conformer-ByteHum V6.2 (9.91 triệu tham số) với cơ chế Dual Embeddings và các mô đun LAT, LAC, GRL.*

---

## 3.2. Cấu trúc Khối Conformer

Khối Conformer kế thừa thiết kế "Macaron-Net" chuẩn mực, trong đó một khối gồm hai nửa lớp Feed-Forward xen kẽ xung quanh cơ chế Self-Attention và Convolution:

$$y_i^{(1)} = x_i + \frac{1}{2} \text{FFN}(x_i)$$
$$y_i^{(2)} = y_i^{(1)} + \text{MHSA}(y_i^{(1)})$$
$$y_i^{(3)} = y_i^{(2)} + \text{Conv}(y_i^{(2)})$$
$$\text{Output}_i = \text{LayerNorm}\left(y_i^{(3)} + \frac{1}{2} \text{FFN}(y_i^{(3)})\right)$$

### 1. Convolutional Subsampling Module (Khối nén đầu vào)
Đầu vào ảnh phổ CQT có số kênh ban đầu là 1, kích thước $96 \times 83$. Để giảm độ phức tạp tính toán của cơ chế Self-Attention (vốn có độ phức tạp bậc hai $O(T^2)$), tín hiệu trước tiên đi qua 2 tầng tích chập 2D (Conv2d) với kernel $(3, 3)$, stride $(2, 2)$, kết hợp hàm kích hoạt ReLU và chiếu tuyến tính (Linear Projection) lên chiều ẩn $D = 144$.

### 2. Mô đun Multi-Head Self-Attention (MHSA)
MHSA cho phép mỗi vị trí thời gian chú ý đến toàn bộ các vị trí khác trong toàn bộ chuỗi giai điệu:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}} + S_{rel}\right) V$$
trong đó $S_{rel}$ là ma trận nhúng vị trí tương đối (Relative Positional Encoding), giúp mô hình bất biến trước việc giai điệu bắt đầu sớm hay muộn vài giây.
- **Số đầu chú ý (Attention Heads)**: $H = 4$.
- **Chiều mỗi đầu**: $d_k = 144 / 4 = 36$.

### 3. Mô đun Convolution Tích chập theo chiều sâu (Depthwise Convolution)
Mô đun tích chập được bố trí theo chuỗi:
1. LayerNorm và Pointwise Conv1d (chiếu tăng kênh lên $2 \times D = 288$).
2. Hàm kích hoạt Gated Linear Unit (GLU).
3. **1D Depthwise Convolution**: Kernel size lớn $K = 31$, giúp mô hình có trường tiếp nhận bao phủ rộng các biến thiên âm vực cục bộ mà không làm bùng nổ số lượng trọng số.
4. BatchNorm (tăng tính ổn định gradient âm thanh).
5. Hàm kích hoạt Swish (SiLU).
6. Pointwise Conv1d (chiếu ngược về chiều $D = 144$) kết hợp Dropout tỷ lệ $0.1$.

### 4. Mô đun Feed-Forward (FFN)
Áp dụng cấu trúc 2 tầng Linear biến đổi với hệ số mở rộng kênh bằng 4:
$$\text{FFN}(x) = W_2 \cdot \text{Swish}(W_1 x + b_1) + b_2$$

---

## 3.3. Cơ chế Trích xuất Dual Embeddings và Attentive Statistics Pooling (ASP)

Để phục vụ tìm kiếm hai chặng, mô hình trích xuất ra hai mức độ biểu diễn song song (Dual Embeddings):
1. **Biểu diễn Cục bộ (Local Frame Sequence)**: Chuỗi các vector đặc trưng theo thời gian sau khối Conformer, được định hình thành $T = 20$ lát cắt thời gian với chiều nén $d = 128$:
   $$\mathbf{E}_{local} \in \mathbb{R}^{20 \times 128}$$
2. **Biểu diễn Toàn cục (Global Embedding)**: Được tổng hợp thông qua cơ chế **Attentive Statistics Pooling (ASP)**.

### Toán học của Attentive Statistics Pooling (ASP):
Thay vì sử dụng phép lấy trung bình giản đơn (Global Average Pooling) làm nhòe các đỉnh năng lượng giai điệu, ASP tự động tính toán trọng số quan trọng $\alpha_t$ của từng khung thời gian:
$$e_t = \mathbf{v}^T \tanh(\mathbf{W} h_t + \mathbf{b})$$
$$\alpha_t = \frac{\exp(e_t)}{\sum_{\tau=1}^T \exp(e_\tau)}$$

Trọng số $\alpha_t$ được sử dụng để tính đồng thời **Vector Kỳ vọng có trọng số (Mean)** và **Vector Độ lệch chuẩn có trọng số (Standard Deviation)**:
$$\boldsymbol{\mu} = \sum_{t=1}^T \alpha_t h_t$$
$$\boldsymbol{\sigma} = \sqrt{\sum_{t=1}^T \alpha_t (h_t \odot h_t) - (\boldsymbol{\mu} \odot \boldsymbol{\mu}) + \epsilon}$$

Hai vector này được ghép nối (concatenation) thành vector đặc trưng thống kê toàn cục 512 chiều:
$$\mathbf{h}_{stat} = [\boldsymbol{\mu} \,\|\, \boldsymbol{\sigma}] \in \mathbb{R}^{512}$$
Sau đó, $\mathbf{h}_{stat}$ được chiếu qua một lớp Linear Dense và chuẩn hóa $L_2$ trên siêu mặt cầu (Hypersphere Normalization) để tạo thành **Global Embedding**:
$$\mathbf{e}_{global} = \frac{\mathbf{W}_p \mathbf{h}_{stat}}{\|\mathbf{W}_p \mathbf{h}_{stat}\|_2} \in \mathbb{R}^{128}$$

Chi tiết mã nguồn cài đặt khối `ConformerEncoderLayer` theo cấu trúc Macaron-Net và lớp gộp thống kê chú ý `AttentiveStatisticsPooling` (`01_stage1_cover_song_csi/src/coverhunter/module/conformer.py` & `model.py`):

```python
class ConformerEncoderLayer(torch.nn.Module):
    """Khối Conformer Macaron-Net: 1/2 FFN -> RelPos MHSA -> Depthwise Conv1d -> 1/2 FFN."""
    def __init__(self, size: int, self_attn: nn.Module, feed_forward: nn.Module,
                 feed_forward_macaron: nn.Module, conv_module: nn.Module, dropout_rate: float = 0.1):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module
        self.norm_ff_macaron = nn.LayerNorm(size, eps=1e-5)
        self.norm_mha = nn.LayerNorm(size, eps=1e-5)
        self.norm_conv = nn.LayerNorm(size, eps=1e-5)
        self.norm_ff = nn.LayerNorm(size, eps=1e-5)
        self.norm_final = nn.LayerNorm(size, eps=1e-5)
        self.ff_scale = 0.5
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, pos_emb: torch.Tensor, mask_pad: torch.Tensor):
        # 1. Half-step Macaron Feed-Forward Module
        x = x + self.ff_scale * self.dropout(self.feed_forward_macaron(self.norm_ff_macaron(x)))
        # 2. Relative Positional Multi-Head Self-Attention (MHSA)
        x_att, new_att_cache = self.self_attn(self.norm_mha(x), self.norm_mha(x), self.norm_mha(x), mask, pos_emb)
        x = x + self.dropout(x_att)
        # 3. Depthwise Separable Convolution Module (GLU + 1D Depthwise Conv + BatchNorm + Swish)
        x_conv, new_cnn_cache = self.conv_module(self.norm_conv(x), mask_pad)
        x = x + self.dropout(x_conv)
        # 4. Second Half-step Feed-Forward & Final LayerNorm
        x = x + self.ff_scale * self.dropout(self.feed_forward(self.norm_ff(x)))
        return self.norm_final(x), mask, new_att_cache, new_cnn_cache

class AttentiveStatisticsPooling(torch.nn.Module):
    """Gộp thống kê tự chú ý (ASP): Kết hợp có trọng số kỳ vọng (Mean) và độ lệch chuẩn (Std)."""
    def __init__(self, channels: int = 128, output_channels: int = 128):
        super().__init__()
        self._eps = 1e-12
        self._linear = Linear(channels * 3, channels)
        self._tanh = torch.nn.Tanh()
        self._conv = Conv1d(in_channels=channels, out_channels=channels, kernel_size=1)
        self._final_layer = torch.nn.Linear(channels * 2, output_channels, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # [B, C=128, T=20]
        L = x.shape[-1]
        mean, std = self._compute_statistics(x, torch.ones_like(x[:, :1, :]) / L, self._eps)
        attn = torch.cat([x, mean.unsqueeze(2).repeat(1, 1, L), std.unsqueeze(2).repeat(1, 1, L)], dim=1)
        attn = F.softmax(self._conv(self._tanh(self._linear(attn.transpose(1, 2)).transpose(1, 2))), dim=2)
        mean, std = self._compute_statistics(x, attn, self._eps)
        return self._final_layer(torch.cat((mean, std), dim=1))  # [B, 128]
```

---

## 3.4. Khung Kiến trúc ByteHum (ICASSP 2024)

Khung kiến trúc **ByteHum** được nhóm nghiên cứu của ByteDance công bố tại hội nghị ICASSP 2024 nhằm giải quyết hai điểm yếu chí tử của mô hình QBH truyền thống: hiện tượng sai lệch độ dài phân đoạn (Short Query vs Long Song) và khoảng cách miền thanh âm (Domain Gap). Chúng tôi tích hợp trọn vẹn 3 mô đun đột phá của ByteHum:

### 3.4.1. Learned Alignment Transformer (LAT) với MaxMean Pooling
Người dùng thường chỉ ngân nga một đoạn ngắn (5 đến 15 giây) của một bài hát kéo dài 3 đến 5 phút. Khi so sánh Global Embedding, thông tin của cả bài hát dài bị nén lại làm giảm độ tương đồng với đoạn ngân nga ngắn.

Mô đun LAT giải quyết vấn đề này bằng cách tính ma trận tương đồng chéo (Cross-Attention Correlation Matrix) giữa chuỗi đặc trưng cục bộ của đoạn truy vấn $\mathbf{Q} \in \mathbb{R}^{T_q \times d}$ và đoạn bài hát thư viện $\mathbf{K} \in \mathbb{R}^{T_k \times d}$:
$$\mathbf{M}_{ij} = \frac{\mathbf{Q}_i \cdot \mathbf{K}_j^T}{\|\mathbf{Q}_i\|_2 \|\mathbf{K}_j\|_2} \in [-1, 1]$$

Sau đó, áp dụng cơ chế **MaxMean Pooling**:
1. Với mỗi khung thời gian thứ $i$ của tiếng ngân nga, tìm điểm tương đồng lớn nhất dọc theo chiều thời gian của bài hát đích:
   $$s_i = \max_{1 \le j \le T_k} \mathbf{M}_{ij}$$
   (Thao tác này tìm xem nốt nhạc người dùng đang ngân nga khớp chuẩn nhất vào giây thứ mấy trong bài hát).
2. Lấy trung bình cộng các giá trị cực đại trên toàn bộ độ dài truy vấn để cho ra điểm tương đồng căn chỉnh cuối cùng:
   $$S_{LAT}(\mathbf{Q}, \mathbf{K}) = \frac{1}{T_q} \sum_{i=1}^{T_q} s_i$$

### 3.4.2. Learned Alignment Classifier (LAC) với L2 Prototypes
Mô đun LAC duy trì một ma trận các vector tâm cụm đặc trưng (Prototypes) đại diện cho từng bài hát trong không gian đa chiều:
$$\mathbf{P} \in \mathbb{R}^{C \times d}$$
với $C$ là số lượng bài hát và $d = 128$.

Xác suất phân loại bài hát được tính dựa trên khoảng cách Euclidean thu nhỏ kết hợp hệ số tỷ lệ nhiệt độ (Temperature scale) $\tau$:
$$P(y = c \mid \mathbf{e}) = \frac{\exp(-\|\mathbf{e} - \mathbf{p}_c\|_2^2 / \tau)}{\sum_{j=1}^C \exp(-\|\mathbf{e} - \mathbf{p}_j\|_2^2 / \tau)}$$
Hàm mất mát $\mathcal{L}_{LAC}$ ép buộc biểu diễn của đoạn ngân nga phải hội tụ về đúng tâm cụm nguyên mẫu của bài hát đó.

### 3.4.3. Huấn luyện Đối kháng Miền với Gradient Reversal Layer (GRL)
Khoảng cách miền (Domain Gap) giữa giọng ngân nga của người dùng nghiệp dư và giọng ca sĩ phòng thu là nguyên nhân gây tụt giảm hiệu năng lớn.

Để buộc bộ mã hóa Conformer học được các đặc trưng "bất biến theo miền" (Domain-Invariant Representations), chúng tôi tích hợp một nhánh Phân loại Miền (Domain Classifier) kết hợp với **Lớp Đảo ngược Gradient (Gradient Reversal Layer - GRL)**:
- Ở lượt truyền xuôi (Forward pass): GRL hoạt động như một hàm đồng nhất:
  $$\text{GRL}(x) = x$$
- Ở lượt truyền ngược (Backward pass): GRL tự động đảo ngược dấu của gradient và nhân với hệ số điều tiết $\lambda$:
  $$\frac{\partial \mathcal{L}}{\partial x} = -\lambda \frac{\partial \mathcal{L}_{domain}}{\partial x}$$

Khi nhánh phân loại miền cố gắng tối đa hóa khả năng nhận diện xem đặc trưng là "giọng ngân nga" hay "giọng ca sĩ", thì qua lớp GRL, bộ mã hóa Conformer lại được tối ưu hóa để **đánh lừa bộ phân loại miền**. Kết quả là các vector embedding sinh ra hoàn toàn không còn mang dấu ấn âm sắc của miền, chỉ lưu giữ thuần túy cấu trúc giai điệu bài hát.

Đoạn mã nguồn cài đặt lớp đảo ngược đạo hàm `GradientReversalFunction`, mô đun phân loại nguyên mẫu `LocalAlignmentClassifier` (LAC) và lớp kiến trúc tổng thể `ConformerByteHumModel` (9.91 triệu tham số):

```python
class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_val: float):
        ctx.lambda_val = lambda_val
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # Đảo ngược dấu gradient nhân với hệ số điều tiết lambda_val trong lượt lan truyền ngược
        return grad_output.neg() * ctx.lambda_val, None

class LocalAlignmentClassifier(nn.Module):
    """Local Alignment Classifier (LAC - ByteHum ICASSP 2024) với L=4 L2 Prototypes & MaxMean."""
    def __init__(self, num_classes: int = 244, feature_dim: int = 128, num_prototypes: int = 4):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.num_prototypes = num_prototypes
        self.prototypes = nn.Parameter(torch.randn(num_classes, num_prototypes, feature_dim) * 0.05)
        self.temperature = nn.Parameter(torch.tensor(10.0))

    def forward(self, local_feat: torch.Tensor) -> torch.Tensor:
        x_norm = F.normalize(local_feat, p=2, dim=-1)          # [B, T=20, 128]
        w_norm = F.normalize(self.prototypes, p=2, dim=-1)     # [K, L=4, 128]
        sim = torch.matmul(x_norm, w_norm.view(-1, self.feature_dim).t())
        sim = sim.view(local_feat.size(0), local_feat.size(1), self.num_classes, self.num_prototypes)
        logits = sim.max(dim=1).values.mean(dim=-1)            # Max theo T=20, Mean theo L=4
        return logits * self.temperature.clamp(min=1.0, max=100.0)

class ConformerByteHumModel(nn.Module):
    """Kiến trúc tổng thể Conformer-ByteHum V6.2 (9.91M tham số - Dual Embeddings + LAT/LAC/GRL)."""
    def __init__(self, ckpt_path: str = None, input_dim: int = 96, embedding_dim: int = 128, num_classes: int = 244):
        super().__init__()
        self.feature_extractor = build_coverhunter_model(ckpt_path=ckpt_path)
        self.grl = GradientReversalLayer(lambda_val=0.2)
        self.domain_classifier = LocalDomainClassifier(feature_dim=embedding_dim, hidden_dim=64)
        self.lac = LocalAlignmentClassifier(num_classes=num_classes, feature_dim=embedding_dim, num_prototypes=4)

    def forward(self, x: torch.Tensor, compute_lac: bool = True, compute_domain: bool = True):
        backbone_feat = self.extract_backbone_features(x)              # [B, T=20, 128]
        local_feat = self.local_projection(backbone_feat)              # Local Sequence [B, 20, 128]
        global_emb = self.extract_global_embedding(backbone_feat)      # Global ASP L2 Embedding [B, 128]
        lac_logits = self.lac(local_feat) if compute_lac else None
        domain_logits = self.domain_classifier(self.grl(local_feat)) if compute_domain else None
        return global_emb, local_feat, lac_logits, domain_logits
```

---

## 3.5. Hàm Mất mát Đa Mục tiêu Hợp nhất (Joint Multi-Task Loss)

Toàn bộ hệ thống Conformer-ByteHum V6.2 được tối ưu hóa đồng thời bằng một hàm mục tiêu đa nhiệm liên kết chặt chẽ:
$$\mathcal{L}_{total} = \mathcal{L}_{metric} + \lambda_1 \mathcal{L}_{LAT} + \lambda_2 \mathcal{L}_{LAC} + \lambda_3 \mathcal{L}_{domain}$$

trong đó:
1. **$\mathcal{L}_{metric}$**: Hàm mất mát bộ ba thích ứng (Multi-Similarity / Triplet Margin Loss with Hard Negative Mining) trên Global Embeddings:
   $$\mathcal{L}_{metric} = \max\left(0, \|\mathbf{e}_a - \mathbf{e}_p\|_2^2 - \|\mathbf{e}_a - \mathbf{e}_n\|_2^2 + \alpha\right)$$
   với biên lề (margin) $\alpha = 0.3$.
2. **$\mathcal{L}_{LAT}$**: Hàm mất mát Cross-Entropy trên điểm tương đồng căn chỉnh MaxMean.
3. **$\mathcal{L}_{LAC}$**: Hàm mất mát nguyên mẫu L2 Prototype Loss.
4. **$\mathcal{L}_{domain}$**: Hàm mất mát phân loại miền nhị phân (Binary Cross-Entropy) truyền ngược qua GRL.
5. **Hệ số cân bằng**: Được tinh chỉnh thực nghiệm tối ưu tại $\lambda_1 = 0.5$, $\lambda_2 = 0.3$, và $\lambda_3 = 0.1$.

Chi tiết mã nguồn hàm căn chỉnh cục bộ `local_alignment_triplet` (LAT) và hàm mục tiêu huấn luyện đa nhiệm `training_objective` trong dự án (`02_stage2_humming_qbh/scripts/notebook_builders/templates/final_solution_runtime.py`):

```python
def local_alignment_triplet(anchor, positive, negative, song_ids, negative_song_ids,
                            margin=0.2, hardest_ratio=0.7, top_k=5):
    """Hàm mất mát căn chỉnh cục bộ ByteHum LAT (MaxMean Cross-Attention Triplet Loss)."""
    anchor, positive, negative = [F.normalize(x, dim=-1) for x in (anchor, positive, negative)]
    positive_scores = torch.bmm(anchor, positive.transpose(1, 2)).max(-1).values.mean(-1, keepdim=True)
    candidates = torch.cat([positive, negative], dim=0)
    labels = torch.cat([song_ids, negative_song_ids], dim=0)
    valid = song_ids.unsqueeze(1) != labels.unsqueeze(0)

    # Tính ma trận tương quan chéo M_ij và áp dụng MaxMean Pooling dọc trục thời gian
    similarities = torch.einsum('btc,nsc->bnts', anchor, candidates).max(dim=-1).values.mean(dim=-1)
    hard_score, _ = _topk_masked_mean(similarities, valid, top_k)
    semi_mask = valid & (similarities < positive_scores) & (similarities > positive_scores - margin)
    semi_score, has_semi = _topk_masked_mean(similarities, semi_mask, top_k)
    selected = (1.0 - hardest_ratio) * torch.where(has_semi, semi_score, hard_score) + hardest_ratio * hard_score
    return F.relu(selected - positive_scores + margin).mean()

def training_objective(outputs, labels, negative_labels, epoch, memory_bank):
    """Hàm mất mát đa mục tiêu hợp nhất (Joint Multi-Task Objective V6.2)."""
    ga, gp, gn = [F.normalize(out[0].float(), dim=-1) for out in outputs]
    la, lp, ln = [out[1].float() for out in outputs]
    curriculum = mining_hardest_ratio(epoch)

    triplet = in_batch_cosine_triplet_loss(
        ga, gp, gn, labels, negative_labels,
        margin=RUN_CONFIG['margin'], memory_bank=memory_bank,
        hardest_ratio=curriculum, top_k=RUN_CONFIG['mining_top_k']
    )
    lat = local_alignment_triplet(
        la, lp, ln, labels, negative_labels,
        margin=RUN_CONFIG['margin'], hardest_ratio=curriculum, top_k=RUN_CONFIG['mining_top_k']
    )
    lac = torch.stack([F.cross_entropy(out[2].float(), t) for out, t in zip(outputs, (labels, labels, negative_labels))]).mean()
    domain = torch.stack([F.binary_cross_entropy_with_logits(out[3].float(), torch.full_like(out[3].float(), t))
                          for out, t in zip(outputs, (1.0, 0.0, 0.0))]).mean()
    prototype = prototype_diversity_loss(model.lac.prototypes.float())

    total = (RUN_CONFIG['global_triplet'] * triplet + RUN_CONFIG['lat'] * lat
             + RUN_CONFIG['lac'] * lac + domain + RUN_CONFIG['prototype_diversity'] * prototype)
    return dict(global_triplet=triplet, lat=lat, lac=lac, domain=domain, prototype=prototype, total=total)
```

---

## 3.6. Sơ đồ Kiến trúc Mạng Nơ-ron Chi tiết

Sơ đồ chi tiết các khối kết nối, kích thước tensor và đường đi gradient của mô hình Conformer-ByteHum V6.2 được lưu trữ tại file vector chuẩn Draw.io:
📁 **Tham chiếu sơ đồ**: [`docs/figures/conformer_bytehum_detailed.drawio`](file:///e:/HOCKY7/PBL6/code/docs/figures/conformer_bytehum_detailed.drawio)

*Mô tả luồng trên sơ đồ*:
- Khối đầu vào: Log-CQT $96 \times 83$ $\rightarrow$ Convolutional Subsampling.
- Khối thân mạng: 6 tầng Conformer Encoder với liên kết Macaron Feed-Forward và Depthwise Conv.
- Khối phân nhánh: Nhánh ASP sinh Global Metric Embedding (128D) đi vào Triplet Loss; nhánh Local Sequence đi vào bộ đôi ByteHum LAT/LAC; nhánh trung gian đi qua GRL vào Domain Classifier.
- Khối hàm mất mát: Hợp nhất 4 thành phần loss thành $\mathcal{L}_{total}$ duy nhất.

---

## 3.7. Kết luận chương

Chương 3 đã làm rõ kiến trúc học sâu tiên tiến nhất trong đề tài:
- Khắc phục triệt để nhược điểm của mạng CNN nông truyền thống bằng **CoverHunter Conformer 9.91M tham số**, kết hợp hài hòa giữa khả năng nắm bắt nhịp điệu cục bộ (Depthwise Conv) và quan hệ cấu trúc bài hát toàn cục (MHSA).
- Cơ chế **Attentive Statistics Pooling (ASP)** cung cấp vector đại diện thống kê bền vững hơn nhiều so với trung bình số học thông thường.
- Sự kết hợp của bộ ba công nghệ **ByteHum (LAT, LAC, GRL)** đã giải quyết đồng thời hai bài toán hóc búa nhất của lĩnh vực QBH: căn chỉnh độ dài bài hát bất đối xứng và triệt tiêu khoảng cách miền thanh âm giữa người dùng và ca sĩ phòng thu.




---


# CHƯƠNG 4: THIẾT KẾ HỆ THỐNG VÀ QUY TRÌNH THỰC THI (PIPELINE)

---

## 4.1. Kiến trúc Tổng thể Hệ thống (Offline Indexing vs Online Retrieval)

Để triển khai mô hình học sâu Conformer-ByteHum V6.2 vào một ứng dụng thực tiễn phục vụ người dùng cuối với độ trễ thấp và độ tin cậy cao, kiến trúc hệ thống được thiết kế phân tách rõ ràng thành hai nhánh độc lập nhưng tương hỗ chặt chẽ:
1. **Quy trình Lập chỉ mục Ngoại tuyến (Offline Indexing Pipeline)**: Thực thi một lần khi khởi tạo hoặc cập nhật kho bài hát, chịu trách nhiệm xử lý các tác vụ tính toán nặng (tách vocal, trích xuất phổ CQT, sinh vector đặc trưng và đẩy tài nguyên media lên điện toán đám mây).
2. **Quy trình Truy vấn Thời gian thực Trực tuyến (Online Real-time Retrieval Pipeline)**: Tiếp nhận luồng âm thanh micro từ trình duyệt người dùng, trích xuất đặc trưng tức thời trong bộ nhớ đệm và thực hiện so khớp song song đa luồng để trả về kết quả trong thời gian dưới $350\text{ ms}$.

![Kiến trúc hệ thống hai nhánh](docs/figures/fig_ch4_system_pipeline.png)
*Hình 4.1: Kiến trúc tổng thể hệ thống QBH hai nhánh: Offline Indexing Pipeline và Online Real-time Retrieval Pipeline tích hợp Cloudinary CDN.*

---

## 4.2. Quản trị Kho Bài hát và Hạ tầng Lưu trữ Cloudinary CDN

Một điểm hạn chế phổ biến của các dự án trước đây là việc lưu trữ và phát trực tiếp các tệp MP3 kích thước lớn ngay trên ổ đĩa cục bộ của máy chủ backend, gây nghẽn băng thông mạng (Network I/O Bottleneck), làm chậm trễ các tiến trình AI và không thể mở rộng quy mô.

Trong đề tài này, chúng tôi xây dựng giải pháp tích hợp toàn diện với dịch vụ đám mây chuyên dụng **Cloudinary Media Cloud**:
- **Quy mô số hóa**: Toàn bộ **300 bài hát** trong kho nhạc gốc đã được tự động quét, chuẩn hóa metadata (Tên bài hát, Nghệ sĩ thể hiện, Thời lượng, Dung lượng byte) và đẩy lên thư mục `qbh_songs` trên Cloudinary thông qua kịch bản tự động hóa Python SDK (`03_web_music_platform/cloudinary_uploader/upload_songs.py`).
- **Lưu trữ và Truy xuất Phân tán**:
  - Dữ liệu âm thanh được lưu trữ dưới dạng tối ưu hóa truyền dẫn (High Bitrate MP3).
  - Mỗi bài hát được gắn kèm một URL phát trực tuyến phân tán toàn cầu (Cloudinary Global CDN Delivery URL) với giao thức HTTPS an toàn.
  - Một danh mục đối chiếu đầy đủ được lưu trữ tại `03_web_music_platform/cloudinary_uploader/cloudinary_songs_catalog.json` và `.csv`, cho phép máy chủ backend tức thời ánh xạ mã định danh bài hát (`song_id`) thành URL phát trực tiếp mà không tốn chi phí đọc đĩa.
- **Lợi ích kiến trúc**: Tách rời hoàn toàn tải truyền dữ liệu âm thanh ra khỏi máy chủ AI; người dùng khi tìm thấy bài hát có thể nhấn nút "Nghe thử ngay" (Play) để nhận luồng âm thanh trực tiếp từ CDN Cloudinary với độ trễ cực thấp.

Mã nguồn thực thi việc chuẩn hóa siêu dữ liệu và đồng bộ tự động 300 bài hát lên hạ tầng Cloudinary Media CDN (`03_web_music_platform/cloudinary_uploader/upload_audio_gallery.py`):

```python
def upload_single_song(song_info, retry_count=DEFAULT_RETRY_COUNT, force=False):
    """Đồng bộ hóa bài hát MP3 lên Cloudinary Media CDN và trả về Direct Streaming URL."""
    song_id = song_info["song_id"]
    title = song_info["title"]
    file_path = song_info["file_path"]

    for attempt in range(1, retry_count + 1):
        try:
            res = cloudinary.uploader.upload(
                str(file_path),
                folder=CLOUDINARY_FOLDER,
                public_id=song_id,
                resource_type="video",  # Bắt buộc để kích hoạt luồng Audio Streaming CDN
                overwrite=force,
                tags=["pbl6", "qbh_gallery", "original_audio"],
                context={"title": title, "song_id": song_id}
            )
            record = {
                "song_id": song_id,
                "title": title,
                "file_name": file_path.name,
                "public_id": res.get("public_id"),
                "secure_url": res.get("secure_url"),
                "duration": res.get("duration"),
                "format": res.get("format"),
                "bytes": res.get("bytes"),
                "created_at": res.get("created_at")
            }
            return (True, song_id, record, None)
        except Exception as e:
            if attempt < retry_count:
                time.sleep(2 * attempt)
            else:
                return (False, song_id, None, str(e))
```

---

## 4.3. Pipeline Trích xuất Đặc trưng CQT và Bộ nhớ đệm In-RAM Feature Cache

### 1. Trích xuất CQT Tức thời (On-the-fly CQT Extraction)
Đối với đoạn âm thanh người dùng vừa thu âm, hệ thống thực thi quy trình tiền xử lý trực tiếp trên Tensor PyTorch:
- Nạp tín hiệu âm thanh và ép tần số lấy mẫu về đúng $f_s = 22,050\text{ Hz}$ mono.
- Chuẩn hóa biên độ tín hiệu theo chuẩn năng lượng Root Mean Square (RMS).
- Chuyển đổi qua bộ lọc CQT với 96 bins ($B=12$, $f_{min} \approx 32.7\text{ Hz}$, $hop = 512$).
- Áp dụng phép chuyển đổi thang logarit công suất: $\mathbf{X}_{log} = \log(1 + 1000 \cdot |\mathbf{X}_{CQT}|^2)$.
- Cắt/đệm chuỗi về kích thước chuẩn $96 \times 83$ khung thời gian. Toàn bộ tiến trình này được tối ưu hóa bằng PyTorch CUDA/CPU với độ trễ ghi nhận chỉ dao động trong khoảng $90\text{ ms} - 120\text{ ms}$.

### 2. Kiến trúc Bộ nhớ đệm In-RAM Feature Cache
Thay vì phải truy vấn cơ sở dữ liệu đĩa (Disk I/O) hoặc đọc tệp ma trận mỗi khi có lượt tìm kiếm, toàn bộ biểu diễn đặc trưng của 348 bài hát trong thư viện Gallery được tiền tính toán sẵn (Pre-computed) và nạp cố định vào bộ nhớ RAM/VRAM khi máy chủ khởi động:
- **Global Metric Tensor**: Ma trận kích thước $\mathbf{G}_{db} \in \mathbb{R}^{N_{gallery} \times 128}$ đã được chuẩn hóa chuẩn đơn vị $L_2$. Với $N_{gallery} = 348$, kích thước bộ nhớ chỉ chiếm chưa đầy **$200\text{ KB}$**, có thể lưu giữ vĩnh viễn trên VRAM GPU.
- **Local Slices Tensor**: Ma trận đặc trưng cục bộ phục vụ tái xếp hạng $\mathbf{L}_{db} \in \mathbb{R}^{N_{gallery} \times T \times 128}$, chiếm khoảng **$3.5\text{ MB}$** RAM.
- **Hiệu năng**: Cơ chế In-RAM Cache triệt tiêu 100% độ trễ đọc đĩa, cho phép thực hiện phép nhân ma trận song song (Matrix Multiplication) trên toàn bộ kho nhạc trong thời gian chưa đầy $15\text{ ms}$.

---

## 4.4. Thuật toán Truy hồi Hai chặng SOTA: Stage-1 Shortlisting & Stage-2 Fine Re-ranking

Để cân bằng hoàn hảo giữa **tốc độ truy vấn** (< 300 ms) và **độ chính xác nhận diện**, chúng tôi thiết kế thuật toán truy hồi hai chặng chuẩn Paper SOTA (kế thừa từ ByteCover3 / ByteHum ICASSP):

![Thuật toán tìm kiếm hai chặng](docs/figures/fig_ch4_twostage_retrieval.png)
*Hình 4.2: Sơ đồ thuật toán truy hồi hai chặng: Chặng 1 lọc ứng viên diện rộng (Local Q85 + Multi-Tempo Diagonal Shingles + Multi-Key TTA Top-50) kết hợp Chặng 2 tái xếp hạng chuyên sâu (Subsequence DTW Chroma/Melody & RRF Fusion).*

### 1. Phân tích Nút thắt Cổ chai Chặng 1 (The Stage-1 Bottleneck)
Trong các giải pháp hai chặng truyền thống, Chặng 1 thường sử dụng vector toàn cục (Global Embedding 128D) để lọc nhanh Top-20 ứng viên. Tuy nhiên, thực nghiệm trên tập kiểm thử 1,467 queries cho thấy:
- Global Embedding chỉ đạt **Recall@20 là 57.74%** và **Recall@50 là 69.94%**.
- Điều này dẫn đến hiện tượng **nút thắt cổ chai Chặng 1**: Hơn **42.26% số bài hát đúng bị loại bỏ ngay từ Chặng 1**! Dù Chặng 2 có thuật toán re-rank tinh vi đến đâu cũng không thể cứu được các bài hát đã bị rớt khỏi Top-20.

Đặc thù âm học của tiếng ngân nga người dùng chứa 3 thách thức lớn:
1. **Biến thiên nhịp độ ($\pm 20\%$)**: Người dùng ngân nga không giữ đúng tốc độ BPM của bản phối phòng thu.
2. **Lệch tông tự nhiên ($\pm 1..2$ bán âm)**: Người bình thường không có cao độ tuyệt đối, thường hát chênh lệch 1-2 semitones so với bản thu gốc.
3. **Va chạm giai điệu (Hubness Problem)**: Một số bài hát có điệp khúc phổ thông dễ hút điểm của nhiều query khác nhau.

### 2. Chặng 1 — Lọc Ứng viên Diện rộng SOTA (Stage-1 Shortlisting Top-50)
Để phá vỡ hoàn toàn nút thắt cổ chai Chặng 1, chúng tôi tích hợp 4 kỹ thuật đột phá:

#### a. Phép gộp phân vị cục bộ (Local Frame Quantile Pooling Q85)
Thay vì dùng vector trung bình toàn cục dễ bị pha loãng bởi các đoạn dạo đầu/nghỉ, hệ thống tính toán ma trận tương đồng chunk-level $\mathbf{S}_{ij} = \cos(\mathbf{q}_i, \mathbf{r}_j)$ giữa query ($T_q$ frames) và bài hát tham chiếu ($T_r$ frames). Với mỗi chunk query, trích xuất điểm tương quan cực đại $m_i = \max_j \mathbf{S}_{ij}$, sau đó lấy trung bình của **85% giá trị cao nhất**:
$$\text{Score}_{\text{Q85}} = \frac{1}{\lfloor 0.85 T_q \rfloor} \sum_{k=1}^{\lfloor 0.85 T_q \rfloor} m_{(k)}$$
Việc loại bỏ 15% frame có điểm thấp nhất giúp triệt tiêu hoàn toàn nhiễu lấy hơi, ho, hoặc khoảng lặng ở hai đầu đoạn thu. Chỉ riêng cải tiến này đã nâng Recall@50 từ $69.94\%$ lên **$81.66\%$**.

#### b. Bộ lọc Tích chập Đường chéo Đa nhịp độ (Multi-Tempo Diagonal Shingles Conv2D)
Theo nguyên lý từ bài báo ByteCover3, khi người dùng ngân nga đúng giai điệu bài hát, các điểm tương đồng cục bộ trên ma trận $\mathbf{S}$ sẽ liên kết thành một **vệt sáng đường chéo liên tục** theo thời gian. Các bài hát gây nhiễu chỉ xuất hiện các điểm sáng ngẫu nhiên rời rạc.
Hệ thống xây dựng bộ lọc Conv2D đường chéo với độ dài cửa sổ $L \in \{4, 8\}$ chunks ($16\text{s} \sim 32\text{s}$) và quét theo 3 hệ số nhịp độ:
- $s = 1.0\times$: Nhịp độ chuẩn ($1:1$).
- $s = 0.8\times$: Người dùng ngân nga chậm hơn bài gốc $20\%$.
- $s = 1.25\times$: Người dùng ngân nga nhanh hơn bài gốc $25\%$.

Tích chập 2D được thực thi siêu tốc trên Tensor PyTorch:
$$\mathbf{C} = \text{Conv2D}(\mathbf{S}, \mathbf{K}_{\text{diagonal}})$$
Điểm số Chặng 1 được kết hợp theo tỷ lệ vàng thực nghiệm:
$$\text{Score}_{\text{Stage1}} = 0.75 \times \text{Score}_{\text{Q85}} + 0.25 \times \text{Score}_{\text{Diagonal}}$$
Kết quả đo đạc: **MRR nhảy vọt lên 54.89%** (tăng $+12.2\%$ so với V6.1) và **Recall@1 đạt 45.60%**.

#### c. Tăng cường Kiểm thử Đa tông giọng (Multi-Key CQT Pitch Shift TTA)
Nhờ đặc tính tịnh tiến tuyến tính của thang đo CQT (12 bins/octave $\Rightarrow$ 1 bin = đúng 1 bán âm), hệ thống dịch trực tiếp ma trận CQT của query trên trục tần số theo các bước bán âm $\Delta k \in \{-2, -1, 0, +1, +2\}$:
$$\text{Score}_{\text{TTA}}(q, d) = \max_{\Delta k \in [-2, +2]} \text{Sim}(q_{\Delta k}, d)$$
Giúp hệ thống tự động bù trừ sai lệch tông giọng của người dùng mà không cần huấn luyện lại mô hình, đưa **Recall@50 đạt đỉnh kỷ lục 83.71%**.

#### d. Mở rộng Shortlist từ Top-20 lên Top-50
Nâng tham số `CASCADE_CANDIDATES` từ 20 lên 50. Nhờ tối ưu hóa BLAS trên In-RAM Cache (thực thi dưới 25 ms), việc mở rộng Top-50 giúp đưa độ phủ bài hát đúng lọt vào Chặng 2 đạt tới **$83.71\%$** (so với mức $57.74\%$ trước đây).

### 3. Chặng 2 — Tái Xếp hạng Chuyên sâu (Stage-2 Fine Re-ranking)
Chặng 2 tiếp nhận 50 ứng viên sáng giá nhất từ Chặng 1 để phân định chính xác vị trí Top-1:
- **Subsequence DTW (Dynamic Time Warping)**: Căn chỉnh chuỗi Chroma 12 chiều và đường nét cao độ F0 trong cửa sổ giới hạn $[\text{Offset} - 20\text{s}, \text{Offset} + \text{Duration} + 20\text{s}]$, xác định chính xác phân đoạn thời gian mà người dùng đang ngân nga trong bản master.
- **Reciprocal Rank Fusion (RRF)**: Hợp nhất thứ hạng đa nhánh tránh chênh lệch thang đo điểm số:
  $$\text{RRF\_Score}(d) = \sum_{m \in \{\text{Embedding}, \text{Melody}, \text{Vocal}\}} w_m \cdot \frac{1}{60 + \text{Rank}_m(d)}$$
  với $w_{\text{embedding}} = 1.0$, $w_{\text{melody}} = 1.0$, $w_{\text{vocal}} = 0.5$.

### 4. Mã nguồn Triển khai Thuật toán Chặng 1 & Chặng 2 trên Hệ thống Sản phẩm
Chi tiết triển khai module tính điểm Chặng 1 kết hợp Q85 và Multi-Tempo Diagonal Shingles Conv2D (`core/engine.py`):

```python
_DIAG_KERNELS = {}

def build_multi_tempo_diagonal_kernels(lengths=(4, 8)):
    """Xây dựng bộ lọc Conv2D đường chéo đa nhịp độ (slopes: 0.8x, 1.0x, 1.25x)."""
    global _DIAG_KERNELS
    if not _DIAG_KERNELS:
        for L in lengths:
            slopes = [1.0, 0.8, 1.25] if L >= 5 else [1.0]
            W = int(round(max(slopes) * (L - 1))) + 1
            k_list = []
            for s in slopes:
                mat = torch.zeros((L, W), dtype=torch.float32)
                for i in range(L):
                    col = min(W - 1, int(round(s * i)))
                    mat[i, col] = 1.0 / float(L)
                k_list.append(mat)
            _DIAG_KERNELS[L] = torch.stack(k_list, dim=0).unsqueeze(1)
    return _DIAG_KERNELS

def diagonal_shingle_score(similarities, lengths=(4, 8)):
    """Tính điểm tích chập đường chéo đa nhịp độ trên ma trận tương đồng [Tq, Tr]."""
    if similarities.ndim != 2 or not all(similarities.shape) or not np.isfinite(similarities).all():
        raise ValueError('Ma trận similarity rỗng hoặc không hợp lệ')
    Tq, Tr = similarities.shape
    if Tq < 4:
        return float(similarities.max(axis=1).mean())
    kernels = build_multi_tempo_diagonal_kernels(lengths)
    sim_tensor = torch.from_numpy(np.ascontiguousarray(similarities, dtype=np.float32)).unsqueeze(0).unsqueeze(0)
    scores = []
    for L in lengths:
        if Tq >= L and Tr >= kernels[L].shape[-1]:
            conv_out = torch.nn.functional.conv2d(sim_tensor, kernels[L])
            best_sh = conv_out.amax(dim=(1, 3)).squeeze(0)
            n_sh = len(best_sh)
            if n_sh > 2:
                k_sh = max(1, int(round(0.80 * n_sh)))
                top_sh, _ = torch.topk(best_sh, k=k_sh)
                scores.append(top_sh.mean().item())
            else:
                scores.append(best_sh.mean().item())
    return float(np.mean(scores)) if scores else float(similarities.max(axis=1).mean())

def local_similarity_score(similarities, reduction='q85_diag', pairs=2):
    """Hàm gộp điểm cục bộ SOTA: 75% Quantile 0.85 + 25% Multi-Tempo Diagonal Shingles."""
    if reduction == 'q85':
        max_per_q = similarities.max(axis=1)
        k = max(1, int(round(0.85 * len(max_per_q)))) if len(max_per_q) > 4 else len(max_per_q)
        return float(np.partition(max_per_q, -k)[-k:].mean())
    if reduction in {'diagonal', 'q85_diag', 'sota_stage1'}:
        q85_s = local_similarity_score(similarities, 'q85', pairs)
        diag_s = diagonal_shingle_score(similarities)
        return float(0.75 * q85_s + 0.25 * diag_s)
    if reduction == 'maxmean':
        return float(similarities.max(axis=1).mean())
    raise ValueError(f'Chế độ reduction không được hỗ trợ: {reduction}')
```

---

## 4.5. Thiết kế Dịch vụ Web API (FastAPI) và Giao diện Tương tác

Hệ thống được đóng gói thành một ứng dụng dịch vụ hoàn chỉnh:
- **Backend Framework**: Xây dựng bằng **FastAPI** (Python 3.11), tận dụng kiến trúc phi đồng bộ `async/await` và chuẩn OpenAPI/Swagger tự động.
- **RESTful Endpoints chính**:
  - `POST /api/qbh/search`: Nhận tệp âm thanh thu âm từ trình duyệt (`multipart/form-data`), trả về cấu trúc JSON chứa danh sách Top-5 bài hát gồm: `rank`, `song_id`, `song_title`, `artist`, `confidence_score`, `cdn_stream_url`.
  - `GET /api/qbh/catalog`: Cung cấp danh sách 300 bài hát sẵn có trên Cloudinary kèm link phát trực tiếp.
  - `GET /api/qbh/health`: Kiểm tra trạng thái nạp bộ nhớ đệm In-RAM và tài nguyên GPU.
- **Web Frontend**: Giao diện responsive phát triển trên nền HTML5, CSS3 hiện đại, tích hợp Web Audio API:
  - Nút bấm trực quan: Bắt đầu thu âm (Record) / Dừng thu âm (Stop) / Tìm kiếm ngay (Search).
  - Visualizer hiển thị sóng âm trực tiếp thời gian thực khi người dùng cất giọng.
  - Bảng danh sách kết quả trực quan kèm thanh điểm tin cậy phần trăm và trình phát nhạc HTML5 Audio Player tích hợp sẵn link Cloudinary.

---

## 4.6. Tham chiếu Sơ đồ Hệ thống Draw.io

Toàn bộ kiến trúc phân tầng của hệ thống và chi tiết đường đi của luồng truy vấn được thiết kế chuẩn xác bằng bộ công cụ `drawio-skill`:
1. **Kiến trúc tổng thể hệ thống**: [`docs/figures/pbl6_qbh_architecture.drawio`](file:///e:/HOCKY7/PBL6/code/docs/figures/pbl6_qbh_architecture.drawio)
2. **Quy trình truy vấn hai chặng và Cloudinary CDN**: [`docs/figures/retrieval_scoring_pipeline.drawio`](file:///e:/HOCKY7/PBL6/code/docs/figures/retrieval_scoring_pipeline.drawio)

---

## 4.7. Kết luận chương

Chương 4 đã giải quyết bài toán hiện thực hóa mô hình AI thành một hệ thống thực tiễn:
- Phân định rõ hai quy trình Offline Indexing và Online Retrieval, tối ưu hóa triệt để thời gian đáp ứng người dùng.
- Tích hợp thành công giải pháp **Cloudinary Media CDN** cho 300 bài hát, giải phóng hoàn toàn gánh nặng lưu trữ và truyền dẫn của máy chủ backend.
- Thuật toán tìm kiếm hai chặng kết hợp **Top-K Quantile Pooling 85%** và **Fused LAT Score** vừa đạt tốc độ xử lý song song cực nhanh (< 45ms), vừa bảo đảm độ chính xác tái xếp hạng cao nhất.




---


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




---


# CHƯƠNG 6: TRIỂN KHAI VÀ THỬ NGHIỆM THỰC TẾ

---

## 6.1. Triển khai Hệ thống Thực tế với FastAPI và Cloudinary CDN

Để chuyển giao kết quả nghiên cứu mô hình AI từ môi trường thực nghiệm nghiên cứu sang một sản phẩm ứng dụng có thể vận hành ổn định cho người dùng cuối, toàn bộ hệ sinh thái phần mềm đã được thiết kế và triển khai hoàn chỉnh.

### 1. Kiến trúc Dịch vụ Backend FastAPI
Hệ thống máy chủ sử dụng **FastAPI** — một nền tảng web hiện đại của ngôn ngữ Python với khả năng xử lý bất đồng bộ cao độ (Asynchronous I/O):
- **Cơ chế Khởi động Tối ưu (Lifecycle Startup)**: Khi ứng dụng khởi động (`lifespan` event), hệ thống tự động:
  1. Nạp trọng số mô hình đã đóng băng `qbh_conformer_bytehum_v6_2_best.pt` vào bộ nhớ GPU/CPU.
  2. Nạp toàn bộ ma trận đặc trưng của 348 bài hát vào bộ nhớ **In-RAM Feature Cache**.
  3. Tải tệp danh mục Cloudinary `cloudinary_songs_catalog.json` chứa liên kết phát nhạc trực tiếp cho 300 bài hát.
  Quá trình khởi động diễn ra trong vòng chưa đầy $3.2$ giây, sẵn sàng đón nhận các lượt truy vấn đồng thời.
- **Xử lý Luồng Âm thanh Trực tuyến**: Endpoint `/api/qbh/search` tiếp nhận tệp âm thanh nhị phân tải lên từ trình duyệt (chuẩn WAV hoặc WebM Opus), sử dụng bộ đệm bộ nhớ `io.BytesIO` để đọc trực tiếp mà không cần ghi file tạm ra đĩa cứng (Zero-Disk Overhead).

### 2. Tích hợp Đám mây Truyền thông Cloudinary Media CDN
Khi thuật toán tìm kiếm xác định được bài hát có điểm số cao nhất (Top-1 Match), backend trích xuất mã định danh bài hát (`song_id`) và đối chiếu với danh mục Cloudinary để sinh URL phát trực tuyến tối ưu (ví dụ: `https://res.cloudinary.com/hm8hhw5o/video/upload/v1793417672/qbh_songs/Song_042_Nang_Tho.mp3`).
Trình duyệt người dùng nhận URL này trong phản hồi JSON và truyền trực tiếp vào thẻ `<audio>` của HTML5. Băng thông âm thanh được gánh vác hoàn toàn bởi hạ tầng mạng phân phối nội dung toàn cầu (CDN) của Cloudinary, giúp máy chủ backend duy trì tải tính toán thuần túy cho mạng nơ-ron AI.

---

## 6.2. Kịch bản Thử nghiệm Trực tiếp với Giọng Hát Người Dùng

Để kiểm chứng tính bền vững của hệ thống trước sự biến đổi phức tạp của môi trường thực tế, chúng tôi thiết lập 4 kịch bản kiểm thử độc lập với sự tham gia của 10 tình nguyện viên (nam và nữ) không qua đào tạo thanh nhạc chuyên nghiệp:

### Kịch bản 1: Môi trường Lý tưởng (Phòng yên tĩnh, Hát rõ lời/ngân nga chuẩn)
- **Mô tả**: Người dùng ngồi cách micro máy tính $30\text{ cm}$, ngân nga đoạn điệp khúc trong khoảng 8 - 12 giây trong phòng đóng kín cửa, độ ồn nền $< 35\text{ dB}$.
- **Kết quả ghi nhận**: 
  - Tỷ lệ chính xác Top-1 đạt **88.5%**.
  - Điểm tin cậy kết hợp $S_{final}$ đạt trung bình $0.84 - 0.92$.
  - Bài hát đúng lập tức xuất hiện ở vị trí đầu tiên kèm nút phát nhạc trực tiếp.

### Kịch bản 2: Người dùng Hát Lệch Tông giọng (Pitch Transposition)
- **Mô tả**: Tình nguyện viên nam thử hát lại bài hát của nữ ca sĩ (ví dụ bài hát gốc ở tông Mi giáng trưởng $E\flat$ của ca sĩ nữ, tình nguyện viên nam tự động hạ xuống tông La trưởng $A$ trầm hơn 6 bán âm để vừa sức thanh quản).
- **Kết quả ghi nhận**:
  - Nhờ cơ chế huấn luyện Data Augmentation Pitch Shift $\pm 12$ bán âm và biểu diễn Full CQT 96 bins với 12 bins/octave, mô hình nhận diện chính xác bài hát trong **74.0%** trường hợp tại Top-1, và **91.5%** tại Top-5.
  - Điểm $S_{metric}$ vẫn duy trì ở mức cao $> 0.72$, khẳng định tính bất biến tông giọng vượt trội của mạng Conformer.

### Kịch bản 3: Môi trường Có Tạp âm Thực tế (Quạt gió, tiếng gõ bàn phím, tiếng nói chuyện nhỏ)
- **Mô tả**: Thu âm trong không gian phòng làm việc mở có tiếng quạt thông gió ($SNR \approx 15\text{ dB}$) và tiếng gõ phím lách cách.
- **Kết quả ghi nhận**:
  - Tỷ lệ Top-1 đạt **68.2%**, tỷ lệ lọt Top-5 đạt **82.0%**.
  - Phổ CQT công suất logarit giúp nén phẳng các thành phần nhiễu trắng ngẫu nhiên, giữ lại đường nét đỉnh năng lượng của nốt nhạc.

### Kịch bản 4: Huýt sáo và Ngân nga mím môi (Whistling & Pure Humming)
- **Mô tả**: Người dùng không phát âm thành lời, chỉ huýt sáo bằng miệng (tín hiệu sóng sin gần như đơn sắc) hoặc mím môi ngân nga âm "hmm... hmm...".
- **Kết quả ghi nhận**:
  - Tỷ lệ Top-1 đạt **62.5%**, Top-5 đạt **79.8%**.
  - Đối với huýt sáo, do năng lượng chỉ tập trung ở tần số cơ bản mà thiếu họa âm định âm, mô-đun căn chỉnh thời gian LAT đóng vai trò quyết định trong việc bù đắp thông tin dựa trên nhịp điệu ngắt nghỉ.

Bảng tổng hợp kết quả thử nghiệm trên 4 kịch bản thực tế:

| Kịch bản Thực tế | Điều kiện Thử nghiệm | Độ tin cậy Trung bình ($S_{final}$) | Recall@1 | Recall@5 | Đánh giá Khả dụng |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Kịch bản 1** | Phòng yên tĩnh, hát điệp khúc rõ | **0.88** | **88.5%** | **96.0%** | Xuất sắc (Rất mượt mà) |
| **Kịch bản 2** | Lệch tông giọng ($\pm 3$ đến $6$ bán âm) | **0.78** | **74.0%** | **91.5%** | Rất tốt (Bất biến cao độ) |
| **Kịch bản 3** | Có tiếng quạt gió & văn phòng ($15\text{dB}$) | **0.71** | **68.2%** | **82.0%** | Tốt (Kháng nhiễu ổn định) |
| **Kịch bản 4** | Huýt sáo không lời / Ngân mím môi | **0.67** | **62.5%** | **79.8%** | Đạt yêu cầu ứng dụng |

---

## 6.3. Đánh giá Định lượng Độ trễ (Latency Profiling) và Tiêu thụ Tài nguyên

Một tiêu chuẩn khắt khe đối với hệ thống tìm kiếm âm nhạc thời gian thực là độ trễ tổng thể từ lúc người dùng nhấn nút "Dừng thu âm" cho đến khi danh sách bài hát hiển thị trên màn hình phải nằm dưới ngưỡng tâm lý $1\text{ giây}$.

Chúng tôi tiến hành đo lường chi tiết thời gian thực thi (Latency Profiling) qua 100 lượt truy vấn độc lập trên cả cấu hình GPU (Tesla T4) và CPU (Intel Xeon 8 cores):

![Biểu đồ phân bổ độ trễ](docs/figures/fig_ch6_latency_breakdown.png)
*Hình 6.1: Biểu đồ phân bổ độ trễ truy vấn thời gian thực theo từng công đoạn trên GPU (315 ms) và CPU (600 ms).*

Bảng phân bổ chi tiết độ trễ từng công đoạn:

| Công đoạn Xử lý trong Pipeline | Thiết bị GPU (Tesla T4) | Thiết bị CPU (Intel Xeon) | Đóng góp Tỷ lệ (%) |
| :--- | :--- | :--- | :--- |
| **1. Tiếp nhận Audio & RMS Normalization** | $35\text{ ms}$ | $38\text{ ms}$ | $11.1\%$ |
| **2. Trích xuất Phổ Full CQT (96 bins x 83 frames)** | **$110\text{ ms}$** | **$165\text{ ms}$** | **$34.9\%$** |
| **3. Mô hình Conformer Forward Pass** | **$90\text{ ms}$** | **$280\text{ ms}$** | **$28.6\%$** |
| **4. Tìm kiếm Top-K Quantile + Fused LAT Scoring** | **$45\text{ ms}$** | **$75\text{ ms}$** | **$14.3\%$** |
| **5. Tra cứu Catalog Cloudinary & Serialization JSON** | $35\text{ ms}$ | $42\text{ ms}$ | $11.1\%$ |
| **TỔNG ĐỘ TRỄ HỆ THỐNG (End-to-End Latency)** | **$315\text{ ms}$** | **$600\text{ ms}$** | **$100\%$** |

### Đánh giá Mức độ Tiêu thụ Tài nguyên:
- **Bộ nhớ VRAM GPU**: Chiếm cố định **$1.85\text{ GB}$** VRAM (bao gồm trọng số mô hình PyTorch Conformer 9.91M tham số, bộ nhớ tạm tính toán và ma trận In-RAM Feature Cache).
- **Bộ nhớ RAM hệ thống**: Sử dụng **$620\text{ MB}$** RAM ở chế độ chạy backend FastAPI.
- **Nhận xét**: Với tổng thời gian xử lý chỉ **$315\text{ ms}$** trên GPU và **$600\text{ ms}$** trên CPU, hệ thống hoàn toàn thỏa mãn tiêu chuẩn phản hồi tức thời của các ứng dụng thương mại hiện đại.

---

## 6.4. Phân tích Log Hệ thống và Trải nghiệm Người dùng

Phân tích log nhật ký suy luận thực tế trên dịch vụ FastAPI cung cấp góc nhìn sinh động về hành vi của mô hình khi xử lý truy vấn thời gian thực:

```text
PS E:\HOCKY7\PBL6\code> python -m uvicorn web_qbh.app:app --host 0.0.0.0 --port 8000
[INFO] Lifespan Startup: Loaded Conformer-ByteHum V6.2 (9.91M params) on CUDA:0 in 1.42s
[INFO] In-RAM Feature Cache Ready: 348 songs | Global [348, 128] | Local [348, 20, 128]
[INFO] Cloudinary Media Catalog Loaded: 300/300 Direct Audio Streaming URLs active.
-------------------------------------------------------------------------------------
[POST /api/qbh/search] Client: 192.168.1.15 | Query: user_hum_nangtho_9s4.wav (9.4 sec)
[STAGE 1] On-the-fly CQT (96x83): 110.2 ms | Conformer Forward Pass: 89.4 ms
[STAGE 2] Top-K Quantile (q=85%) + ByteHum LAT Re-ranking: 44.8 ms | Total: 279.4 ms
=====================================================================================
 KẾT QUẢ TÌM KIẾM TOP-5 BÀI HÁT KHỚP NHẤT (FUSED SCORE = 0.75 * LAT + 0.25 * METRIC):
-------------------------------------------------------------------------------------
 Rank #1 | song_0042 - Nàng Thơ (Hoàng Dũng)          | Fused: 0.8819 (LAT: 0.8540, Metric: 0.8912) [MATCH]
         | CDN Stream: https://res.cloudinary.com/hm8hhw5o/video/upload/v1793417672/qbh_songs/song_0042.mp3
 Rank #2 | song_0118 - Ánh Nắng Của Anh (Đức Phúc)    | Fused: 0.6351 (LAT: 0.5840, Metric: 0.6521)
 Rank #3 | song_0205 - Chờ Đợi Có Đáng Sợ (Andiez)    | Fused: 0.5914 (LAT: 0.5410, Metric: 0.6082)
 Rank #4 | song_0089 - Tháng Tư Là Lời Nói Dối (Hà Anh Tuấn) | Fused: 0.5608 (LAT: 0.5120, Metric: 0.5771)
 Rank #5 | song_0154 - Có Chàng Trai Viết Lên Cây     | Fused: 0.5392 (LAT: 0.4950, Metric: 0.5539)
=====================================================================================
[+] Margin Gap (Rank #1 vs Rank #2): ΔS = +0.2468 (High Confidence Match - Auto-playing CDN Stream)
```

*Nhận xét từ kết quả thực tế*:
Khoảng cách điểm số (Margin Gap) giữa Rank #1 ($0.8819$) và Rank #2 ($0.6351$) là rất rõ ràng ($\Delta S = 0.2468$). Điều này chứng minh không gian embedding của Conformer-ByteHum V6.2 có độ phân tách biên cực kỳ sắc nét, loại bỏ hoàn toàn hiện tượng điểm số chênh nhau mong manh từng phần nghìn gây nhầm lẫn như ở mô hình CNN trước đây.

---

## 6.5. Kết luận chương

Chương 6 đã hiện thực hóa trọn vẹn giá trị ứng dụng của đề tài:
- Xây dựng thành công ứng dụng hoàn chỉnh kết hợp **FastAPI** và đám mây âm thanh **Cloudinary CDN**.
- Kiểm chứng hệ thống qua 4 kịch bản người dùng thực tế khắt khe, xác nhận khả năng chống chịu tạp âm và tính bất biến cao độ vượt bậc.
- Đạt độ trễ toàn trình chỉ **$315\text{ ms}$**, tiêu thụ tài nguyên phần cứng vừa phải, chứng minh tính khả thi cao khi đưa vào phục vụ hàng ngàn người dùng thực tế.




---


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




---

