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

