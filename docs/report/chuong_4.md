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

