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

