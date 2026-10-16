import random
import numpy as np
import torch
import torch.nn.functional as F
import librosa
import soundfile as sf


class AudioPreprocessor:
    """Tiền xử lý âm thanh: Resample về 22,050 Hz, chuẩn hóa biên độ và cắt Sliding Window 10s / hop 3s."""
    def __init__(self, sample_rate: int = 22050, chunk_duration: float = 10.0, hop_duration: float = 3.0):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.hop_duration = hop_duration
        self.chunk_samples = int(chunk_duration * sample_rate)
        self.hop_samples = int(hop_duration * sample_rate)

    def load_audio(self, audio_path: str) -> np.ndarray:
        try:
            y, sr = sf.read(audio_path, always_2d=False, dtype='float32')
            if len(y.shape) > 1:
                y = np.mean(y, axis=1)
            if sr != self.sample_rate:
                y = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)
            max_val = np.max(np.abs(y))
            if max_val > 1e-6:
                y = y / max_val
            return y
        except Exception:
            y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            max_val = np.max(np.abs(y))
            if max_val > 1e-6:
                y = y / max_val
            return y

    def slice_into_chunks(self, y: np.ndarray) -> list:
        if len(y) <= self.chunk_samples:
            return [np.pad(y, (0, self.chunk_samples - len(y)), mode='reflect')]
        chunks = []
        for start in range(0, len(y) - self.chunk_samples + 1, self.hop_samples):
            chunks.append(y[start : start + self.chunk_samples])
        if len(y) > self.chunk_samples and (len(y) - start) > (self.chunk_samples // 3):
            chunks.append(y[-self.chunk_samples:])
        return chunks


class CQTExtractor:
    """Trích xuất Constant-Q Transform (CQT) phổ tần số âm nhạc (84 bins, 12 bins/octave, hop 512)."""
    def __init__(self, sample_rate: int = 22050, n_bins: int = 84, bins_per_octave: int = 12, hop_length: int = 512, target_frames: int = 431):
        self.sample_rate = sample_rate
        self.n_bins = n_bins
        self.bins_per_octave = bins_per_octave
        self.hop_length = hop_length
        self.target_frames = target_frames

    def extract(self, y: np.ndarray) -> torch.Tensor:
        cqt = librosa.cqt(
            y,
            sr=self.sample_rate,
            hop_length=self.hop_length,
            n_bins=self.n_bins,
            bins_per_octave=self.bins_per_octave
        )
        cqt_mag = np.abs(cqt)
        cqt_db = librosa.amplitude_to_db(cqt_mag, ref=np.max)

        mean = np.mean(cqt_db)
        std = np.std(cqt_db) + 1e-6
        cqt_norm = (cqt_db - mean) / std

        tensor = torch.tensor(cqt_norm, dtype=torch.float32).unsqueeze(0)

        # Căn chỉnh chiều thời gian về đúng target_frames (431)
        cur_frames = tensor.shape[-1]
        if cur_frames < self.target_frames:
            tensor = F.pad(tensor, (0, self.target_frames - cur_frames), mode='reflect')
        elif cur_frames > self.target_frames:
            tensor = tensor[:, :, :self.target_frames]

        return tensor


class RandomPitchShiftCQT:
    """Dịch chuyển ngẫu nhiên trục tần số CQT để mô phỏng sự thay đổi tone giọng giữa các bản cover."""
    def __init__(self, max_shift_bins: int = 3, prob: float = 0.5):
        self.max_shift_bins = max_shift_bins
        self.prob = prob

    def __call__(self, cqt_tensor: torch.Tensor) -> torch.Tensor:
        if random.random() > self.prob or self.max_shift_bins <= 0:
            return cqt_tensor
        shift = random.randint(-self.max_shift_bins, self.max_shift_bins)
        if shift == 0:
            return cqt_tensor
        C, F_dim, T_dim = cqt_tensor.shape
        shifted = torch.zeros_like(cqt_tensor)
        if shift > 0:
            shifted[:, shift:, :] = cqt_tensor[:, :F_dim - shift, :]
        else:
            shifted[:, :F_dim + shift, :] = cqt_tensor[:, -shift:, :]
        return shifted
