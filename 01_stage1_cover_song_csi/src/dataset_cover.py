import os
import random
import torch
from torch.utils.data import Dataset, Sampler
from typing import Dict, List, Optional
from .transforms import AudioPreprocessor, CQTExtractor, RandomPitchShiftCQT


class CoverSongDataset(Dataset):
    """
    Dataset cho bài toán Cover Song Identification (CSI):
    Hỗ trợ cả 2 định dạng:
    1. Thư mục phân tầng theo tác phẩm: root_dir/song_title/performance_1.mp3, performance_2.mp3 ...
    2. Cấu trúc Covers80 (hai thư mục con hoặc danh sách bài hát đối sánh).
    """
    def __init__(
        self,
        dataset_dir: str,
        audio_preprocessor: AudioPreprocessor,
        cqt_extractor: CQTExtractor,
        pitch_shift: Optional[RandomPitchShiftCQT] = None,
        cqt_cache_dir: Optional[str] = None,
        is_training: bool = True
    ):
        self.dataset_dir = dataset_dir
        self.audio_prep = audio_preprocessor
        self.cqt_ext = cqt_extractor
        self.pitch_shift = pitch_shift
        self.cqt_cache_dir = cqt_cache_dir
        self.is_training = is_training

        self.cliques: Dict[str, List[str]] = {} # work_id -> list of file paths
        self.samples: List[tuple] = []          # list of (work_id, file_path)
        self.work_to_idx: Dict[str, int] = {}

        self._scan_dataset()

    def _scan_dataset(self):
        if not os.path.exists(self.dataset_dir):
            return

        # 1. Quét tìm tất cả các file âm thanh
        audio_exts = ('.mp3', '.wav', '.flac', '.ogg', '.m4a')
        subdirs = [d for d in os.listdir(self.dataset_dir) if os.path.isdir(os.path.join(self.dataset_dir, d))]

        if len(subdirs) > 0 and any(len([f for f in os.listdir(os.path.join(self.dataset_dir, d)) if f.endswith(audio_exts)]) > 0 for d in subdirs):
            for d in sorted(subdirs):
                d_path = os.path.join(self.dataset_dir, d)
                files = [os.path.join(d_path, f) for f in os.listdir(d_path) if f.lower().endswith(audio_exts)]
                if len(files) > 0:
                    self.cliques[d] = files
        else:
            # Nếu tất cả các file nằm trong cùng 1 thư mục với tiền tố/hậu tố
            all_files = sorted([f for f in os.listdir(self.dataset_dir) if f.lower().endswith(audio_exts)])
            for f in all_files:
                # Trích xuất work_id từ tên file (ví dụ: '01_Yesterday_cover.mp3' -> 'Yesterday')
                base = os.path.splitext(f)[0]
                work_id = base.split('+')[0] if '+' in base else base.split('_')[0]
                full_p = os.path.join(self.dataset_dir, f)
                if work_id not in self.cliques:
                    self.cliques[work_id] = []
                self.cliques[work_id].append(full_p)

        self.work_to_idx = {w: i for i, w in enumerate(sorted(self.cliques.keys()))}
        for w, paths in self.cliques.items():
            for p in paths:
                self.samples.append((w, p))

    def __len__(self) -> int:
        return len(self.samples)

    def _get_cqt(self, path: str) -> torch.Tensor:
        cqt_tensor = None
        if self.cqt_cache_dir:
            rel = os.path.relpath(path, self.dataset_dir)
            cache_p = os.path.join(self.cqt_cache_dir, os.path.splitext(rel)[0] + '.pt')
            if os.path.exists(cache_p):
                try:
                    cqt_tensor = torch.load(cache_p, map_location='cpu', weights_only=True)
                except Exception:
                    cqt_tensor = None

        if cqt_tensor is None:
            y = self.audio_prep.load_audio(path)
            chunks = self.audio_prep.slice_into_chunks(y)
            chunk = random.choice(chunks) if (self.is_training and len(chunks) > 1) else chunks[0]
            cqt_tensor = self.cqt_ext.extract(chunk)
            # Tự động lưu cache nếu có thư mục
            if self.cqt_cache_dir:
                try:
                    os.makedirs(os.path.dirname(cache_p), exist_ok=True)
                    torch.save(cqt_tensor, cache_p)
                except Exception:
                    pass
        elif cqt_tensor.dim() == 4: # (N, 1, 84, 431)
            idx = random.randint(0, cqt_tensor.shape[0] - 1) if (self.is_training and cqt_tensor.shape[0] > 1) else 0
            cqt_tensor = cqt_tensor[idx]

        if self.is_training and self.pitch_shift:
            cqt_tensor = self.pitch_shift(cqt_tensor)

        return cqt_tensor

    def __getitem__(self, idx: int) -> dict:
        work_id, path = self.samples[idx]
        cqt = self._get_cqt(path)
        label = self.work_to_idx[work_id]

        return {
            'cqt': cqt,
            'label': torch.tensor(label, dtype=torch.long),
            'work_id': work_id,
            'path': path
        }


class BalancedBatchSampler(Sampler):
    """
    Sampler cân bằng cho Online Triplet Mining:
    Mỗi batch sẽ chọn P tác phẩm (work_ids), mỗi tác phẩm lấy đúng K bản cover (tổng Batch Size = P * K).
    Đảm bảo 100% trong mỗi batch luôn có các cặp Positive hợp lệ để tính Triplet Loss!
    """
    def __init__(self, dataset: CoverSongDataset, n_classes: int = 8, n_samples: int = 2):
        self.dataset = dataset
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.cliques = {w: paths for w, paths in dataset.cliques.items() if len(paths) >= n_samples}
        self.available_works = list(self.cliques.keys())

    def __iter__(self):
        # Tạo batch liên tục bằng cách chọn ngẫu nhiên các clique đủ số lượng mẫu
        num_batches = max(1, len(self.dataset) // (self.n_classes * self.n_samples))
        for _ in range(num_batches):
            selected_works = random.sample(self.available_works, min(self.n_classes, len(self.available_works)))
            batch_indices = []
            for w in selected_works:
                paths = random.sample(self.cliques[w], self.n_samples)
                for p in paths:
                    idx = [i for i, sample in enumerate(self.dataset.samples) if sample[1] == p][0]
                    batch_indices.append(idx)
            yield batch_indices

    def __len__(self):
        return max(1, len(self.dataset) // (self.n_classes * self.n_samples))


CoverSongTripletDataset = CoverSongDataset
