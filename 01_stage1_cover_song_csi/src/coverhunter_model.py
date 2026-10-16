#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CoverHunter Conformer Model Wrapper for Stage 1 (CSI) & Stage 2 (QBH)
Wraps official Tsinghua/MIREX CoverHunter architecture to load 265/265 pretrained tensors.
"""

import os
import torch
import torch.nn as nn
import numpy as np

from src.coverhunter.model import Model
from src.coverhunter.cqt import PyCqt, shorter
from src.coverhunter.utils import load_hparams

DEFAULT_CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../checkpoints/pretrain_model/config/hparams.yaml"))
DEFAULT_CKPT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../checkpoints/coverhunter_shs100k_pretrained.pth"))


class CoverHunterCQT:
    """
    Trích xuất đặc trưng Constant-Q Transform (CQT) chuẩn hóa theo CoverHunter:
    - Sample rate: 16,000 Hz
    - n_bins: 96 (12 bins/octave, 8 octaves từ 32 Hz đến 8000 Hz)
    - hop_size: 0.04s (25 fps)
    - Nén thời gian: shorter(mean_size=3)
    - Chuẩn hóa dB theo biên độ cực đại: 20 * log10(cqt / max_cqt)
    """
    def __init__(self, sample_rate: int = 16000, hop_size: float = 0.04, octave_resolution: int = 12, min_freq: float = 32, max_freq: float = 8000, mean_size: int = 3):
        self.sample_rate = sample_rate
        self.mean_size = mean_size
        self.py_cqt = PyCqt(
            sample_rate=sample_rate,
            hop_size=hop_size,
            octave_resolution=octave_resolution,
            min_freq=min_freq,
            max_freq=max_freq
        )

    def extract(self, audio: np.ndarray) -> torch.Tensor:
        """
        Trích xuất đặc trưng CQT từ tín hiệu âm thanh 1D numpy array.
        Trả về tensor shape: [time_frames, 96]
        """
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()
        feat = self.py_cqt.compute_cqt(audio, feat_dim_first=False) # (T, 96)
        if self.mean_size > 1:
            feat = shorter(feat, mean_size=self.mean_size)
        return torch.from_numpy(feat).float()


def build_coverhunter_model(ckpt_path: str = None, config_path: str = None, device: str = "cpu") -> Model:
    """
    Khởi tạo mô hình CoverHunter Conformer chính thức (9.91M tham số, 6 khối Conformer, Attention Pooling 128-D).
    Nếu truyền ckpt_path, sẽ nạp trọn vẹn 265/265 tensor từ checkpoint SHS100K.
    """
    if config_path is None or not os.path.exists(config_path):
        config_path = DEFAULT_CONFIG_PATH
    if not os.path.exists(config_path):
        # Fallback to src/coverhunter/hparams.yaml
        config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "coverhunter/hparams.yaml"))

    hp = load_hparams(config_path)
    model = Model(hp)

    if ckpt_path is None:
        ckpt_path = DEFAULT_CKPT_PATH

    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("generator", ckpt.get("model_state_dict", ckpt))
        model.load_state_dict(state_dict)
        total_p = sum(p.numel() for p in model.parameters())
        print(f"[+] Loaded 265/265 tensors ({total_p:,} params) successfully from:\n    -> {ckpt_path}")
    else:
        print(f"[!] Warning: Checkpoint not found at {ckpt_path}. Model randomly initialized.")

    model = model.to(device)
    return model


# Alias for backward compatibility
CoverHunterConformer = Model
