"""PyTorch Dataset đọc manifest.csv / vn_real_manifest.csv."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from src.preprocess import load_clean_rgb


class WasteDataset(Dataset):
    """Trả về (ảnh tensor, nhãn 3 nhóm, nhãn con fine_id hoặc -1, chỉ số dòng)."""

    def __init__(self, df: pd.DataFrame, root: str | Path, transform):
        self.paths = [str(Path(root) / p) for p in df['path']]
        self.vn = df['vn_class_id'].to_numpy(dtype=np.int64)
        self.fine = df['fine_id'].to_numpy(dtype=np.int64) if 'fine_id' in df else np.full(len(df), -1)
        self.tf = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return self.tf(load_clean_rgb(self.paths[i])), self.vn[i], self.fine[i], i


def make_loader(df, root, transform, batch_size=64, shuffle=False, workers=4, drop_last=False):
    return DataLoader(WasteDataset(df, root, transform), batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, pin_memory=torch.cuda.is_available(), drop_last=drop_last,
                      persistent_workers=workers > 0)


def class_weights(labels: np.ndarray, n_classes: int) -> torch.Tensor:
    """Trọng số nghịch tần suất: N / (K * n_k)."""
    counts = np.bincount(labels, minlength=n_classes).astype(np.float64)
    w = len(labels) / (n_classes * np.maximum(counts, 1))
    w[counts == 0] = 0.0
    return torch.tensor(w, dtype=torch.float32)
