"""Tiện ích dùng chung cho train / calibrate / evaluate."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from src.data.dataset import make_loader
from src.preprocess import get_transforms


def get_device(pref: str = 'auto') -> torch.device:
    if pref != 'auto':
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device('cuda')
    # MPS (GPU Apple) KHÔNG tự chọn: torch mới bị crash MPS trên macOS 14. Muốn dùng thì truyền --device mps.
    return torch.device('cpu')


def seed_everything(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def default_workers() -> int:
    return min(4, os.cpu_count() or 2)


@torch.no_grad()
def predict_logits(model, df, root, device, batch_size=128, workers=None, eval_mode='letterbox', amp=True):
    """Logits thô (float32, numpy) của model trên df, theo đúng thứ tự dòng của df."""
    if len(df) == 0:
        return np.zeros((0, model.head[-1].out_features), np.float32)
    workers = default_workers() if workers is None else workers
    tf = get_transforms(False, model.img_size, eval_mode=eval_mode)
    loader = make_loader(df, root, tf, batch_size=batch_size, shuffle=False, workers=workers)
    model.eval().to(device)
    out = []
    use_amp = amp and device.type == 'cuda'
    for x, *_ in loader:
        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            out.append(model(x.to(device, non_blocking=True)).float().cpu())
    return torch.cat(out).numpy()


def vn_probs_np(model, logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    with torch.no_grad():
        return model.vn_probs(torch.from_numpy(logits), T).numpy()     # tính trên CPU


def smoke_subset(df, n_per_class: int = 8, col: str = 'vn_class_id', seed: int = 42):
    """Lấy vài ảnh mỗi lớp để chạy thử nhanh (bắt lỗi code, không đo chất lượng)."""
    if len(df) == 0:
        return df
    return (df.sample(frac=1, random_state=seed).groupby(col, group_keys=False).head(n_per_class)
              .reset_index(drop=True))


def dump_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=_np_default), encoding='utf-8')


def _np_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))
