"""Suy luận 1 ảnh - đường đi DUY NHẤT mà demo dùng (cùng preprocess + vn_probs với evaluate)."""
from __future__ import annotations

import numpy as np
import torch

from src.decision import decide
from src.preprocess import get_transforms, load_clean_rgb


@torch.no_grad()
def predict_one(model, src, T: float = 1.0, device: str | torch.device = 'cpu'):
    """src: đường dẫn / bytes / file-like. Trả (P_vn[3], fine_p[30] hoặc None, tensor đầu vào 1x3xHxW)."""
    img = load_clean_rgb(src)
    x = get_transforms(False, model.img_size)(img).unsqueeze(0).to(device)
    logits = model.eval().to(device)(x).float().cpu()
    P = model.vn_probs(logits, T)[0].numpy()
    fp = model.fine_probs(logits, T)
    return P, (None if fp is None else fp[0].numpy()), x.cpu()


def classify(model, src, temp: dict, device='cpu', tau_override: float | None = None,
             min_conf: float = 0.0) -> dict:
    """Ảnh -> quyết định đầy đủ theo luật ở src/decision.py."""
    P, fp, x = predict_one(model, src, float(temp['T']), device)
    tau = float(temp['tau_recycle']) if tau_override is None else float(tau_override)
    out = decide(P, tau, fp, model.fine_labels if fp is not None else None, min_conf=min_conf)
    out['input_tensor'] = x
    return out


def probs_equal(a: np.ndarray, b: np.ndarray, tol: float = 1e-4) -> bool:
    return bool(np.abs(np.asarray(a) - np.asarray(b)).max() < tol)
