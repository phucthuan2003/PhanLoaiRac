"""Luật ra quyết định (SPEC v2 mục 5.1) - dùng chung cho evaluate.py và app/demo.py."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.labels import HAZARD_FINE, VN_DISPLAY_FINE, VN_NAMES

DEFAULT_TAU = 0.75
MSG_UNCERTAIN = ('Không chắc chắn — vui lòng kiểm tra chất liệu; nếu là xốp hoặc dính bẩn '
                 'hãy bỏ vào nhóm Rác sinh hoạt khác.')
MSG_CLEAN = 'Vui lòng đổ thức ăn thừa / tráng sạch trước khi bỏ vào thùng tái chế.'
MSG_HAZARD = 'Rác nguy hại hộ gia đình — thu gom riêng, không bỏ chung thùng.'
MSG_LIMIT = ('Lưu ý: mô hình chưa được học ảnh pin / thiết bị điện tử nên KHÔNG cảnh báo được các vật này. '
             'Pin phải luôn thu gom riêng.')


def load_temperature(path: str | Path | None) -> dict:
    if path and Path(path).exists():
        return json.loads(Path(path).read_text(encoding='utf-8'))
    return {'T': 1.0, 'tau_recycle': DEFAULT_TAU}


def decide(p_vn: np.ndarray, tau: float, fine_p: np.ndarray | None = None, fine_labels: list[str] | None = None,
           top_k: int = 3) -> dict:
    """Quyết định cho MỘT ảnh. p_vn: xác suất 3 nhóm đã chia nhiệt T."""
    p_vn = np.asarray(p_vn, dtype=float)
    k = int(p_vn.argmax())
    out = dict(vn_class=k, vn_name=VN_NAMES[k], prob=float(p_vn[k]), probs=p_vn.tolist(),
               status='OK', messages=[], hazard=False, top_fine=[])
    if k == 0 and p_vn[0] < tau:
        out['status'] = 'UNCERTAIN'
        out['messages'].append(MSG_UNCERTAIN)
    elif k == 0:
        out['messages'].append(MSG_CLEAN)
    if fine_p is not None and fine_labels is not None:       # chỉ ở mode fine_grained
        idx = np.argsort(-np.asarray(fine_p))[:top_k]
        out['top_fine'] = [dict(label=fine_labels[i], name=VN_DISPLAY_FINE.get(fine_labels[i], fine_labels[i]),
                                prob=float(fine_p[i])) for i in idx]
        if fine_labels[idx[0]] in HAZARD_FINE:
            out['hazard'] = True
            out['messages'].insert(0, MSG_HAZARD)
    return out
