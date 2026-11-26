"""Luật ra quyết định (SPEC v2 mục 5.1) - dùng chung cho evaluate.py và app/demo.py."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.labels import HAZARD_FINE, VN_DISPLAY_FINE, VN_NAMES

DEFAULT_TAU = 0.75
DEFAULT_MIN_CONF = 0.60
MSG_UNCERTAIN = ('Không chắc chắn — vui lòng kiểm tra chất liệu; nếu là xốp hoặc dính bẩn '
                 'hãy bỏ vào nhóm Rác sinh hoạt khác.')
MSG_OOD_UNCERTAIN = ('Độ tự tin thấp trên cả 3 nhóm — ảnh có thể là vật ngoài miền rác sinh hoạt, '
                     'quá nhỏ, nền quá rối hoặc bị mờ/thiếu sáng. Vui lòng chụp gần 1 vật thể rõ nét hơn.')
MSG_CLEAN = 'Vui lòng đổ thức ăn thừa / tráng sạch và tách vật liệu ghép (ống hút, nắp, đá thừa) trước khi bỏ thùng tái chế.'
MSG_HAZARD = 'Rác nguy hại hộ gia đình — thu gom riêng, không bỏ chung thùng.'
MSG_LIMIT = ('Lưu ý: mô hình chỉ phân loại 1 vật/ảnh, chưa học trạng thái sạch–bẩn và chưa có ảnh pin / '
             'thiết bị điện tử trong tập train nên KHÔNG cảnh báo được pin. Pin luôn phải thu gom riêng.')


def load_temperature(path: str | Path | None) -> dict:
    if path and Path(path).exists():
        return json.loads(Path(path).read_text(encoding='utf-8'))
    return {'T': 1.0, 'tau_recycle': DEFAULT_TAU}


def decide(p_vn: np.ndarray, tau: float, fine_p: np.ndarray | None = None, fine_labels: list[str] | None = None,
           top_k: int = 3, min_conf: float = 0.0) -> dict:
    """Quyết định cho MỘT ảnh. p_vn: xác suất 3 nhóm đã chia nhiệt T.

    min_conf > 0 kích hoạt lớp chặn OOD/mơ hồ chung cho cả 3 lớp (mặc định 0.0 để giữ
    tương thích ngược với evaluate.py; demo dùng DEFAULT_MIN_CONF = 0.60).
    """
    p_vn = np.asarray(p_vn, dtype=float)
    k = int(p_vn.argmax())
    out = dict(vn_class=k, vn_name=VN_NAMES[k], prob=float(p_vn[k]), probs=p_vn.tolist(),
               status='OK', uncertain_reason=None, messages=[], hazard=False, top_fine=[])
    if k == 0 and p_vn[0] < tau:
        out['status'] = 'UNCERTAIN'
        out['uncertain_reason'] = 'low_recycle_conf'
        out['messages'].append(MSG_UNCERTAIN)
    elif min_conf > 0.0 and p_vn[k] < min_conf:
        out['status'] = 'UNCERTAIN'
        out['uncertain_reason'] = 'low_overall_conf'
        out['messages'].append(MSG_OOD_UNCERTAIN)
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
