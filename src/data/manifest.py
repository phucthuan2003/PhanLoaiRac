"""Đọc manifest và chọn tập con cho từng split / tầng đánh giá. Chỉ dùng pandas (không cần torch)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.labels import fine_labels

TRAIN_SPLITS = ('train', 'val', 'test_in')
UNUSABLE = ('dup', 'exclude')


def load_manifest(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False, low_memory=False,
                     dtype={'split_raw': str, 'orig_split': str, 'context': str, 'reason': str})
    for c in ('vn_class_id', 'is_hazardous', 'mo_ho', 'cross_fine', 'group_id', 'is_rep'):
        if c in df:
            df[c] = pd.to_numeric(df[c]).astype(int)
    if 'in_zip' not in df:
        df['in_zip'] = 1
    df['in_zip'] = pd.to_numeric(df['in_zip'].replace('', 0)).astype(int)
    labs = fine_labels()
    fid = {lab: i for i, lab in enumerate(labs)}
    df['fine_id'] = [fid.get(lab, -1) if ds == 'rhw' else -1 for ds, lab in zip(df['dataset'], df['orig_label'])]
    return df


def select(df: pd.DataFrame, split: str, mode: str = 'flat', allow_leaky: bool = False) -> pd.DataFrame:
    """Chọn các dòng cho một split.

    - Sạch (mặc định): dùng cột `split` (đã khử trùng). Ở mode fine_grained, bỏ dòng cross_fine
      khỏi train/val (ảnh trùng mang 2 nhãn con mâu thuẫn).
    - allow_leaky (leakage ablation): train/val/test_in lấy theo `split_raw` = chia ngẫu nhiên
      toàn bộ 7.500 ảnh rhw/default, KHÔNG khử trùng -> ảnh gần trùng nằm ở cả train lẫn test.
    """
    df = df[df['in_zip'] == 1]
    if allow_leaky and split in TRAIN_SPLITS:
        sub = df[df['split_raw'] == split]
    else:
        sub = df[df['split'] == split]
    if mode == 'fine_grained' and split in ('train', 'val') and not allow_leaky:
        sub = sub[sub['cross_fine'] == 0]
    if mode == 'fine_grained' and split in ('train', 'val'):
        sub = sub[sub['fine_id'] >= 0]
    return sub.reset_index(drop=True)


def eval_tiers(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Tầng 1-3 của SPEC v2 mục 5.2 (tầng 4 VN-Real đọc từ vn_real_manifest.csv riêng)."""
    d = df[(df['in_zip'] == 1) & ~df['split'].isin(UNUSABLE)]
    gd = d[d['split'] == 'test_cross_gd']
    tiers = {
        'T1_test_in': d[d['split'] == 'test_in'],
        'T2_test_shift': d[d['split'] == 'test_shift'],
        'T3_gd_full': gd,
        'T3_gd_no_trash': gd[gd['orig_label'] != 'trash'],
        'T3_nextech': d[d['split'] == 'test_cross_nextech'],
        'T3_trashnet': d[d['split'] == 'trashnet_only'],
    }
    return {k: v.reset_index(drop=True) for k, v in tiers.items() if len(v)}


def load_vn_real(path: str | Path) -> pd.DataFrame:
    """vn_real_manifest.csv: path, vn_class_id, item_name, is_dirty, bg_type, is_vn_specific[, annotator]."""
    df = pd.read_csv(path, keep_default_na=False)
    need = {'path', 'vn_class_id', 'is_dirty', 'is_vn_specific'}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f'vn_real_manifest thiếu cột: {sorted(miss)}')
    for c in ('vn_class_id', 'is_dirty', 'is_vn_specific'):
        df[c] = pd.to_numeric(df[c]).astype(int)
    if '\\' in ''.join(df['path']):
        raise ValueError('vn_real_manifest có đường dẫn chứa "\\" - phải là đường dẫn POSIX')
    df['dataset'] = 'vn_real'
    df['orig_label'] = df.get('item_name', '')
    df['fine_id'] = -1
    df['mo_ho'] = df.get('mo_ho', 0)
    return df


def vn_real_tiers(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out = {'T4_vn_real': df}
    for d in (0, 1):
        for v in (0, 1):
            sub = df[(df['is_dirty'] == d) & (df['is_vn_specific'] == v)]
            if len(sub):
                out[f'T4_dirty{d}_vn{v}'] = sub
    return {k: v.reset_index(drop=True) for k, v in out.items()}
