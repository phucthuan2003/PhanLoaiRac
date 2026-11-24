"""Gom kết quả mọi lần train thành bảng (SV2: leakage ablation, flat vs fine; SV3: so 3 backbone).

  python -m src.engine.benchmark --runs_dir /content/drive/MyDrive/PhanLoaiRac/runs [--latency]

Ghi <runs_dir>/runs_table.csv: mỗi dòng 1 run với backbone, mode, leaky, aug, loss, số tham số,
val bal_acc, test_in (cách chia của run), test_shift, thời gian train và (tuỳ chọn) độ trễ suy luận
batch 1 ảnh 256x256 trên CPU và GPU.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch

from src.models.net import load_checkpoint


@torch.no_grad()
def latency_ms(model, device: str, img_size: int = 256, n: int = 30, warmup: int = 5) -> float:
    model = model.to(device).eval()
    x = torch.randn(1, 3, img_size, img_size, device=device)
    for _ in range(warmup):
        model(x)
    if device == 'cuda':
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        model(x)
    if device == 'cuda':
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1000


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs_dir', required=True)
    ap.add_argument('--latency', action='store_true', help='đo thêm độ trễ suy luận')
    ap.add_argument('--include_smoke', action='store_true')
    a = ap.parse_args(argv)
    rows = []
    for f in sorted(Path(a.runs_dir).glob('*/summary.json')):
        s = json.loads(f.read_text(encoding='utf-8'))
        ar = s['args']
        if ar.get('smoke') and not a.include_smoke:
            continue
        r = dict(run=s['run'], backbone=ar['backbone'], mode=ar['mode'], leaky=ar['allow_leaky'], aug=ar['aug'],
                 loss=ar['loss'], params_M=round(s['params'] / 1e6, 2), best_epoch=s['best_epoch'],
                 val_bal_acc=s['best_val_bal_acc'], train_min=round(s['train_minutes'], 1),
                 test_in_bal_acc=s['test_in_own_split']['bal_acc'], test_in_macro_f1=s['test_in_own_split']['macro_f1'],
                 test_shift_bal_acc=s['test_shift']['bal_acc'], test_shift_macro_f1=s['test_shift']['macro_f1'])
        r['gap_in_minus_shift'] = r['test_in_bal_acc'] - r['test_shift_bal_acc']
        ev = f.parent / 'eval_letterbox' / 'summary.csv'
        if ev.exists():
            e = pd.read_csv(ev, index_col=0)
            for t in e.index:
                r[f'{t}_bal_acc'] = e.loc[t, 'bal_acc']
        if a.latency:
            m, _ = load_checkpoint(f.parent / 'best_model.pth')
            r['cpu_ms'] = round(latency_ms(m, 'cpu', m.img_size), 1)
            if torch.cuda.is_available():
                r['gpu_ms'] = round(latency_ms(m, 'cuda', m.img_size), 2)
        rows.append(r)
    if not rows:
        print('Không có summary.json nào trong', a.runs_dir); return
    df = pd.DataFrame(rows).sort_values(['leaky', 'mode', 'backbone', 'aug'])
    out = Path(a.runs_dir) / 'runs_table.csv'
    df.to_csv(out, index=False)
    pd.set_option('display.width', 200)
    print(df.round(4).to_string(index=False))
    print('Đã lưu ->', out)


if __name__ == '__main__':
    main()
