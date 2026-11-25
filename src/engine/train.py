"""Huấn luyện LP-FT (SPEC v2 mục 4.2).

Ví dụ (Colab):
  python -m src.engine.train --manifest /content/manifest_colab.csv --data_root /content/data \
      --out_dir /content/drive/MyDrive/PhanLoaiRac/runs --backbone convnext_tiny --mode flat
  ... --allow_leaky            # leakage ablation (chia ngẫu nhiên không khử trùng)
  ... --smoke                  # vài ảnh mỗi lớp, 2 epoch: chỉ để bắt lỗi code

Kết quả trong <out_dir>/<run_name>/: best_model.pth, history.csv, summary.json
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.dataset import class_weights, make_loader
from src.data.manifest import load_manifest, select
from src.engine.common import (default_workers, dump_json, get_device, predict_logits, seed_everything,
                               smoke_subset, vn_probs_np)
from src.metrics import bal_acc, macro_f1
from src.models.net import BACKBONES, WasteNet, count_params, save_checkpoint
from src.preprocess import get_transforms


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma: float = 2.0, label_smoothing: float = 0.0):
        super().__init__()
        self.register_buffer('weight', weight if weight is not None else None)
        self.gamma, self.ls = gamma, label_smoothing

    def forward(self, logits, y):
        ce = F.cross_entropy(logits, y, weight=self.weight, reduction='none', label_smoothing=self.ls)
        pt = torch.exp(-F.cross_entropy(logits, y, reduction='none'))
        return ((1 - pt) ** self.gamma * ce).mean()


def build_loss(kind: str, targets: np.ndarray, n_out: int, smoothing: float):
    w = class_weights(targets, n_out)
    if kind == 'weighted_ce':
        return nn.CrossEntropyLoss(weight=w, label_smoothing=smoothing)
    if kind == 'focal':
        return FocalLoss(weight=w, gamma=2.0, label_smoothing=smoothing)
    if kind == 'ce':
        return nn.CrossEntropyLoss(label_smoothing=smoothing)
    raise ValueError(kind)


def run_name(a) -> str:
    if a.run_name:
        return a.run_name
    return f"{a.backbone}_{a.mode}_{'leaky' if a.allow_leaky else 'clean'}_{a.aug}_{a.loss}{'_smoke' if a.smoke else ''}"


def evaluate_split(model, df, root, device, a, eval_mode='letterbox'):
    logits = predict_logits(model, df, root, device, a.batch * 2, a.workers, eval_mode)
    P = vn_probs_np(model, logits)
    y = df['vn_class_id'].to_numpy()
    pred = P.argmax(1)
    return dict(n=len(df), bal_acc=bal_acc(y, pred), macro_f1=macro_f1(y, pred), accuracy=float((pred == y).mean()))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--out_dir', default='runs')
    ap.add_argument('--run_name', default='')
    ap.add_argument('--backbone', default='convnext_tiny', choices=BACKBONES)
    ap.add_argument('--mode', default='flat', choices=['flat', 'fine_grained'])
    ap.add_argument('--loss', default='weighted_ce', choices=['weighted_ce', 'focal', 'ce'])
    ap.add_argument('--aug', default='advanced', choices=['basic', 'advanced'])
    ap.add_argument('--allow_leaky', action='store_true', help='leakage ablation: dùng split_raw, không khử trùng')
    ap.add_argument('--epochs', type=int, default=18)
    ap.add_argument('--lp_epochs', type=int, default=3, help='số epoch đầu chỉ train head (backbone đóng băng)')
    ap.add_argument('--lr_head', type=float, default=1e-3)
    ap.add_argument('--lr', type=float, default=3e-5)
    ap.add_argument('--wd', type=float, default=1e-2)
    ap.add_argument('--label_smoothing', type=float, default=0.0)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--img_size', type=int, default=256)
    ap.add_argument('--patience', type=int, default=5, help='early stopping (tính trong giai đoạn fine-tune)')
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--allow_cpu', action='store_true', help='cho phép train bằng CPU (rất chậm; mặc định báo lỗi)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--no_pretrained', action='store_true')
    ap.add_argument('--smoke', action='store_true', help='chạy thử: vài ảnh mỗi lớp, 2 epoch')
    a = ap.parse_args(argv)
    a.workers = default_workers() if a.workers is None else a.workers
    if a.smoke:
        a.epochs, a.lp_epochs, a.batch = 2, 1, min(a.batch, 16)

    seed_everything(a.seed)
    device = get_device(a.device)
    if device.type == 'cpu' and not (a.smoke or a.allow_cpu):
        raise SystemExit('LỖI: không thấy GPU (CUDA) nên sẽ phải train bằng CPU, chậm hơn ~30 lần.\n'
                         'Trên Colab: Runtime -> Change runtime type -> T4 GPU, rồi chạy lại từ đầu.\n'
                         'Nếu thật sự muốn dùng CPU, thêm --allow_cpu.')
    out = Path(a.out_dir) / run_name(a)
    out.mkdir(parents=True, exist_ok=True)

    df = load_manifest(a.manifest)
    tr = select(df, 'train', a.mode, a.allow_leaky)
    va = select(df, 'val', a.mode, a.allow_leaky)
    if a.smoke:
        col = 'vn_class_id' if a.mode == 'flat' else 'fine_id'
        tr, va = smoke_subset(tr, 4 if a.mode == 'fine_grained' else 8, col), smoke_subset(va, 8)
    tcol = 'vn_class_id' if a.mode == 'flat' else 'fine_id'
    targets = tr[tcol].to_numpy()
    print(f'Thiết bị {device} | train {len(tr)} | val {len(va)} | mode {a.mode} | leaky {a.allow_leaky}', flush=True)

    model = WasteNet(a.backbone, a.mode, pretrained=not a.no_pretrained, img_size=a.img_size).to(device)
    n_out = model.head[-1].out_features
    crit = build_loss(a.loss, targets, n_out, a.label_smoothing).to(device)
    loader = make_loader(tr, a.data_root, get_transforms(True, a.img_size, aug=a.aug), a.batch, shuffle=True,
                         workers=a.workers, drop_last=len(tr) > a.batch)
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    def make_opt(phase):
        if phase == 'lp':
            model.freeze_backbone(True)
            return torch.optim.AdamW(model.head.parameters(), lr=a.lr_head, weight_decay=a.wd), None
        model.freeze_backbone(False)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
        n_ft = max(1, a.epochs - a.lp_epochs)
        return opt, torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_ft * len(loader))

    phase = 'lp' if a.lp_epochs > 0 else 'ft'
    opt, sched = make_opt(phase)
    best, best_ep, bad_epochs, hist = -math.inf, 0, 0, []
    t_start = time.time()
    for ep in range(1, a.epochs + 1):
        if phase == 'lp' and ep > a.lp_epochs:
            phase = 'ft'; opt, sched = make_opt(phase)       # tạo lại optimizer khi mở khoá backbone
        model.train(); t0 = time.time(); tot, seen = 0.0, 0
        for x, yvn, yfine, _ in loader:
            y = (yvn if a.mode == 'flat' else yfine).to(device, non_blocking=True)
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
                loss = crit(model(x).float(), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            if sched is not None:
                sched.step()
            tot += loss.item() * len(y); seen += len(y)
        logits = predict_logits(model, va, a.data_root, device, a.batch * 2, a.workers)
        P = vn_probs_np(model, logits)
        yv = va['vn_class_id'].to_numpy(); pv = P.argmax(1)
        vb, vf = bal_acc(yv, pv), macro_f1(yv, pv)
        row = dict(epoch=ep, phase=phase, train_loss=tot / max(seen, 1), val_bal_acc=vb, val_macro_f1=vf,
                   lr=opt.param_groups[0]['lr'], sec=time.time() - t0)
        hist.append(row)
        flag = ''
        if vb > best:
            best, best_ep, bad_epochs = vb, ep, 0
            save_checkpoint(model, out / 'best_model.pth', epoch=ep, val_bal_acc=vb, args=vars(a))
            flag = '  <- lưu'
        elif phase == 'ft':
            bad_epochs += 1
        print(f"Epoch {ep:2d}/{a.epochs} [{phase}] loss {row['train_loss']:.4f}  val bal_acc {vb:.4f}  "
              f"macro-F1 {vf:.4f}  ({row['sec']:.0f}s){flag}", flush=True)
        pd.DataFrame(hist).to_csv(out / 'history.csv', index=False)
        if phase == 'ft' and bad_epochs >= a.patience:
            print(f'Early stopping: {a.patience} epoch không cải thiện'); break

    # ---- đánh giá nhanh checkpoint tốt nhất (cho bảng leakage ablation) ----
    # test_in_own_split: test_in theo cách chia của chính run (run leaky -> có ảnh gần trùng với train => "ảo").
    # test_shift: ảnh real_world KHÔNG có bản trùng nào trong rhw/default -> thước đo trung thực cho cả 2 run.
    from src.models.net import load_checkpoint
    model, _ = load_checkpoint(out / 'best_model.pth', map_location=device)
    model.to(device)
    t_own = select(df, 'test_in', 'flat', a.allow_leaky)
    t_shift = select(df, 'test_shift', 'flat', False)
    if a.smoke:
        t_own, t_shift = smoke_subset(t_own, 8), smoke_subset(t_shift, 8)
    summary = dict(run=run_name(a), args=vars(a), device=str(device), params=count_params(model),
                   best_epoch=best_ep, best_val_bal_acc=best, train_minutes=(time.time() - t_start) / 60,
                   n_train=len(tr), n_val=len(va),
                   test_in_own_split=evaluate_split(model, t_own, a.data_root, device, a),
                   test_shift=evaluate_split(model, t_shift, a.data_root, device, a))
    dump_json(summary, out / 'summary.json')
    print('test_in (cách chia của run):', summary['test_in_own_split'])
    print('test_shift:', summary['test_shift'])
    print('Đã lưu ->', out)


if __name__ == '__main__':
    main()
