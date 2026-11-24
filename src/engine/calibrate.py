"""Hiệu chuẩn nhiệt độ T và chọn ngưỡng tau_recycle trên tập val SẠCH (SPEC v2 mục 5.1).

  python -m src.engine.calibrate --ckpt runs/<run>/best_model.pth --manifest ... --data_root ...
-> <thư mục ckpt>/temperature.json  (T, ece_before, ece_after, tau_recycle, bảng quét tau)

T được tối ưu bằng L-BFGS (tối thiểu NLL của xác suất 3 nhóm) trên log T để luôn có T > 0.
Ở mode fine_grained, logits 30 lớp được chia cho T TRƯỚC khi softmax và gộp qua M.
Chọn tau: nhỏ nhất (giữ coverage cao nhất) sao cho contamination <= mục tiêu và coverage >= min.
Nếu ngay khi KHÔNG từ chối contamination trên val đã <= mục tiêu thì val không phân biệt được các tau
-> giữ tau mặc định 0.75 và ghi rõ lý do (xem risk-coverage từng tầng trong evaluate).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.data.manifest import load_manifest, select
from src.decision import DEFAULT_TAU
from src.engine.common import default_workers, dump_json, get_device, predict_logits, smoke_subset, vn_probs_np
from src.metrics import accept_mask, contamination_rate, ece, nll, risk_coverage
from src.models.net import load_checkpoint


def fit_temperature(model, logits: np.ndarray, y: np.ndarray, max_iter: int = 100) -> float:
    L = torch.from_numpy(logits).float()
    yt = torch.from_numpy(y).long()
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn='strong_wolfe')
    M = model.M.detach().cpu()

    def probs(t):
        p = torch.softmax(L / t, dim=1)
        return p if model.mode == 'flat' else p @ M

    def closure():
        opt.zero_grad()
        p = probs(log_t.exp())
        loss = -torch.log(p[torch.arange(len(yt)), yt].clamp_min(1e-12)).mean()
        loss.backward()
        return loss

    opt.step(closure)
    T = float(log_t.exp().item())
    if not (0.05 <= T <= 20):          # model gần như đoán bừa (vd. chạy smoke) -> T vô nghĩa, kẹp lại
        print(f'CẢNH BÁO: T = {T:.3g} nằm ngoài [0.05, 20] - model chưa học được gì? Kẹp T vào khoảng này.')
        T = min(max(T, 0.05), 20.0)
    return T


def choose_tau(P: np.ndarray, y: np.ndarray, target: float, min_cov: float):
    sweep = risk_coverage(P, y)
    base = contamination_rate(y, P.argmax(1))
    if np.isnan(base) or base <= target:
        return DEFAULT_TAU, (f'contamination khi không từ chối = {base:.4f} <= mục tiêu {target}: val không phân biệt '
                             f'được các tau -> giữ mặc định {DEFAULT_TAU}'), sweep
    ok = [r for r in sweep if r['contamination'] <= target and r['coverage'] >= min_cov]
    if ok:
        r = min(ok, key=lambda r: r['tau'])
        return r['tau'], f'tau nhỏ nhất đạt contamination <= {target} với coverage >= {min_cov}', sweep
    feas = [r for r in sweep if r['coverage'] >= min_cov]
    r = min(feas, key=lambda r: (r['contamination'], -r['coverage']))
    return r['tau'], f'không tau nào đạt mục tiêu {target}; chọn contamination thấp nhất với coverage >= {min_cov}', sweep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--out', default=None, help='mặc định: temperature.json cạnh checkpoint')
    ap.add_argument('--target_contamination', type=float, default=0.05)
    ap.add_argument('--min_coverage', type=float, default=0.85)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args(argv)
    device = get_device(a.device)
    model, ck = load_checkpoint(a.ckpt, map_location='cpu')
    model.to(device)

    va = select(load_manifest(a.manifest), 'val', 'flat', False)
    if a.smoke:
        va = smoke_subset(va, 8)
    logits = predict_logits(model, va, a.data_root, device, a.batch, default_workers() if a.workers is None else a.workers)
    y = va['vn_class_id'].to_numpy()
    P0 = vn_probs_np(model, logits, 1.0)
    T = fit_temperature(model, logits, y)
    P1 = vn_probs_np(model, logits, T)
    tau, reason, sweep = choose_tau(P1, y, a.target_contamination, a.min_coverage)
    acc = accept_mask(P1, tau)
    res = dict(T=T, ece_before=ece(P0, y), ece_after=ece(P1, y), nll_before=nll(P0, y), nll_after=nll(P1, y),
               tau_recycle=tau, tau_reason=reason, target_contamination=a.target_contamination,
               min_coverage=a.min_coverage, val_coverage_at_tau=float(acc.mean()),
               val_contamination_at_tau=contamination_rate(y, P1.argmax(1), acc),
               n_val=int(len(y)), ckpt=str(a.ckpt), mode=model.mode, backbone=model.backbone_name, sweep=sweep)
    out = Path(a.out) if a.out else Path(a.ckpt).with_name('temperature.json')
    dump_json(res, out)
    print(f"T = {T:.4f} | ECE {res['ece_before']:.4f} -> {res['ece_after']:.4f} | tau = {tau} ({reason})")
    print('Đã lưu ->', out)


if __name__ == '__main__':
    main()
