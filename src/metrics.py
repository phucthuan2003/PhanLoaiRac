"""Chỉ số đánh giá (chỉ numpy). Chỉ số chính: balanced accuracy (không phụ thuộc tỷ lệ lớp)."""
from __future__ import annotations

import numpy as np

K = 3


def confusion(y, pred, k: int = K) -> np.ndarray:
    y, pred = np.asarray(y, int), np.asarray(pred, int)
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    return cm


def per_class(y, pred, k: int = K) -> dict:
    cm = confusion(y, pred, k)
    tp = np.diag(cm).astype(float)
    with np.errstate(invalid='ignore', divide='ignore'):
        prec = np.where(cm.sum(0) > 0, tp / cm.sum(0), 0.0)
        rec = np.where(cm.sum(1) > 0, tp / cm.sum(1), np.nan)
        f1 = np.where((prec + np.nan_to_num(rec)) > 0, 2 * prec * np.nan_to_num(rec) / (prec + np.nan_to_num(rec)), 0.0)
    present = cm.sum(1) > 0
    f1 = np.where(present, f1, np.nan)
    prec = np.where(present, prec, np.nan)
    return dict(precision=prec, recall=rec, f1=f1, support=cm.sum(1), present=present, cm=cm)


def bal_acc(y, pred, k: int = K) -> float:
    pc = per_class(y, pred, k)
    return float(np.nanmean(pc['recall'][pc['present']])) if pc['present'].any() else float('nan')


def macro_f1(y, pred, k: int = K) -> float:
    pc = per_class(y, pred, k)
    return float(np.nanmean(pc['f1'][pc['present']])) if pc['present'].any() else float('nan')


def accept_mask(P: np.ndarray, tau: float) -> np.ndarray:
    """Luật từ chối (SPEC 5.1): bị từ chối khi argmax = Tái chế nhưng P[0] < tau."""
    pred = P.argmax(1)
    return ~((pred == 0) & (P[:, 0] < tau))


def contamination_rate(y, pred, accepted=None) -> float:
    """% mẫu thật nhóm 1-2 (không bị từ chối) bị đoán là nhóm 0 -> làm bẩn thùng tái chế."""
    y, pred = np.asarray(y), np.asarray(pred)
    m = y != 0
    if accepted is not None:
        m &= np.asarray(accepted, bool)
    return float((pred[m] == 0).mean()) if m.any() else float('nan')


def missed_recycle_rate(y, pred, accepted=None) -> float:
    y, pred = np.asarray(y), np.asarray(pred)
    m = y == 0
    if accepted is not None:
        m &= np.asarray(accepted, bool)
    return float((pred[m] != 0).mean()) if m.any() else float('nan')


def ece(P: np.ndarray, y, n_bins: int = 15) -> float:
    conf = P.max(1); correct = (P.argmax(1) == np.asarray(y)).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(e)


def nll(P: np.ndarray, y) -> float:
    return float(-np.log(np.clip(P[np.arange(len(P)), np.asarray(y)], 1e-12, 1)).mean())


def risk_coverage(P: np.ndarray, y, taus=None) -> list[dict]:
    """Quét ngưỡng tau của luật từ chối: coverage, balanced acc và contamination trên phần được nhận."""
    taus = np.round(np.arange(0.34, 1.0, 0.01), 2) if taus is None else taus
    y = np.asarray(y); pred = P.argmax(1); out = []
    for t in taus:
        acc = accept_mask(P, t)
        out.append(dict(tau=float(t), coverage=float(acc.mean()),
                        selective_bal_acc=bal_acc(y[acc], pred[acc]) if acc.any() else float('nan'),
                        selective_risk=float((pred[acc] != y[acc]).mean()) if acc.any() else float('nan'),
                        contamination=contamination_rate(y, pred, acc)))
    return out


def bootstrap_ci(y, pred, fn, n: int = 1000, seed: int = 42, alpha: float = 0.05):
    y, pred = np.asarray(y), np.asarray(pred)
    if len(y) == 0:
        return [float('nan'), float('nan')]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        v = fn(y[i], pred[i])
        if not np.isnan(v):
            vals.append(v)
    return [float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))] if vals else [float('nan')] * 2


def full_report(y, P: np.ndarray, tau: float, n_boot: int = 0) -> dict:
    """Mọi chỉ số bắt buộc của SPEC 5.2 cho một tập."""
    y = np.asarray(y, int); pred = P.argmax(1); acc = accept_mask(P, tau)
    pc = per_class(y, pred)
    r = dict(
        n=int(len(y)),
        bal_acc=bal_acc(y, pred), macro_f1=macro_f1(y, pred), accuracy=float((pred == y).mean()),
        per_class_precision=[None if np.isnan(v) else float(v) for v in pc['precision']],
        per_class_recall=[None if np.isnan(v) else float(v) for v in pc['recall']],
        per_class_f1=[None if np.isnan(v) else float(v) for v in pc['f1']],
        support=pc['support'].tolist(),
        confusion_matrix=pc['cm'].tolist(),
        contamination_rate=contamination_rate(y, pred, acc),
        contamination_rate_no_abstain=contamination_rate(y, pred),
        missed_recycle_rate=missed_recycle_rate(y, pred, acc),
        ece=ece(P, y), nll=nll(P, y),
        tau=float(tau), abstention_rate=float(1 - acc.mean()),
        selective_bal_acc=bal_acc(y[acc], pred[acc]) if acc.any() else float('nan'),
        selective_macro_f1=macro_f1(y[acc], pred[acc]) if acc.any() else float('nan'),
    )
    if n_boot:
        r['bal_acc_ci95'] = bootstrap_ci(y, pred, bal_acc, n_boot)
        r['macro_f1_ci95'] = bootstrap_ci(y, pred, macro_f1, n_boot)
        r['accuracy_ci95'] = bootstrap_ci(y, pred, lambda a, b: float((a == b).mean()), n_boot)
    return r
