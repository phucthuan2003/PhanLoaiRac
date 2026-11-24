"""Đánh giá 4 tầng (SPEC v2 mục 5.2) bằng MỘT lệnh.

  python -m src.engine.evaluate --ckpt runs/<run>/best_model.pth --manifest ... --data_root ... \
      [--vn_real_manifest vn_real/vn_real_manifest.csv --vn_real_root vn_real] [--out_dir results/<run>]

Ghi ra <out_dir>/:
  results.json          mọi chỉ số từng tầng (bal_acc, macro_f1, P/R/F1 từng lớp, CM, contamination, ECE,
                        abstention, selective_*, CI bootstrap 95%)
  summary.csv           bảng gọn tầng x chỉ số + mức giảm so với tầng 1
  per_label_recall.csv  tỷ lệ đúng theo nhãn gốc (vd. GD plastic / trash) - phân tích tầng 3
  risk_coverage.json/.png   đường risk-coverage RIÊNG từng tầng
  confusion_matrices.png
  predictions.csv       từng ảnh: nhãn, dự đoán, xác suất, bị từ chối? (dùng cho Grad-CAM theo nhóm lỗi)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.manifest import eval_tiers, load_manifest, load_vn_real, vn_real_tiers
from src.decision import load_temperature
from src.engine.common import default_workers, dump_json, get_device, predict_logits, smoke_subset, vn_probs_np
from src.labels import VN_NAMES
from src.metrics import accept_mask, full_report, risk_coverage
from src.models.net import load_checkpoint

KEYS = ['n', 'bal_acc', 'macro_f1', 'accuracy', 'contamination_rate', 'missed_recycle_rate', 'ece',
        'abstention_rate', 'selective_bal_acc']


def plot_cms(reports: dict, path: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names = list(reports)
    cols = min(4, len(names)); rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4 * rows), squeeze=False)
    lab = [VN_NAMES[i] for i in range(3)]
    for ax in axes.ravel():
        ax.axis('off')
    for ax, n in zip(axes.ravel(), names):
        cm = np.array(reports[n]['confusion_matrix'], float)
        with np.errstate(invalid='ignore', divide='ignore'):
            cmn = cm / cm.sum(1, keepdims=True)
        ax.axis('on'); ax.imshow(np.nan_to_num(cmn), vmin=0, vmax=1, cmap='Blues')
        for i in range(3):
            for j in range(3):
                ax.text(j, i, '-' if np.isnan(cmn[i, j]) else f'{cmn[i, j]:.2f}\n({int(cm[i, j])})', ha='center',
                        va='center', fontsize=8, color='white' if np.nan_to_num(cmn[i, j]) > .5 else 'black')
        ax.set_xticks(range(3), lab, rotation=25, fontsize=8); ax.set_yticks(range(3), lab, fontsize=8)
        ax.set_xlabel('Dự đoán'); ax.set_ylabel('Thực tế')
        ax.set_title(f"{n}\nbal_acc {reports[n]['bal_acc']:.3f}", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_risk_coverage(curves: dict, path: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for n, c in curves.items():
        cov = [r['coverage'] for r in c]
        a1.plot(cov, [r['selective_risk'] for r in c], label=n)
        a2.plot(cov, [r['contamination'] for r in c], label=n)
    a1.set_xlabel('Coverage'); a1.set_ylabel('Selective risk (tỷ lệ sai trên ca được nhận)')
    a2.set_xlabel('Coverage'); a2.set_ylabel('Contamination rate')
    a1.set_title('Risk–coverage theo tầng'); a2.set_title('Contamination–coverage theo tầng')
    a2.legend(fontsize=7); fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--temperature', default=None, help='mặc định: temperature.json cạnh checkpoint (không có -> T=1)')
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--vn_real_manifest', default=None)
    ap.add_argument('--vn_real_root', default=None)
    ap.add_argument('--out_dir', default=None, help='mặc định: <thư mục ckpt>/eval_<eval_mode>')
    ap.add_argument('--eval_mode', default='letterbox', choices=['letterbox', 'center_crop'])
    ap.add_argument('--tiers', nargs='*', default=None, help='chỉ chạy các tầng này (tên trong results.json)')
    ap.add_argument('--bootstrap', type=int, default=1000)
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--device', default='auto')
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args(argv)
    device = get_device(a.device)
    workers = default_workers() if a.workers is None else a.workers
    model, ck = load_checkpoint(a.ckpt, map_location='cpu')
    model.to(device)
    tp = a.temperature or Path(a.ckpt).with_name('temperature.json')
    temp = load_temperature(tp)
    T, tau = float(temp['T']), float(temp['tau_recycle'])
    out = Path(a.out_dir) if a.out_dir else Path(a.ckpt).parent / f'eval_{a.eval_mode}'
    out.mkdir(parents=True, exist_ok=True)
    print(f'Model {model.backbone_name}/{model.mode} | T={T:.4f} tau={tau} | eval_mode={a.eval_mode} | {device}')

    tiers = eval_tiers(load_manifest(a.manifest))
    roots = {k: a.data_root for k in tiers}
    agreement = None
    if a.vn_real_manifest:
        vdf = load_vn_real(a.vn_real_manifest)
        if 'annotator2_vn_class_id' in vdf and (vdf['annotator2_vn_class_id'].astype(str) != '').all():
            from sklearn.metrics import cohen_kappa_score
            y2 = pd.to_numeric(vdf['annotator2_vn_class_id']).astype(int)
            agreement = dict(percent_agree=float((y2 == vdf['vn_class_id']).mean()),
                             cohen_kappa=float(cohen_kappa_score(vdf['vn_class_id'], y2)))
            print('VN-Real - đồng thuận 2 người gán nhãn:', agreement)
        vr = vn_real_tiers(vdf)
        tiers.update(vr)
        roots.update({k: a.vn_real_root or str(Path(a.vn_real_manifest).parent) for k in vr})
    if a.tiers:
        tiers = {k: v for k, v in tiers.items() if k in a.tiers}
    if a.smoke:
        tiers = {k: smoke_subset(v, 8) for k, v in tiers.items()}

    reports, curves, preds, per_label = {}, {}, [], []
    cache = {}
    for name, df in tiers.items():
        key = (roots[name], tuple(df['path']))
        if key not in cache:          # T4_vn_real và các ô con dùng chung ảnh -> chỉ suy luận 1 lần
            logits = predict_logits(model, df, roots[name], device, a.batch, workers, a.eval_mode)
            cache[key] = logits
        logits = cache[key]
        P = vn_probs_np(model, logits, T)
        y = df['vn_class_id'].to_numpy()
        rep = full_report(y, P, tau, a.bootstrap)
        reports[name] = rep; curves[name] = risk_coverage(P, y)
        print(f"{name:18s} n={rep['n']:6d}  bal_acc {rep['bal_acc']:.4f}  macro-F1 {rep['macro_f1']:.4f}  "
              f"contam {rep['contamination_rate']:.4f}  abstain {rep['abstention_rate']:.3f}", flush=True)
        pr = P.argmax(1); acc = accept_mask(P, tau)
        d = pd.DataFrame(dict(tier=name, path=df['path'], dataset=df['dataset'], orig_label=df['orig_label'],
                              y=y, pred=pr, p0=P[:, 0], p1=P[:, 1], p2=P[:, 2], accepted=acc.astype(int)))
        if model.mode == 'fine_grained':
            with torch.no_grad():
                fp = model.fine_probs(torch.from_numpy(logits), T).numpy()
            d['top_fine'] = [model.fine_labels[i] for i in fp.argmax(1)]
            d['top_fine_prob'] = fp.max(1)
        if name.startswith('T4_dirty'):
            continue                   # ô con của VN-Real đã có trong T4_vn_real
        preds.append(d)
        g = d.assign(correct=(d.y == d.pred)).groupby(['dataset', 'orig_label', 'y'])
        per_label.append(g.agg(n=('correct', 'size'), recall=('correct', 'mean'),
                               pred_tai_che=('pred', lambda s: (s == 0).mean()),
                               pred_thuc_pham=('pred', lambda s: (s == 1).mean()),
                               pred_khac=('pred', lambda s: (s == 2).mean())).reset_index().assign(tier=name))

    meta = dict(ckpt=str(a.ckpt), backbone=model.backbone_name, mode=model.mode, T=T, tau=tau,
                eval_mode=a.eval_mode, bootstrap=a.bootstrap, smoke=a.smoke, vn_real_label_agreement=agreement)
    dump_json(dict(meta=meta, tiers=reports), out / 'results.json')
    dump_json(curves, out / 'risk_coverage.json')
    summ = pd.DataFrame({k: {m: reports[k][m] for m in KEYS} for k in reports}).T
    if 'T1_test_in' in summ.index:
        for m in ('bal_acc', 'macro_f1'):
            summ[f'drop_{m}_vs_T1'] = summ.loc['T1_test_in', m] - summ[m]
    for k in reports:
        if 'bal_acc_ci95' in reports[k]:
            summ.loc[k, 'bal_acc_ci95'] = '[{:.3f}, {:.3f}]'.format(*reports[k]['bal_acc_ci95'])
    summ.to_csv(out / 'summary.csv')
    pd.concat(preds).to_csv(out / 'predictions.csv', index=False)
    pd.concat(per_label).sort_values(['tier', 'recall']).to_csv(out / 'per_label_recall.csv', index=False)
    plot_cms(reports, out / 'confusion_matrices.png')
    plot_risk_coverage(curves, out / 'risk_coverage.png')
    print(summ[['n', 'bal_acc', 'macro_f1', 'contamination_rate', 'abstention_rate']].round(4).to_string())
    print('Đã lưu ->', out)


if __name__ == '__main__':
    main()
