"""Grad-CAM tự chọn target layer theo tên backbone (SPEC v2 mục 5.3).

Dùng trong code:
  cam = GradCAM(model); heat, P = cam(x, target_class=None, T=1.0); rgb = overlay(img01, heat)

CLI - bộ ảnh Grad-CAM theo 4 nhóm lỗi, từ predictions.csv của evaluate.py:
  python -m src.explain.gradcam --ckpt runs/<run>/best_model.pth --predictions runs/<run>/eval_letterbox/predictions.csv \
      --data_root /content/data [--vn_real_root vn_real] --out_dir runs/<run>/gradcam
Nhóm lỗi:
  1_nham_vao_tai_che   thật là Thực phẩm/Rác khác, đoán Tái chế và KHÔNG bị từ chối (làm bẩn thùng tái chế)
  2_bo_sot_tai_che     thật là Tái chế, đoán nhóm khác
  3_nham_thuc_pham_khac  nhầm giữa Thực phẩm và Rác khác
  4_bi_tu_choi         bị luật UNCERTAIN từ chối
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.decision import load_temperature
from src.labels import VN_NAMES
from src.preprocess import denormalize, get_transforms, load_clean_rgb


def target_layer(model):
    b, name = model.backbone, model.backbone_name
    if name.startswith('mobilenetv3'):
        return b.blocks[-1]            # conv_head của timm nằm SAU global pool (1x1) nên không dùng được
    if name.startswith('convnext'):
        return b.stages[-1].blocks[-1]
    if name.startswith('swin'):
        return b.layers[-1].blocks[-1]  # đầu ra (B, H, W, C) -> cần reshape
    raise ValueError(f'Chưa khai báo target layer cho {name}')


class GradCAM:
    def __init__(self, model):
        self.model = model.eval()
        self.nhwc = model.backbone_name.startswith('swin')
        self._act = None
        self._grad = None
        self.h = target_layer(model).register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        self._act = out
        if out.requires_grad:
            out.register_hook(lambda g: setattr(self, '_grad', g))

    def remove(self):
        self.h.remove()
        self._act = None
        self._grad = None
        self.model.zero_grad(set_to_none=True)

    def __call__(self, x: torch.Tensor, target_class: int | None = None, T: float = 1.0):
        """x: 1x3xHxW đã chuẩn hoá. Trả (heatmap HxW trong [0,1], P_vn[3]). Điểm = log P_vn[c]."""
        dev = next(self.model.parameters()).device
        x = x.to(dev)
        try:
            with torch.enable_grad():
                self.model.zero_grad(set_to_none=True)
                logits = self.model(x)
                P = self.model.vn_probs(logits, T)
                c = int(P[0].argmax()) if target_class is None else int(target_class)
                torch.log(P[0, c].clamp_min(1e-12)).backward()
            A, G = self._act.detach(), self._grad.detach()
            if self.nhwc:
                A, G = A.permute(0, 3, 1, 2), G.permute(0, 3, 1, 2)
            w = G.mean(dim=(2, 3), keepdim=True)
            cam = F.relu((w * A).sum(1, keepdim=True))
            cam = F.interpolate(cam, size=x.shape[-2:], mode='bilinear', align_corners=False)[0, 0]
            cam = cam - cam.min()
            cam = cam / cam.max() if cam.max() > 0 else cam
            return cam.cpu().numpy(), P[0].detach().cpu().numpy()
        finally:
            # Giải phóng tensor trung gian & gradient ngay sau mỗi ảnh để chống tràn RAM trên CPU server
            self._act = None
            self._grad = None
            self.model.zero_grad(set_to_none=True)


def overlay(img01: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    import matplotlib
    heat = matplotlib.colormaps['jet'](cam)[..., :3]
    return ((1 - alpha) * img01 + alpha * heat).clip(0, 1)


GROUPS = {
    '1_nham_vao_tai_che': lambda d: (d.y != 0) & (d.pred == 0) & (d.accepted == 1),
    '2_bo_sot_tai_che': lambda d: (d.y == 0) & (d.pred != 0),
    '3_nham_thuc_pham_khac': lambda d: (d.y != 0) & (d.pred != 0) & (d.y != d.pred),
    '4_bi_tu_choi': lambda d: d.accepted == 0,
}


def main(argv=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    from src.models.net import load_checkpoint

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--predictions', required=True)
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--vn_real_root', default=None)
    ap.add_argument('--temperature', default=None)
    ap.add_argument('--out_dir', default=None)
    ap.add_argument('--tiers', nargs='*', default=None)
    ap.add_argument('--per_group', type=int, default=8)
    ap.add_argument('--device', default='auto')
    a = ap.parse_args(argv)
    from src.engine.common import get_device
    dev = get_device(a.device)
    model, _ = load_checkpoint(a.ckpt, map_location=dev)
    model.to(dev)
    temp = load_temperature(a.temperature or Path(a.ckpt).with_name('temperature.json'))
    T = float(temp['T'])
    d = pd.read_csv(a.predictions)
    if a.tiers:
        d = d[d.tier.isin(a.tiers)]
    out = Path(a.out_dir) if a.out_dir else Path(a.ckpt).parent / 'gradcam'
    out.mkdir(parents=True, exist_ok=True)
    tf = get_transforms(False, model.img_size)
    cam = GradCAM(model)
    for g, fn in GROUPS.items():
        sub = d[fn(d)].copy()
        if sub.empty:
            print(f'{g}: không có ảnh'); continue
        sub['conf'] = sub[['p0', 'p1', 'p2']].max(1)
        sub = sub.sort_values('conf', ascending=g == '4_bi_tu_choi').head(a.per_group)   # lỗi tự tin nhất trước
        fig, axes = plt.subplots(len(sub), 2, figsize=(6, 3 * len(sub)), squeeze=False)
        for (_, r), (ax1, ax2) in zip(sub.iterrows(), axes):
            root = a.vn_real_root if str(r.tier).startswith('T4') and a.vn_real_root else a.data_root
            x = tf(load_clean_rgb(Path(root) / r.path)).unsqueeze(0)
            heat, P = cam(x, target_class=int(r.pred), T=T)
            img = denormalize(x[0])
            ax1.imshow(img); ax2.imshow(overlay(img, heat))
            for ax in (ax1, ax2):
                ax.axis('off')
            ax1.set_title(f'{r.tier} | {str(r.orig_label)[:26]}', fontsize=8)
            ax2.set_title(f'Thật: {VN_NAMES[int(r.y)]} | Đoán: {VN_NAMES[int(r.pred)]} ({P.max():.2f})', fontsize=8)
        fig.suptitle(g); fig.tight_layout(); fig.savefig(out / f'{g}.png', dpi=110); plt.close(fig)
        sub.to_csv(out / f'{g}.csv', index=False)
        print(f'{g}: {len(sub)} ảnh -> {out / (g + ".png")}')
    cam.remove()


if __name__ == '__main__':
    main()
