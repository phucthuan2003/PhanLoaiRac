"""Backbone timm + 2 chế độ đầu ra (SPEC v2 mục 4.2).

flat:          3 logits -> softmax -> P_vn
fine_grained:  30 logits -> softmax p -> P_vn = p @ M   (M: ma trận 0/1 cố định, 30 x 3)
Hàm vn_probs() là cách DUY NHẤT để ra xác suất 3 nhóm (dùng chung cho calibrate / evaluate / demo).
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.labels import fine_labels, mapping_matrix

BACKBONES = ('mobilenetv3_large_100', 'convnext_tiny', 'swinv2_tiny_window8_256')


class WasteNet(nn.Module):
    def __init__(self, backbone: str = 'convnext_tiny', mode: str = 'flat', pretrained: bool = True,
                 img_size: int = 256, drop: float = 0.2):
        super().__init__()
        import timm
        if mode not in ('flat', 'fine_grained'):
            raise ValueError(f'mode không hợp lệ: {mode}')
        self.backbone_name, self.mode, self.img_size = backbone, mode, img_size
        kw = dict(pretrained=pretrained, num_classes=0)
        if backbone.startswith('swin'):
            kw['img_size'] = img_size
        self.backbone = timm.create_model(backbone, **kw)
        self.fine_labels = fine_labels()
        n_out = 3 if mode == 'flat' else len(self.fine_labels)
        # mobilenetv3 có conv_head sau pooling (1280) khác num_features (960) -> lấy đúng số chiều đầu ra
        feat = getattr(self.backbone, 'head_hidden_size', None) or self.backbone.num_features
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(feat, n_out))
        self.register_buffer('M', torch.from_numpy(mapping_matrix(self.fine_labels)), persistent=False)

    def forward(self, x):
        return self.head(self.backbone(x))

    def vn_probs(self, logits: torch.Tensor, T: float | torch.Tensor = 1.0) -> torch.Tensor:
        """Xác suất 3 nhóm sau khi chia nhiệt T (áp lên logits gốc, TRƯỚC khi gộp)."""
        p = F.softmax(logits.float() / T, dim=1)
        return p if self.mode == 'flat' else p @ self.M.to(p.device)

    def fine_probs(self, logits: torch.Tensor, T: float = 1.0):
        return None if self.mode == 'flat' else F.softmax(logits.float() / T, dim=1)

    # ---- LP-FT ----
    def freeze_backbone(self, frozen: bool = True):
        for p in self.backbone.parameters():
            p.requires_grad = not frozen
        self._frozen = frozen

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and getattr(self, '_frozen', False):
            self.backbone.eval()      # giai đoạn LP: giữ BatchNorm/Dropout của backbone ở chế độ eval
        return self

    def config(self) -> dict:
        return dict(backbone=self.backbone_name, mode=self.mode, img_size=self.img_size, fine_labels=self.fine_labels)


def save_checkpoint(model: WasteNet, path: str | Path, **extra):
    torch.save({'state_dict': model.state_dict(), 'config': model.config(), **extra}, path)


def load_checkpoint(path: str | Path, map_location='cpu') -> tuple[WasteNet, dict]:
    ck = torch.load(path, map_location=map_location, weights_only=False)
    cfg = ck['config']
    model = WasteNet(cfg['backbone'], cfg['mode'], pretrained=False, img_size=cfg.get('img_size', 256))
    if cfg['mode'] == 'fine_grained' and cfg.get('fine_labels') and cfg['fine_labels'] != model.fine_labels:
        raise ValueError('Danh sách nhãn con trong checkpoint khác configs/label_mapping.csv')
    model.load_state_dict(ck['state_dict'])
    return model.eval(), ck


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
