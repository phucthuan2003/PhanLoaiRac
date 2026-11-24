"""Tiền xử lý DÙNG CHUNG cho train / evaluate / demo (SPEC v2 mục 4.1).

Không được viết lại các bước này ở nơi khác: mọi chỗ đọc ảnh phải gọi load_clean_rgb()
và get_transforms() để demo cho cùng kết quả với lúc đánh giá.
"""
from __future__ import annotations

import io
import random
from pathlib import Path
from typing import BinaryIO, Union

from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None
IMG_SIZE = 256
GRAY = (128, 128, 128)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_clean_rgb(src: Union[str, Path, bytes, BinaryIO]) -> Image.Image:
    """Đọc ảnh -> RGB đã xoay đúng theo EXIF; kênh alpha được ghép lên nền xám 128.

    Nhận đường dẫn, bytes hoặc file-like (vd. UploadedFile của Streamlit).
    """
    if isinstance(src, (bytes, bytearray)):
        src = io.BytesIO(src)
    with Image.open(src) as img:
        img.seek(0)                      # GIF/MPO/WebP động: lấy khung đầu
        img = ImageOps.exif_transpose(img)
        if img.mode in ('RGBA', 'LA') or (img.mode == 'P' and 'transparency' in img.info):
            img = img.convert('RGBA')
            bg = Image.new('RGBA', img.size, GRAY + (255,))
            img = Image.alpha_composite(bg, img)
        return img.convert('RGB')


class LetterboxSquare:
    """Đệm ảnh thành hình vuông (giữ nguyên tỷ lệ vật thể), vật nằm giữa."""

    def __init__(self, fill=GRAY):
        self.fill = tuple(fill)

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w == h:
            return img
        s = max(w, h)
        out = Image.new('RGB', (s, s), self.fill)
        out.paste(img, ((s - w) // 2, (s - h) // 2))
        return out

    def __repr__(self):
        return f'LetterboxSquare(fill={self.fill})'


class RandomLetterbox:
    """Lúc train: với xác suất p, đệm ảnh về hình vuông rồi THU NHỎ vật vào khung lớn hơn
    (giả lập ảnh điện thoại 4:3 / vật nhỏ có viền xám như lúc eval). Không dùng RandomApply
    để giữ được ảnh PIL."""

    def __init__(self, p=0.3, max_extra=0.35, fill=GRAY):
        self.p, self.max_extra, self.fill = p, max_extra, tuple(fill)

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        w, h = img.size
        s = int(max(w, h) * (1 + random.uniform(0, self.max_extra)))
        out = Image.new('RGB', (s, s), self.fill)
        out.paste(img, (random.randint(0, s - w), random.randint(0, s - h)))
        return out

    def __repr__(self):
        return f'RandomLetterbox(p={self.p})'


def get_transforms(is_train: bool, img_size: int = IMG_SIZE, aug: str = 'advanced', eval_mode: str = 'letterbox'):
    """Trả về torchvision transform.

    aug (chỉ khi train): 'basic'   = RandomResizedCrop + lật ngang (mốc so sánh cho SV3)
                         'advanced'= pipeline đầy đủ trong SPEC v2 (mặc định)
    eval_mode:           'letterbox' (mặc định, dùng cho evaluate và demo)
                         'center_crop' (chỉ cho ablation letterbox vs center-crop)
    """
    import torchvision.transforms as T   # import muộn: phần PIL ở trên dùng được khi chưa cài torch

    norm = [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if is_train:
        if aug == 'basic':
            return T.Compose([T.RandomResizedCrop(img_size, scale=(0.65, 1.0)), T.RandomHorizontalFlip(0.5), *norm])
        if aug != 'advanced':
            raise ValueError(f'aug không hợp lệ: {aug}')
        return T.Compose([
            RandomLetterbox(p=0.3),
            T.RandomResizedCrop(img_size, scale=(0.65, 1.0)),
            T.RandomHorizontalFlip(0.5),
            T.ColorJitter(0.3, 0.3, 0.25, 0.05),
            T.RandomApply([T.GaussianBlur(3, sigma=(0.1, 1.2))], p=0.2),
            *norm,
            T.RandomErasing(p=0.25, scale=(0.02, 0.15)),
        ])
    if eval_mode == 'letterbox':
        return T.Compose([LetterboxSquare(), T.Resize((img_size, img_size)), *norm])
    if eval_mode == 'center_crop':
        return T.Compose([T.Resize(img_size), T.CenterCrop(img_size), *norm])
    raise ValueError(f'eval_mode không hợp lệ: {eval_mode}')


def denormalize(t):
    """Tensor đã chuẩn hoá (3,H,W) -> numpy HxWx3 trong [0,1] (để vẽ Grad-CAM)."""
    import torch
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (t.detach().cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
