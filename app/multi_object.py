"""Isolated Multi-Object Waste Detector + Visual Callout Annotator for Demo UI.

Pipeline (2-stage, keeps core src/ classifier 100% untouched):
  Stage 1: Localize multiple waste items using FasterRCNN-MobileNetV3-320 (COCO)
           excluding macro background/furniture/person classes, with an automatic
           OpenCV contour fallback for non-COCO items (e.g. styrofoam, paper, bags).
  Stage 2: Batch-classify every cropped item through the team's trained WasteNet
           (ConvNeXt-Tiny + Temperature Scaling + Contamination-Safe Decision Rule).
  Stage 3: Draw clean color-coded boxes, numbered badges (#1, #2...), and pointer
           labels directly on the image so users understand at a glance.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import nms
from torchvision.transforms import functional as TF

from src.decision import decide
from src.preprocess import get_transforms

# Visual palette per bin (high contrast, intuitive, accessible)
BIN_STYLE = {
    0: {
        'short': 'TÁI CHẾ',
        'icon': '♻️',
        'hex': '#0284c7',
        'rgb': (2, 132, 199),
        'bg_light': '#e0f2fe',
        'hint': 'Tráng sạch, để khô',
    },
    1: {
        'short': 'THỰC PHẨM',
        'icon': '🍎',
        'hex': '#16a34a',
        'rgb': (22, 163, 74),
        'bg_light': '#dcfce7',
        'hint': 'Tách túi nilon',
    },
    2: {
        'short': 'RÁC KHÁC',
        'icon': '🗑️',
        'hex': '#475569',
        'rgb': (71, 85, 105),
        'bg_light': '#f1f5f9',
        'hint': 'Pin thu gom riêng',
    },
    'UNCERTAIN': {
        'short': 'KIỂM TRA',
        'icon': '⚠️',
        'hex': '#d97706',
        'rgb': (217, 119, 6),
        'bg_light': '#fef3c7',
        'hint': 'Xốp/bẩn → Rác khác',
    },
}

# COCO class IDs to ignore (people, vehicles, large furniture, appliances, animals)
IGNORE_COCO_IDS = {
    1,   # person
    2, 3, 4, 5, 6, 7, 8, 9,  # vehicles
    15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,  # animals
    62, 63, 64, 65, 67, 70,  # chair, couch, potted plant, bed, dining table, toilet
    72, 73, 78, 79, 80, 81, 82,  # tv, laptop, microwave, oven, toaster, sink, refrigerator
}


def load_object_detector(device: str = 'cpu'):
    """Load lightweight FasterRCNN-MobileNetV3-320 detector (pretrained COCO)."""
    from torchvision.models.detection import (
        FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
        fasterrcnn_mobilenet_v3_large_320_fpn,
    )

    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    det = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
    det.to(device).eval()
    return det


def _fallback_contour_boxes(img: Image.Image, max_boxes: int = 6) -> List[Tuple[int, int, int, int]]:
    """Lightweight OpenCV region proposals when COCO misses non-COCO waste (styrofoam, bags, paper)."""
    rgb = np.asarray(img)
    h, w = rgb.shape[:2]
    scale = min(1.0, 512.0 / max(h, w))
    small = cv2.resize(rgb, (max(1, int(w * scale)), max(1, int(h * scale))))
    sh, sw = small.shape[:2]
    frame_area = float(sh * sw)

    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 130)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    raw_boxes = []
    scores = []
    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = float(bw * bh)
        ratio = area / frame_area
        if ratio < 0.025 or ratio > 0.82:
            continue
        x1 = int(x / scale)
        y1 = int(y / scale)
        x2 = int((x + bw) / scale)
        y2 = int((y + bh) / scale)
        raw_boxes.append([x1, y1, x2, y2])
        scores.append(area)

    if not raw_boxes:
        return []
    bt = torch.tensor(raw_boxes, dtype=torch.float32)
    st = torch.tensor(scores, dtype=torch.float32)
    keep = nms(bt, st, iou_threshold=0.40)[:max_boxes]
    return [tuple(int(v) for v in raw_boxes[i]) for i in keep.tolist()]


@torch.no_grad()
def propose_waste_boxes(
    detector,
    img: Image.Image,
    score_thresh: float = 0.30,
    max_objects: int = 8,
    use_contour_fallback: bool = True,
    device: str = 'cpu',
) -> Tuple[List[Tuple[int, int, int, int]], str]:
    """Return list of (x1, y1, x2, y2) boxes and source tag ('detector' | 'contour' | 'full_frame')."""
    w, h = img.size
    frame_area = float(w * h)

    # Run detector on a bounded-size image for speed on 12-48MP phone photos
    scale = min(1.0, 800.0 / max(w, h))
    work = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.BILINEAR) if scale < 1.0 else img
    tensor = TF.to_tensor(work).to(device)
    out = detector([tensor])[0]

    boxes = out['boxes'].cpu()
    scores = out['scores'].cpu()
    labels = out['labels'].cpu()

    valid_boxes = []
    valid_scores = []
    for box, sc, lb in zip(boxes, scores, labels):
        if float(sc) < score_thresh:
            continue
        if int(lb) in IGNORE_COCO_IDS:
            continue
        x1, y1, x2, y2 = (box / scale).tolist()
        x1 = max(0, min(w - 1, int(round(x1))))
        y1 = max(0, min(h - 1, int(round(y1))))
        x2 = max(x1 + 1, min(w, int(round(x2))))
        y2 = max(y1 + 1, min(h, int(round(y2))))
        area_ratio = ((x2 - x1) * (y2 - y1)) / frame_area
        if area_ratio < 0.012 or area_ratio > 0.88:
            continue
        valid_boxes.append([x1, y1, x2, y2])
        valid_scores.append(float(sc))

    if valid_boxes:
        bt = torch.tensor(valid_boxes, dtype=torch.float32)
        st = torch.tensor(valid_scores, dtype=torch.float32)
        keep = nms(bt, st, iou_threshold=0.45)[:max_objects]
        # Sort left-to-right, top-to-bottom so item numbers #1, #2... read naturally
        selected = [tuple(valid_boxes[i]) for i in keep.tolist()]
        selected.sort(key=lambda b: (b[1] // max(1, h // 4), b[0]))
        return selected, 'detector'

    if use_contour_fallback:
        c_boxes = _fallback_contour_boxes(img, max_boxes=min(max_objects, 6))
        if c_boxes:
            c_boxes.sort(key=lambda b: (b[1] // max(1, h // 4), b[0]))
            return c_boxes, 'contour'

    return [(0, 0, w, h)], 'full_frame'


def _expand_box(box: Tuple[int, int, int, int], w: int, h: int, pad_frac: float = 0.08) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    pw = int((x2 - x1) * pad_frac)
    ph = int((y2 - y1) * pad_frac)
    return max(0, x1 - pw), max(0, y1 - ph), min(w, x2 + pw), min(h, y2 + ph)


@torch.no_grad()
def detect_and_classify_multi(
    model,
    detector,
    img: Image.Image,
    temp: dict,
    tau_recycle: float,
    min_conf: float = 0.60,
    score_thresh: float = 0.30,
    max_objects: int = 8,
    use_contour_fallback: bool = True,
    device: str = 'cpu',
) -> Tuple[List[Dict], str]:
    """Detect all waste items in img and batch-classify each into 3 VN bins."""
    w, h = img.size
    boxes, source = propose_waste_boxes(
        detector,
        img,
        score_thresh=score_thresh,
        max_objects=max_objects,
        use_contour_fallback=use_contour_fallback,
        device=device,
    )
    tf = get_transforms(False, model.img_size)
    crops = []
    tensors = []
    for box in boxes:
        cbox = _expand_box(box, w, h, pad_frac=0.08)
        crop = img.crop(cbox)
        crops.append(crop)
        tensors.append(tf(crop))

    batch = torch.stack(tensors, dim=0).to(device)
    T_val = float(temp.get('T', 1.0))
    logits = model.eval().to(device)(batch).float().cpu()
    vn_probs = model.vn_probs(logits, T_val).numpy()
    fine_probs = model.fine_probs(logits, T_val)
    fine_np = None if fine_probs is None else fine_probs.numpy()

    items: List[Dict] = []
    for idx, (box, crop, p) in enumerate(zip(boxes, crops, vn_probs), start=1):
        fp = None if fine_np is None else fine_np[idx - 1]
        res = decide(
            p,
            tau_recycle,
            fp,
            model.fine_labels if fp is not None else None,
            min_conf=min_conf,
        )
        res['id'] = idx
        res['box'] = box
        res['crop'] = crop
        res['style_key'] = 'UNCERTAIN' if res['status'] == 'UNCERTAIN' else res['vn_class']
        items.append(res)

    return items, source


def _load_unicode_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a TTF font supporting Vietnamese accents (Windows / Linux)."""
    candidates = [
        'C:/Windows/Fonts/segoeui.ttf',
        'C:/Windows/Fonts/arial.ttf',
        'C:/Windows/Fonts/tahoma.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def draw_annotated_image(img: Image.Image, items: List[Dict], max_side: int = 1100) -> Image.Image:
    """Draw clean color-coded boxes + pointer tags (#1 TÁI CHẾ 94%) pointing straight to each item."""
    w0, h0 = img.size
    scale = min(1.0, max_side / float(max(w0, h0)))
    canvas = img.resize((max(1, int(w0 * scale)), max(1, int(h0 * scale))), Image.Resampling.LANCZOS) if scale < 1.0 else img.copy()
    w, h = canvas.size
    draw = ImageDraw.Draw(canvas, 'RGBA')

    font_size = max(14, min(22, int(min(w, h) * 0.032)))
    font = _load_unicode_font(font_size)
    thick = max(3, int(min(w, h) * 0.006))

    used_label_rects: List[Tuple[int, int, int, int]] = []

    for item in items:
        x1, y1, x2, y2 = [int(round(v * scale)) for v in item['box']]
        style = BIN_STYLE[item['style_key']]
        rgb = style['rgb']
        label_txt = f" #{item['id']} {style['short']} {item['prob']:.0%} "

        # Subtle tinted fill + crisp rounded bounding box
        draw.rounded_rectangle([x1, y1, x2, y2], radius=8, fill=rgb + (28,), outline=rgb + (255,), width=thick)

        # Measure label pill
        tb = draw.textbbox((0, 0), label_txt, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        pad_x, pad_y = 6, 4
        pill_w, pill_h = tw + 2 * pad_x, th + 2 * pad_y

        # Place pill directly above box (or inside top-left if near top edge) & nudge to avoid overlap
        lx = max(4, min(w - pill_w - 4, x1))
        ly = y1 - pill_h - 6 if y1 - pill_h - 6 >= 4 else min(h - pill_h - 4, y1 + 6)
        for _ in range(6):
            overlap = any(
                not (lx + pill_w < rx1 or lx > rx2 or ly + pill_h < ry1 or ly > ry2)
                for rx1, ry1, rx2, ry2 in used_label_rects
            )
            if not overlap:
                break
            ly = min(h - pill_h - 4, ly + pill_h + 4)
        used_label_rects.append((lx, ly, lx + pill_w, ly + pill_h))

        # Pointer line connecting pill to top-center of object when separated
        anchor_x = max(x1 + 6, min(x2 - 6, lx + pill_w // 2))
        anchor_y = y1 if ly + pill_h <= y1 else y1 + 4
        if abs(anchor_y - (ly + pill_h)) > 2 and ly + pill_h <= y1:
            draw.line([(lx + pill_w // 2, ly + pill_h), (anchor_x, anchor_y)], fill=rgb + (255,), width=max(2, thick - 1))
            draw.ellipse([anchor_x - 4, anchor_y - 4, anchor_x + 4, anchor_y + 4], fill=rgb + (255,))

        draw.rounded_rectangle([lx, ly, lx + pill_w, ly + pill_h], radius=6, fill=rgb + (240,))
        draw.text((lx + pad_x, ly + pad_y - 1), label_txt, font=font, fill=(255, 255, 255, 255))

    return canvas
