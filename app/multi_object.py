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

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import nms

from src.decision import decide
from src.preprocess import GRAY, IMAGENET_MEAN, IMAGENET_STD

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

_NORM_CACHE: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}


def _get_norm_tensors(dev: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    key = str(dev)
    cached = _NORM_CACHE.get(key)
    if cached is None:
        mean = torch.tensor(IMAGENET_MEAN, device=dev, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=dev, dtype=torch.float32).view(1, 3, 1, 1)
        cached = (mean, std)
        _NORM_CACHE[key] = cached
    return cached


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    if device is None or str(device) == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(device)


def load_object_detector(device: str | torch.device | None = None):
    """Load lightweight FasterRCNN-MobileNetV3-320 detector (pretrained COCO)."""
    from torchvision.models.detection import (
        FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
        fasterrcnn_mobilenet_v3_large_320_fpn,
    )

    dev = resolve_device(device)
    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    det = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
    det.to(dev).eval()
    return det


def _to_rgb_array(img: Union[Image.Image, np.ndarray]) -> np.ndarray:
    if isinstance(img, np.ndarray):
        return img
    return np.asarray(img)


def _letterbox_resize_rgb(crop_rgb: np.ndarray, img_size: int) -> np.ndarray:
    """Fast OpenCV equivalent of LetterboxSquare(GRAY) + Resize((img_size, img_size))."""
    h, w = crop_rgb.shape[:2]
    if h != w:
        s = max(h, w)
        top = (s - h) // 2
        bottom = s - h - top
        left = (s - w) // 2
        right = s - w - left
        crop_rgb = cv2.copyMakeBorder(
            crop_rgb, top, bottom, left, right, cv2.BORDER_CONSTANT, value=GRAY
        )
    if crop_rgb.shape[0] != img_size or crop_rgb.shape[1] != img_size:
        interp = cv2.INTER_AREA if crop_rgb.shape[0] > img_size else cv2.INTER_LINEAR
        crop_rgb = cv2.resize(crop_rgb, (img_size, img_size), interpolation=interp)
    return crop_rgb


def preprocess_rgb_batch(crops_rgb: List[np.ndarray], img_size: int, dev: torch.device) -> torch.Tensor:
    """Letterbox + resize on CPU (fast uint8 OpenCV), convert & normalize on GPU."""
    resized = np.stack([_letterbox_resize_rgb(c, img_size) for c in crops_rgb], axis=0)
    t = torch.from_numpy(resized).to(dev, non_blocking=(dev.type == 'cuda'))
    t = t.permute(0, 3, 1, 2).float().div_(255.0)
    mean, std = _get_norm_tensors(dev)
    return t.sub_(mean).div_(std)


def _fallback_contour_boxes(rgb: np.ndarray, max_boxes: int = 6) -> List[Tuple[int, int, int, int]]:
    """Lightweight OpenCV region proposals when COCO misses non-COCO waste (styrofoam, bags, paper)."""
    h, w = rgb.shape[:2]
    scale = min(1.0, 512.0 / max(h, w))
    small = cv2.resize(rgb, (max(1, int(w * scale)), max(1, int(h * scale)))) if scale < 1.0 else rgb
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
    img: Union[Image.Image, np.ndarray],
    score_thresh: float = 0.30,
    max_objects: int = 8,
    use_contour_fallback: bool = True,
    device: str | torch.device | None = None,
    det_max_side: int = 640,
) -> Tuple[List[Tuple[int, int, int, int]], str]:
    """Return list of (x1, y1, x2, y2) boxes and source tag ('detector' | 'contour' | 'full_frame')."""
    dev = resolve_device(device) if device is not None else next(detector.parameters()).device
    rgb = _to_rgb_array(img)
    h, w = rgb.shape[:2]
    frame_area = float(w * h)

    # Fast OpenCV downscale + transfer uint8 to GPU (4x less PCIe bandwidth than float32)
    scale = min(1.0, float(det_max_side) / max(w, h))
    work = cv2.resize(rgb, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_LINEAR) if scale < 1.0 else rgb
    tensor = torch.from_numpy(work).to(dev, non_blocking=(dev.type == 'cuda')).permute(2, 0, 1).float().div_(255.0)
    # FP32 is faster than FP16 autocast on FasterRCNN-MobileNetV3 (avoids ~3.8s cold init & cast overhead)
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
        selected = [tuple(valid_boxes[i]) for i in keep.tolist()]
        selected.sort(key=lambda b: (b[1] // max(1, h // 4), b[0]))
        return selected, 'detector'

    if use_contour_fallback:
        c_boxes = _fallback_contour_boxes(rgb, max_boxes=min(max_objects, 6))
        if c_boxes:
            c_boxes.sort(key=lambda b: (b[1] // max(1, h // 4), b[0]))
            return c_boxes, 'contour'

    return [(0, 0, w, h)], 'full_frame'


def _expand_box(box: Tuple[int, int, int, int], w: int, h: int, pad_frac: float = 0.08) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    pw = int((x2 - x1) * pad_frac)
    ph = int((y2 - y1) * pad_frac)
    return max(0, x1 - pw), max(0, y1 - ph), min(w, x2 + pw), min(h, y2 + ph)


_CUDA_FIXED_BATCH = 6


@torch.no_grad()
def forward_wastenet_fast(model, batch: torch.Tensor, dev: torch.device) -> torch.Tensor:
    """Run WasteNet in FP32 with static CUDA batch shapes (1 or 6) to avoid ~1.5s cuDNN replanning per new N."""
    model.eval()
    if next(model.parameters()).device != dev:
        model.to(dev)
    n = batch.shape[0]
    if dev.type != 'cuda' or n == 1 or n == _CUDA_FIXED_BATCH:
        return model(batch).float().cpu()
    if n < _CUDA_FIXED_BATCH:
        pad = torch.zeros((_CUDA_FIXED_BATCH - n, *batch.shape[1:]), device=dev, dtype=batch.dtype)
        return model(torch.cat([batch, pad], dim=0))[:n].float().cpu()
    # Chunk larger batches in fixed size 6
    chunks = []
    for i in range(0, n, _CUDA_FIXED_BATCH):
        sub = batch[i : i + _CUDA_FIXED_BATCH]
        if sub.shape[0] < _CUDA_FIXED_BATCH:
            pad = torch.zeros((_CUDA_FIXED_BATCH - sub.shape[0], *sub.shape[1:]), device=dev, dtype=sub.dtype)
            chunks.append(model(torch.cat([sub, pad], dim=0))[: sub.shape[0]].float().cpu())
        else:
            chunks.append(model(sub).float().cpu())
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def warmup_gpu_models(model, detector, device: str | torch.device) -> None:
    """One-time CUDA warmup for static shapes (bs=1, bs=6, detector) so live streaming starts instantly."""
    dev = resolve_device(device)
    if dev.type != 'cuda':
        return
    model.eval().to(dev)
    sz = getattr(model, 'img_size', 256)
    for bs in (1, _CUDA_FIXED_BATCH):
        _ = model(torch.zeros((bs, 3, sz, sz), device=dev, dtype=torch.float32))
    if detector is not None:
        detector.eval().to(dev)
        _ = detector([torch.zeros((3, 384, 512), device=dev, dtype=torch.float32)])
    torch.cuda.synchronize()


@torch.no_grad()
def detect_and_classify_multi(
    model,
    detector,
    img: Union[Image.Image, np.ndarray],
    temp: dict,
    tau_recycle: float,
    min_conf: float = 0.60,
    score_thresh: float = 0.30,
    max_objects: int = 8,
    use_contour_fallback: bool = True,
    device: str | torch.device | None = None,
    return_crops: bool = True,
    det_max_side: int = 640,
) -> Tuple[List[Dict], str]:
    """Detect all waste items in img and batch-classify each into 3 VN bins."""
    dev = resolve_device(device) if device is not None else next(model.parameters()).device
    rgb = _to_rgb_array(img)
    h, w = rgb.shape[:2]
    boxes, source = propose_waste_boxes(
        detector,
        rgb,
        score_thresh=score_thresh,
        max_objects=max_objects,
        use_contour_fallback=use_contour_fallback,
        device=dev,
        det_max_side=det_max_side,
    )

    crops_np: List[np.ndarray] = []
    for box in boxes:
        cx1, cy1, cx2, cy2 = _expand_box(box, w, h, pad_frac=0.08)
        crops_np.append(rgb[cy1:cy2, cx1:cx2])

    batch = preprocess_rgb_batch(crops_np, model.img_size, dev)
    T_val = float(temp.get('T', 1.0))
    logits = forward_wastenet_fast(model, batch, dev)
    vn_probs = model.vn_probs(logits, T_val).numpy()
    fine_probs = model.fine_probs(logits, T_val)
    fine_np = None if fine_probs is None else fine_probs.numpy()

    items: List[Dict] = []
    for idx, (box, crop_arr, p) in enumerate(zip(boxes, crops_np, vn_probs), start=1):
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
        if return_crops:
            res['crop'] = Image.fromarray(crop_arr)
        res['style_key'] = 'UNCERTAIN' if res['status'] == 'UNCERTAIN' else res['vn_class']
        items.append(res)

    return items, source


@lru_cache(maxsize=16)
def _load_unicode_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load & cache a TTF font supporting Vietnamese accents (Windows / Linux)."""
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


@lru_cache(maxsize=256)
def _render_label_pill(label_txt: str, rgb: Tuple[int, int, int], font_size: int) -> np.ndarray:
    """Render & cache a crisp Vietnamese label pill bitmap (H, W, 3) RGB uint8."""
    font = _load_unicode_font(font_size)
    tmp = Image.new('RGB', (1, 1))
    tb = ImageDraw.Draw(tmp).textbbox((0, 0), label_txt, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    pad_x, pad_y = 6, 4
    pw, ph = tw + 2 * pad_x, th + 2 * pad_y
    pill = Image.new('RGB', (pw, ph), rgb)
    draw = ImageDraw.Draw(pill)
    draw.text((pad_x - tb[0], pad_y - tb[1]), label_txt, font=font, fill=(255, 255, 255))
    return np.asarray(pill)


def draw_annotated_rgb_fast(
    rgb: np.ndarray,
    items: List[Dict],
    max_side: int | None = 720,
) -> np.ndarray:
    """Ultra-fast OpenCV + cached Vietnamese pill annotator (~1-2 ms/frame vs ~25 ms PIL)."""
    h0, w0 = rgb.shape[:2]
    scale = min(1.0, float(max_side) / max(w0, h0)) if max_side else 1.0
    if scale < 1.0:
        canvas = cv2.resize(
            rgb,
            (max(1, int(w0 * scale)), max(1, int(h0 * scale))),
            interpolation=cv2.INTER_LINEAR,
        )
    else:
        canvas = rgb.copy()

    h, w = canvas.shape[:2]
    font_size = max(14, min(20, int(min(w, h) * 0.032)))
    thick = max(2, int(min(w, h) * 0.005))
    used_label_rects: List[Tuple[int, int, int, int]] = []

    for item in items:
        x1, y1, x2, y2 = [int(round(v * scale)) for v in item['box']]
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(x1 + 1, min(w, x2))
        y2 = max(y1 + 1, min(h, y2))

        style = BIN_STYLE[item['style_key']]
        col = style['rgb']
        label_txt = f" #{item['id']} {style['short']} {item['prob']:.0%} "

        # Subtle tinted fill inside box ROI (10% tint) + crisp border
        roi = canvas[y1:y2, x1:x2]
        if roi.size > 0:
            tint = np.empty_like(roi)
            tint[:] = col
            cv2.addWeighted(tint, 0.12, roi, 0.88, 0, dst=roi)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), col, thickness=thick, lineType=cv2.LINE_AA)

        pill = _render_label_pill(label_txt, col, font_size)
        ph, pw = pill.shape[:2]
        lx = max(4, min(w - pw - 4, x1))
        ly = y1 - ph - 6 if y1 - ph - 6 >= 4 else min(h - ph - 4, y1 + 6)
        for _ in range(6):
            overlap = any(
                not (lx + pw < rx1 or lx > rx2 or ly + ph < ry1 or ly > ry2)
                for rx1, ry1, rx2, ry2 in used_label_rects
            )
            if not overlap:
                break
            ly = min(h - ph - 4, ly + ph + 4)
        used_label_rects.append((lx, ly, lx + pw, ly + ph))

        # Pointer line when label sits above box
        anchor_x = max(x1 + 6, min(x2 - 6, lx + pw // 2))
        if ly + ph <= y1 and (y1 - (ly + ph)) > 2:
            cv2.line(canvas, (lx + pw // 2, ly + ph), (anchor_x, y1), col, max(2, thick - 1), cv2.LINE_AA)
            cv2.circle(canvas, (anchor_x, y1), 4, col, -1, cv2.LINE_AA)

        # Paste pill (clamped safely)
        px2 = min(w, lx + pw)
        py2 = min(h, ly + ph)
        if lx >= 0 and ly >= 0 and px2 > lx and py2 > ly:
            canvas[ly:py2, lx:px2] = pill[: py2 - ly, : px2 - lx]

    return canvas


def draw_annotated_image(img: Image.Image, items: List[Dict], max_side: int = 1100) -> Image.Image:
    """Draw clean color-coded boxes + pointer tags (#1 TÁI CHẾ 94%) for still photos."""
    rgb = _to_rgb_array(img)
    annotated_rgb = draw_annotated_rgb_fast(rgb, items, max_side=max_side)
    return Image.fromarray(annotated_rgb)
