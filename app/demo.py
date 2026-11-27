"""Demo Streamlit trực quan (Multi-Object + Isolated Realtime Tab + Visual-First UX).

  streamlit run app/demo.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import streamlit as st

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from app.multi_object import (  # noqa: E402
    BIN_STYLE,
    detect_and_classify_multi,
    draw_annotated_image,
    load_object_detector,
    warmup_gpu_models,
)
from app.realtime_tab import render_realtime_tab  # noqa: E402
from src.decision import DEFAULT_MIN_CONF, load_temperature  # noqa: E402
from src.explain.gradcam import GradCAM, overlay  # noqa: E402
from src.inference import classify  # noqa: E402
from src.labels import VN_NAMES  # noqa: E402
from src.preprocess import (  # noqa: E402
    assess_image_quality,
    denormalize,
    downscale_for_display,
    load_clean_rgb,
)

BACKUP_DIR = REPO / 'results_drive' / 'demo_samples'

st.set_page_config(page_title='Phân loại rác 3 nhóm', page_icon='♻️', layout='wide')


@st.cache_resource(show_spinner='Đang nạp mô hình phân loại...')
def load_model(ckpt: str, temp_path: str, device: str):
    from src.models.net import load_checkpoint
    model, ck = load_checkpoint(ckpt, map_location=device)
    model = model.to(device).eval()
    warmup_gpu_models(model, None, device)
    return model, load_temperature(temp_path), ck.get('epoch'), ck.get('val_bal_acc')


@st.cache_resource(show_spinner='Đang nạp bộ khoanh vùng nhiều vật thể...')
def load_detector(device: str):
    det = load_object_detector(device=device)
    if device == 'cuda':
        import torch
        with torch.no_grad():
            for h, w in ((384, 512), (288, 512)):  # 4:3 webcam + 16:9 video
                _ = det([torch.zeros((3, h, w), device=device, dtype=torch.float32)])
        torch.cuda.synchronize()
    return det


def resolve_runtime_device(pref: str = 'auto') -> tuple[str, str]:
    import torch

    if pref == 'auto':
        use_cuda = torch.cuda.is_available()
    else:
        use_cuda = pref == 'cuda' and torch.cuda.is_available()
    if use_cuda:
        return 'cuda', f'🚀 GPU: {torch.cuda.get_device_name(0)}'
    return 'cpu', '💻 CPU'


def render_legend_bar() -> None:
    """One-row visual color legend so users immediately understand the 3 bins without reading."""
    pills = ''.join(
        f"<span style='display:inline-flex;align-items:center;gap:6px;padding:6px 14px;"
        f"border-radius:999px;background:{BIN_STYLE[k]['bg_light']};border:1.5px solid {BIN_STYLE[k]['hex']};"
        f"color:{BIN_STYLE[k]['hex']};font-weight:700;font-size:14px;margin-right:8px;margin-bottom:6px'>"
        f"{BIN_STYLE[k]['icon']} {BIN_STYLE[k]['short']}</span>"
        for k in (0, 1, 2, 'UNCERTAIN')
    )
    st.markdown(f"<div style='margin-bottom:10px'>{pills}</div>", unsafe_allow_html=True)


def render_quality_badges(quality: dict) -> None:
    badges = []
    if quality['is_low_light']:
        badges.append('🌙 Thiếu sáng')
    if quality['is_overexposed']:
        badges.append('☀️ Lóa sáng (dễ nhầm xốp/nhựa)')
    if quality['is_blurry']:
        badges.append('💨 Ảnh hơi mờ')
    if badges:
        st.warning(' · '.join(badges))


def render_multi_result(
    model,
    detector,
    clean_img,
    temp: dict,
    tau_recycle: float,
    min_conf: float,
    score_thresh: float,
    show_cam: bool,
    device: str = 'cpu',
) -> None:
    t0 = time.perf_counter()
    items, source = detect_and_classify_multi(
        model=model,
        detector=detector,
        img=clean_img,
        temp=temp,
        tau_recycle=tau_recycle,
        min_conf=min_conf,
        score_thresh=score_thresh,
        max_objects=8,
        use_contour_fallback=True,
        device=device,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    annotated = draw_annotated_image(clean_img, items, max_side=1100)

    c_img, c_sum = st.columns([1.35, 1.0])
    with c_img:
        st.image(annotated, width='stretch')
        st.caption(f"Phát hiện **{len(items)} vật** · `{elapsed_ms:.0f} ms` · nguồn khung: `{source}`")

    with c_sum:
        # 1. Big at-a-glance bin summary cards (count + item IDs)
        grouped: Dict = {0: [], 1: [], 2: [], 'UNCERTAIN': []}
        for it in items:
            grouped[it['style_key']].append(it)

        for key in (0, 1, 2, 'UNCERTAIN'):
            grp = grouped[key]
            if not grp:
                continue
            s = BIN_STYLE[key]
            id_tags = ' '.join(
                f"<span style='background:{s['hex']};color:white;padding:2px 8px;border-radius:6px;"
                f"font-weight:700;font-size:13px;margin-right:4px'>#{it['id']} ({it['prob']:.0%})</span>"
                for it in grp
            )
            st.markdown(
                f"<div style='padding:12px 14px;margin-bottom:10px;border-radius:12px;"
                f"background:{s['bg_light']};border-left:6px solid {s['hex']}'>"
                f"<div style='display:flex;justify-content:space-between;align-items:center'>"
                f"<span style='font-size:20px;font-weight:800;color:{s['hex']}'>{s['icon']} {s['short']}</span>"
                f"<span style='font-size:14px;font-weight:700;color:{s['hex']}'>{len(grp)} món</span></div>"
                f"<div style='margin-top:6px'>{id_tags}</div>"
                f"<div style='margin-top:4px;font-size:12px;color:#475569'>{s['hint']}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        # 2. Compact visual crop gallery (each item thumbnail -> bin badge)
        st.markdown('**Chi tiết từng vật**')
        cols = st.columns(min(3, max(1, len(items))))
        for idx, it in enumerate(items):
            s = BIN_STYLE[it['style_key']]
            with cols[idx % len(cols)]:
                st.image(downscale_for_display(it['crop'], max_side=240), width='stretch')
                st.markdown(
                    f"<div style='text-align:center;padding:4px 6px;border-radius:8px;"
                    f"background:{s['hex']};color:white;font-weight:700;font-size:12px;margin-top:-10px;margin-bottom:10px'>"
                    f"#{it['id']} {s['icon']} {s['short']} · {it['prob']:.0%}</div>",
                    unsafe_allow_html=True,
                )

    if show_cam and items:
        with st.expander('🔥 Bản đồ nhiệt Grad-CAM (từng vật)', expanded=True):
            chosen_id = st.selectbox(
                'Chọn vật để xem Grad-CAM',
                [it['id'] for it in items],
                format_func=lambda i: f"Vật #{i} — {BIN_STYLE[items[i-1]['style_key']]['short']}",
            )
            target_it = items[chosen_id - 1]
            from src.preprocess import get_transforms
            x = get_transforms(False, model.img_size)(target_it['crop']).unsqueeze(0)
            cam = GradCAM(model)
            try:
                heat, _ = cam(x, target_class=target_it['vn_class'], T=float(temp['T']))
            finally:
                cam.remove()
            img01 = denormalize(x[0])
            g1, g2 = st.columns(2)
            g1.image(img01, caption=f'Vật #{chosen_id} (256×256)', width='stretch')
            g2.image(overlay(img01, heat), caption=f"Grad-CAM: {VN_NAMES[target_it['vn_class']]}", width='stretch')


def render_single_result(
    model,
    raw_bytes: bytes,
    clean_img,
    temp: dict,
    tau_recycle: float,
    min_conf: float,
    show_cam: bool,
    device: str = 'cpu',
) -> None:
    t0 = time.perf_counter()
    res = classify(model, raw_bytes, temp, device=device, tau_override=tau_recycle, min_conf=min_conf)
    infer_ms = (time.perf_counter() - t0) * 1000.0
    style_key = 'UNCERTAIN' if res['status'] == 'UNCERTAIN' else res['vn_class']
    s = BIN_STYLE[style_key]

    c1, c2 = st.columns([1.1, 1.0])
    with c1:
        st.image(downscale_for_display(clean_img, max_side=1000), width='stretch')
    with c2:
        st.markdown(
            f"<div style='padding:20px;border-radius:14px;background:{s['hex']};color:white;margin-bottom:14px'>"
            f"<div style='font-size:13px;opacity:0.9'>BỎ VÀO THÙNG</div>"
            f"<div style='font-size:34px;font-weight:800;margin:4px 0'>{s['icon']} {s['short']}</div>"
            f"<div style='font-size:16px'>Độ tin cậy: <b>{res['prob']:.0%}</b> · {s['hint']}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        for i, p in enumerate(res['probs']):
            st.progress(float(p), text=f"{BIN_STYLE[i]['icon']} {BIN_STYLE[i]['short']} · {p:.0%}")
        st.caption(f"⏱️ `{infer_ms:.0f} ms` · Lưu ý: Pin luôn thu gom riêng.")

    if show_cam:
        cam = GradCAM(model)
        try:
            heat, _ = cam(res['input_tensor'], target_class=res['vn_class'], T=float(temp['T']))
        finally:
            cam.remove()
        img01 = denormalize(res['input_tensor'][0])
        g1, g2 = st.columns(2)
        g1.image(img01, caption='Đầu vào 256×256', width='stretch')
        g2.image(overlay(img01, heat), caption=f"Grad-CAM: {VN_NAMES[res['vn_class']]}", width='stretch')


def main() -> None:
    st.title('♻️ Phân loại Rác 3 Nhóm')
    render_legend_bar()

    # Clean, minimal sidebar
    default_ckpt = os.environ.get('WASTE_CKPT', str(REPO / 'checkpoints' / 'best_model.pth'))
    default_dev_pref = os.environ.get('WASTE_DEVICE', 'auto').lower()
    if default_dev_pref not in ('auto', 'cuda', 'cpu'):
        default_dev_pref = 'auto'
    with st.sidebar:
        st.header('⚙️ Điều chỉnh nhanh')
        tau_recycle = st.slider('Ngưỡng Tái chế (τ)', 0.45, 0.95, 0.75, 0.01)
        min_conf = st.slider('Chặn vật lạ / mơ hồ', 0.00, 0.85, DEFAULT_MIN_CONF, 0.05)
        det_thresh = st.slider('Độ nhạy khoanh nhiều rác', 0.15, 0.70, 0.30, 0.05)
        show_cam = st.toggle('🔥 Bật Grad-CAM', value=False)
        with st.expander('🔧 Nâng cao (Checkpoint & Thiết bị)', expanded=False):
            ckpt = st.text_input('Checkpoint (.pth)', default_ckpt)
            temp_path = st.text_input('temperature.json', str(Path(ckpt).with_name('temperature.json')))
            dev_pref = st.selectbox(
                'Thiết bị suy luận',
                ['auto', 'cuda', 'cpu'],
                index=['auto', 'cuda', 'cpu'].index(default_dev_pref),
                format_func=lambda d: {'auto': 'Tự động (ưu tiên GPU)', 'cuda': 'GPU (CUDA)', 'cpu': 'CPU'}[d],
            )

    if not Path(ckpt).exists():
        st.error(f'Không thấy checkpoint: `{ckpt}`')
        st.stop()

    device, device_label = resolve_runtime_device(dev_pref)
    model, temp, ep, vba = load_model(ckpt, temp_path, device)
    detector = load_detector(device)
    st.sidebar.caption(
        f"{device_label} · `{model.backbone_name}` ({model.mode}) · Val `{vba:.1%}` · `T={float(temp['T']):.2f}`"
        if vba is not None else f"{device_label} · `{model.backbone_name}`"
    )

    tab_img, tab_cam, tab_rt, tab_guide = st.tabs([
        '🖼️ Nhận diện ảnh',
        '📷 Chụp ảnh',
        '🎥 Realtime (Thử nghiệm)',
        '📘 Tra nhanh 3 thùng',
    ])

    with tab_img:
        r1, r2 = st.columns([2.2, 1.2])
        with r1:
            up = st.file_uploader(
                'Thả ảnh rác vào đây (JPG, PNG, WEBP, HEIC)',
                type=['jpg', 'jpeg', 'png', 'webp', 'bmp', 'heic', 'heif'],
                label_visibility='collapsed',
            )
        with r2:
            detect_mode = st.radio(
                'Chế độ',
                ['📦 Nhiều rác', '🎯 1 vật'],
                horizontal=True,
                label_visibility='collapsed',
            )

        # Optional quick backup sample selector if demo_samples/ has files
        raw_bytes = up.getvalue() if up is not None else None
        if raw_bytes is None and BACKUP_DIR.exists():
            samples = sorted(
                p for p in BACKUP_DIR.glob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
            )
            if samples:
                c_s1, c_s2 = st.columns([2.2, 1.0])
                chosen = c_s1.selectbox('Hoặc chọn ảnh mẫu dự phòng:', samples, format_func=lambda p: p.name)
                if c_s2.button('▶️ Chạy ảnh mẫu', use_container_width=True):
                    raw_bytes = chosen.read_bytes()

        if raw_bytes is not None:
            clean_img = load_clean_rgb(raw_bytes)
            render_quality_badges(assess_image_quality(clean_img))
            if detect_mode.startswith('📦'):
                render_multi_result(model, detector, clean_img, temp, tau_recycle, min_conf, det_thresh, show_cam, device=device)
            else:
                render_single_result(model, raw_bytes, clean_img, temp, tau_recycle, min_conf, show_cam, device=device)

    with tab_cam:
        c_cam1, c_cam2 = st.columns([1.2, 1.5])
        with c_cam1:
            cam_on = st.toggle('📷 Bật camera chụp ảnh', value=False, key='snap_cam_on')
        with c_cam2:
            cam_mode = st.radio('Chế độ chụp', ['📦 Nhiều rác', '🎯 1 vật'], horizontal=True, label_visibility='collapsed')
        if cam_on:
            # Đảm bảo dừng Tab Realtime nếu đang bật để tránh tranh chấp webcam
            st.session_state['rt_running'] = False
            shot = st.camera_input('Chụp ảnh rác (yêu cầu localhost hoặc HTTPS)')
            if shot is not None:
                b = shot.getvalue()
                clean_img = load_clean_rgb(b)
                render_quality_badges(assess_image_quality(clean_img))
                if cam_mode.startswith('📦'):
                    render_multi_result(model, detector, clean_img, temp, tau_recycle, min_conf, det_thresh, show_cam, device=device)
                else:
                    render_single_result(model, b, clean_img, temp, tau_recycle, min_conf, show_cam, device=device)
        else:
            st.info('Gạt **📷 Bật camera chụp ảnh** khi cần dùng (tắt mặc định để không tự mở cam và tránh đụng Tab Realtime).')

    with tab_rt:
        render_realtime_tab(model, detector, temp, tau_recycle, min_conf, device=device)

    with tab_guide:
        g1, g2, g3 = st.columns(3)
        cards = [
            (0, ['🧴 Chai / hộp nhựa sạch', '🥫 Lon nhôm / kim loại', '📦 Giấy / bìa carton khô', '🍾 Chai lọ thủy tinh', '👕 Quần áo / giày cũ']),
            (1, ['🍚 Thức ăn thừa', '🍌 Vỏ rau củ quả', '🥚 Vỏ trứng', '☕ Bã cà phê / túi trà']),
            (2, ['🥡 Hộp xốp / cốc xốp', '🥤 Ống hút, dao nĩa 1 lần', '☕ Cốc giấy tráng nhựa', '🛍️ Túi rác / đồ dính bẩn', '🔋 Pin, bình xịt (gom riêng)']),
        ]
        for col, (k, bullets) in zip((g1, g2, g3), cards):
            s = BIN_STYLE[k]
            li = ''.join(f"<div style='padding:5px 0;font-size:15px'>{b}</div>" for b in bullets)
            col.markdown(
                f"<div style='padding:18px;border-radius:14px;background:{s['bg_light']};"
                f"border-top:6px solid {s['hex']};min-height:240px'>"
                f"<div style='font-size:22px;font-weight:800;color:{s['hex']};margin-bottom:10px'>"
                f"{s['icon']} {s['short']}</div>{li}</div>",
                unsafe_allow_html=True,
            )


main()
