"""Demo Streamlit chạy LOCAL (SPEC v2 mục 5.3).

  streamlit run app/demo.py
  (hoặc đặt biến môi trường WASTE_CKPT=checkpoints/best_model.pth trước khi chạy)

Cần: best_model.pth và temperature.json (cùng thư mục) lấy từ Drive sau khi train + calibrate.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import streamlit as st

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.decision import MSG_LIMIT, load_temperature  # noqa: E402
from src.explain.gradcam import GradCAM, overlay  # noqa: E402
from src.inference import classify  # noqa: E402
from src.labels import VN_NAMES  # noqa: E402
from src.preprocess import denormalize, load_clean_rgb  # noqa: E402

COLORS = {0: '#1b7f3b', 1: '#a05a00', 2: '#5a5a5a'}
BINS = {0: 'Thùng TÁI CHẾ', 1: 'Thùng THỰC PHẨM', 2: 'Thùng RÁC SINH HOẠT KHÁC'}

st.set_page_config(page_title='Phân loại rác 3 nhóm', page_icon='♻️', layout='wide')


@st.cache_resource(show_spinner='Đang nạp mô hình...')
def load_model(ckpt: str, temp_path: str):
    from src.models.net import load_checkpoint
    model, ck = load_checkpoint(ckpt, map_location='cpu')
    return model, load_temperature(temp_path), ck.get('epoch'), ck.get('val_bal_acc')


def sidebar():
    st.sidebar.header('Cấu hình')
    default = os.environ.get('WASTE_CKPT', str(REPO / 'checkpoints' / 'best_model.pth'))
    ckpt = st.sidebar.text_input('Checkpoint (.pth)', default)
    temp = st.sidebar.text_input('temperature.json', str(Path(ckpt).with_name('temperature.json')))
    show_cam = st.sidebar.toggle('Hiện bản đồ nhiệt Grad-CAM', value=True)
    return ckpt, temp, show_cam


def main():
    st.title('♻️ Phân loại rác sinh hoạt theo 3 nhóm')
    st.caption('Theo Điều 75 Luật BVMT 2020 và hướng dẫn kỹ thuật kèm Công văn 9368/BTNMT-KSONMT. '
               'Mỗi ảnh nên chỉ có MỘT vật.')
    ckpt, temp_path, show_cam = sidebar()
    if not Path(ckpt).exists():
        st.error(f'Không thấy checkpoint: {ckpt}. Tải best_model.pth + temperature.json từ Drive về thư mục checkpoints/.')
        st.stop()
    model, temp, ep, vba = load_model(ckpt, temp_path)
    st.sidebar.markdown(f'**Model:** `{model.backbone_name}` / `{model.mode}`  \n'
                        f'epoch {ep}, val bal_acc {vba:.3f}  \nT = {float(temp["T"]):.3f}, '
                        f'τ = {float(temp["tau_recycle"]):.2f}' if vba is not None else '')
    if not Path(temp_path).exists():
        st.sidebar.warning('Chưa có temperature.json → dùng T = 1, τ = 0.75 (chưa hiệu chuẩn).')

    tab_up, tab_cam = st.tabs(['📁 Tải ảnh lên', '📷 Chụp bằng webcam'])
    with tab_up:
        up = st.file_uploader('Chọn ảnh', type=['jpg', 'jpeg', 'png', 'webp', 'bmp'])
    with tab_cam:
        shot = st.camera_input('Chụp ảnh vật cần phân loại')
    f = shot or up
    if f is None:
        st.info('Tải ảnh lên hoặc chụp ảnh để bắt đầu.')
        st.stop()

    data = f.getvalue()
    res = classify(model, data, temp)
    k = res['vn_class']
    c1, c2 = st.columns([1, 1])
    with c1:
        st.image(load_clean_rgb(data), caption='Ảnh gốc (đã xoay theo EXIF)', width='stretch')
    with c2:
        if res['hazard']:
            st.error('⚠️ ' + res['messages'][0])
        if res['status'] == 'UNCERTAIN':
            st.warning(f"**KHÔNG CHẮC CHẮN** (dự đoán nghiêng về {VN_NAMES[k]}, {res['prob']:.0%})")
        else:
            st.markdown(f"<div style='padding:14px;border-radius:10px;background:{COLORS[k]};color:white'>"
                        f"<div style='font-size:14px'>Bỏ vào</div><div style='font-size:28px;font-weight:700'>"
                        f"{BINS[k]}</div><div>Độ tin cậy {res['prob']:.0%}</div></div>", unsafe_allow_html=True)
        for m in res['messages']:
            if not (res['hazard'] and m == res['messages'][0]):
                st.info(m)
        st.subheader('Xác suất 3 nhóm')
        for i, p in enumerate(res['probs']):
            st.progress(float(p), text=f'{VN_NAMES[i]}: {p:.1%}')
        if res['top_fine']:
            st.subheader('Top-3 loại vật (nhãn con)')
            for t in res['top_fine']:
                st.write(f"- {t['name']} (`{t['label']}`): {t['prob']:.1%}")
        st.caption(MSG_LIMIT)

    if show_cam:
        st.subheader('Grad-CAM: vùng ảnh mô hình dựa vào')
        cam = GradCAM(model)
        try:
            heat, _ = cam(res['input_tensor'], target_class=k, T=float(temp['T']))
        finally:
            cam.remove()
        img = denormalize(res['input_tensor'][0])
        g1, g2 = st.columns(2)
        g1.image(img, caption=f'Đầu vào mô hình ({model.img_size}×{model.img_size}, letterbox)', width='stretch')
        g2.image(overlay(img, heat), caption=f'Grad-CAM cho lớp "{VN_NAMES[k]}"', width='stretch')


main()
