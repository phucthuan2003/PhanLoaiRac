"""Isolated Realtime Tab for Streamlit Demo (Thread-safe CameraWorker + @st.fragment).

Why a dedicated CameraWorker thread?
On Windows, OpenCV DirectShow/MSMF uses COM and crashes Python silently (exit code 1)
if Streamlit interrupts a blocking ``while True: cap.read()`` loop on rerun/Stop.
Keeping open/read/release strictly on one background thread + pulling latest frames
via ``@st.fragment(run_every=...)`` eliminates the crash and keeps UI buttons instant.
"""
from __future__ import annotations

import os
import tempfile
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from app.multi_object import BIN_STYLE, detect_and_classify_multi, draw_annotated_image
from src.decision import decide
from src.preprocess import get_transforms

os.environ.setdefault('OPENCV_VIDEOIO_PRIORITY_OBSENSOR', '0')
os.environ.setdefault('OPENCV_LOG_LEVEL', 'SILENT')


class CameraWorker:
    """Owns cv2.VideoCapture strictly on a single background thread (Windows COM-safe)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest_rgb: np.ndarray | None = None
        self._error: str | None = None
        self._ended: bool = False
        self._source_desc: str = ''

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop_evt.is_set()

    def start(self, source_type: str, cam_idx: int = 0, ip_url: str = '', video_path: str | None = None) -> None:
        self.stop()
        self._stop_evt.clear()
        with self._lock:
            self._latest_rgb = None
            self._error = None
            self._ended = False
            self._source_desc = {'video': 'Video', 'ip_cam': 'Phone Cam', 'webcam': 'Webcam'}.get(source_type, 'Live')
        self._thread = threading.Thread(
            target=self._loop,
            args=(source_type, int(cam_idx), (ip_url or '').strip(), video_path),
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.5)
        self._thread = None

    def get_state(self):
        with self._lock:
            frame = None if self._latest_rgb is None else self._latest_rgb.copy()
            return frame, self._error, self._ended, self._source_desc

    def _open_cap(self, source_type: str, cam_idx: int, ip_url: str, video_path: str | None):
        if source_type == 'video' and video_path:
            cap = cv2.VideoCapture(video_path)
            return (cap, None) if cap.isOpened() else (None, 'Không đọc được file video này.')

        if source_type == 'ip_cam':
            if not ip_url:
                return None, 'Vui lòng nhập URL luồng video (VD: http://192.168.1.5:8080/video).'
            cap = cv2.VideoCapture(ip_url, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                cap = cv2.VideoCapture(ip_url)
            return (cap, None) if cap.isOpened() else (None, f'Không kết nối được tới `{ip_url}`.')

        for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF):
            try:
                cap = cv2.VideoCapture(cam_idx, backend)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    ok, frame = False, None
                    for _ in range(4):
                        ok, frame = cap.read()
                        time.sleep(0.03)
                    if ok and frame is not None and float(frame.std()) > 1.0:
                        return cap, None
                cap.release()
            except Exception:
                continue

        return None, (
            'Không mở được webcam máy tính (kiểm tra đã tắt cam ở tab khác hoặc ứng dụng khác chưa). '
            'Bạn có thể dùng **🎞️ Video quay sẵn (MP4)** hoặc **📱 Camera ĐT (IP URL)**.'
        )

    def _loop(self, source_type: str, cam_idx: int, ip_url: str, video_path: str | None) -> None:
        cap, err = self._open_cap(source_type, cam_idx, ip_url, video_path)
        if cap is None:
            with self._lock:
                self._error = err
                self._ended = True
            return

        # Delay nhỏ khi phát video file để giống tốc độ thật
        vid_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0) if source_type == 'video' else 0.0
        vid_sleep = 1.0 / max(min(vid_fps, 30.0), 10.0) if source_type == 'video' else 0.015

        try:
            while not self._stop_evt.is_set():
                ok, bgr = cap.read()
                if not ok or bgr is None:
                    with self._lock:
                        self._ended = True
                        if source_type != 'video':
                            self._error = 'Mất tín hiệu camera, đã tự dừng.'
                    break
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                with self._lock:
                    self._latest_rgb = rgb
                time.sleep(vid_sleep)
        finally:
            try:
                cap.release()
            except Exception:
                pass


@st.cache_resource
def get_camera_worker() -> CameraWorker:
    return CameraWorker()


def render_realtime_tab(model, detector, temp: dict, tau_recycle: float, min_conf: float) -> None:
    """Render an isolated, crash-safe realtime recognition tab."""
    worker = get_camera_worker()
    if 'rt_running' not in st.session_state:
        st.session_state['rt_running'] = False
    if 'rt_prob_hist' not in st.session_state:
        st.session_state['rt_prob_hist'] = deque(maxlen=3)

    c_src, c_mode, c_btn = st.columns([1.8, 1.3, 1.0])
    with c_src:
        src_choice = st.radio(
            'Nguồn Realtime',
            ['💻 Webcam máy tính', '🎞️ Video quay sẵn (MP4)', '📱 Camera ĐT (IP URL)'],
            horizontal=True,
            label_visibility='collapsed',
        )
    with c_mode:
        rt_mode = st.radio(
            'Chế độ',
            ['🎯 1 vật (Nhanh)', '📦 Nhiều vật'],
            horizontal=True,
            label_visibility='collapsed',
        )

    source_type = 'webcam'
    cam_idx = 0
    ip_url = ''
    video_path = None

    c_opt1, c_opt2 = st.columns([2.2, 1.0])
    with c_opt1:
        if src_choice.startswith('🎞️'):
            source_type = 'video'
            up_vid = st.file_uploader(
                'Thả video quay rác vào đây (MP4, MOV, AVI)',
                type=['mp4', 'mov', 'avi', 'mkv'],
                label_visibility='collapsed',
            )
            if up_vid is not None:
                tmp_path = Path(tempfile.gettempdir()) / f'waste_rt_{up_vid.name}'
                if not tmp_path.exists() or tmp_path.stat().st_size != len(up_vid.getvalue()):
                    tmp_path.write_bytes(up_vid.getvalue())
                video_path = str(tmp_path)
        elif src_choice.startswith('📱'):
            source_type = 'ip_cam'
            ip_url = st.text_input(
                'URL Camera điện thoại',
                placeholder='http://192.168.1.10:8080/video',
                label_visibility='collapsed',
            )
        else:
            source_type = 'webcam'
            cam_idx = int(st.number_input('Webcam ID (thường là 0)', min_value=0, max_value=3, value=0, step=1))

    with c_opt2:
        score_th = st.slider('Độ nhạy khung', 0.20, 0.70, 0.35, 0.05)

    with c_btn:
        b1, b2 = st.columns(2)
        if b1.button('▶️ Chạy', use_container_width=True, type='primary'):
            if source_type == 'video' and not video_path:
                st.warning('Hãy tải lên 1 file video `.mp4`/`.mov` trước.')
            else:
                st.session_state['rt_prob_hist'].clear()
                st.session_state['rt_running'] = True
                worker.start(source_type, cam_idx=cam_idx, ip_url=ip_url, video_path=video_path)
        if b2.button('⏹️ Dừng', use_container_width=True):
            st.session_state['rt_running'] = False
            worker.stop()

    if not st.session_state['rt_running']:
        if worker.is_running():
            worker.stop()
        col_v, col_c = st.columns([1.6, 1.0])
        col_v.info('Chọn nguồn rồi bấm **▶️ Chạy** để bắt đầu nhận diện trực tiếp.')
        col_c.markdown(
            "<div style='padding:16px;border-radius:12px;background:#f8fafc;border:1px solid #e2e8f0'>"
            "<div style='font-weight:700;font-size:15px;margin-bottom:6px'>💡 3 nguồn Realtime</div>"
            "<div style='font-size:13px;color:#334155;line-height:1.6'>"
            "1. <b>💻 Webcam:</b> Mở trực tiếp camera laptop (chạy luồng riêng chống crash).<br>"
            "2. <b>🎞️ Video MP4:</b> Thả clip quay sẵn từ điện thoại.<br>"
            "3. <b>📱 Camera ĐT:</b> Dán URL từ app <i>IP Webcam</i> / <i>DroidCam</i>."
            "</div></div>",
            unsafe_allow_html=True,
        )
        return

    is_multi = rt_mode.startswith('📦')
    tf = get_transforms(False, model.img_size)
    T_val = float(temp.get('T', 1.0))

    @st.fragment(run_every=0.35 if is_multi else 0.20)
    def _live_view():
        if not st.session_state.get('rt_running', False):
            return
        rgb, err, ended, tag = worker.get_state()
        col_v, col_c = st.columns([1.6, 1.0])

        if err:
            st.session_state['rt_running'] = False
            worker.stop()
            col_v.warning(err)
            return

        if rgb is None:
            if ended:
                st.session_state['rt_running'] = False
                worker.stop()
                col_v.success('✅ Đã phát xong video Realtime. Bấm **▶️ Chạy** để xem lại.')
            else:
                col_v.info('Đang khởi động luồng camera...')
            return

        t0 = time.perf_counter()
        pil_img = Image.fromarray(rgb)

        if is_multi:
            items, _ = detect_and_classify_multi(
                model=model,
                detector=detector,
                img=pil_img,
                temp=temp,
                tau_recycle=tau_recycle,
                min_conf=min_conf,
                score_thresh=score_th,
                max_objects=6,
                use_contour_fallback=False,
                device='cpu',
            )
            annotated = draw_annotated_image(pil_img, items, max_side=720)
            ms = (time.perf_counter() - t0) * 1000.0
            col_v.image(annotated, caption=f'{tag} · Nhiều vật · {ms:.0f} ms/khung', width='stretch')

            counts = {0: [], 1: [], 2: [], 'UNCERTAIN': []}
            for it in items:
                counts[it['style_key']].append(f"#{it['id']}")
            blocks = []
            for key in (0, 1, 2, 'UNCERTAIN'):
                ids = counts[key]
                if not ids:
                    continue
                s = BIN_STYLE[key]
                blocks.append(
                    f"<div style='padding:12px 14px;margin-bottom:10px;border-radius:10px;"
                    f"background:{s['bg_light']};border-left:6px solid {s['hex']}'>"
                    f"<div style='font-size:18px;font-weight:700;color:{s['hex']}'>"
                    f"{s['icon']} {s['short']} · {len(ids)} món</div>"
                    f"<div style='font-size:13px;color:#334155'>Vật: {', '.join(ids)} — {s['hint']}</div>"
                    f"</div>"
                )
            col_c.markdown(
                ''.join(blocks) if blocks else '<p>Đưa vật thể rác vào khung hình...</p>',
                unsafe_allow_html=True,
            )
        else:
            import torch

            with torch.no_grad():
                x = tf(pil_img).unsqueeze(0)
                logits = model.eval()(x).float()
                p_now = model.vn_probs(logits, T_val)[0].numpy()
            hist = st.session_state['rt_prob_hist']
            hist.append(p_now)
            p_smooth = np.mean(hist, axis=0)
            res = decide(p_smooth, tau_recycle, min_conf=min_conf)
            style_key = 'UNCERTAIN' if res['status'] == 'UNCERTAIN' else res['vn_class']
            s = BIN_STYLE[style_key]

            w, h = pil_img.size
            vis = draw_annotated_image(
                pil_img,
                [{**res, 'id': 1, 'box': (int(w * 0.12), int(h * 0.10), int(w * 0.88), int(h * 0.90)), 'style_key': style_key}],
                max_side=720,
            )
            ms = (time.perf_counter() - t0) * 1000.0
            col_v.image(vis, caption=f'{tag} · 1 vật · {ms:.0f} ms/khung', width='stretch')

            bars = ''.join(
                f"<div style='margin-top:8px'>"
                f"<div style='display:flex;justify-content:space-between;font-size:13px;font-weight:600'>"
                f"<span>{BIN_STYLE[i]['icon']} {BIN_STYLE[i]['short']}</span><span>{p_smooth[i]:.0%}</span></div>"
                f"<div style='background:#e2e8f0;border-radius:6px;height:10px;overflow:hidden'>"
                f"<div style='width:{p_smooth[i]*100:.1f}%;background:{BIN_STYLE[i]['hex']};height:100%'></div>"
                f"</div></div>"
                for i in (0, 1, 2)
            )
            col_c.markdown(
                f"<div style='padding:18px;border-radius:14px;background:{s['hex']};color:white;margin-bottom:12px'>"
                f"<div style='font-size:13px;opacity:0.9'>PHÂN LOẠI TRỰC TIẾP</div>"
                f"<div style='font-size:30px;font-weight:800;margin:4px 0'>{s['icon']} {s['short']}</div>"
                f"<div style='font-size:15px'>Độ tin cậy: <b>{res['prob']:.0%}</b> · {s['hint']}</div>"
                f"</div>{bars}",
                unsafe_allow_html=True,
            )

        if ended:
            st.session_state['rt_running'] = False
            worker.stop()
            st.success('✅ Đã phát xong video Realtime.')

    _live_view()
