"""Isolated Realtime Tab for Streamlit Demo (Synced GPU st.empty() Stream + Full MP4 Export).

Why this architecture?
  1. PyTorch/cuDNN on Windows initializes per-thread CUDA/cuDNN state (~1.5s cold penalty
     per new thread) AND a zero-sleep PyTorch background thread starves Streamlit's Tornado
     WebSocket loop of the Python GIL (causing frames to jump 16 -> 97 -> EOF).
  2. Running GPU inference directly on Streamlit's already-warmed ScriptRunner thread via
     st.empty() placeholders + yielding 15 ms to Tornado after each frame guarantees:
       - Instant startup (<0.2s, zero background-thread CUDA re-init)
       - Every rendered frame actually reaches the browser smoothly (~15-20 FPS live!)
       - Zero MediaFileManager 404s (inline data:image/jpeg;base64)
       - Clean Windows webcam safety: ONLY lightweight cv2.VideoCapture for Webcam/IP-Cam
         lives on a tiny capture-only thread (COM-safe on Windows), never touching PyTorch.
  3. Full MP4 Export Mode processes every frame of an uploaded MP4 at full GPU speed (~25 FPS),
     encodes a browser-playable H.264 MP4 (Windows MSMF avc1), and plays it natively at 30 FPS.
"""
from __future__ import annotations

import base64
import os
import tempfile
import threading
import time
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st
import torch

from app.multi_object import (
    BIN_STYLE,
    detect_and_classify_multi,
    draw_annotated_rgb_fast,
    forward_wastenet_fast,
    preprocess_rgb_batch,
    resolve_device,
)
from src.decision import decide

os.environ.setdefault('OPENCV_VIDEOIO_PRIORITY_OBSENSOR', '0')
os.environ.setdefault('OPENCV_LOG_LEVEL', 'SILENT')


def _resize_max_side(bgr: np.ndarray, max_side: int) -> np.ndarray:
    h, w = bgr.shape[:2]
    if max(h, w) <= max_side:
        return bgr
    scale = float(max_side) / max(h, w)
    return cv2.resize(
        bgr,
        (max(2, int(w * scale) & ~1), max(2, int(h * scale) & ~1)),
        interpolation=cv2.INTER_LINEAR,
    )


def _encode_jpeg_data_uri(rgb: np.ndarray, quality: int = 78) -> str:
    """Encode RGB array directly to inline data:image/jpeg;base64 URI (bypasses MediaFileManager)."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode('.jpg', bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return ''
    b64 = base64.b64encode(buf.tobytes()).decode('ascii')
    return f'data:image/jpeg;base64,{b64}'


def _open_h264_writer(out_path: str, fps: float, size: Tuple[int, int]) -> Tuple[Optional[cv2.VideoWriter], str]:
    """Open a browser-playable H.264 VideoWriter (Windows MSMF avc1 first, then fallbacks)."""
    w, h = size[0] & ~1, size[1] & ~1
    attempts = []
    if os.name == 'nt':
        attempts.append((cv2.CAP_MSMF, 'avc1', 'MSMF/H.264'))
    attempts.extend([
        (cv2.CAP_ANY, 'avc1', 'H.264'),
        (cv2.CAP_ANY, 'mp4v', 'MPEG-4'),
    ])
    for api, fourcc, label in attempts:
        try:
            writer = cv2.VideoWriter(out_path, api, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
            if writer.isOpened():
                return writer, label
            writer.release()
        except Exception:
            continue
    return None, ''


@torch.no_grad()
def _infer_single_rgb(
    model,
    rgb: np.ndarray,
    T_val: float,
    dev: torch.device,
) -> np.ndarray:
    batch = preprocess_rgb_batch([rgb], model.img_size, dev)
    logits = forward_wastenet_fast(model, batch, dev)
    return model.vn_probs(logits, T_val)[0].numpy()


class LiveCameraCapture:
    """Lightweight capture-only thread for Webcam/IP-Cam (Windows DirectShow/MSMF COM-safe).

    Does ZERO PyTorch work so it never competes for GPU/cuDNN or starves the GIL.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest_bgr: np.ndarray | None = None
        self._seq: int = 0
        self._error: str | None = None
        self._ended: bool = False

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop_evt.is_set()

    def start(self, source_type: str, cam_idx: int = 0, ip_url: str = '') -> None:
        self.stop()
        self._stop_evt.clear()
        with self._lock:
            self._latest_bgr = None
            self._seq = 0
            self._error = None
            self._ended = False
        self._thread = threading.Thread(
            target=self._loop,
            args=(source_type, int(cam_idx), (ip_url or '').strip()),
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.5)
        self._thread = None

    def get_latest(self):
        with self._lock:
            frame = None if self._latest_bgr is None else self._latest_bgr.copy()
            return frame, self._seq, self._error, self._ended

    def _open_cap(self, source_type: str, cam_idx: int, ip_url: str):
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
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
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

    def _loop(self, source_type: str, cam_idx: int, ip_url: str) -> None:
        cap, err = self._open_cap(source_type, cam_idx, ip_url)
        if cap is None:
            with self._lock:
                self._error = err
                self._ended = True
            return
        try:
            while not self._stop_evt.is_set():
                ok, bgr = cap.read()
                if not ok or bgr is None:
                    with self._lock:
                        self._ended = True
                        self._error = 'Mất tín hiệu camera, đã tự dừng.'
                    break
                bgr = _resize_max_side(bgr, max_side=720)
                with self._lock:
                    self._latest_bgr = bgr
                    self._seq += 1
                time.sleep(0.02)
        finally:
            try:
                cap.release()
            except Exception:
                pass


@st.cache_resource
def get_live_camera_capture() -> LiveCameraCapture:
    return LiveCameraCapture()


def _render_frame_html(data_uri: str, caption: str) -> str:
    return (
        f"<img src='{data_uri}' alt='Realtime frame' "
        f"style='width:100%;max-height:72vh;object-fit:contain;border-radius:10px;"
        f"background:#0f172a;display:block' />"
        f"<div style='font-size:12px;color:#64748b;margin-top:6px;text-align:center'>{caption}</div>"
    )


def _render_multi_cards_html(counts: Dict) -> str:
    blocks = []
    for key in (0, 1, 2, 'UNCERTAIN'):
        ids = counts.get(key, [])
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
    return ''.join(blocks) if blocks else '<p style="color:#64748b">Đưa vật thể rác vào khung hình...</p>'


def _render_single_card_html(style_key, prob: float, probs: List[float]) -> str:
    s = BIN_STYLE[style_key]
    bars = ''.join(
        f"<div style='margin-top:8px'>"
        f"<div style='display:flex;justify-content:space-between;font-size:13px;font-weight:600'>"
        f"<span>{BIN_STYLE[i]['icon']} {BIN_STYLE[i]['short']}</span><span>{probs[i]:.0%}</span></div>"
        f"<div style='background:#e2e8f0;border-radius:6px;height:10px;overflow:hidden'>"
        f"<div style='width:{probs[i]*100:.1f}%;background:{BIN_STYLE[i]['hex']};height:100%'></div>"
        f"</div></div>"
        for i in (0, 1, 2)
    )
    return (
        f"<div style='padding:18px;border-radius:14px;background:{s['hex']};color:white;margin-bottom:12px'>"
        f"<div style='font-size:13px;opacity:0.9'>PHÂN LOẠI TRỰC TIẾP</div>"
        f"<div style='font-size:30px;font-weight:800;margin:4px 0'>{s['icon']} {s['short']}</div>"
        f"<div style='font-size:15px'>Độ tin cậy: <b>{prob:.0%}</b> · {s['hint']}</div>"
        f"</div>{bars}"
    )


def _process_one_frame(
    bgr: np.ndarray,
    model,
    detector,
    temp: dict,
    dev: torch.device,
    is_multi: bool,
    tau_recycle: float,
    min_conf: float,
    score_th: float,
    prob_hist: deque,
    cached_items: Optional[List[Dict]] = None,
    run_detect: bool = True,
    det_max_side: int = 512,
) -> Tuple[str, str, Optional[List[Dict]]]:
    """Run inference on 1 BGR frame on the already-warmed GPU thread and return (data_uri, side_html, items)."""
    bgr_s = _resize_max_side(bgr, max_side=720)
    rgb = cv2.cvtColor(bgr_s, cv2.COLOR_BGR2RGB)

    if is_multi:
        if run_detect or cached_items is None:
            items, _ = detect_and_classify_multi(
                model=model,
                detector=detector,
                img=rgb,
                temp=temp,
                tau_recycle=tau_recycle,
                min_conf=min_conf,
                score_thresh=score_th,
                max_objects=6,
                use_contour_fallback=False,
                device=dev,
                return_crops=False,
                det_max_side=det_max_side,
            )
        else:
            items = cached_items
        vis_rgb = draw_annotated_rgb_fast(rgb, items, max_side=None)
        counts = {0: [], 1: [], 2: [], 'UNCERTAIN': []}
        for it in items:
            counts[it['style_key']].append(f"#{it['id']}")
        return _encode_jpeg_data_uri(vis_rgb, quality=78), _render_multi_cards_html(counts), items

    T_val = float(temp.get('T', 1.0))
    p_now = _infer_single_rgb(model, rgb, T_val, dev)
    prob_hist.append(p_now)
    p_smooth = np.mean(prob_hist, axis=0)
    res = decide(p_smooth, tau_recycle, min_conf=min_conf)
    style_key = 'UNCERTAIN' if res['status'] == 'UNCERTAIN' else res['vn_class']
    h, w = rgb.shape[:2]
    vis_rgb = draw_annotated_rgb_fast(
        rgb,
        [{**res, 'id': 1, 'box': (int(w * 0.12), int(h * 0.10), int(w * 0.88), int(h * 0.90)), 'style_key': style_key}],
        max_side=None,
    )
    return (
        _encode_jpeg_data_uri(vis_rgb, quality=78),
        _render_single_card_html(style_key, float(res['prob']), p_smooth.tolist()),
        None,
    )


def _stream_video_live_synced(
    video_path: str,
    model,
    detector,
    temp: dict,
    dev: torch.device,
    is_multi: bool,
    tau_recycle: float,
    min_conf: float,
    score_th: float,
    detect_stride: int,
    frame_slot,
    info_slot,
) -> None:
    """Stream MP4 live on the warmed GPU thread via st.empty() — zero GIL starvation, every frame shown."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        frame_slot.error('Không đọc được file video này.')
        st.session_state['rt_running'] = False
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    vid_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    vid_fps = max(10.0, min(vid_fps, 60.0))
    mode_txt = 'Nhiều vật' if is_multi else '1 vật'
    prob_hist: deque = deque(maxlen=3)
    cached_items: Optional[List[Dict]] = None
    fps_ema = 0.0
    frame_idx = 0
    step = 0
    last_tick = time.perf_counter()

    try:
        while st.session_state.get('rt_running', False):
            t0 = time.perf_counter()
            ok, bgr = cap.read()
            if not ok or bgr is None:
                st.session_state['rt_running'] = False
                info_slot.success(f'✅ Đã phát xong {frame_idx} khung hình. Bấm **▶️ Xem trực tiếp** để xem lại.')
                break
            frame_idx += 1

            data_uri, side_html, cached_items = _process_one_frame(
                bgr=bgr,
                model=model,
                detector=detector,
                temp=temp,
                dev=dev,
                is_multi=is_multi,
                tau_recycle=tau_recycle,
                min_conf=min_conf,
                score_th=score_th,
                prob_hist=prob_hist,
                cached_items=cached_items,
                run_detect=(step % max(1, detect_stride) == 0),
                det_max_side=512,
            )
            infer_ms = (time.perf_counter() - t0) * 1000.0
            step += 1

            now = time.perf_counter()
            inst_fps = 1.0 / max(now - last_tick, 1e-4)
            last_tick = now
            fps_ema = inst_fps if fps_ema == 0.0 else (0.8 * fps_ema + 0.2 * inst_fps)

            prog_txt = f' · khung {frame_idx}/{total_frames}' if total_frames > 0 else f' · khung {frame_idx}'
            caption = f'Video · {mode_txt} · {fps_ema:.1f} FPS ({infer_ms:.0f} ms/khung){prog_txt}'
            frame_slot.markdown(_render_frame_html(data_uri, caption), unsafe_allow_html=True)
            info_slot.markdown(side_html, unsafe_allow_html=True)

            # Yield briefly so Tornado flushes the WebSocket delta cleanly to the browser every single frame
            elapsed = time.perf_counter() - t0
            target_dt = 1.0 / vid_fps
            sleep_s = max(0.015, target_dt - elapsed)
            time.sleep(sleep_s)
    finally:
        cap.release()


def _stream_camera_live_synced(
    cam: LiveCameraCapture,
    tag: str,
    model,
    detector,
    temp: dict,
    dev: torch.device,
    is_multi: bool,
    tau_recycle: float,
    min_conf: float,
    score_th: float,
    frame_slot,
    info_slot,
    mirror: bool = False,
) -> None:
    """Stream Webcam / IP-Cam on the warmed GPU thread reading from COM-safe LiveCameraCapture."""
    mode_txt = 'Nhiều vật' if is_multi else '1 vật'
    prob_hist: deque = deque(maxlen=3)
    last_seq = -1
    fps_ema = 0.0
    last_tick = time.perf_counter()
    wait_start = time.perf_counter()

    while st.session_state.get('rt_running', False):
        bgr, seq, err, ended = cam.get_latest()
        if err:
            st.session_state['rt_running'] = False
            cam.stop()
            frame_slot.warning(err)
            return
        if bgr is None:
            if ended:
                st.session_state['rt_running'] = False
                cam.stop()
                return
            if time.perf_counter() - wait_start > 10.0:
                st.session_state['rt_running'] = False
                cam.stop()
                frame_slot.warning('Quá thời gian chờ tín hiệu camera. Hãy kiểm tra lại thiết bị/URL.')
                return
            frame_slot.info('Đang mở tín hiệu camera...')
            time.sleep(0.05)
            continue

        if seq == last_seq:
            time.sleep(0.015)
            continue
        last_seq = seq

        if mirror:
            bgr = cv2.flip(bgr, 1)

        t0 = time.perf_counter()
        data_uri, side_html, _ = _process_one_frame(
            bgr=bgr,
            model=model,
            detector=detector,
            temp=temp,
            dev=dev,
            is_multi=is_multi,
            tau_recycle=tau_recycle,
            min_conf=min_conf,
            score_th=score_th,
            prob_hist=prob_hist,
            run_detect=True,
            det_max_side=512,
        )
        infer_ms = (time.perf_counter() - t0) * 1000.0
        now = time.perf_counter()
        inst_fps = 1.0 / max(now - last_tick, 1e-4)
        last_tick = now
        fps_ema = inst_fps if fps_ema == 0.0 else (0.8 * fps_ema + 0.2 * inst_fps)

        caption = f'{tag} · {mode_txt} · {fps_ema:.1f} FPS ({infer_ms:.0f} ms/khung)'
        frame_slot.markdown(_render_frame_html(data_uri, caption), unsafe_allow_html=True)
        info_slot.markdown(side_html, unsafe_allow_html=True)
        time.sleep(0.015)


def _process_video_offline_gpu(
    video_path: str,
    model,
    detector,
    temp: dict,
    dev: torch.device,
    is_multi: bool,
    tau_recycle: float,
    min_conf: float,
    score_th: float,
    detect_stride: int = 1,
    max_side: int = 960,
) -> Tuple[Optional[str], Dict]:
    """Process every frame of an MP4 on GPU and write a smooth, browser-playable H.264 MP4."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, {'error': 'Không mở được file video đầu vào.'}

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    fps = max(10.0, min(fps, 60.0))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if w0 <= 0 or h0 <= 0:
        cap.release()
        return None, {'error': 'Không đọc được kích thước video.'}

    scale = min(1.0, float(max_side) / max(w0, h0))
    out_w = max(2, int(w0 * scale) & ~1)
    out_h = max(2, int(h0 * scale) & ~1)

    out_path = str(Path(tempfile.gettempdir()) / f'waste_annotated_{int(time.time() * 1000)}.mp4')
    writer, codec_name = _open_h264_writer(out_path, fps, (out_w, out_h))
    if writer is None:
        cap.release()
        return None, {'error': 'Không khởi tạo được bộ mã hóa video MP4 trên hệ thống.'}

    T_val = float(temp.get('T', 1.0))
    prob_hist: deque = deque(maxlen=3)
    last_items: List[Dict] = []
    bin_frame_hits = {0: 0, 1: 0, 2: 0, 'UNCERTAIN': 0}

    prog = st.progress(0.0, text='Đang khởi động xử lý video trên GPU...')
    t_start = time.perf_counter()
    processed = 0

    try:
        while True:
            ok, bgr = cap.read()
            if not ok or bgr is None:
                break
            if bgr.shape[1] != out_w or bgr.shape[0] != out_h:
                bgr = cv2.resize(bgr, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

            if is_multi:
                if processed % max(1, detect_stride) == 0 or not last_items:
                    items, _ = detect_and_classify_multi(
                        model=model,
                        detector=detector,
                        img=rgb,
                        temp=temp,
                        tau_recycle=tau_recycle,
                        min_conf=min_conf,
                        score_thresh=score_th,
                        max_objects=8,
                        use_contour_fallback=False,
                        device=dev,
                        return_crops=False,
                        det_max_side=640,
                    )
                    last_items = items
                else:
                    items = last_items
                for it in items:
                    bin_frame_hits[it['style_key']] += 1
                vis_rgb = draw_annotated_rgb_fast(rgb, items, max_side=None)
            else:
                p_now = _infer_single_rgb(model, rgb, T_val, dev)
                prob_hist.append(p_now)
                p_smooth = np.mean(prob_hist, axis=0)
                res = decide(p_smooth, tau_recycle, min_conf=min_conf)
                style_key = 'UNCERTAIN' if res['status'] == 'UNCERTAIN' else res['vn_class']
                bin_frame_hits[style_key] += 1
                vis_rgb = draw_annotated_rgb_fast(
                    rgb,
                    [{**res, 'id': 1, 'box': (int(out_w * 0.12), int(out_h * 0.10), int(out_w * 0.88), int(out_h * 0.90)), 'style_key': style_key}],
                    max_side=None,
                )

            writer.write(cv2.cvtColor(vis_rgb, cv2.COLOR_RGB2BGR))
            processed += 1

            if processed % 8 == 0 or processed == total_frames:
                elapsed = max(time.perf_counter() - t_start, 1e-3)
                proc_fps = processed / elapsed
                if total_frames > 0:
                    frac = min(1.0, processed / total_frames)
                    eta = max(0.0, (total_frames - processed) / max(proc_fps, 0.1))
                    prog.progress(
                        frac,
                        text=f'🚀 Đang xử lý khung {processed}/{total_frames} · {proc_fps:.1f} FPS · còn ~{eta:.1f}s',
                    )
                else:
                    prog.progress(0.5, text=f'🚀 Đang xử lý khung {processed} · {proc_fps:.1f} FPS')
    finally:
        cap.release()
        writer.release()

    total_sec = time.perf_counter() - t_start
    prog.progress(1.0, text=f'✅ Hoàn tất {processed} khung hình trong {total_sec:.1f}s ({processed / max(total_sec, 1e-3):.1f} FPS)')
    return out_path, {
        'frames': processed,
        'fps': fps,
        'proc_fps': processed / max(total_sec, 1e-3),
        'elapsed_s': total_sec,
        'codec': codec_name,
        'hits': bin_frame_hits,
        'size_wh': (out_w, out_h),
    }


def render_realtime_tab(
    model,
    detector,
    temp: dict,
    tau_recycle: float,
    min_conf: float,
    device: str | None = None,
) -> None:
    """Render a smooth synced GPU realtime tab + full-MP4 GPU export."""
    dev = resolve_device(device) if device is not None else next(model.parameters()).device
    cam = get_live_camera_capture()
    if 'rt_running' not in st.session_state:
        st.session_state['rt_running'] = False

    c_src, c_mode = st.columns([1.8, 1.3])
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

    is_multi = rt_mode.startswith('📦')
    source_type = 'webcam'
    cam_idx = 0
    ip_url = ''
    video_path = None
    up_vid_id = None
    mirror_cam = False

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
                up_vid_id = getattr(up_vid, 'file_id', f'{up_vid.name}_{up_vid.size}')
                if st.session_state.get('rt_uploaded_vid_id') != up_vid_id:
                    tmp_path = Path(tempfile.gettempdir()) / f'waste_rt_{up_vid.name}'
                    tmp_path.write_bytes(up_vid.getvalue())
                    st.session_state['rt_uploaded_vid_id'] = up_vid_id
                    st.session_state['rt_uploaded_vid_path'] = str(tmp_path)
                    st.session_state.pop('rt_exported_video', None)
                video_path = st.session_state.get('rt_uploaded_vid_path')
        elif src_choice.startswith('📱'):
            source_type = 'ip_cam'
            c_ip1, c_ip2 = st.columns([2.2, 1.0])
            ip_url = c_ip1.text_input(
                'URL Camera điện thoại',
                placeholder='http://192.168.1.10:8080/video',
                label_visibility='collapsed',
            )
            mirror_cam = c_ip2.toggle('🪞 Lật gương', value=False)
        else:
            source_type = 'webcam'
            c_w1, c_w2 = st.columns([1.3, 1.0])
            cam_idx = int(c_w1.number_input('Webcam ID (thường là 0)', min_value=0, max_value=3, value=0, step=1))
            mirror_cam = c_w2.toggle('🪞 Lật gương', value=True)

    with c_opt2:
        score_th = st.slider('Độ nhạy khung', 0.20, 0.70, 0.35, 0.05)

    # Action bar
    if source_type == 'video':
        b1, b2, b3, b4 = st.columns([1.15, 0.9, 1.65, 0.9])
        start_live = b1.button('▶️ Xem trực tiếp', use_container_width=True)
        stop_live = b2.button('⏹️ Dừng', use_container_width=True)
        export_mp4 = b3.button('🎬 Xử lý & xuất MP4 mượt (GPU)', use_container_width=True, type='primary')
        stride = b4.selectbox(
            'Nhịp detect',
            [1, 2, 3],
            index=0,
            format_func=lambda k: f'Mỗi {k} khung',
            label_visibility='collapsed',
            help='Mỗi 1 khung = chính xác từng khung; Mỗi 2–3 khung = nhanh hơn 2–3×',
        )
    else:
        b1, b2 = st.columns([1.2, 1.0])
        start_live = b1.button('▶️ Chạy', use_container_width=True, type='primary')
        stop_live = b2.button('⏹️ Dừng', use_container_width=True)
        export_mp4 = False
        stride = 1

    if start_live:
        if source_type == 'video' and not video_path:
            st.warning('Hãy tải lên 1 file video `.mp4`/`.mov` trước.')
        else:
            st.session_state['rt_running'] = True
            if source_type != 'video':
                cam.start(source_type, cam_idx=cam_idx, ip_url=ip_url)
            else:
                cam.stop()

    if stop_live:
        st.session_state['rt_running'] = False
        cam.stop()

    # Option 2: Full GPU offline MP4 export & smooth 25-30 FPS native playback
    if export_mp4:
        st.session_state['rt_running'] = False
        cam.stop()
        if not video_path:
            st.warning('Hãy tải lên 1 file video `.mp4`/`.mov` trước.')
        else:
            out_path, stats = _process_video_offline_gpu(
                video_path=video_path,
                model=model,
                detector=detector,
                temp=temp,
                dev=dev,
                is_multi=is_multi,
                tau_recycle=tau_recycle,
                min_conf=min_conf,
                score_th=score_th,
                detect_stride=stride,
            )
            if out_path is None:
                st.error(stats.get('error', 'Xử lý video thất bại.'))
            else:
                st.session_state['rt_exported_video'] = {
                    'path': out_path,
                    'stats': stats,
                    'source_id': up_vid_id,
                }

    # Display exported smooth MP4 player when not running live stream
    exported = st.session_state.get('rt_exported_video')
    if (
        source_type == 'video'
        and not st.session_state['rt_running']
        and exported
        and exported.get('source_id') == up_vid_id
        and Path(exported['path']).exists()
    ):
        stats = exported['stats']
        col_v, col_c = st.columns([1.6, 1.0])
        with col_v:
            st.video(exported['path'])
            w_out, h_out = stats['size_wh']
            st.caption(
                f"🎬 Video mượt **{stats['fps']:.0f} FPS** ({w_out}×{h_out}, {stats['codec']}) · "
                f"{stats['frames']} khung xử lý trong `{stats['elapsed_s']:.1f}s` (`{stats['proc_fps']:.1f} FPS` trên `{dev}`)"
            )
        with col_c:
            st.markdown('**📊 Tổng hợp nhận diện toàn video**')
            hits = stats['hits']
            total_hits = max(1, sum(hits.values()))
            for key in (0, 1, 2, 'UNCERTAIN'):
                cnt = hits[key]
                if cnt <= 0:
                    continue
                s = BIN_STYLE[key]
                pct = cnt / total_hits * 100.0
                st.markdown(
                    f"<div style='padding:10px 14px;margin-bottom:8px;border-radius:10px;"
                    f"background:{s['bg_light']};border-left:6px solid {s['hex']}'>"
                    f"<div style='font-size:16px;font-weight:700;color:{s['hex']}'>"
                    f"{s['icon']} {s['short']} · {pct:.0f}% ({cnt} lượt/khung)</div>"
                    f"<div style='font-size:12px;color:#334155'>{s['hint']}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            st.download_button(
                '⬇️ Tải video kết quả (.mp4)',
                data=Path(exported['path']).read_bytes(),
                file_name='phan_loai_rac_gpu.mp4',
                mime='video/mp4',
                use_container_width=True,
            )
        return

    if not st.session_state['rt_running']:
        if cam.is_running():
            cam.stop()
        col_v, col_c = st.columns([1.6, 1.0])
        if source_type == 'video':
            col_v.info(
                'Chọn **🎬 Xử lý & xuất MP4 mượt (GPU)** để xem mượt 30 FPS không rớt khung hình, '
                'hoặc bấm **▶️ Xem trực tiếp** để phát trực tiếp từng khung bằng GPU.'
            )
        else:
            col_v.info('Chọn nguồn rồi bấm **▶️ Chạy** để bắt đầu nhận diện trực tiếp.')
        col_c.markdown(
            "<div style='padding:16px;border-radius:12px;background:#f8fafc;border:1px solid #e2e8f0'>"
            "<div style='font-weight:700;font-size:15px;margin-bottom:6px'>💡 3 nguồn Realtime</div>"
            "<div style='font-size:13px;color:#334155;line-height:1.6'>"
            "1. <b>💻 Webcam:</b> Mở trực tiếp camera laptop (chạy GPU đồng bộ, an toàn trên Windows).<br>"
            "2. <b>🎞️ Video MP4:</b> Xem trực tiếp (~15–20 FPS) hoặc xuất video mượt chuẩn 30 FPS.<br>"
            "3. <b>📱 Camera ĐT:</b> Dán URL từ app <i>IP Webcam</i> / <i>DroidCam</i>."
            "</div></div>",
            unsafe_allow_html=True,
        )
        return

    # Option 1: Synced GPU streaming directly on warmed ScriptRunner thread via st.empty()
    col_v, col_c = st.columns([1.6, 1.0])
    frame_slot = col_v.empty()
    info_slot = col_c.empty()

    if source_type == 'video':
        if not video_path:
            st.session_state['rt_running'] = False
            frame_slot.warning('Hãy tải lên 1 file video `.mp4`/`.mov` trước.')
            return
        _stream_video_live_synced(
            video_path=video_path,
            model=model,
            detector=detector,
            temp=temp,
            dev=dev,
            is_multi=is_multi,
            tau_recycle=tau_recycle,
            min_conf=min_conf,
            score_th=score_th,
            detect_stride=stride,
            frame_slot=frame_slot,
            info_slot=info_slot,
        )
    else:
        tag = 'Phone Cam' if source_type == 'ip_cam' else 'Webcam'
        if not cam.is_running():
            cam.start(source_type, cam_idx=cam_idx, ip_url=ip_url)
        _stream_camera_live_synced(
            cam=cam,
            tag=tag,
            model=model,
            detector=detector,
            temp=temp,
            dev=dev,
            is_multi=is_multi,
            tau_recycle=tau_recycle,
            min_conf=min_conf,
            score_th=score_th,
            frame_slot=frame_slot,
            info_slot=info_slot,
            mirror=mirror_cam,
        )
