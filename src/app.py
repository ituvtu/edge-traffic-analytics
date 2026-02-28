from __future__ import annotations

import argparse
import ctypes
import csv
import datetime as dt
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from src.analytics.color_detector import ColorResult, VehicleColorDetector
from src.analytics.visualizer import Visualizer
from src.config import AppConfig, get_config
from src.inference.onnx_yolo import ONNXYoloDetector
from src.inference.plate_recognizer import PlateRecognizer
from src.io.rtsp_capture import RTSPCapture
from src.pipeline.plate_ocr import PlateOCRPipeline, PlateResult
from src.pipeline.vehicle_tracker import VehicleTracker


# ---------------------------------------------------------------------------
# Async (non-blocking) video writer
# ---------------------------------------------------------------------------

class _AsyncVideoWriter:
    """
    Writes annotated frames to disk in a dedicated background thread so that
    disk I/O never stalls the main processing loop.

    Frames that arrive faster than the writer can handle are silently dropped
    (queue.Full) rather than accumulating unboundedly in memory.
    """

    _SENTINEL = object()

    def __init__(self, path: Path, fourcc: int, fps: float, size: tuple[int, int]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = cv2.VideoWriter(str(path), fourcc, fps, size)
        self._queue: queue.Queue = queue.Queue(maxsize=16)
        self._thread = threading.Thread(target=self._run, daemon=True, name="video-writer")
        self._thread.start()

    def write(self, frame: np.ndarray) -> None:
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            pass  # prefer dropping a frame over blocking the main loop

    def release(self) -> None:
        self._queue.put(self._SENTINEL)
        self._thread.join(timeout=10)
        self._writer.release()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is self._SENTINEL:
                break
            self._writer.write(item)


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _ensure_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "track_id", "plate", "color", "confidence"])


def _append_csv(path: Path, track_id: int, plate: str | None, color: str, conf: float) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([dt.datetime.now().isoformat(), track_id, plate or "", color, f"{conf:.4f}"])


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_app(config: AppConfig) -> None:
    if not config.rtsp_url:
        raise ValueError("RTSP_URL is empty. Set RTSP_URL env var or pass --rtsp")

    profile = config.runtime_profiles.get(config.profile, config.runtime_profiles["cpu"])

    capture = RTSPCapture(
        url=config.rtsp_url,
        frame_skip=config.frame_skip,
        reconnect_initial_delay_sec=config.reconnect_initial_delay_sec,
        reconnect_max_delay_sec=config.reconnect_max_delay_sec,
    )
    vehicle_detector = ONNXYoloDetector(
        model_path=config.car_model_path,
        providers=profile.providers,
        conf_threshold=config.vehicle_conf_threshold,
        nms_threshold=config.nms_threshold,
        input_size=config.input_size,
        allowed_class_ids=config.vehicle_class_ids,
    )
    plate_detector = ONNXYoloDetector(
        model_path=config.plate_model_path,
        providers=profile.providers,
        conf_threshold=config.plate_conf_threshold,
        nms_threshold=config.nms_threshold,
        input_size=config.input_size,
    )
    tracker = VehicleTracker()
    color_detector = VehicleColorDetector()
    ocr_engine = PlateRecognizer(
        det_model_path=str(config.ocr_det_path),
        rec_model_path=str(config.ocr_rec_path),
        char_dict_path=str(config.ocr_char_dict_path),
        regex_patterns=config.ocr_regex_patterns,
    )
    plate_pipeline = PlateOCRPipeline(plate_detector=plate_detector, ocr_engine=ocr_engine)

    _ensure_csv(config.output_csv_path)

    # Deferred initialisation — requires the first real frame for frame size
    visualizer: Visualizer | None = None
    video_writer: _AsyncVideoWriter | None = None
    disp_w: int = 0
    disp_h: int = 0

    _WIN = "car-tracking"
    # Real screen resolution (Win32); fall back to 1920×1080 on non-Windows
    try:
        _SCR_W = ctypes.windll.user32.GetSystemMetrics(0)
        _SCR_H = ctypes.windll.user32.GetSystemMetrics(1)
    except Exception:
        _SCR_W, _SCR_H = 1920, 1080

    def _open_window() -> None:
        """Create a resizable window at initial size, centered on screen."""
        cv2.namedWindow(_WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(_WIN, disp_w, disp_h)
        cx = (_SCR_W - disp_w) // 2
        cy = (_SCR_H - disp_h) // 2
        cv2.moveWindow(_WIN, cx, cy)

    def _ensure_window() -> None:
        """Recreate the window if the user closed it."""
        if cv2.getWindowProperty(_WIN, cv2.WND_PROP_VISIBLE) < 1:
            _open_window()

    def _letterbox(img: np.ndarray, cw: int, ch: int) -> np.ndarray:
        """Fit img into a cw×ch canvas with black bars — never stretch."""
        ih, iw = img.shape[:2]
        scale = min(cw / iw, ch / ih)
        fit_w, fit_h = max(1, int(iw * scale)), max(1, int(ih * scale))
        resized = cv2.resize(img, (fit_w, fit_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((ch, cw, 3), dtype=np.uint8)
        x0 = (cw - fit_w) // 2
        y0 = (ch - fit_h) // 2
        canvas[y0:y0 + fit_h, x0:x0 + fit_w] = resized
        return canvas

    # FPS tracking (exponential moving average, α = 0.1)
    fps_ema: float = 0.0
    t_prev: float = time.perf_counter()

    # Track which IDs have already been written to CSV to avoid duplicates
    csv_written: set[int] = set()

    capture.start()
    try:
        while True:
            frame = capture.read()
            if frame is None:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            # ── Lazy one-time setup ─────────────────────────────────────
            h, w = frame.shape[:2]
            if visualizer is None:
                visualizer = Visualizer(h, w)
                # Canvas: 70 % of screen height, width from video aspect ratio
                disp_h = int(_SCR_H * 0.70)
                disp_w = int(disp_h * w / h)
                _open_window()
            if video_writer is None and config.output_video_path is not None:
                fourcc: int = cv2.VideoWriter.fourcc(*"mp4v")
                video_writer = _AsyncVideoWriter(
                    config.output_video_path, fourcc, fps=25.0, size=(w, h)
                )

            # ── Analytics (on the clean/unmodified frame) ───────────────
            detections = vehicle_detector.detect(frame)
            tracks = tracker.update(detections, (h, w))

            color_results: dict[int, ColorResult] = {}
            plate_results: dict[int, PlateResult] = {}

            for track in tracks:
                crop = tracker.crop_from_track(frame, track)

                # Both detectors are cached by track_id internally
                color_result = color_detector.detect(track.track_id, crop)
                plate_result = plate_pipeline.get_plate_for_track(track.track_id, crop)

                color_results[track.track_id] = color_result
                plate_results[track.track_id] = plate_result

                # Write to CSV only on first confirmed plate per track
                if plate_result.plate and track.track_id not in csv_written:
                    _append_csv(
                        config.output_csv_path,
                        track_id=track.track_id,
                        plate=plate_result.plate,
                        color=color_result.color_name,
                        conf=plate_result.confidence,
                    )
                    csv_written.add(track.track_id)

            # ── FPS ──────────────────────────────────────────────────────
            t_now = time.perf_counter()
            instant_fps = 1.0 / max(t_now - t_prev, 1e-6)
            fps_ema = 0.9 * fps_ema + 0.1 * instant_fps
            t_prev = t_now

            # ── Visualisation (on a copy) ────────────────────────────────
            visualized = visualizer.draw_frame(
                frame, tracks, color_results, plate_results, fps=fps_ema
            )

            # Full-resolution frame goes to the file writer
            if video_writer is not None:
                video_writer.write(visualized)

            # Letterbox into whatever size the window currently is
            _ensure_window()
            rect = cv2.getWindowImageRect(_WIN)   # (x, y, w, h)
            cw, ch = max(1, rect[2]), max(1, rect[3])
            display = _letterbox(visualized, cw, ch)
            cv2.imshow(_WIN, display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        capture.stop()
        if video_writer is not None:
            video_writer.release()
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Edge car-tracking MVP")
    parser.add_argument("--rtsp",    type=str, default=None, help="RTSP URL")
    parser.add_argument("--profile", choices=["cpu", "gpu"], default=None, help="Runtime profile")
    parser.add_argument("--csv",     type=str, default=None, help="CSV output path")
    parser.add_argument("--output",  type=str, default=None, help="Output video file (mp4)")
    return parser.parse_args()


def build_config_from_args() -> AppConfig:
    args = parse_args()
    config = get_config()
    if args.rtsp:
        config.rtsp_url = args.rtsp
    if args.profile:
        config.profile = args.profile
    if args.csv:
        config.output_csv_path = Path(args.csv)
    if args.output:
        config.output_video = args.output
    return config

