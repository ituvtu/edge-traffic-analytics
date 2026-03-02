from __future__ import annotations

import argparse
import collections
import ctypes
import csv
import datetime
import logging
import queue
import re
import signal
import sys
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
from src.utils.profiler import PipelineProfiler

_log = logging.getLogger(__name__)


class _AsyncVideoWriter:

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
            pass

    def release(self) -> None:
        # Drain any pending frames so the put(sentinel) never blocks on a full queue.
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._queue.put(self._SENTINEL)
        self._thread.join(timeout=15)
        self._writer.release()

    def _run(self) -> None:
        while True:
            item = self._queue.get()
            if item is self._SENTINEL:
                break
            self._writer.write(item)


# How long a track must be absent from active detections before it is
# considered to have left the camera FOV and is flushed to disk.
TRACK_EXPIRATION_SEC: float = 5.0


def _append_vehicles_to_csv(path: Path, entries: list[dict]) -> None:
    """Append *entries* to the CSV report, writing the header when needed."""
    if not entries:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    need_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if need_header:
            writer.writerow(["Timestamp", "License_Plate", "Color"])
        for entry in sorted(entries, key=lambda e: e["timestamp_sort"]):
            writer.writerow([entry["timestamp_display"], entry["plate"], entry["color"]])
        f.flush()


_PLATE_RE = re.compile(r'^[A-ZА-ЯІЇЄ0-9]{4,10}$')


def _is_valid_plate(plate: str) -> bool:
    cleaned = plate.strip().upper().replace(" ", "").replace("-", "")
    if not _PLATE_RE.match(cleaned):
        return False
    if not any(c.isalpha() for c in cleaned):
        return False
    if not any(c.isdigit() for c in cleaned):
        return False
    if len(set(cleaned)) == 1:
        return False
    return True


def run_app(config: AppConfig) -> None:
    if not config.rtsp_url:
        raise ValueError("RTSP_URL is empty. Set RTSP_URL env var or pass --rtsp")

    profile = config.runtime_profiles.get(config.profile, config.runtime_profiles["cpu"])
    _HEADLESS: bool = config.headless_mode

    capture = RTSPCapture(
        url=config.rtsp_url,
        frame_skip=config.frame_skip,
        reconnect_initial_delay_sec=config.reconnect_initial_delay_sec,
        reconnect_max_delay_sec=config.reconnect_max_delay_sec,
        loop_file=not _HEADLESS,
    )
    is_live_stream = capture.is_live_stream
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

    profiler = PipelineProfiler()

    plate_pipeline = PlateOCRPipeline(
        plate_detector=plate_detector,
        ocr_engine=ocr_engine,
        profiler=profiler,
    )

    session_vehicles: dict[int, dict] = {}
    track_last_seen: dict[int, float] = {}
    color_tally_session: dict[str, int] = collections.defaultdict(int)
    color_counted_ids: set[int] = set()
    plate_seen_counts: dict[str, int] = collections.defaultdict(int)

    visualizer: Visualizer | None = None
    video_writer: _AsyncVideoWriter | None = None
    disp_w: int = 0
    disp_h: int = 0

    _WIN = "car-tracking"
    _WIN_ANA = "analytics"
    _ana_opened: bool = False
    try:
        _SCR_W = ctypes.windll.user32.GetSystemMetrics(0)
        _SCR_H = ctypes.windll.user32.GetSystemMetrics(1)
    except Exception:
        _log.debug("GetSystemMetrics unavailable (non-Windows?), defaulting to 1920x1080")
        _SCR_W, _SCR_H = 1920, 1080

    def _open_window() -> None:
        cv2.namedWindow(_WIN, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(_WIN, disp_w, disp_h)
        cx = (_SCR_W - disp_w) // 2
        cy = (_SCR_H - disp_h) // 2
        cv2.moveWindow(_WIN, cx, cy)

    def _ensure_analytics_window(frame: np.ndarray) -> None:
        nonlocal _ana_opened
        cx_main = (_SCR_W - disp_w) // 2
        cy_main = (_SCR_H - disp_h) // 2
        need_create = (
            not _ana_opened
            or cv2.getWindowProperty(_WIN_ANA, cv2.WND_PROP_VISIBLE) < 1
        )
        if need_create:
            cv2.namedWindow(_WIN_ANA, cv2.WINDOW_NORMAL)
            fh, fw = frame.shape[:2]
            cv2.resizeWindow(_WIN_ANA, fw, fh)
            cv2.moveWindow(_WIN_ANA, cx_main + disp_w + 12, cy_main)
            _ana_opened = True
        ana_rect = cv2.getWindowImageRect(_WIN_ANA)
        ana_cw, ana_ch = max(1, ana_rect[2]), max(1, ana_rect[3])
        cv2.imshow(_WIN_ANA, _letterbox(frame, ana_cw, ana_ch))

    def _ensure_window() -> None:
        if cv2.getWindowProperty(_WIN, cv2.WND_PROP_VISIBLE) < 1:
            _open_window()

    def _letterbox(img: np.ndarray, cw: int, ch: int) -> np.ndarray:
        ih, iw = img.shape[:2]
        scale = min(cw / iw, ch / ih)
        fit_w, fit_h = max(1, int(iw * scale)), max(1, int(ih * scale))
        resized = cv2.resize(img, (fit_w, fit_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((ch, cw, 3), dtype=np.uint8)
        x0 = (cw - fit_w) // 2
        y0 = (ch - fit_h) // 2
        canvas[y0:y0 + fit_h, x0:x0 + fit_w] = resized
        return canvas

    fps_ema: float = 0.0
    t_prev: float = time.perf_counter()
    t_frame_start: float = t_prev

    _src_fps: float = 25.0

    _last_loops: int = 0
    _LARGE_VEHICLE: frozenset[int] = frozenset({5})

    keyboard_stop = threading.Event()

    def _keyboard_listener() -> None:
        try:
            import msvcrt
            while not keyboard_stop.is_set():
                if msvcrt.kbhit():
                    if msvcrt.getch().lower() == b"q":
                        keyboard_stop.set()
                        break
                time.sleep(0.05)
        except ImportError:
            for line in sys.stdin:
                if line.strip().lower() == "q":
                    keyboard_stop.set()
                    break

    if not _HEADLESS:
        threading.Thread(target=_keyboard_listener, daemon=True).start()
        _log.info("Press Q to stop the pipeline.")
    else:
        _log.info("Pipeline running in headless mode. Press Ctrl+C to stop.")

    is_running = True

    def handle_signal(signum: int, _frame: object) -> None:
        nonlocal is_running
        _log.info("Received signal %s, shutting down...", signum)
        is_running = False

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    capture.start()
    _fps_probe_end = time.perf_counter() + 2.0
    while time.perf_counter() < _fps_probe_end and capture.source_fps == 0.0:
        time.sleep(0.02)
    if capture.source_fps > 0.0:
        _src_fps = capture.source_fps
    stream_start_time: float = 0.0
    written_video_msec: float = 0.0
    frame_duration_msec: float = 0.0
    try:
        while is_running and not keyboard_stop.is_set():
            profiler.start("frame_read")
            frame = capture.read()
            profiler.stop("frame_read")

            if frame is None and capture.eof_reached:
                break

            current_loops = capture.stats.loops
            if current_loops != _last_loops:
                profiler.new_epoch()
                tracker.reset()
                color_detector.reset_cache()
                plate_pipeline.ocr_cache.clear()
                color_counted_ids.clear()
                if config.reset_stats_on_loop:
                    color_tally_session.clear()
                    session_vehicles.clear()
                    plate_seen_counts.clear()
                track_last_seen.clear()
                _last_loops = current_loops

            if frame is None:
                if not _HEADLESS and cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            h, w = frame.shape[:2]
            t_frame_start = time.perf_counter()
            if visualizer is None:
                visualizer = Visualizer(h, w)
                disp_h = int(_SCR_H * 0.70)
                disp_w = int(disp_h * w / h)
                if not _HEADLESS:
                    _open_window()
            if video_writer is None and config.output_video_path is not None:
                fourcc: int = cv2.VideoWriter.fourcc(*"mp4v")
                writer_fps = 60.0
                video_writer = _AsyncVideoWriter(
                    config.output_video_path, fourcc, fps=writer_fps, size=(w, h)
                )
                stream_start_time = time.time()
                written_video_msec = 0.0
                frame_duration_msec = 1000.0 / max(writer_fps, 1.0)

            profiler.start("frame_resize")
            max_side = max(h, w)
            det_scale = min(1000.0 / max_side, 1.0)
            if det_scale < 1.0:
                det_frame = cv2.resize(
                    frame,
                    (int(w * det_scale), int(h * det_scale)),
                    interpolation=cv2.INTER_LINEAR,
                )
            else:
                det_frame = frame
            profiler.stop("frame_resize")

            profiler.start("car_detect")
            detections = vehicle_detector.detect(det_frame)
            if det_scale < 1.0:
                inv = 1.0 / det_scale
                for d in detections:
                    d.x1 *= inv
                    d.y1 *= inv
                    d.x2 *= inv
                    d.y2 *= inv
            profiler.stop("car_detect")
            profiler.start("tracker_update")
            tracks = tracker.update(detections, (h, w))
            profiler.stop("tracker_update")

            active_ids = {t.track_id for t in tracks}
            _now = time.perf_counter()
            for _tid in active_ids:
                track_last_seen[_tid] = _now
            plate_pipeline.prune(active_ids)

            color_results: dict[int, ColorResult] = {}
            plate_results: dict[int, PlateResult] = {}
            car_tracks: list = []

            for track in tracks:
                if track.class_id in _LARGE_VEHICLE:
                    continue

                car_tracks.append(track)
                x1, y1, x2, y2 = track.bbox
                car_w, car_h = x2 - x1, y2 - y1

                crop = tracker.crop_from_track(frame, track)

                profiler.start("color_detect")
                color_result = color_detector.detect(track.track_id, crop)
                profiler.stop("color_detect")

                if track.track_id not in color_counted_ids and color_result.color_name != "unknown":
                    color_tally_session[color_result.color_name] += 1
                    color_counted_ids.add(track.track_id)

                # Skip OCR for vehicles too far / too small to have a readable plate.
                # Trigger only when the car bbox is wide enough (> 400 px) OR its
                # area exceeds 120 000 px² — whichever happens first for a given lens.
                if car_w <= config.ocr_min_vehicle_width and car_w * car_h <= config.ocr_min_vehicle_area:
                    color_results[track.track_id] = color_result
                    plate_results[track.track_id] = PlateResult(None, 0.0)
                    continue

                plate_result = plate_pipeline.get_plate_for_track(track.track_id, crop)

                color_results[track.track_id] = color_result
                plate_results[track.track_id] = plate_result

                if plate_result.plate:
                    pos_ms = capture.frame_pos_ms
                    if is_live_stream:
                        timestamp_display = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    else:
                        total_sec = int(pos_ms / 1000)
                        timestamp_display = f"{total_sec // 60:02d}:{total_sec % 60:02d}"
                    existing = session_vehicles.get(track.track_id)
                    if existing is None:
                        plate_seen_counts[plate_result.plate] += 1
                        session_vehicles[track.track_id] = {
                            "plate": plate_result.plate,
                            "color": color_result.color_name,
                            "timestamp_sort": (time.time() * 1000.0) if is_live_stream else pos_ms,
                            "timestamp_display": timestamp_display,
                            "confidence": plate_result.confidence,
                        }
                    else:
                        # Update plate fields only if confidence improved.
                        if plate_result.confidence > existing["confidence"]:
                            existing["plate"] = plate_result.plate
                            existing["confidence"] = plate_result.confidence
                            existing["timestamp_sort"] = (time.time() * 1000.0) if is_live_stream else pos_ms
                            existing["timestamp_display"] = timestamp_display
                        # Update color independently: fill in "unknown" as soon
                        # as a valid color reading becomes available.
                        if existing["color"] == "unknown" and color_result.color_name != "unknown":
                            existing["color"] = color_result.color_name

            # ── GC: flush expired tracks to CSV and free memory ─────────
            _expired_ids = [
                _tid for _tid, _entry in session_vehicles.items()
                if _now - track_last_seen.get(_tid, 0.0) > TRACK_EXPIRATION_SEC
            ]
            if _expired_ids:
                _append_vehicles_to_csv(
                    config.report_csv_path,
                    [session_vehicles[_tid] for _tid in _expired_ids],
                )
                for _tid in _expired_ids:
                    del session_vehicles[_tid]
                    track_last_seen.pop(_tid, None)
                    plate_pipeline.ocr_attempts.pop(_tid, None)

            # ── FPS ──────────────────────────────────────────────────────
            t_now = time.perf_counter()
            instant_fps = 1.0 / max(t_now - t_prev, 1e-6)
            fps_ema = 0.9 * fps_ema + 0.1 * instant_fps
            t_prev = t_now
            profiler.tick()

            color_tally = dict(color_tally_session)
            top_plates = [
                (entry["plate"], entry["color"], plate_seen_counts[entry["plate"]])
                for entry in sorted(session_vehicles.values(), key=lambda e: e["timestamp_sort"], reverse=True)[:10]
            ]

            profiler.start("visualization")
            visualized = visualizer.draw_frame(
                frame, car_tracks, color_results, plate_results,
                fps=fps_ema,
            )
            profiler.stop("visualization")

            analytics_frame = visualizer.make_analytics_frame(color_tally, top_plates)
            if not _HEADLESS:
                _ensure_analytics_window(analytics_frame)

            if video_writer is not None:
                if is_live_stream:
                    current_msec = (time.time() - stream_start_time) * 1000.0
                else:
                    current_msec = capture.frame_pos_ms
                while written_video_msec <= current_msec:
                    video_writer.write(visualized)
                    written_video_msec += frame_duration_msec

            # Compensated delay: wait only the time remaining in the current
            # frame slot so playback matches the source speed on-screen.
            _elapsed_ms = (time.perf_counter() - t_frame_start) * 1000.0
            _wait_ms = max(1, int(1000.0 / max(_src_fps, 1.0) - _elapsed_ms))
            if not _HEADLESS:
                _ensure_window()
                rect = cv2.getWindowImageRect(_WIN)
                cw, ch = max(1, rect[2]), max(1, rect[3])
                display = _letterbox(visualized, cw, ch)
                cv2.imshow(_WIN, display)
                if cv2.waitKey(_wait_ms) & 0xFF == ord("q"):
                    break
            else:
                time.sleep(_wait_ms / 1000.0)

    finally:
        capture.stop()
        if video_writer is not None:
            video_writer.release()
        if not _HEADLESS:
            cv2.destroyAllWindows()
        _append_vehicles_to_csv(config.report_csv_path, list(session_vehicles.values()))
        profiler.print_report()
        profiler.save_report(config.profiler_report_path)
        _log.info("Results saved — Report: %s | Profiler: %s", config.report_csv_path, config.profiler_report_path)
        if config.output_video_path is not None:
            _log.info("Video saved: %s", config.output_video_path)


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

