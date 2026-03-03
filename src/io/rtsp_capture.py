from __future__ import annotations

import argparse
import logging
import os
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np

_log = logging.getLogger(__name__)


@dataclass(slots=True)
class CaptureStats:
    frames_received: int = 0
    reconnects: int = 0
    read_failures: int = 0
    open_failures: int = 0
    loops: int = 0


class RTSPCapture:
    """Unified capture for RTSP streams and local video files.

    A background reader thread continuously decodes frames and overwrites
    a single-slot buffer (``_latest_frame``).  The consumer always gets
    the *freshest* frame; anything the consumer cannot process in time is
    silently dropped — exactly like a real IP camera.

    For **local files** the reader inserts a ``time.sleep(1/fps)`` after
    every decoded frame so the file is played back at its native frame
    rate instead of being consumed at disk speed.  This makes a local
    ``.mp4`` behave identically to a live RTSP camera for downstream
    processing and recording.
    """

    def __init__(
        self,
        url: str,
        frame_skip: int = 1,
        reconnect_initial_delay_sec: float = 1.0,
        reconnect_max_delay_sec: float = 8.0,
        loop_sleep_sec: float = 0.002,
        open_timeout_sec: float = 5.0,
        read_timeout_sec: float = 5.0,
        stale_frame_timeout_sec: float = 6.0,
        loop_file: bool = True,
    ) -> None:
        self.url = url
        self.frame_skip = max(frame_skip, 1)
        self.reconnect_initial_delay_sec = reconnect_initial_delay_sec
        self.reconnect_max_delay_sec = reconnect_max_delay_sec
        self.loop_sleep_sec = max(loop_sleep_sec, 0.0)
        self.open_timeout_sec = max(open_timeout_sec, 0.1)
        self.read_timeout_sec = max(read_timeout_sec, 0.1)
        self.stale_frame_timeout_sec = max(stale_frame_timeout_sec, 0.5)
        self._is_file: bool = not url.lower().startswith(("rtsp://", "rtsps://", "http://", "https://"))
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._latest_frame: np.ndarray | None = None
        self._latest_frame_pos_ms: float = 0.0
        self._last_frame_ts: float = 0.0
        self._capture: cv2.VideoCapture | None = None
        self.stats = CaptureStats()
        self.source_fps: float = 0.0
        self._loop_file = loop_file
        self._eof_reached = False

    @property
    def is_live_stream(self) -> bool:
        return not self._is_file

    @property
    def eof_reached(self) -> bool:
        return self._eof_reached

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2)
        self._close_capture()

    def read(self) -> np.ndarray | None:
        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def read_with_pos(self) -> tuple[np.ndarray | None, float]:
        """Return *(frame, position_ms)* atomically under a single lock."""
        with self._lock:
            if self._latest_frame is None:
                return None, 0.0
            return self._latest_frame.copy(), self._latest_frame_pos_ms

    @property
    def frame_pos_ms(self) -> float:
        with self._lock:
            return self._latest_frame_pos_ms

    def _open_capture(self) -> cv2.VideoCapture:
        params: list[int] = []
        if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
            params.extend([cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(self.open_timeout_sec * 1000)])
        if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
            params.extend([cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(self.read_timeout_sec * 1000)])

        # Force FFMPEG-level timeout and TCP transport for RTSP streams.
        # - rtsp_transport;tcp avoids UDP packet loss / grey-frame artefacts.
        # - timeout;5000000 (µs) ensures a hard 5-second connection deadline so
        #   that a dead camera unblocks the reader thread and lets Exponential
        #   Backoff take over rather than hanging for 30-60 s.
        # These env-var options work across all OpenCV / FFMPEG versions and act
        # as a reliable fallback alongside the newer CAP_PROP_*_TIMEOUT_MSEC API.
        if not self._is_file:
            timeout_us = int(self.open_timeout_sec * 1_000_000)
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
                f"rtsp_transport;tcp|timeout;{timeout_us}"
            )

        if params:
            cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG, params)
        else:
            cap = cv2.VideoCapture(self.url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _close_capture(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def _schedule_reconnect(self, delay: float) -> float:
        self.stats.reconnects += 1
        with self._lock:
            self._latest_frame = None
        self._close_capture()
        time.sleep(delay)
        return min(delay * 2, self.reconnect_max_delay_sec)

    def _reader_loop(self) -> None:
        frame_idx = 0
        delay = self.reconnect_initial_delay_sec

        while not self._stop_event.is_set():
            if self._capture is None or not self._capture.isOpened():
                self._capture = self._open_capture()
                if not self._capture.isOpened():
                    self.stats.open_failures += 1
                    delay = self._schedule_reconnect(delay)
                    if self.loop_sleep_sec > 0:
                        time.sleep(self.loop_sleep_sec)
                    continue
                delay = self.reconnect_initial_delay_sec
                self._last_frame_ts = time.monotonic()
                if self.source_fps == 0.0:
                    reported = self._capture.get(cv2.CAP_PROP_FPS)
                    if reported and reported > 0:
                        self.source_fps = reported

            _frame_t0 = time.monotonic()
            ok, frame = self._capture.read()
            if not ok or frame is None:
                self.stats.read_failures += 1
                if self._is_file:
                    if not self._loop_file:
                        with self._lock:
                            self._latest_frame = None
                        self._eof_reached = True
                        self._stop_event.set()
                        break
                    self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.stats.loops += 1
                    frame_idx = 0
                    self._last_frame_ts = time.monotonic()
                    continue
                now = time.monotonic()
                if (now - self._last_frame_ts) >= self.stale_frame_timeout_sec:
                    delay = self._schedule_reconnect(delay)
                elif self.loop_sleep_sec > 0:
                    time.sleep(self.loop_sleep_sec)
                continue

            frame_idx += 1
            self._last_frame_ts = time.monotonic()
            if frame_idx % self.frame_skip != 0:
                if self.loop_sleep_sec > 0:
                    time.sleep(self.loop_sleep_sec)
                continue

            pos_ms = self._capture.get(cv2.CAP_PROP_POS_MSEC)

            # Unified single-slot: always overwrite with the latest frame.
            with self._lock:
                self._latest_frame = frame
                self._latest_frame_pos_ms = pos_ms
            self.stats.frames_received += 1

            if self._is_file:
                # Compensated sleep: subtract the time already spent on
                # decoding / locking so each frame period is exactly 1/fps
                # regardless of how long the decode took.  A flat
                # ``time.sleep(1/fps)`` would overshoot by the decode
                # duration, causing slow-motion on machines where the
                # decode cost is non-trivial (e.g. Windows + large files).
                fps = self.source_fps if self.source_fps > 0 else 25.0
                _decode_elapsed = time.monotonic() - _frame_t0
                _sleep_time = max(0.0, 1.0 / fps - _decode_elapsed)
                if _sleep_time > 0:
                    time.sleep(_sleep_time)
            elif self.loop_sleep_sec > 0:
                time.sleep(self.loop_sleep_sec)


def _parse_local_test_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local RTSPCapture test runner")
    parser.add_argument("--rtsp", required=True, help="RTSP URL to test")
    parser.add_argument("--frame-skip", type=int, default=1, help="Use every N-th frame")
    parser.add_argument("--reconnect-initial", type=float, default=1.0, help="Initial reconnect delay in seconds")
    parser.add_argument("--reconnect-max", type=float, default=8.0, help="Maximum reconnect delay in seconds")
    parser.add_argument("--loop-sleep", type=float, default=0.002, help="Reader loop sleep in seconds")
    parser.add_argument("--open-timeout", type=float, default=5.0, help="Open timeout in seconds")
    parser.add_argument("--read-timeout", type=float, default=5.0, help="Read timeout in seconds")
    parser.add_argument("--stale-timeout", type=float, default=6.0, help="Stale-frame timeout in seconds")
    parser.add_argument("--window", default="rtsp-capture-test", help="OpenCV window name")
    return parser.parse_args()


def _run_local_test() -> None:
    args = _parse_local_test_args()
    capture = RTSPCapture(
        url=args.rtsp,
        frame_skip=args.frame_skip,
        reconnect_initial_delay_sec=args.reconnect_initial,
        reconnect_max_delay_sec=args.reconnect_max,
        loop_sleep_sec=args.loop_sleep,
        open_timeout_sec=args.open_timeout,
        read_timeout_sec=args.read_timeout,
        stale_frame_timeout_sec=args.stale_timeout,
    )

    capture.start()
    last_stats_print_ts = time.monotonic()

    try:
        while True:
            frame = capture.read()
            if frame is not None:
                cv2.imshow(args.window, frame)

            now = time.monotonic()
            if now - last_stats_print_ts >= 2.0:
                stats = capture.stats
                _log.info(
                    "frames_received=%d reconnects=%d open_failures=%d read_failures=%d",
                    stats.frames_received, stats.reconnects,
                    stats.open_failures, stats.read_failures,
                )
                last_stats_print_ts = now

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            time.sleep(0.001)
    finally:
        capture.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    _run_local_test()
