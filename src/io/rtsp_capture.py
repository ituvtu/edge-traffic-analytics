from __future__ import annotations

import argparse
import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class CaptureStats:
    frames_received: int = 0
    reconnects: int = 0
    read_failures: int = 0
    open_failures: int = 0


class RTSPCapture:
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
    ) -> None:
        self.url = url
        self.frame_skip = max(frame_skip, 1)
        self.reconnect_initial_delay_sec = reconnect_initial_delay_sec
        self.reconnect_max_delay_sec = reconnect_max_delay_sec
        self.loop_sleep_sec = max(loop_sleep_sec, 0.0)
        self.open_timeout_sec = max(open_timeout_sec, 0.1)
        self.read_timeout_sec = max(read_timeout_sec, 0.1)
        self.stale_frame_timeout_sec = max(stale_frame_timeout_sec, 0.5)
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._latest_frame: np.ndarray | None = None
        self._last_frame_ts: float = 0.0
        self._capture: cv2.VideoCapture | None = None
        self.stats = CaptureStats()

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

    def _open_capture(self) -> cv2.VideoCapture:
        params: list[int] = []
        if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
            params.extend([cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(self.open_timeout_sec * 1000)])
        if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
            params.extend([cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(self.read_timeout_sec * 1000)])

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

            ok, frame = self._capture.read()
            if not ok or frame is None:
                self.stats.read_failures += 1
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

            with self._lock:
                self._latest_frame = frame
            self.stats.frames_received += 1

            if self.loop_sleep_sec > 0:
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
                print(
                    "frames_received=", stats.frames_received,
                    "reconnects=", stats.reconnects,
                    "open_failures=", stats.open_failures,
                    "read_failures=", stats.read_failures,
                    sep="",
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
