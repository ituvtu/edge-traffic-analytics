"""
Visualizer — draws detection/tracking results onto video frames.

Rules:
  * Only cv2.rectangle / cv2.putText (no heavy graphics libs).
  * Always operates on a copy of the frame so upstream analytics
    receive clean pixels.
  * Font scale and line thickness scale dynamically with frame resolution.
"""
from __future__ import annotations

import cv2
import numpy as np

from src.analytics.color_detector import ColorResult
from src.pipeline.plate_ocr import PlateResult
from src.pipeline.vehicle_tracker import Track

# BGR palette indexed by color name
_PALETTE: dict[str, tuple[int, int, int]] = {
    "black":   (30,  30,  30),
    "white":   (230, 230, 230),
    "gray":    (128, 128, 128),
    "red":     (20,  20,  220),
    "orange":  (0,   140, 255),
    "yellow":  (0,   255, 255),
    "green":   (0,   200,   0),
    "blue":    (220,  90,   0),
    "purple":  (180,  40, 180),
    "unknown": (150, 150, 150),
}
_SHADOW: tuple[int, int, int] = (15, 15, 15)
_FPS_COLOR: tuple[int, int, int] = (0, 230, 0)
_TEXT_COLOR: tuple[int, int, int] = (255, 255, 255)  # always-white labels


class Visualizer:
    """
    Stateless renderer — create once per stream, call ``draw_frame`` every tick.

    Parameters
    ----------
    frame_height, frame_width:
        Dimensions of the video source, used to scale typography/lines
        relative to a 960-pixel reference side.
    """

    def __init__(self, frame_height: int, frame_width: int) -> None:
        ref = max(frame_height, frame_width)
        scale = ref / 960.0

        self._font       = cv2.FONT_HERSHEY_SIMPLEX
        self._font_scale = max(0.38, round(0.52 * scale, 2))
        self._thick      = max(1, round(1.6 * scale))
        self._box_thick  = max(1, round(2.2 * scale))
        self._pad        = max(4, round(8 * scale))

        # Pre-compute text line height (based on a capital "A")
        (_, lh), baseline = cv2.getTextSize(
            "A", self._font, self._font_scale, self._thick
        )
        self._line_h = lh + baseline + self._pad

    # ------------------------------------------------------------------
    def draw_frame(
        self,
        frame: np.ndarray,
        tracks: list[Track],
        color_results: dict[int, ColorResult],
        plate_results: dict[int, PlateResult],
        fps: float = 0.0,
    ) -> np.ndarray:
        """
        Render all annotations onto a *copy* of ``frame`` and return it.

        The original ``frame`` is never mutated so OCR / color analytics
        that run after this call still receive pristine pixels.
        """
        out = frame.copy()

        for track in tracks:
            self._draw_track(out, track, color_results, plate_results)

        if fps > 0.0:
            self._draw_fps(out, fps)

        return out

    # ------------------------------------------------------------------
    def _draw_track(
        self,
        out: np.ndarray,
        track: Track,
        color_results: dict[int, ColorResult],
        plate_results: dict[int, PlateResult],
    ) -> None:
        x1, y1, x2, y2 = map(int, track.bbox)
        color_res = color_results.get(track.track_id, ColorResult("unknown", 0.0))
        plate_res = plate_results.get(track.track_id, PlateResult(None, 0.0))

        raw_bgr = _PALETTE.get(color_res.color_name, _PALETTE["unknown"])

        # Bounding box — vehicle color
        cv2.rectangle(out, (x1, y1), (x2, y2), raw_bgr, self._box_thick)

        # Build label lines (top to bottom visual order)
        lines: list[str] = [f"ID {track.track_id}", color_res.color_name]
        if plate_res.plate:
            conf_pct = int(plate_res.confidence * 100)
            lines.append(f"{plate_res.plate}  {conf_pct}%")

        # Stack lines upward from the top edge of the box
        for i, text in enumerate(reversed(lines)):
            ty = max(self._line_h, y1) - i * self._line_h - self._pad
            # Dark shadow + white text = readable on any background
            cv2.putText(
                out, text, (x1 + 1, ty + 1),
                self._font, self._font_scale, _SHADOW,
                self._thick + 1, cv2.LINE_AA,
            )
            cv2.putText(
                out, text, (x1, ty),
                self._font, self._font_scale, _TEXT_COLOR,
                self._thick, cv2.LINE_AA,
            )

    def _draw_fps(self, out: np.ndarray, fps: float) -> None:
        h, w = out.shape[:2]
        text = f"FPS {fps:.1f}"
        (tw, _), _ = cv2.getTextSize(text, self._font, self._font_scale, self._thick)
        fx = w - tw - self._pad * 3
        fy = self._line_h + self._pad
        cv2.putText(
            out, text, (fx + 1, fy + 1),
            self._font, self._font_scale, _SHADOW,
            self._thick + 1, cv2.LINE_AA,
        )
        cv2.putText(
            out, text, (fx, fy),
            self._font, self._font_scale, _FPS_COLOR,
            self._thick, cv2.LINE_AA,
        )
