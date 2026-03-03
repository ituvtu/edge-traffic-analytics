from __future__ import annotations

import cv2
import numpy as np

from src.analytics.color_detector import ColorResult
from src.pipeline.plate_ocr import PlateResult
from src.pipeline.vehicle_tracker import Track

_PALETTE: dict[str, tuple[int, int, int]] = {
    "black":   (30,  30,  30),
    "white":   (230, 230, 230),
    "grey":    (128, 128, 128),
    "red":     (20,  20,  220),
    "orange":  (0,   140, 255),
    "yellow":  (0,   255, 255),
    "lime":    (0,   255, 100),
    "green":   (0,   200,   0),
    "cyan":    (210, 210,   0),
    "blue":    (220,  90,   0),
    "purple":  (180,  40, 180),
    "brown":   (30,   60, 100),
    "beige":   (185, 215, 230),
    "unknown": (150, 150, 150),
}
_SHADOW: tuple[int, int, int] = (15, 15, 15)
_FPS_COLOR: tuple[int, int, int] = (0, 230, 0)
_TEXT_COLOR: tuple[int, int, int] = (255, 255, 255)
_PANEL_ALPHA: float = 0.72


class Visualizer:
    def __init__(self, frame_height: int, frame_width: int) -> None:
        ref = max(frame_height, frame_width)
        scale = ref / 960.0

        self._font       = cv2.FONT_HERSHEY_SIMPLEX
        self._font_scale = max(0.38, round(0.52 * scale, 2))
        self._thick      = max(1, round(1.6 * scale))
        self._box_thick  = max(1, round(2.2 * scale))
        self._pad        = max(4, round(8 * scale))

        (_, lh), baseline = cv2.getTextSize(
            "A", self._font, self._font_scale, self._thick
        )
        self._line_h = lh + baseline + self._pad
        self._panel_w = max(200, int(270 * scale))

    def draw_frame(
        self,
        frame: np.ndarray,
        tracks: list[Track],
        color_results: dict[int, ColorResult],
        plate_results: dict[int, PlateResult],
        fps: float = 0.0,
    ) -> np.ndarray:
        out = frame.copy()

        for track in tracks:
            self._draw_track(out, track, color_results, plate_results)

        if fps > 0.0:
            self._draw_fps(out, fps)

        return out

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
        cv2.rectangle(out, (x1, y1), (x2, y2), raw_bgr, self._box_thick)

        lines: list[str] = [f"ID:{track.track_id} {color_res.color_name}"]
        if plate_res.plate:
            conf_pct = int(plate_res.confidence * 100)
            lines.append(f"{plate_res.plate}  {conf_pct}%")

        for i, text in enumerate(reversed(lines)):
            ty = max(self._line_h, y1) - i * self._line_h - self._pad
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

    def make_analytics_frame(
        self,
        color_tally: dict[str, int],
        top_plates: list[tuple[str, str, int]],
    ) -> np.ndarray:
        bar_fixed_w = 200
        label_area_w = 160
        canvas_w = self._pad * 2 + bar_fixed_w + self._pad + label_area_w

        active_colors = sorted(
            [(c, n) for c, n in color_tally.items() if n > 0],
            key=lambda x: x[1],
            reverse=True,
        )
        capped_plates = top_plates[:10]

        rows_hist = len(active_colors)
        rows_plates = len(capped_plates)
        section_gap = self._line_h if rows_hist > 0 and rows_plates > 0 else 0

        canvas_h = (
            (self._line_h + rows_hist * (self._line_h + 2) if rows_hist else 0)
            + section_gap
            + (self._line_h + rows_plates * (self._line_h + 2) if rows_plates else 0)
            + self._pad * 4
        )
        canvas_h = max(canvas_h, self._line_h * 14)

        canvas = np.full((canvas_h, canvas_w, 3), (59, 41, 30), dtype=np.uint8)
        cursor_y = self._pad

        if rows_hist > 0:
            max_count = max(n for _, n in active_colors)
            cv2.putText(
                canvas, "Color distribution",
                (self._pad, cursor_y + self._line_h - self._pad // 2),
                self._font, self._font_scale * 0.75, _TEXT_COLOR, 1, cv2.LINE_AA,
            )
            cursor_y += self._line_h
            for color_name, count in active_colors:
                bar_w = max(1, int(bar_fixed_w * count / max_count))
                bar_color = _PALETTE.get(color_name, _PALETTE["unknown"])
                cv2.rectangle(
                    canvas,
                    (self._pad, cursor_y),
                    (self._pad + bar_w, cursor_y + self._line_h - 4),
                    bar_color, -1,
                )
                cv2.putText(
                    canvas, f"{color_name} {count}",
                    (self._pad + bar_fixed_w + self._pad, cursor_y + self._line_h - 4),
                    self._font, self._font_scale * 0.70, _TEXT_COLOR, 1, cv2.LINE_AA,
                )
                cursor_y += self._line_h + 2

        if rows_plates > 0:
            cursor_y += section_gap
            cv2.putText(
                canvas, "Top plates",
                (self._pad, cursor_y + self._line_h - self._pad // 2),
                self._font, self._font_scale * 0.75, _TEXT_COLOR, 1, cv2.LINE_AA,
            )
            cursor_y += self._line_h
            for plate, color, count in capped_plates:
                dot_color = _PALETTE.get(color, _PALETTE["unknown"])
                dot_cy = cursor_y + (self._line_h - 4) // 2
                cv2.circle(canvas, (self._pad + 5, dot_cy), 5, dot_color, -1)
                cv2.putText(
                    canvas, f"{plate}  ({count}x)",
                    (self._pad + 16, cursor_y + self._line_h - 4),
                    self._font, self._font_scale * 0.78, _TEXT_COLOR, 1, cv2.LINE_AA,
                )
                cursor_y += self._line_h + 2

        return canvas
