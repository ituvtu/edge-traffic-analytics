from __future__ import annotations

import cv2
import numpy as np

from src.analytics.color_detector import ColorResult
from src.pipeline.plate_ocr import PlateResult
from src.pipeline.vehicle_tracker import Track

_BOX_THICKNESS: int = 2
_LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
_LABEL_FONT_SCALE: float = 0.5
_LABEL_THICKNESS: int = 2
_LABEL_OFFSET_Y: int = 8


def _color_to_bgr(color_name: str) -> tuple[int, int, int]:
    palette = {
        "black": (30, 30, 30),
        "white": (230, 230, 230),
        "gray": (128, 128, 128),
        "red": (20, 20, 220),
        "orange": (0, 140, 255),
        "yellow": (0, 255, 255),
        "green": (0, 200, 0),
        "blue": (220, 90, 0),
        "purple": (180, 40, 180),
        "unknown": (150, 150, 150),
    }
    return palette.get(color_name, (150, 150, 150))


def draw_overlay(
    frame: np.ndarray,
    tracks: list[Track],
    color_results: dict[int, ColorResult],
    plate_results: dict[int, PlateResult],
) -> np.ndarray:
    output = frame.copy()
    for track in tracks:
        x1, y1, x2, y2 = map(int, track.bbox)
        color = color_results.get(track.track_id, ColorResult("unknown", 0.0))
        plate = plate_results.get(track.track_id, PlateResult(None, 0.0))

        box_color = _color_to_bgr(color.color_name)
        cv2.rectangle(output, (x1, y1), (x2, y2), box_color, _BOX_THICKNESS)

        label = f"ID:{track.track_id} plate:{plate.plate or '-'} color:{color.color_name}"
        cv2.putText(output, label, (x1, max(0, y1 - _LABEL_OFFSET_Y)),
                    _LABEL_FONT, _LABEL_FONT_SCALE, box_color, _LABEL_THICKNESS)
    return output
