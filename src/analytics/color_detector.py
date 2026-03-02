from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass


@dataclass(slots=True)
class ColorResult:
    color_name: str
    confidence: float


class ColorDetector:
    def __init__(self) -> None:
        self.color_ranges: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
            "white": [
                (np.array([0, 0, 175]), np.array([180, 25, 255])),
            ],
            "black": [
                (np.array([0, 0, 0]), np.array([179, 255, 55])),
            ],
            "grey": [
                (np.array([0, 0, 50]), np.array([180, 45, 174])),
            ],
            "red": [
                (np.array([0, 100, 100]), np.array([10, 255, 255])),
                (np.array([160, 100, 100]), np.array([179, 255, 255])),
            ],
            "orange": [
                (np.array([10, 150, 150]), np.array([20, 255, 255])),
            ],
            "yellow": [
                (np.array([20, 100, 100]), np.array([35, 255, 255])),
            ],
            "lime": [
                (np.array([40, 150, 150]), np.array([65, 255, 255])),
            ],
            "green": [
                (np.array([65, 100, 50]), np.array([86, 255, 255])),
            ],
            "cyan": [
                (np.array([85, 100, 100]), np.array([100, 255, 255])),
            ],
            "blue": [
                (np.array([100, 100, 60]), np.array([140, 255, 255])),
            ],
            "purple": [
                (np.array([130, 80, 50]), np.array([155, 255, 255])),
            ],
            "brown": [
                (np.array([8, 100, 40]), np.array([18, 200, 130])),
            ],
            "beige": [
                (np.array([15, 20, 180]), np.array([30, 70, 235])),
            ],
        }

    def detect_color(self, image: np.ndarray) -> str:
        if image is None or image.size == 0:
            return "unknown"

        h = image.shape[0]
        image = image[:max(1, int(h * 0.75)), :]

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        best_count = 0
        best_color = "unknown"
        for color_name, ranges in self.color_ranges.items():
            count = sum(cv2.countNonZero(cv2.inRange(hsv, lo, hi)) for lo, hi in ranges)
            if count > best_count:
                best_count = count
                best_color = color_name

        return best_color


class VehicleColorDetector:
    def __init__(self) -> None:
        self._detector = ColorDetector()
        self._cache: dict[int, ColorResult] = {}

    def detect(self, track_id: int, crop: np.ndarray) -> ColorResult:
        if track_id in self._cache:
            return self._cache[track_id]
        color_name = self._detector.detect_color(crop)
        result = ColorResult(
            color_name=color_name,
            confidence=0.0 if color_name == "unknown" else 1.0,
        )
        self._cache[track_id] = result
        return result

    def reset_cache(self) -> None:
        self._cache.clear()
