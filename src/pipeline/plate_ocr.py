from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.inference.onnx_yolo import ONNXYoloDetector
from src.inference.plate_recognizer import PlateRecognizer


@dataclass(slots=True)
class PlateResult:
    plate: str | None
    confidence: float


class PlateOCRPipeline:
    def __init__(self, plate_detector: ONNXYoloDetector, ocr_engine: PlateRecognizer) -> None:
        self.plate_detector = plate_detector
        self.ocr_engine = ocr_engine
        self._cache: dict[int, PlateResult] = {}

    def get_plate_for_track(self, track_id: int, vehicle_crop: np.ndarray) -> PlateResult:
        if track_id in self._cache:
            return self._cache[track_id]
        if vehicle_crop.size == 0:
            return PlateResult(None, 0.0)

        plate_detections = self.plate_detector.detect(vehicle_crop)
        if not plate_detections:
            return PlateResult(None, 0.0)

        best_det = max(plate_detections, key=lambda d: d.score)
        x1, y1, x2, y2 = map(int, [best_det.x1, best_det.y1, best_det.x2, best_det.y2])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(vehicle_crop.shape[1], x2), min(vehicle_crop.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return PlateResult(None, 0.0)

        plate_crop = vehicle_crop[y1:y2, x1:x2]

        # YOLO already isolated the plate — feed it directly to the recognizer.
        # Running DBNet detection again on an already-cropped tiny image is
        # unreliable and redundant; skip to recognition + validation.
        plate_text, ocr_conf = self.ocr_engine.recognize_and_validate(plate_crop)
        result = PlateResult(plate=plate_text, confidence=ocr_conf)
        if plate_text:
            self._cache[track_id] = result
        return result
