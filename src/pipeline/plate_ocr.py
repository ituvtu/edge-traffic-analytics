from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from src.inference.onnx_yolo import ONNXYoloDetector
from src.inference.plate_recognizer import PlateRecognizer

if TYPE_CHECKING:
    from src.utils.profiler import PipelineProfiler


@dataclass(slots=True)
class PlateResult:
    plate: str | None
    confidence: float


class PlateOCRPipeline:
    def __init__(
        self,
        plate_detector: ONNXYoloDetector,
        ocr_engine: PlateRecognizer,
        profiler: "PipelineProfiler | None" = None,
    ) -> None:
        self.plate_detector = plate_detector
        self.ocr_engine = ocr_engine
        self._profiler = profiler
        self.ocr_cache: dict[int, PlateResult] = {}
        self.ocr_attempts: dict[int, int] = {}
        self.ocr_max_attempts: int = 10

    def prune(self, active_ids: set[int]) -> None:
        stale = self.ocr_cache.keys() - active_ids
        for tid in stale:
            del self.ocr_cache[tid]
            self.ocr_attempts.pop(tid, None)

    def get_plate_for_track(self, track_id: int, vehicle_crop: np.ndarray) -> PlateResult:
        if track_id in self.ocr_cache:
            return self.ocr_cache[track_id]
        self.ocr_attempts[track_id] = self.ocr_attempts.get(track_id, 0) + 1
        if self.ocr_attempts[track_id] > self.ocr_max_attempts:
            negative = PlateResult(plate=None, confidence=0.0)
            self.ocr_cache[track_id] = negative
            return negative
        if vehicle_crop.size == 0:
            return PlateResult(None, 0.0)

        if self._profiler:
            self._profiler.start("plate_detect")
        plate_detections = self.plate_detector.detect(vehicle_crop)
        if self._profiler:
            self._profiler.stop("plate_detect")

        if not plate_detections:
            return PlateResult(None, 0.0)

        best_det = max(plate_detections, key=lambda d: d.score)
        x1, y1, x2, y2 = map(int, [best_det.x1, best_det.y1, best_det.x2, best_det.y2])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(vehicle_crop.shape[1], x2), min(vehicle_crop.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return PlateResult(None, 0.0)

        plate_crop = vehicle_crop[y1:y2, x1:x2]

        if self._profiler:
            self._profiler.start("text_recognize")
        plate_text, ocr_conf = self.ocr_engine.recognize_and_validate(plate_crop)
        if self._profiler:
            self._profiler.stop("text_recognize")

        result = PlateResult(plate=plate_text, confidence=ocr_conf)
        if plate_text:
            self.ocr_cache[track_id] = result
        return result
