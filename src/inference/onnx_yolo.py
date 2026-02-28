from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


@dataclass(slots=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int


class YOLOv10ONNX:
    def __init__(
        self,
        model_path: str | Path,
        conf_threshold: float,
        providers: list[str] | None = None,
        input_size: tuple[int, int] = (640, 640),
        allowed_class_ids: tuple[int, ...] | None = (2, 3, 5, 7),
    ) -> None:
        self.model_path = str(model_path)
        self.conf_threshold = conf_threshold
        self.input_size = input_size
        self.allowed_class_ids = allowed_class_ids
        
        # Enforce CPU execution for edge compatibility
        self.session = ort.InferenceSession(
            self.model_path, 
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float, float]:
        """
        Optimized CPU preprocessing.
        """
        h, w = frame.shape[:2]
        input_w, input_h = self.input_size
        
        # Fast resizing
        resized = cv2.resize(frame, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
        
        # Vectorized normalization and transposition (HWC -> CHW)
        # Replacing slow loop-based operations with direct numpy manipulation
        img = resized.astype(np.float32)
        img /= 255.0
        
        # Add batch dimension: (1, 3, 640, 640)
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)

        scale_x = w / float(input_w)
        scale_y = h / float(input_h)
        return img, scale_x, scale_y

    def run(self, input_tensor: np.ndarray) -> np.ndarray:
        outputs = self.session.run(None, {self.input_name: input_tensor})
        return outputs[0]

    def postprocess(
        self,
        raw_output: np.ndarray,
        original_shape: tuple[int, ...],
        scale_x: float,
        scale_y: float,
    ) -> np.ndarray:
        """
        Post-processing adapted for YOLOv10 (NMS-free) output.
        Format varies by export, typically (1, 300, 6) or (1, 6, 300).
        Goal: Return Nx6 matrix [x1, y1, x2, y2, conf, class_id]
        """
        # Remove batch dim: (1, 300, 6) -> (300, 6)
        pred = np.squeeze(raw_output)

        # Handle different output shapes if necessary (transpose if 6xN)
        if pred.shape[0] == 6 and pred.shape[1] > 6:
            pred = pred.T

        # If empty or wrong shape
        if pred.ndim < 2 or pred.shape[1] < 6:
             return np.empty((0, 6), dtype=np.float32)

        # Filtering mask (vectorized is faster than loop)
        # Columns: 0=x1, 1=y1, 2=x2, 3=y2, 4=conf, 5=class
        
        # 1. Confidence threshold
        mask = pred[:, 4] >= self.conf_threshold
        pred = pred[mask]

        # 2. Class filtering
        if self.allowed_class_ids is not None:
             # Check if class_id (col 5) is in allowed list
             class_mask = np.isin(pred[:, 5], self.allowed_class_ids)
             pred = pred[class_mask]

        if len(pred) == 0:
            return np.empty((0, 6), dtype=np.float32)

        # 3. Rescaling coordinates
        # Using in-place multiplication for speed
        pred[:, 0] *= scale_x  # x1
        pred[:, 1] *= scale_y  # y1
        pred[:, 2] *= scale_x  # x2
        pred[:, 3] *= scale_y  # y2

        # 4. Clipping to image boundaries
        h, w = original_shape[:2]
        np.clip(pred[:, 0], 0, w, out=pred[:, 0])
        np.clip(pred[:, 2], 0, w, out=pred[:, 2])
        np.clip(pred[:, 1], 0, h, out=pred[:, 1])
        np.clip(pred[:, 3], 0, h, out=pred[:, 3])

        return pred

    def detect(self, frame: np.ndarray) -> np.ndarray:
        input_tensor, scale_x, scale_y = self.preprocess(frame)
        raw_output = self.run(input_tensor)
        return self.postprocess(raw_output, frame.shape[:2], scale_x, scale_y)


class ONNXYoloDetector:
    def __init__(
        self,
        model_path: str | Path,
        providers: list[str] | None,
        conf_threshold: float,
        nms_threshold: float,
        input_size: tuple[int, int] = (640, 640),
        allowed_class_ids: tuple[int, ...] | None = None,
    ) -> None:
        # Compatibility wrapper: providers arg is ignored, enforced CPU
        self._detector = YOLOv10ONNX(
            model_path=model_path,
            conf_threshold=conf_threshold,
            input_size=input_size,
            allowed_class_ids=allowed_class_ids,
        )

    def detect_matrix(self, frame: np.ndarray) -> np.ndarray:
        return self._detector.detect(frame)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        matrix = self.detect_matrix(frame)
        # Convert matrix to list of Detection objects
        return [
            Detection(
                x1=row[0], y1=row[1], x2=row[2], y2=row[3],
                score=row[4], class_id=int(row[5])
            )
            for row in matrix
        ]