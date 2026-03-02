from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

def _create_session_options() -> ort.SessionOptions:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 4
    return options

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
        self.session = ort.InferenceSession(
            self.model_path,
            sess_options=_create_session_options(),
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name

    def preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float, float]:
        h, w = frame.shape[:2]
        input_w, input_h = self.input_size
        resized = cv2.resize(frame, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
        img = resized.astype(np.float32)
        img /= 255.0
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
        pred = np.squeeze(raw_output)

        if pred.shape[0] == 6 and pred.shape[1] > 6:
            pred = pred.T

        if pred.ndim < 2 or pred.shape[1] < 6:
            return np.empty((0, 6), dtype=np.float32)

        mask = pred[:, 4] >= self.conf_threshold
        pred = pred[mask]

        if self.allowed_class_ids is not None:
            class_mask = np.isin(pred[:, 5], self.allowed_class_ids)
            pred = pred[class_mask]

        if len(pred) == 0:
            return np.empty((0, 6), dtype=np.float32)

        pred[:, 0] *= scale_x
        pred[:, 1] *= scale_y
        pred[:, 2] *= scale_x
        pred[:, 3] *= scale_y

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
        return [
            Detection(
                x1=row[0], y1=row[1], x2=row[2], y2=row[3],
                score=row[4], class_id=int(row[5])
            )
            for row in matrix
        ]
