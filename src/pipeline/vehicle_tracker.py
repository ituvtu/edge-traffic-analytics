from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from src.inference.onnx_yolo import Detection
from src.pipeline.tracker_core import BYTETracker


def _bbox_iou(
    ax1: float, ay1: float, ax2: float, ay2: float,
    bx1: float, by1: float, bx2: float, by2: float,
) -> float:
    """Fast axis-aligned IoU for class-id matching."""
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    if inter == 0.0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0.0 else 0.0


@dataclass
class Track:
    track_id: int
    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int = -1
    age: int = 0
    lost: int = 0


class ByteTrackArgs:
    def __init__(self, track_thresh=0.5, track_buffer=30, match_thresh=0.8, mot20=False):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.mot20 = mot20


class VehicleTracker:
    _MIN_CLASS_MATCH_IOU: float = 0.15  # minimum bbox overlap to accept class-id assignment
    def __init__(
        self,
        track_thresh: float = 0.5,
        track_buffer: int = 90,
        match_thresh: float = 0.65,
        frame_rate: int = 30
    ) -> None:
        self._track_thresh = track_thresh
        self._track_buffer = track_buffer
        self._match_thresh = match_thresh
        self._frame_rate = frame_rate
        args = ByteTrackArgs(
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh
        )
        self.tracker = BYTETracker(args, frame_rate=frame_rate)
        self._track_class: dict[int, int] = {}

    def reset(self) -> None:
        args = ByteTrackArgs(
            track_thresh=self._track_thresh,
            track_buffer=self._track_buffer,
            match_thresh=self._match_thresh,
        )
        self.tracker = BYTETracker(args, frame_rate=self._frame_rate)
        self._track_class.clear()

    def update(self, detections: List[Detection], frame_shape: tuple[int, int]) -> List[Track]:
        if not detections:
            dets = np.zeros((0, 5), dtype=np.float32)
        else:
            dets = np.array([
                [d.x1, d.y1, d.x2, d.y2, d.score]
                for d in detections
            ], dtype=np.float32)

        height, width = frame_shape
        img_info = [height, width, 1.0]
        img_size = [height, width]

        online_targets = self.tracker.update(dets, img_info, img_size)

        tracks = []
        for t in online_targets:
            tlbr = t.tlbr
            track = Track(
                track_id=t.track_id,
                bbox=(float(tlbr[0]), float(tlbr[1]), float(tlbr[2]), float(tlbr[3])),
                confidence=float(t.score),
                class_id=-1,
                age=0,
                lost=0,
            )
            tracks.append(track)

        if detections:
            for track in tracks:
                x1, y1, x2, y2 = track.bbox
                best_iou = self._MIN_CLASS_MATCH_IOU
                for det in detections:
                    iou = _bbox_iou(x1, y1, x2, y2, det.x1, det.y1, det.x2, det.y2)
                    if iou > best_iou:
                        best_iou = iou
                        self._track_class[track.track_id] = det.class_id

        for track in tracks:
            track.class_id = self._track_class.get(track.track_id, -1)

        return tracks

    @staticmethod
    def crop_from_track(frame: np.ndarray, track: Track) -> np.ndarray:
        x1, y1, x2, y2 = map(int, track.bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return np.empty((0, 0, 3), dtype=np.uint8)
        return frame[y1:y2, x1:x2]
