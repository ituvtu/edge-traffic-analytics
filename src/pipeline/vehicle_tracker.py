from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from src.inference.onnx_yolo import Detection
from src.pipeline.tracker_core import BYTETracker


@dataclass
class Track:
    track_id: int
    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int = -1  # Placeholder, ByteTrack doesn't inherently track class without modification
    age: int = 0
    lost: int = 0


class ByteTrackArgs:
    def __init__(self, track_thresh=0.5, track_buffer=30, match_thresh=0.8, mot20=False):
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.mot20 = mot20


class VehicleTracker:
    def __init__(
        self,
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        frame_rate: int = 30
    ) -> None:
        args = ByteTrackArgs(
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh
        )
        self.tracker = BYTETracker(args, frame_rate=frame_rate)

    def update(self, detections: List[Detection], frame_shape: tuple[int, int]) -> List[Track]:
        """
        Update tracker with detections.
        
        Args:
            detections: List of Detection objects from inference
            frame_shape: (height, width) of the frame
        """
        if not detections:
            # Propagate tracks even if no detections
            # ByteTracker handles empty detections internally, but we need to pass empty array
            dets = np.zeros((0, 5), dtype=np.float32)
        else:
            # Convert detections to numpy array [x1, y1, x2, y2, score]
            dets = np.array([
                [d.x1, d.y1, d.x2, d.y2, d.score] 
                for d in detections
            ], dtype=np.float32)

        # ByteTracker.update method signature:
        # def update(self, output_results, img_info, img_size):
        # img_info: [height, width, scale_factor] - scale_factor seems unused in simplified version or we pass 1.0
        # img_size: [height, width]
        
        height, width = frame_shape
        img_info = [height, width, 1.0] # Scale is 1.0 as we handle scaling in inference or before
        img_size = [height, width]

        online_targets = self.tracker.update(dets, img_info, img_size)

        tracks = []
        for t in online_targets:
            tlbr = t.tlbr
            # ByteTrack STrack doesn't store class_id by default unless we modify it.
            # Only score and bbox.
            track = Track(
                track_id=t.track_id,
                bbox=(float(tlbr[0]), float(tlbr[1]), float(tlbr[2]), float(tlbr[3])),
                confidence=float(t.score),
                class_id=-1, # Unknown
                age=0, # Not exposed directly
                lost=0 # Not exposed directly
            )
            tracks.append(track)
            
        return tracks

    @staticmethod
    def crop_from_track(frame: np.ndarray, track: Track) -> np.ndarray:
        x1, y1, x2, y2 = map(int, track.bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return np.empty((0, 0, 3), dtype=np.uint8)
        return frame[y1:y2, x1:x2]
