import cv2
import numpy as np
from dataclasses import dataclass
from collections import Counter


@dataclass(slots=True)
class ColorResult:
    color_name: str
    confidence: float


class ColorDetector:
    """
    Detects the dominant color of a vehicle using HSV color space thresholding.
    Optimized for CPU usage, avoiding heavy ML models.
    """

    def __init__(self):
        # Define HSV color ranges using dictionaries.
        # Format: Lower Bound, Upper Bound for Hue, Saturation, Value.
        # HSV in OpenCV: H [0, 179], S [0, 255], V [0, 255]
        self.color_ranges = {
            "white": [
                (np.array([0, 0, 200]), np.array([179, 30, 255]))  # Low saturation, high value
            ],
            "black": [
                 (np.array([0, 0, 0]), np.array([179, 255, 30])) # Low value regardless of saturation/hue
            ],
            "gray": [
                 (np.array([0, 0, 50]), np.array([179, 50, 220])) # Low saturation, medium value
            ],
            "red": [
                (np.array([0, 100, 100]), np.array([10, 255, 255])),
                (np.array([160, 100, 100]), np.array([179, 255, 255]))
            ],
            "blue": [
                (np.array([100, 150, 0]), np.array([140, 255, 255]))
            ],
            "green": [
                (np.array([36, 100, 100]), np.array([86, 255, 255]))
            ],
            "yellow": [
                (np.array([20, 100, 100]), np.array([35, 255, 255]))
            ]
        }
        
    def detect_color(self, image: np.ndarray) -> str:
        """
        Detects the dominant color in the given image.
        
        Args:
            image: BGR image (numpy array)
            
        Returns:
            The name of the dominant color.
        """
        if image is None or image.size == 0:
            return "unknown"

        # Crop central 50%
        h, w = image.shape[:2]
        crop_h, crop_w = int(h * 0.5), int(w * 0.5)
        start_y, start_x = int(h * 0.25), int(w * 0.25)
        
        center_crop = image[start_y:start_y+crop_h, start_x:start_x+crop_w]
        
        if center_crop.size == 0:
             # Fallback if crop failed somehow (e.g. tiny image)
             center_crop = image

        # Convert to HSV
        hsv_img = cv2.cvtColor(center_crop, cv2.COLOR_BGR2HSV)
        
        max_pixels = 0
        dominant_color = "unknown"
        
        for color_name, ranges in self.color_ranges.items():
            mask_count = 0
            for (lower, upper) in ranges:
                mask = cv2.inRange(hsv_img, lower, upper)
                mask_count += cv2.countNonZero(mask)
            
            if mask_count > max_pixels:
                max_pixels = mask_count
                dominant_color = color_name
                
        return dominant_color


class VehicleColorDetector:
    """Wrapper for ColorDetector to integrate with application tracks."""
    def __init__(self) -> None:
        self._detector = ColorDetector()
        self._cache: dict[int, ColorResult] = {}

    def detect(self, track_id: int, crop: np.ndarray) -> ColorResult:
        if track_id in self._cache:
            return self._cache[track_id]
        
        color_name = self._detector.detect_color(crop)
        # Simple confidence metric could be added, but passing 1.0/0.0 for now basic implementation
        confidence = 0.0 if color_name == "unknown" else 1.0 
        
        result = ColorResult(color_name=color_name, confidence=confidence)
        self._cache[track_id] = result
        return result
