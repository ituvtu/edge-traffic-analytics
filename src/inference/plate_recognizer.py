import cv2
import numpy as np
import onnxruntime as ort
import re
from typing import List, Optional, Tuple


class PlateRecognizer:
    def __init__(
        self,
        det_model_path: str,
        rec_model_path: str,
        char_dict_path: str,
        det_db_thresh: float = 0.3,
        det_db_box_thresh: float = 0.6,
        det_db_unclip_ratio: float = 1.5,
        rec_batch_num: int = 6,
        regex_patterns: Optional[List[str]] = None,
    ):
        providers = ["CPUExecutionProvider"]
        
        # Load Detection Model
        self.det_sess = ort.InferenceSession(det_model_path, providers=providers)
        self.det_input_name = self.det_sess.get_inputs()[0].name

        # Load Recognition Model
        self.rec_sess = ort.InferenceSession(rec_model_path, providers=providers)
        self.rec_input_name = self.rec_sess.get_inputs()[0].name
        
        # Load Character Dictionary
        self.character_str = []
        with open(char_dict_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                line = line.strip("\n").replace("\r", "")
                self.character_str.append(line)
        # Space is appended as an extra entry; the true CTC blank is the last
        # logit index (num_classes - 1) resolved at decode time, not via this list.
        self.character_str.append(" ")

        # Parameters
        self.det_db_thresh = det_db_thresh
        self.det_db_box_thresh = det_db_box_thresh
        self.det_db_unclip_ratio = det_db_unclip_ratio
        self.rec_image_shape = [3, 48, 320]

        # Compile validation patterns (support Latin + Cyrillic plates)
        _patterns = regex_patterns or [r"^[A-ZА-ЯІЇЄ0-9]{3,10}$"]
        self._plate_patterns = [re.compile(p) for p in _patterns]

    def _resize_img(self, img: np.ndarray, limit_side_len: int = 960) -> Tuple[np.ndarray, float, float]:
        """Resize image for detection model (multiple of 32)."""
        h, w, c = img.shape
        ratio = 1.0
        if max(h, w) > limit_side_len:
            if h > w:
                ratio = float(limit_side_len) / h
            else:
                ratio = float(limit_side_len) / w
        else:
            ratio = 1.0
            
        resize_h = int(h * ratio)
        resize_w = int(w * ratio)
        
        resize_h = max(int(round(resize_h / 32) * 32), 32)
        resize_w = max(int(round(resize_w / 32) * 32), 32)
        
        img = cv2.resize(img, (resize_w, resize_h))
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)
        return img, ratio_h, ratio_w

    def _normalize(self, img: np.ndarray, mean: List[float], std: List[float]) -> np.ndarray:
        img = img.astype(np.float32)
        img /= 255.0
        img -= mean
        img /= std
        return img

    def _box_score_fast(self, bitmap: np.ndarray, _box: np.ndarray) -> float:
        """Calculate score of a box based on the bitmap."""
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, [box.reshape(-1, 1, 2).astype(np.int32)], (1,))
        return cv2.mean(bitmap[ymin : ymax + 1, xmin : xmax + 1], mask)[0]

    def detect_text(self, img: np.ndarray) -> List[np.ndarray]:
        """Run text detection model."""
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Preprocessing
        img_resize, ratio_h, ratio_w = self._resize_img(img)
        
        # Mean/Std for detection generally: [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img_norm = self._normalize(img_resize, mean, std)
        
        # HWC -> CHW -> NCHW
        img_norm = img_norm.transpose(2, 0, 1)
        img_norm = np.expand_dims(img_norm, axis=0)
        
        # Inference
        outputs = self.det_sess.run(None, {self.det_input_name: img_norm})
        
        # Postprocessing
        # Output is usually probability map: [1, 1, H, W]
        # Depending on model export, sometimes [1, H, W]
        pred = outputs[0]
        if len(pred.shape) == 4:
            pred = pred[0, 0, :, :]
        elif len(pred.shape) == 3:
            pred = pred[0, :, :]
            
        segmentation = pred > self.det_db_thresh
        
        boxes = []
        bitmap = (pred * 255).astype(np.uint8) # for box score
        
        # Find contours
        # cv2.findContours on binary mask
        mask = (segmentation * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            points = contour.reshape((-1, 2))
            if points.shape[0] < 4: 
                continue
                
            score = self._box_score_fast(pred, points)
            if self.det_db_box_thresh > score:
                continue
                
            # Get minAreaRect
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            
            # Unclip (expand) - Need simple implementation if pyclipper not allowed
            # Using simple expansion logic:
            # Expand box by det_db_unclip_ratio relative to its size
            # Since we have minAreaRect, we can just expand width/height
            center, size, angle = rect
            w_rect, h_rect = size
            # Approx logic from DBNet: distance = area * unclip_ratio / perimeter
            area = w_rect * h_rect
            perimeter = 2 * (w_rect + h_rect)
            distance = area * self.det_db_unclip_ratio / perimeter
            
            # New size
            new_w = w_rect + 2 * distance
            new_h = h_rect + 2 * distance
            new_rect = (center, (new_w, new_h), angle)
            box = cv2.boxPoints(new_rect)
            
            box = np.array(box).reshape(-1, 2)
            
            # Rescale back to original image
            box[:, 0] /= ratio_w
            box[:, 1] /= ratio_h
            
            # Clip to image bounds
            h_orig, w_orig = img.shape[:2]
            box[:, 0] = np.clip(box[:, 0], 0, w_orig)
            box[:, 1] = np.clip(box[:, 1], 0, h_orig)
            
            boxes.append(box.astype(np.int32))
            
        return boxes

    def recognize_text(self, img_crop: np.ndarray) -> Tuple[str, float]:
        """Run text recognition model on cropped image."""
        # Convert BGR to RGB
        img_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)
        
        h, w = img_crop.shape[:2]
        
        # Resize for recognition (maintain aspect ratio, pad if needed)
        # Target shape: 3, 48, 320 (CHW)
        target_h = 48
        target_w = 320
        
        ratio = w / float(h)
        # Usually we resize height to 48 and scale width proportionally
        new_w = int(target_h * ratio)
        if new_w > target_w:
            new_w = target_w
            
        img_resize = cv2.resize(img_crop, (new_w, target_h))
        
        # Preprocessing: normalize -> pad
        # Rec mean/std typically just 0.5, 0.5, 0.5 for PaddleOCR rec models
        img_norm = self._normalize(img_resize, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        
        # HWC -> CHW
        img_norm = img_norm.transpose(2, 0, 1)
        
        # Pad width to 320
        padded_img = np.zeros((3, target_h, target_w), dtype=np.float32)
        padded_img[:, :, :new_w] = img_norm
        
        # NCHW
        padded_img = np.expand_dims(padded_img, axis=0)
        
        # Inference
        outputs = self.rec_sess.run(None, {self.rec_input_name: padded_img})
        preds = outputs[0] # Shape: [Batch, Time, NumClasses] -> usually [1, 80, 6625]
        
        # CTC Decoding
        text, conf = self._ctc_decode(preds)
        return text, conf
        
    def _ctc_decode(self, preds: np.ndarray) -> Tuple[str, float]:
        """Greedy CTC decoder implemented in pure NumPy.

        PaddleOCR convention (PP-OCRv3/v4 rec):
          - logit index 0  → CTC blank
          - logit index i  → character_str[i - 1]   (1-based shift)
        The model outputs N+1 logits for a dictionary of N characters.
        """
        if len(preds.shape) == 3:
            preds = preds[0]  # [1, T, C] -> [T, C]

        indices = np.argmax(preds, axis=1)   # [T]
        max_probs = np.max(preds, axis=1)    # [T]
        blank_idx = 0                        # PaddleOCR always puts blank at 0

        char_list: List[str] = []
        conf_list: List[float] = []
        prev_idx = -1

        for i, idx in enumerate(indices):
            idx = int(idx)
            if idx == blank_idx:
                prev_idx = -1  # reset duplicate suppression on blank
                continue
            if idx != prev_idx:
                char_idx = idx - 1  # shift: logit[i] → character_str[i-1]
                if 0 <= char_idx < len(self.character_str):
                    char = self.character_str[char_idx]
                    if char != " ":
                        char_list.append(char)
                        conf_list.append(float(max_probs[i]))
            prev_idx = idx

        text = "".join(char_list)
        avg_conf = float(np.mean(conf_list)) if conf_list else 0.0
        return text, avg_conf

    def recognize_and_validate(self, plate_crop: np.ndarray) -> Tuple[str, float]:
        """
        Run recognition on an already-isolated plate crop (no DBNet step).

        The image is stretched to the recognition model's expected height so
        all text is visible, then CTC-decoded and regex-validated.
        """
        if plate_crop is None or plate_crop.size == 0:
            return "", 0.0

        # Ensure minimum readable size — stretch short dimension to 48 px
        h, w = plate_crop.shape[:2]
        if h < 8 or w < 8:
            return "", 0.0

        target_h = self.rec_image_shape[1]   # 48
        target_w = self.rec_image_shape[2]   # 320
        scale = target_h / h
        new_w = min(int(w * scale), target_w)
        resized = cv2.resize(plate_crop, (new_w, target_h), interpolation=cv2.INTER_CUBIC)

        # BGR → RGB, normalise, pad to target_w, make NCHW
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        norm = self._normalize(rgb, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        chw = norm.transpose(2, 0, 1)                              # (3, 48, new_w)
        padded = np.zeros((3, target_h, target_w), dtype=np.float32)
        padded[:, :, :new_w] = chw
        inp = np.expand_dims(padded, axis=0)                       # (1, 3, 48, 320)

        outputs = self.rec_sess.run(None, {self.rec_input_name: inp})
        text, conf = self._ctc_decode(outputs[0])

        text_clean = text.replace(" ", "").upper()
        if self._validate_plate(text_clean):
            return text_clean, conf
        return "", 0.0

    def full_pipeline(self, img: np.ndarray) -> Tuple[str, float]:
        """Detect and recognize text in image (e.g. cropped plate)."""
        boxes = self.detect_text(img)
        
        # Sort boxes if multiple found? For license plate, usually centralized one matters.
        # Or simplistic approach: Take largest box or most central.
        if not boxes:
             return "", 0.0
             
        # Take the box with largest area
        best_box = max(boxes, key=lambda b: cv2.contourArea(b))
        
        # Crop using perspective transform
        w = int(np.linalg.norm(best_box[0] - best_box[1]))
        h = int(np.linalg.norm(best_box[0] - best_box[3]))
        src_pts = best_box.astype("float32")
        dst_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype="float32")
        
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        crop = cv2.warpPerspective(img, M, (w, h))
        
        # Recognize
        text, conf = self.recognize_text(crop)
        
        # Validation
        if self._validate_plate(text):
            return text, conf
        return "", 0.0

    def _validate_plate(self, text: str) -> bool:
        """Validate result against any of the configured regex patterns."""
        text = text.replace(" ", "")
        if len(text) < 3 or len(text) > 10:
            return False
        return any(p.match(text) for p in self._plate_patterns)

