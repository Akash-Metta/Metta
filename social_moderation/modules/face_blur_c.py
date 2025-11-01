import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

def _ensure_int_coords(box):
    return tuple(int(round(v)) for v in box)

class FaceBlurrer:
    def __init__(self, detector, config: dict):
        self.detector = detector
        self.device = config.get("device", "cpu")
        self.min_kernel = config["blur"].get("gaussian_min_kernel", 15)
        self.max_kernel = config["blur"].get("gaussian_max_kernel", 121)
        self.blur_method = config["blur"].get("method", "gaussian")
        self.conf_threshold = config["face_detector"].get("conf_threshold", 0.35)
        self.padding_ratio = 0.05

    def _clamp_box(self, x1, y1, x2, y2, H, W):
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        return x1, y1, x2, y2

    def _pad_box(self, x1, y1, x2, y2, H, W):
        padx = int((x2 - x1) * self.padding_ratio)
        pady = int((y2 - y1) * self.padding_ratio)
        return self._clamp_box(x1 - padx, y1 - pady, x2 + padx, y2 + pady, H, W)

    def _adaptive_kernel(self, w, h):
        k = max(self.min_kernel, min(self.max_kernel, int(max(w, h) // 3)))
        if k % 2 == 0: k += 1
        return k

    def _mosaic(self, roi, block_size):
        h, w = roi.shape[:2]
        small = cv2.resize(roi, (max(1, w // block_size), max(1, h // block_size)))
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    def blur_faces(self, image):
        if image is None or image.size == 0:
            return image

        H, W = image.shape[:2]
        detections = self.detector.detect_faces(image, self.conf_threshold)
        for det in detections:
            box = det[:4]
            x1, y1, x2, y2 = _ensure_int_coords(box)
            x1, y1, x2, y2 = self._pad_box(x1, y1, x2, y2, H, W)
            roi = image[y1:y2, x1:x2]
            if roi.size == 0: continue

            if self.blur_method == "gaussian":
                k = self._adaptive_kernel(x2 - x1, y2 - y1)
                blurred = cv2.GaussianBlur(roi, (k, k), 0)
            else:
                blk = max(6, (x2 - x1) // 12)
                blurred = self._mosaic(roi, blk)
            image[y1:y2, x1:x2] = blurred
        return image
