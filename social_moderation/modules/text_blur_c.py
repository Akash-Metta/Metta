import cv2
import numpy as np
import logging
import re

logger = logging.getLogger(__name__)

def normalize_leet(word: str) -> str:
    w = word.lower()
    w = re.sub(r'[\W_]+', '', w)
    subs = {'1': 'i', '!': 'i', '@': 'a', '$': 's', '0': 'o', '3': 'e', '4': 'a', '7': 't', '5': 's'}
    for k, v in subs.items():
        w = w.replace(k, v)
    return w

class TextBlurrer:
    def __init__(self, detector, config: dict, offensive_words=None):
        self.detector = detector
        self.config = config
        self.padding_x = config["text_blur"].get("padding_x_ratio", 0.1)
        self.padding_y = config["text_blur"].get("padding_y_ratio", 0.15)
        self.blur_method = config["blur"].get("method", "gaussian")
        self.gaussian_min = config["blur"].get("gaussian_min_kernel", 15)
        self.gaussian_max = config["blur"].get("gaussian_max_kernel", 121)
        self.offensive = offensive_words or set()

    def _pad_box(self, x1, y1, x2, y2, H, W):
        padx = int((x2 - x1) * self.padding_x)
        pady = int((y2 - y1) * self.padding_y)
        return (max(0, x1 - padx), max(0, y1 - pady),
                min(W, x2 + padx), min(H, y2 + pady))

    def _adaptive_kernel(self, w):
        k = max(self.gaussian_min, min(self.gaussian_max, int(max(3, w // 3))))
        if k % 2 == 0: k += 1
        return k

    def _mosaic(self, roi, block_size):
        h, w = roi.shape[:2]
        small = cv2.resize(roi, (max(1, w // block_size), max(1, h // block_size)))
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

    def blur_text_simple(self, image, conf_threshold=0.5):
        if image is None or image.size == 0:
            return image
        H, W = image.shape[:2]
        detections = self.detector.detect_words_precise(image, conf_threshold)
        for (x1, y1, x2, y2), word, conf in detections:
            norm = normalize_leet(word)
            if any(bad in norm for bad in self.offensive):
                x1, y1, x2, y2 = self._pad_box(x1, y1, x2, y2, H, W)
                roi = image[y1:y2, x1:x2]
                if roi.size == 0: continue
                if self.blur_method == "gaussian":
                    k = self._adaptive_kernel(x2 - x1)
                    blurred = cv2.GaussianBlur(roi, (k, k), 0)
                else:
                    blk = max(4, (x2 - x1) // 8)
                    blurred = self._mosaic(roi, blk)
                image[y1:y2, x1:x2] = blurred
                logger.debug(f"Blurred word '{word}' normalized '{norm}'")
        return image
