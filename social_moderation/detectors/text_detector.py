import easyocr
import cv2
import numpy as np
import logging
import torch
import re

logger = logging.getLogger(__name__)

class TextDetector:
    def __init__(self, languages=['en'], gpu=False):
        """Enhanced EasyOCR text detector with word-level precision"""
        self.gpu = gpu and torch.cuda.is_available()
        logger.info(f"Initializing EasyOCR with GPU: {self.gpu}")
        try:
            self.reader = easyocr.Reader(languages, gpu=self.gpu)
            logger.info("✓ EasyOCR initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            raise

    def detect_text(self, image, confidence_threshold=0.5):
        """Detect text regions - returns full text blocks"""
        if image is None or image.size == 0:
            return []

        try:
            # Convert to RGB for EasyOCR
            if len(image.shape) == 3 and image.shape[2] == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image

            # Perform detection
            results = self.reader.readtext(rgb_image, detail=1)
            
            # Parse results into consistent format
            detections = []
            for (bbox, text, confidence) in results:
                if confidence >= confidence_threshold and text.strip():
                    # Convert bbox from [[x,y], [x,y], ...] to [x1, y1, x2, y2]
                    bbox_array = np.array(bbox)
                    x1, y1 = bbox_array.min(axis=0)
                    x2, y2 = bbox_array.max(axis=0)
                    detections.append(([x1, y1, x2, y2], text.strip(), confidence))
            
            return detections

        except Exception as e:
            logger.error(f"Text detection failed: {e}")
            return []

    def get_text_regions(self, image, confidence_threshold=0.3):
        """Get text regions with enhanced filtering"""
        detections = self.detect_text(image, confidence_threshold)
        return detections

    def detect_text_with_angles(self, image, confidence_threshold=0.5):
        """Detect text with rotation angles"""
        if image is None or image.size == 0:
            return []

        try:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image
            results = self.reader.readtext(rgb_image, detail=1)
            detections = []
            for (bbox, text, confidence) in results:
                if confidence >= confidence_threshold and text.strip():
                    bbox_array = np.array(bbox)
                    x1, y1 = bbox_array.min(axis=0)
                    x2, y2 = bbox_array.max(axis=0)
                    detections.append(([x1, y1, x2, y2], text.strip(), confidence, bbox))
            return detections
        except Exception as e:
            logger.error(f"Text detection with angles failed: {e}")
            return []
