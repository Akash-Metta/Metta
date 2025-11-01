"""
OpenCV Haar Cascade Face Detector (Fallback)
Lightweight CPU-based face detection
"""

import cv2
import logging
import numpy as np
from typing import List, Tuple

logger = logging.getLogger(__name__)

class OpenCVFace:
    """OpenCV Haar Cascade face detector as lightweight fallback."""
    
    def __init__(self):
        """Initialize Haar Cascade classifier."""
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            if self.face_cascade.empty():
                raise IOError("Failed to load Haar cascade")
            
            logger.info("✅ OpenCV Haar Cascade face detector loaded")
            
        except Exception as e:
            logger.error(f"Failed to load OpenCV face detector: {e}")
            raise
    
    def detect_faces(self, image: np.ndarray, 
                    conf_threshold: float = 0.5) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect faces using Haar Cascade.
        
        Args:
            image: Input image (BGR)
            conf_threshold: Unused (kept for API consistency)
            
        Returns:
            List of (x1, y1, x2, y2, confidence)
        """
        if image is None or image.size == 0:
            return []
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            # Convert to (x1, y1, x2, y2, conf) format
            detections = []
            for (x, y, w, h) in faces:
                detections.append((x, y, x + w, y + h, 0.85))  # Fixed confidence
            
            logger.debug(f"OpenCV detected {len(detections)} faces")
            return detections
            
        except Exception as e:
            logger.error(f"OpenCV face detection failed: {e}")
            return []
