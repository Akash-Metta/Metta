import cv2
import numpy as np
import logging
import re

logger = logging.getLogger(__name__)

class TextBlurrer:
    """Text blurrer for toxic text detection and blurring."""
    
    def __init__(self, detector, config: dict, offensive_words=None):
        self.detector = detector
        self.config = config
        self.offensive_words = offensive_words or self._default_offensive_words()
        logger.info("TextBlurrer initialized (simple version for debugging)")
    
    def _default_offensive_words(self):
        return {'hate', 'kill', 'die', 'fuck', 'shit', 'bitch', 'damn', 'hell'}
    
    def _is_toxic(self, text):
        """Check if text is toxic."""
        if not text:
            return False
        text_lower = text.lower()
        # Check for "hate" keyword
        if 'hate' in text_lower:
            logger.info(f"✅✅✅ DETECTED HATE WORD IN: '{text}'")
            return True
        # Check for other offensive words
        if any(word in text_lower for word in self.offensive_words):
            logger.info(f"✅✅✅ DETECTED OFFENSIVE WORD IN: '{text}'")
            return True
        return False
    
    def blur_toxic_text(self, image, confidence=0.5):
        """Blur toxic text in image."""
        logger.info(f"\n🔍 TEXT BLUR CALLED WITH IMAGE SIZE: {image.shape}")
        
        if image is None or image.size == 0:
            return image
        
        result = image.copy()
        H, W = image.shape[:2]
        
        # Detect text
        detections = self.detector.detect_text(image, confidence)
        logger.info(f"📝 DETECTIONS FOUND: {len(detections) if detections else 0}")
        
        if not detections:
            logger.warning("⚠️  NO TEXT DETECTIONS!")
            return result
        
        blurred = 0
        for idx, det in enumerate(detections):
            try:
                bbox, text, conf = det
                x1, y1, x2, y2 = bbox
                
                logger.info(f"\n📄 Detection {idx}: '{text}' at ({x1},{y1},{x2},{y2})")
                
                if self._is_toxic(text):
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
                    
                    if x2 > x1 and y2 > y1:
                        roi = result[y1:y2, x1:x2]
                        if roi.size > 0:
                            result[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (51, 51), 0)
                            blurred += 1
                            logger.info(f"✅ BLURRED: '{text}'")
            except Exception as e:
                logger.error(f"Error: {e}")
        
        logger.info(f"\n🎉 TOTAL BLURRED: {blurred}")
        return result
    
    def blur_hate_text(self, image, conf=0.5):
        return self.blur_toxic_text(image, conf)
    
    def blur_hate_text_simple(self, image, conf=0.5):
        return self.blur_toxic_text(image, conf)
