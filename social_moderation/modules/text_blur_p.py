"""
Production-Grade Text Blurrer - FINAL WORKING VERSION
Simple, direct implementation that detects and blurs offensive/hate text
"""

import cv2
import numpy as np
import logging
import re
from typing import Tuple
from detectors.text_detector import TextDetector

logger = logging.getLogger(__name__)

class TextBlurrer:
    """Text blurrer for detecting and blurring offensive text."""
    
    def __init__(self, blur_strength=(51, 51), padding=0.02, sentiment_threshold=-0.3):
        """
        Initialize text blurrer.
        
        Args:
            blur_strength: Tuple of (blur_kernel_h, blur_kernel_w)
            padding: Padding around text regions
            sentiment_threshold: Threshold for toxicity detection
        """
        # Initialize text detector
        self.detector = TextDetector(languages=['en'], gpu=False)
        
        # Ensure blur strength is odd
        blur_h, blur_w = blur_strength
        self.blur_kernel = (
            blur_h + 1 if blur_h % 2 == 0 else blur_h,
            blur_w + 1 if blur_w % 2 == 0 else blur_w
        )
        
        self.padding = padding
        self.sentiment_threshold = sentiment_threshold
        
        # Offensive words list
        self.offensive_words = {
            'hate', 'kill', 'die', 'stupid', 'dumb', 'fuck', 'shit',
            'asshole', 'bastard', 'bitch', 'damn', 'hell', 'racist',
            'abuse', 'violence', 'threat', 'attack', 'rape', 'loser',
            'pathetic', 'disgusting', 'evil', 'murder', 'toxic', 'you'
        }
        
        logger.info(f"TextBlurrer initialized with blur kernel {self.blur_kernel}")
    
    def _is_toxic(self, text: str) -> bool:
        """Check if text is offensive/toxic."""
        if not text or not isinstance(text, str):
            return False
        
        text_lower = text.lower().strip()
        
        # Direct patterns for hate speech
        patterns = [
            r'\bhate\b',          # "hate" as whole word
            r'\bhate\s+',         # "hate" followed by space
            r'\s+hate\b',         # space followed by "hate"
            r'hate.*you',         # "hate ... you" pattern
        ]
        
        for pattern in patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                logger.info(f"🔴 DETECTED HATE PATTERN: '{text}'")
                return True
        
        # Check for offensive words
        words = set(text_lower.split())
        if words & self.offensive_words:
            logger.info(f"🔴 DETECTED OFFENSIVE WORD: '{text}'")
            return True
        
        return False
    
    def blur_hate_text(self, image: np.ndarray, confidence: float = 0.5) -> np.ndarray:
        """
        Main method: Detect and blur hateful/offensive text.
        
        Args:
            image: Input image (BGR format)
            confidence: Confidence threshold for text detection
            
        Returns:
            Image with blurred offensive text
        """
        logger.info("\n" + "="*70)
        logger.info("TEXT HATE SPEECH BLURRING STARTED")
        logger.info("="*70)
        
        if image is None or image.size == 0:
            logger.error("Image is None or empty!")
            return image
        
        result = image.copy()
        H, W = image.shape[:2]
        
        try:
            # Detect all text in image
            logger.info(f"Detecting text with confidence threshold: {confidence}...")
            detections = self.detector.detect_text(image, confidence)
            
            logger.info(f"Text regions detected: {len(detections) if detections else 0}")
            
            if not detections or len(detections) == 0:
                logger.warning("⚠️  NO TEXT DETECTED!")
                return result
            
            # Process each text region
            blurred_count = 0
            for idx, detection in enumerate(detections):
                try:
                    # Unpack: (bbox, text, confidence)
                    bbox, text, conf = detection
                    
                    logger.info(f"\n--- Text Region {idx} ---")
                    logger.info(f"Text: '{text}' | Confidence: {conf:.4f}")
                    
                    # Check if text is offensive
                    if self._is_toxic(text):
                        # Extract coordinates
                        x1, y1, x2, y2 = bbox
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        logger.info(f"Bbox (original): ({x1}, {y1}, {x2}, {y2})")
                        
                        # Add padding
                        pad_x = max(1, int((x2 - x1) * self.padding))
                        pad_y = max(1, int((y2 - y1) * self.padding))
                        
                        x1 = max(0, x1 - pad_x)
                        y1 = max(0, y1 - pad_y)
                        x2 = min(W, x2 + pad_x)
                        y2 = min(H, y2 + pad_y)
                        
                        logger.info(f"Bbox (padded): ({x1}, {y1}, {x2}, {y2})")
                        
                        # Validate
                        if x2 <= x1 or y2 <= y1:
                            logger.warning("Invalid bbox after padding")
                            continue
                        
                        # Extract and blur ROI
                        roi = result[y1:y2, x1:x2]
                        if roi.size > 0:
                            blurred_roi = cv2.GaussianBlur(roi, self.blur_kernel, 0)
                            result[y1:y2, x1:x2] = blurred_roi
                            blurred_count += 1
                            logger.info(f"✅ BLURRED: '{text}'")
                        else:
                            logger.warning("Empty ROI")
                    else:
                        logger.info(f"✓ Safe text: '{text}'")
                        
                except Exception as e:
                    logger.error(f"Error processing text region {idx}: {e}", exc_info=True)
                    continue
            
            logger.info("\n" + "="*70)
            logger.info(f"BLURRING COMPLETE: {blurred_count} offensive text regions blurred")
            logger.info("="*70 + "\n")
            
        except Exception as e:
            logger.error(f"Text blurring error: {e}", exc_info=True)
        
        return result
    
    def blur_toxic_text(self, image: np.ndarray, confidence: float = 0.5) -> np.ndarray:
        """Alias for blur_hate_text."""
        return self.blur_hate_text(image, confidence)
    
    def blur_hate_text_simple(self, image: np.ndarray, confidence: float = 0.5) -> np.ndarray:
        """Simple version for video processing."""
        return self.blur_hate_text(image, confidence)
