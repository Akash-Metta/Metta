"""
NSFW and Blood Detection Blurrer
"""

import cv2
import numpy as np
import logging
from detectors.nsfw_detector import NSFWDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NSFWBlurrer:
    """Blur blood/NSFW/violence content."""
    
    def __init__(self, blur_strength=(51, 51), blur_type='gaussian', blood_threshold=0.5):
        """Initialize blurrer with blood_threshold parameter."""
        self.detector = NSFWDetector(blood_threshold=blood_threshold)
        
        # Ensure blur strength is odd
        self.blur_strength = (
            blur_strength[0] + 1 if blur_strength[0] % 2 == 0 else blur_strength[0],
            blur_strength[1] + 1 if blur_strength[1] % 2 == 0 else blur_strength[1]
        )
        
        self.blur_type = blur_type
        logger.info(f"✓ NSFWBlurrer initialized ({blur_type}, blood_threshold={blood_threshold})")
    
    def _apply_gaussian_blur(self, image):
        return cv2.GaussianBlur(image, self.blur_strength, 0)
    
    def _apply_pixelate(self, image, pixel_size=20):
        h, w = image.shape[:2]
        small = cv2.resize(image, (w // pixel_size, h // pixel_size), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    
    def _apply_mosaic(self, image, block_size=8):
        h, w = image.shape[:2]
        small = cv2.resize(image, (w // block_size, h // block_size), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    
    def _apply_black_censor(self, image):
        return np.zeros_like(image)
    
    def blur_unsafe_content(self, image, add_warning=True):
        """Detect and blur unsafe content."""
        analysis = self.detector.analyze(image)
        result = image.copy()
        
        if not analysis['is_safe']:
            logger.info(f"🔴 UNSAFE: {analysis['flags']}")
            
            if self.blur_type == 'gaussian':
                result = self._apply_gaussian_blur(result)
            elif self.blur_type == 'pixelate':
                result = self._apply_pixelate(result)
            elif self.blur_type == 'mosaic':
                result = self._apply_mosaic(result)
            elif self.blur_type == 'black':
                result = self._apply_black_censor(result)
            
            if add_warning:
                h, w = result.shape[:2]
                cv2.rectangle(result, (0, 0), (w, 50), (0, 0, 255), -1)
                flag_text = ', '.join(analysis['flags']).upper()
                cv2.putText(result, f"UNSAFE: {flag_text}", 
                           (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            logger.info(f"✅ BLURRED with {self.blur_type}")
        else:
            logger.info("✓ Content is safe")
        
        return {
            'image': result,
            'analysis': analysis
        }
    
    def blur_selective_regions(self, image):
        return self.blur_unsafe_content(image, add_warning=False)
    
    def _apply_blur_effect(self, image):
        return self._apply_gaussian_blur(image)
    
    def get_statistics(self, analysis):
        if not analysis:
            return None
        return analysis
