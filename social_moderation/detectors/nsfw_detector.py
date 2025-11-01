"""
Enhanced NSFW, Violence, and Blood Detection System
Multi-model approach with improved accuracy
"""

import torch
import cv2
import numpy as np
import logging
from transformers import pipeline, AutoModelForImageClassification, AutoFeatureExtractor
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NSFWDetector:
    """Multi-model NSFW, Violence, and Blood detection."""
    
    def __init__(self, nsfw_threshold=0.7, violence_threshold=0.6, blood_threshold=0.5, blood_percentage_threshold=8.0):
        """
        Initialize detector with configurable thresholds.
        
        Args:
            nsfw_threshold: NSFW confidence (0-1)
            violence_threshold: Violence confidence (0-1)
            blood_threshold: Blood/gore confidence (0-1)
            blood_percentage_threshold: Red pixel percentage threshold
        """
        self.nsfw_threshold = nsfw_threshold
        self.violence_threshold = violence_threshold
        self.blood_threshold = blood_threshold
        self.blood_percentage_threshold = blood_percentage_threshold
        
        self.device = 0 if torch.cuda.is_available() else -1
        logger.info(f"✓ Using device: {'GPU' if self.device == 0 else 'CPU'}")
        
        # Initialize models
        self.nsfw_model = None
        self.violence_model = None
        self._init_models()
        
        logger.info(f"✓ NSFW Detector initialized (NSFW:{nsfw_threshold}, Violence:{violence_threshold}, Blood:{blood_threshold})")
    
    def _init_models(self):
        """Initialize classification models."""
        try:
            self.nsfw_model = pipeline(
                "image-classification",
                model="Falconsai/nsfw_image_detection",
                device=self.device
            )
            logger.info("✓ NSFW model loaded")
        except Exception as e:
            logger.warning(f"⚠️  NSFW model failed: {e}")
            self.nsfw_model = None
        
        try:
            self.violence_model = pipeline(
                "image-classification",
                model="microbiophoton/Violence_Detection_Using_Deep_Learning",
                device=self.device
            )
            logger.info("✓ Violence model loaded")
        except Exception as e:
            logger.warning(f"⚠️  Violence model failed: {e}")
            self.violence_model = None
    
    def detect_blood_by_color(self, image):
        """
        Detect blood using HSV color range analysis.
        Returns: (has_blood, blood_percentage, confidence)
        """
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define red color range in HSV
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        
        # Create masks
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        # Calculate percentage
        red_pixels = cv2.countNonZero(red_mask)
        total_pixels = image.shape[0] * image.shape[1]
        blood_percentage = (red_pixels / total_pixels) * 100
        
        # Determine if blood is present
        has_blood = blood_percentage >= self.blood_percentage_threshold
        confidence = min(blood_percentage / 20.0, 1.0)  # Normalize to 0-1
        
        return has_blood, blood_percentage, confidence
    
    def detect_nsfw(self, image):
        """
        Detect NSFW content using model.
        Returns: (is_nsfw, scores_dict)
        """
        if self.nsfw_model is None:
            return False, {}
        
        try:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            results = self.nsfw_model(pil_image)
            
            scores = {r['label']: r['score'] for r in results}
            
            # Check if NSFW score exceeds threshold
            nsfw_score = scores.get('nsfw', 0.0)
            is_nsfw = nsfw_score > self.nsfw_threshold
            
            return is_nsfw, scores
        except Exception as e:
            logger.warning(f"NSFW detection failed: {e}")
            return False, {}
    
    def detect_violence(self, image):
        """
        Detect violence using model.
        Returns: (is_violent, scores_dict)
        """
        if self.violence_model is None:
            return False, {}
        
        try:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            results = self.violence_model(pil_image)
            
            scores = {r['label']: r['score'] for r in results}
            
            # Check violence score
            violence_score = scores.get('Violence', scores.get('violence', 0.0))
            is_violent = violence_score > self.violence_threshold
            
            return is_violent, scores
        except Exception as e:
            logger.warning(f"Violence detection failed: {e}")
            return False, {}
    
    def analyze(self, image):
        """
        Full analysis: NSFW, Violence, and Blood.
        Returns: dict with all detections
        """
        analysis = {
            'is_safe': True,
            'reasons': [],
            'scores': {
                'nsfw': 0.0,
                'violence': 0.0,
                'blood': 0.0
            },
            'flags': []
        }
        
        # Blood detection (color-based, fastest)
        has_blood, blood_pct, blood_conf = self.detect_blood_by_color(image)
        analysis['scores']['blood'] = blood_conf
        
        if has_blood:
            analysis['is_safe'] = False
            analysis['reasons'].append(f"Blood detected ({blood_pct:.1f}%)")
            analysis['flags'].append('blood')
            logger.info(f"🔴 BLOOD DETECTED: {blood_pct:.1f}%")
        
        # NSFW detection
        is_nsfw, nsfw_scores = self.detect_nsfw(image)
        if nsfw_scores:
            analysis['scores']['nsfw'] = nsfw_scores.get('nsfw', 0.0)
        
        if is_nsfw:
            analysis['is_safe'] = False
            analysis['reasons'].append("NSFW content detected")
            analysis['flags'].append('nsfw')
            logger.info("🔴 NSFW CONTENT DETECTED")
        
        # Violence detection
        is_violent, violence_scores = self.detect_violence(image)
        if violence_scores:
            violence_score = violence_scores.get('Violence', violence_scores.get('violence', 0.0))
            analysis['scores']['violence'] = violence_score
        
        if is_violent:
            analysis['is_safe'] = False
            analysis['reasons'].append("Violent content detected")
            analysis['flags'].append('violence')
            logger.info("🔴 VIOLENT CONTENT DETECTED")
        
        return analysis
