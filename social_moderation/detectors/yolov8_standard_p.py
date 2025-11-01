"""
YOLOv8 Standard Face Detector with graceful fallback
Optimized for production with batch processing support
"""

import logging
import numpy as np
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

class YOLOv8StandardFace:
    """YOLOv8 face detector with error handling and batch support."""
    
    def __init__(self, model_path: str = "yolov8n-face.pt", device: str = "cuda"):
        """
        Initialize YOLOv8 face detector.
        
        Args:
            model_path: Path to YOLOv8 face detection weights
            device: "cuda" or "cpu"
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load YOLOv8 model with error handling."""
        try:
            from ultralytics import YOLO
            import torch
            
            # Auto-fallback to CPU if CUDA unavailable
            if self.device == "cuda" and not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                self.device = "cpu"
            
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            
            logger.info(f"✅ YOLOv8 face detector loaded on {self.device}")
            
        except ImportError:
            logger.error("ultralytics not installed. Install: pip install ultralytics")
            raise
        except FileNotFoundError:
            logger.error(f"Model file not found: {self.model_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to load YOLOv8 model: {e}")
            raise
    
    def detect_faces(self, image: np.ndarray, 
                    conf_threshold: float = 0.35) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect faces in image.
        
        Args:
            image: Input image (BGR format)
            conf_threshold: Minimum confidence score
            
        Returns:
            List of detections as (x1, y1, x2, y2, confidence)
        """
        if self.model is None:
            logger.error("Model not loaded")
            return []
        
        if image is None or image.size == 0:
            return []
        
        try:
            # Run inference (verbose=False to suppress output)
            results = self.model.predict(
                image, 
                conf=conf_threshold,
                device=self.device,
                verbose=False
            )
            
            detections = []
            
            # Extract bounding boxes
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                
                for box in boxes:
                    # Get coordinates and confidence
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    
                    detections.append((
                        int(x1), int(y1), int(x2), int(y2), conf
                    ))
            
            logger.debug(f"Detected {len(detections)} faces")
            return detections
            
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            return []
    
    def detect_faces_batch(self, images: List[np.ndarray], 
                          conf_threshold: float = 0.35) -> List[List[Tuple]]:
        """
        Batch face detection for multiple images (faster GPU utilization).
        
        Args:
            images: List of input images
            conf_threshold: Minimum confidence
            
        Returns:
            List of detection lists, one per image
        """
        if self.model is None or not images:
            return [[] for _ in images]
        
        try:
            results = self.model.predict(
                images,
                conf=conf_threshold,
                device=self.device,
                verbose=False
            )
            
            all_detections = []
            
            for result in results:
                detections = []
                boxes = result.boxes
                
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        detections.append((int(x1), int(y1), int(x2), int(y2), conf))
                
                all_detections.append(detections)
            
            return all_detections
            
        except Exception as e:
            logger.error(f"Batch face detection failed: {e}")
            return [[] for _ in images]
