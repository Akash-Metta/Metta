"""
Production-Grade Face Blurrer with Motion Smoothing
Features: Adaptive blur, face tracking, neck coverage, fallback detection
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Dict, Optional
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class FaceTrack:
    """Face tracking data for motion smoothing across frames."""
    track_id: int
    bbox_history: deque = field(default_factory=lambda: deque(maxlen=10))
    confidence_history: deque = field(default_factory=lambda: deque(maxlen=10))
    last_seen_frame: int = 0


def _ensure_int_coords(coords: Tuple) -> Tuple[int, int, int, int]:
    """Ensure coordinates are integers."""
    return tuple(int(round(float(v))) for v in coords[:4])


class FaceBlurrer:
    """Enhanced face blurrer with tracking and adaptive intensity."""
    
    def __init__(self, detector, config: dict):
        """
        Initialize face blurrer.
        
        Args:
            detector: Face detector instance (YOLOv8 or OpenCV)
            config: Configuration dictionary from config.yaml
        """
        self.detector = detector
        self.config = config
        
        # Blur settings
        self.blur_method = config["blur"]["face"].get("method", "gaussian")
        self.adaptive_blur = config["blur"]["face"].get("adaptive_intensity", True)
        self.min_kernel = config["blur"]["face"].get("gaussian_min_kernel", 15)
        self.max_kernel = config["blur"]["face"].get("gaussian_max_kernel", 121)
        self.mosaic_block = config["blur"]["face"].get("mosaic_block_size", 8)
        
        # Detection settings
        self.conf_threshold = config["face_detector"].get("confidence_threshold", 0.35)
        
        # Bbox adjustment settings
        bbox_cfg = config["face_detector"].get("bbox_adjustment", {})
        self.padding_ratio = bbox_cfg.get("padding_ratio", 0.08)
        self.neck_extension = bbox_cfg.get("neck_extension", 0.35)
        self.forehead_pad = bbox_cfg.get("forehead_padding", 0.12)
        
        # Motion smoothing
        smooth_cfg = config["face_detector"].get("motion_smoothing", {})
        self.motion_smoothing = smooth_cfg.get("enabled", True)
        self.iou_threshold = smooth_cfg.get("iou_threshold", 0.5)
        self.smooth_window = smooth_cfg.get("smoothing_window", 5)
        self.max_missing = smooth_cfg.get("max_missing_frames", 30)
        
        # Tracking state
        self.face_tracks: Dict[int, FaceTrack] = {}
        self.next_track_id = 0
        self.frame_count = 0
        
        # Performance metrics
        self.detect_times = deque(maxlen=100)
        self.blur_times = deque(maxlen=100)
        
        logger.info(f"FaceBlurrer initialized (method={self.blur_method}, adaptive={self.adaptive_blur})")
    
    def _calculate_iou(self, box1: Tuple, box2: Tuple) -> float:
        """Calculate Intersection over Union for tracking."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        xi1, yi1 = max(x1_1, x1_2), max(y1_1, y1_2)
        xi2, yi2 = min(x2_1, x2_2), min(y2_1, y2_2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _update_tracks(self, detections: List[Tuple]) -> Dict[int, Tuple]:
        """Update face tracks for motion smoothing."""
        current_tracks = {}
        
        for det in detections:
            x1, y1, x2, y2, conf = det
            best_match_id = None
            best_iou = 0.0
            
            # Match with existing tracks
            for track_id, track in self.face_tracks.items():
                if len(track.bbox_history) > 0:
                    last_bbox = track.bbox_history[-1]
                    iou = self._calculate_iou((x1, y1, x2, y2), last_bbox)
                    
                    if iou > best_iou and iou > self.iou_threshold:
                        best_iou = iou
                        best_match_id = track_id
            
            # Create new track or update existing
            if best_match_id is None:
                track_id = self.next_track_id
                self.next_track_id += 1
                self.face_tracks[track_id] = FaceTrack(track_id=track_id)
            else:
                track_id = best_match_id
            
            track = self.face_tracks[track_id]
            track.bbox_history.append((x1, y1, x2, y2))
            track.confidence_history.append(conf)
            track.last_seen_frame = self.frame_count
            
            current_tracks[track_id] = (x1, y1, x2, y2, conf)
        
        # Remove stale tracks
        stale_ids = [
            tid for tid, track in self.face_tracks.items()
            if self.frame_count - track.last_seen_frame > self.max_missing
        ]
        for tid in stale_ids:
            del self.face_tracks[tid]
            logger.debug(f"Removed stale track {tid}")
        
        return current_tracks
    
    def _smooth_bbox(self, track: FaceTrack) -> Tuple[int, int, int, int]:
        """Apply temporal smoothing using moving average."""
        if len(track.bbox_history) < 2:
            return _ensure_int_coords(track.bbox_history[-1])
        
        # Average over recent frames
        recent = list(track.bbox_history)[-self.smooth_window:]
        x1 = int(np.mean([b[0] for b in recent]))
        y1 = int(np.mean([b[1] for b in recent]))
        x2 = int(np.mean([b[2] for b in recent]))
        y2 = int(np.mean([b[3] for b in recent]))
        
        return (x1, y1, x2, y2)
    
    def _clamp_box(self, x1: int, y1: int, x2: int, y2: int, H: int, W: int) -> Tuple:
        """Clamp bounding box to image boundaries."""
        return (max(0, x1), max(0, y1), min(W, x2), min(H, y2))
    
    def _adjust_bbox_with_neck(self, x1: int, y1: int, x2: int, y2: int, 
                               H: int, W: int) -> Tuple[int, int, int, int]:
        """
        Enhance bbox to cover face + neck comprehensively.
        """
        face_w = x2 - x1
        face_h = y2 - y1
        
        # Horizontal padding
        pad_x = int(face_w * self.padding_ratio)
        
        # Top padding for forehead
        pad_y_top = int(face_h * self.forehead_pad)
        
        # Neck extension (extends below face)
        neck_ext = int(face_h * self.neck_extension)
        
        x1_adj = x1 - pad_x
        x2_adj = x2 + pad_x
        y1_adj = y1 - pad_y_top
        y2_adj = y2 + neck_ext
        
        return self._clamp_box(x1_adj, y1_adj, x2_adj, y2_adj, H, W)
    
    def _adaptive_blur_kernel(self, face_area: int) -> int:
        """
        Calculate adaptive blur intensity based on face size.
        """
        if not self.adaptive_blur:
            return self.min_kernel
        
        # Scale kernel based on area
        if face_area < 5000:  # Small face
            kernel = self.min_kernel
        elif face_area < 20000:  # Medium face
            kernel = (self.min_kernel + self.max_kernel) // 2
        else:  # Large face
            kernel = self.max_kernel
        
        # Ensure odd kernel size
        if kernel % 2 == 0:
            kernel += 1
        
        return max(self.min_kernel, min(self.max_kernel, kernel))
    
    def _apply_gaussian_blur(self, roi: np.ndarray, kernel_size: int) -> np.ndarray:
        """Apply Gaussian blur to ROI."""
        return cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)
    
    def _apply_mosaic(self, roi: np.ndarray, block_size: int) -> np.ndarray:
        """Apply mosaic/pixelation effect to ROI."""
        h, w = roi.shape[:2]
        
        if h < block_size or w < block_size:
            return roi
        
        # Downscale then upscale for pixelation
        small = cv2.resize(
            roi,
            (max(1, w // block_size), max(1, h // block_size)),
            interpolation=cv2.INTER_LINEAR
        )
        
        pixelated = cv2.resize(
            small,
            (w, h),
            interpolation=cv2.INTER_NEAREST
        )
        
        return pixelated
    
    def blur_faces(self, image: np.ndarray) -> np.ndarray:
        """
        Blur all detected faces in image with motion smoothing.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Image with blurred faces
        """
        if image is None or image.size == 0:
            return image
        
        self.frame_count += 1
        result = image.copy()
        H, W = image.shape[:2]
        
        # Detect faces
        import time
        t0 = time.time()
        detections = self.detector.detect_faces(image, self.conf_threshold)
        self.detect_times.append(time.time() - t0)
        
        if not detections:
            return result
        
        # Update tracking
        if self.motion_smoothing:
            tracks = self._update_tracks(detections)
        else:
            tracks = {i: det for i, det in enumerate(detections)}
        
        # Blur each face
        t0 = time.time()
        
        for track_id, det in tracks.items():
            x1, y1, x2, y2, conf = det
            
            # Ensure coordinates are integers
            x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
            
            # Apply motion smoothing
            if self.motion_smoothing and track_id in self.face_tracks:
                x1, y1, x2, y2 = self._smooth_bbox(self.face_tracks[track_id])
            
            # Adjust for neck coverage
            x1_blur, y1_blur, x2_blur, y2_blur = self._adjust_bbox_with_neck(
                x1, y1, x2, y2, H, W
            )
            
            # Ensure blur coordinates are integers
            x1_blur = int(round(x1_blur))
            y1_blur = int(round(y1_blur))
            x2_blur = int(round(x2_blur))
            y2_blur = int(round(y2_blur))
            
            # Extract ROI
            roi = result[y1_blur:y2_blur, x1_blur:x2_blur]
            
            if roi.size == 0:
                continue
            
            # Apply blur
            face_area = (x2_blur - x1_blur) * (y2_blur - y1_blur)
            
            if self.blur_method == "gaussian":
                kernel = self._adaptive_blur_kernel(face_area)
                blurred_roi = self._apply_gaussian_blur(roi, kernel)
            else:  # mosaic
                blurred_roi = self._apply_mosaic(roi, self.mosaic_block)
            
            result[y1_blur:y2_blur, x1_blur:x2_blur] = blurred_roi
        
        self.blur_times.append(time.time() - t0)
        
        # Log performance metrics every 100 frames
        if self.config["system"].get("benchmark_logging", True) and self.frame_count % 100 == 0:
            self._log_performance()
        
        return result
    
    def _log_performance(self):
        """Log performance metrics."""
        if self.detect_times and self.blur_times:
            avg_detect = np.mean(self.detect_times) * 1000
            avg_blur = np.mean(self.blur_times) * 1000
            total_ms = avg_detect + avg_blur
            fps = 1000 / total_ms if total_ms > 0 else 0
            
            logger.info(
                f"Performance [Frame {self.frame_count}]: "
                f"Detection={avg_detect:.2f}ms, Blur={avg_blur:.2f}ms, "
                f"FPS={fps:.1f}, Active Tracks={len(self.face_tracks)}"
            )
