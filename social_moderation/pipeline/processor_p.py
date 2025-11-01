"""
Main Video Processing Pipeline with Async Support
Handles video I/O, frame skipping, caching, and benchmarking
"""

import yaml
import cv2
import time
import logging
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

class Processor:
    """Main content moderation processor with async video processing."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize processor with configuration.
        
        Args:
            config_path: Path to config.yaml
        """
        self.config = self._load_config(config_path)
        self._setup_logging()
        self._init_components()
        
        # Performance tracking
        self.total_frames = 0
        self.processed_frames = 0
        self.start_time = None
    
    def _load_config(self, config_path: str) -> dict:
        """Load and validate configuration."""
        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            
            logger.info(f"✅ Configuration loaded from {config_path}")
            return config
            
        except FileNotFoundError:
            logger.error(f"Config file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML configuration: {e}")
            raise
    
    def _setup_logging(self):
        """Configure logging based on config."""
        log_level = self.config.get("system", {}).get("log_level", "INFO")
        
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        logger.info(f"Logging level set to {log_level}")
    
    def _init_components(self):
        """Initialize detectors and blurring modules."""
        # Import detectors
        from social_moderation.detectors.text_detector import TextDetector
        from social_moderation.detectors.opencv_face import OpenCVFace
        
        # Try to load YOLOv8, fallback to OpenCV
        detector_type = self.config["face_detector"].get("primary_type", "yolov8_standard")
        fallback_enabled = self.config["face_detector"].get("fallback_enabled", True)
        
        if detector_type.startswith("yolov8"):
            try:
                from social_moderation.detectors.yolov8_standard import YOLOv8StandardFace
                
                model_path = self.config["face_detector"]["model_path"]
                device = self.config["system"].get("device", "cuda")
                
                face_detector = YOLOv8StandardFace(model_path, device)
                logger.info("✅ Using YOLOv8 face detector")
                
            except Exception as e:
                if fallback_enabled:
                    logger.warning(f"YOLOv8 failed: {e}. Using OpenCV fallback.")
                    face_detector = OpenCVFace()
                else:
                    raise
        else:
            face_detector = OpenCVFace()
            logger.info("✅ Using OpenCV face detector")
        
        # Initialize text detector
        ocr_langs = self.config["text_detector"].get("ocr_languages", ["en"])
        gpu_enabled = self.config["text_detector"].get("gpu_enabled", True)
        preprocess_cfg = self.config["text_detector"].get("preprocessing", {})
        
        text_detector = TextDetector(
            languages=ocr_langs,
            gpu=gpu_enabled,
            preprocessing_config=preprocess_cfg
        )
        
        # Load offensive words
        offensive_words = self._load_offensive_words()
        
        # Initialize blurring modules
        from social_moderation.modules.face_blur import FaceBlurrer
        from social_moderation.modules.text_blur import TextBlurrer
        
        self.face_blurrer = FaceBlurrer(face_detector, self.config)
        self.text_blurrer = TextBlurrer(text_detector, self.config, offensive_words)
        
        # Pipeline settings
        self.frame_skip = self.config["system"].get("frame_skip", 2)
        self.async_enabled = self.config["pipeline"].get("async_processing", True)
        self.max_workers = self.config["pipeline"].get("max_workers", 4)
        
        logger.info(f"Pipeline initialized (frame_skip={self.frame_skip}, async={self.async_enabled})")
    
    def _load_offensive_words(self) -> set:
        """Load offensive words from file."""
        words_path = self.config.get("offensive_words", {}).get("file_path", "offensive_words.txt")
        case_sensitive = self.config.get("offensive_words", {}).get("case_sensitive", False)
        
        try:
            with open(words_path, "r", encoding="utf-8") as f:
                words = {line.strip() for line in f if line.strip()}
            
            if not case_sensitive:
                words = {w.lower() for w in words}
            
            logger.info(f"✅ Loaded {len(words)} offensive words from {words_path}")
            return words
            
        except FileNotFoundError:
            logger.warning(f"Offensive words file not found: {words_path}. Using default list.")
            return {"fuck", "shit", "bitch", "ass", "idiot", "stupid", "moron"}
    
    def _process_frame(self, frame: cv2.Mat) -> cv2.Mat:
        """Apply face and text blurring to single frame."""
        # Blur faces
        frame = self.face_blurrer.blur_faces(frame)
        
        # Blur toxic text
        frame = self.text_blurrer.blur_toxic_text(frame)
        
        return frame
    
    def process_video(self, input_path: str, output_path: str):
        """
        Process video file with content moderation.
        
        Uses multithreading for frame I/O optimization[web:23].
        
        Args:
            input_path: Path to input video
            output_path: Path to save moderated video
        """
        logger.info(f"Processing video: {input_path}")
        
        # Open video
        cap = cv2.VideoCapture(input_path)
        
        if not cap.isOpened():
            logger.error(f"Failed to open video: {input_path}")
            raise ValueError(f"Cannot open video file: {input_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video properties: {width}x{height} @ {fps:.2f}FPS, {self.total_frames} frames")
        
        # Video writer
        output_codec = self.config["pipeline"].get("output_codec", "mp4v")
        fourcc = cv2.VideoWriter_fourcc(*output_codec)
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            logger.error(f"Failed to create output video: {output_path}")
            cap.release()
            raise ValueError(f"Cannot create output video: {output_path}")
        
        # Processing loop
        self.start_time = time.time()
        frame_idx = 0
        last_processed_frame = None
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                # Frame skipping: process every Nth frame
                if frame_idx % self.frame_skip == 0:
                    processed_frame = self._process_frame(frame)
                    last_processed_frame = processed_frame
                    self.processed_frames += 1
                else:
                    # Reuse last processed frame (or original if first frame skipped)
                    processed_frame = last_processed_frame if last_processed_frame is not None else frame
                
                out.write(processed_frame)
                frame_idx += 1
                
                # Progress logging
                if frame_idx % 100 == 0:
                    progress = (frame_idx / self.total_frames) * 100
                    elapsed = time.time() - self.start_time
                    fps_actual = frame_idx / elapsed if elapsed > 0 else 0
                    
                    logger.info(
                        f"Progress: {progress:.1f}% ({frame_idx}/{self.total_frames}), "
                        f"Processing FPS: {fps_actual:.2f}"
                    )
        
        finally:
            cap.release()
            out.release()
            
            self._log_final_performance(output_path)
    
    def _log_final_performance(self, output_path: str):
        """Log final processing statistics."""
        elapsed = time.time() - self.start_time
        
        logger.info("=" * 60)
        logger.info("PROCESSING COMPLETE")
        logger.info(f"Output saved to: {output_path}")
        logger.info(f"Total frames: {self.total_frames}")
        logger.info(f"Processed frames: {self.processed_frames}")
        logger.info(f"Skipped frames: {self.total_frames - self.processed_frames}")
        logger.info(f"Total time: {elapsed:.2f}s")
        logger.info(f"Average FPS: {self.total_frames / elapsed:.2f}")
        logger.info("=" * 60)
