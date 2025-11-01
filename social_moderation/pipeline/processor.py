import yaml
import cv2
import time
import logging
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

class Processor:
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.cfg = yaml.safe_load(f)

        # --- DETECTORS ---
        from social_moderation.detectors.text_detector import TextDetector
        from social_moderation.detectors.opencv_face import OpenCVFace
        try:
            from social_moderation.detectors.yolov8_standard import YOLOv8StandardFace
            face_detector = YOLOv8StandardFace(self.cfg["face_detector"]["model_path"])
        except Exception:
            logger.warning("YOLOv8 not available, using OpenCV fallback.")
            face_detector = OpenCVFace()

        # --- MODULES ---
        from social_moderation.modules.face_blur import FaceBlurrer
        from social_moderation.modules.text_blur import TextBlurrer

        text_detector = TextDetector(languages=self.cfg["text_detector"].get("ocr_languages", ["en"]))
        offensive_words = self._load_offensive_words()

        self.face_blurrer = FaceBlurrer(face_detector, self.cfg)
        self.text_blurrer = TextBlurrer(text_detector, self.cfg, offensive_words)
        self.frame_skip = self.cfg.get("frame_skip", 3)
        self.executor = ThreadPoolExecutor(max_workers=2)

    def _load_offensive_words(self):
        try:
            with open("offensive_words.txt", "r") as f:
                return set([ln.strip().lower() for ln in f if ln.strip()])
        except FileNotFoundError:
            logger.warning("offensive_words.txt not found, using fallback list.")
            return {"fuck", "shit", "bitch", "ass", "idiot"}

    def process_video(self, input_path, output_path):
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        W, H = int(cap.get(3)), int(cap.get(4))
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        frame_idx = 0
        t0 = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % self.frame_skip == 0:
                processed = self._detect_and_blur(frame)
            else:
                processed = frame

            writer.write(processed)
            frame_idx += 1

        cap.release()
        writer.release()
        logger.info(f"✅ Processed in {time.time() - t0:.2f}s")

    def _detect_and_blur(self, frame):
        frame = self.face_blurrer.blur_faces(frame)
        frame = self.text_blurrer.blur_text_simple(frame)
        return frame
