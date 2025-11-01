# detectors/yolov8_face.py

from ultralytics import YOLO
import cv2
import os
from urllib.request import urlretrieve


class YOLOv8Face:
    def __init__(self, conf=0.5, model_name="yolov8n-face.pt"):
        """
        Enhanced wrapper for YOLOv8 face detection with auto-download.
        
        :param conf: default confidence threshold
        :param model_name: name of YOLOv8 face model
        """
        self.model_name = model_name
        self.conf = conf
        self.model = self._load_model()

    def _load_model(self):
        """Load model, downloading if necessary"""
        model_path = self.model_name
        
        # If model doesn't exist locally, download it
        if not os.path.exists(model_path):
            print(f"Downloading YOLOv8 face model to {model_path}...")
            try:
                # Download from a known source
                model_url = "https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt"
                urlretrieve(model_url, model_path)
                print("Download completed successfully.")
            except Exception as e:
                print(f"Download failed: {e}")
                print("Please download the model manually from the link above.")
                raise
        
        return YOLO(model_path)

    def detect_faces(self, image, conf_threshold=None):
        """
        Detect faces in an image and return bounding boxes WITH confidence scores.
        
        :param image: input image (numpy array, BGR format)
        :param conf_threshold: confidence threshold (uses self.conf if None)
        :return: list of tuples (x1, y1, x2, y2, confidence)
        """
        if conf_threshold is None:
            conf_threshold = self.conf
        
        # Run prediction
        results = self.model.predict(image, conf=conf_threshold, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Get confidence score
                confidence = float(box.conf[0].cpu().numpy())
                
                # Return as tuple with 5 values: (x1, y1, x2, y2, conf)
                detections.append((
                    float(x1), 
                    float(y1), 
                    float(x2), 
                    float(y2), 
                    confidence
                ))
        
        return detections

    def __call__(self, image, conf_threshold=None):
        """
        Make the class callable for convenience.
        """
        return self.detect_faces(image, conf_threshold)
