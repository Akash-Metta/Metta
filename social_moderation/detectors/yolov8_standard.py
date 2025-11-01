# detectors/yolov8_standard.py
from ultralytics import YOLO
import cv2

class YOLOv8StandardFace:
    def __init__(self, conf=0.5):
        """
        Use standard YOLOv8 model for face detection.
        The standard model can detect faces as part of the 'person' class.
        """
        self.conf = conf
        # Load standard YOLOv8 model (will auto-download if needed)
        self.model = YOLO("yolov8n.pt")

    def detect_faces(self, image, confidence_threshold=None):
        """
        Detect faces in an image using standard YOLOv8.
        This looks for the 'person' class and uses the face region.
        """
        detection_conf = confidence_threshold if confidence_threshold is not None else self.conf
        
        results = self.model.predict(image, conf=detection_conf, verbose=False)
        boxes = []
        
        if results and len(results[0].boxes) > 0:
            for i, box in enumerate(results[0].boxes):
                # Check if this is a person detection (class 0 in COCO dataset)
                if box.cls.item() == 0:  # 0 is the person class in COCO
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    # For faces, we'll use the upper portion of the person detection
                    height = y2 - y1
                    # Estimate face region (upper 1/3 of person bounding box)
                    face_y2 = y1 + height * 0.4  # Reduced to better capture just the face
                    boxes.append([int(x1), int(y1), int(x2), int(face_y2)])
                    
        return boxes