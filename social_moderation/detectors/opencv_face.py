# detectors/opencv_face.py
import cv2
import os

class OpenCVFace:
    def __init__(self, conf=0.5):
        """
        Face detection using OpenCV's built-in Haar cascades.
        """
        self.conf = conf
        
        # Load pre-trained Haar cascade for face detection
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if not os.path.exists(cascade_path):
            # Download if needed (though it should be included with OpenCV)
            print("Haar cascade not found. Please ensure OpenCV is properly installed.")
            raise FileNotFoundError("Haar cascade not found")
            
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def detect_faces(self, image, confidence_threshold=None):
        """
        Detect faces using OpenCV's Haar cascades.
        """
        # Convert to grayscale for detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Convert to [x1, y1, x2, y2] format
        boxes = []
        for (x, y, w, h) in faces:
            boxes.append([x, y, x + w, y + h])
            
        return boxes