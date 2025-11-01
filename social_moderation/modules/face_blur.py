import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

class FaceBlurrer:
    def __init__(self, blur_strength=(51, 51), padding=0.05, detector_type="yolov8_standard"):
        """
        Precise face+neck blurring with minimal padding
        :param blur_strength: kernel size for Gaussian blur (must be odd)
        :param padding: minimal padding (5% default) for face region only
        :param detector_type: "yolov8", "opencv", or "yolov8_standard"
        """
        # Initialize detector
        if detector_type == "yolov8":
            from detectors.yolov8_face import YOLOv8Face
            self.detector = YOLOv8Face()
        elif detector_type == "yolov8_standard":
            from detectors.yolov8_standard import YOLOv8StandardFace
            self.detector = YOLOv8StandardFace()
        elif detector_type == "opencv":
            from detectors.opencv_face import OpenCVFace
            self.detector = OpenCVFace()
        else:
            raise ValueError(f"Unknown detector type: {detector_type}")
        
        # Ensure blur strength values are odd
        self.blur_strength = (
            blur_strength[0] + 1 if blur_strength[0] % 2 == 0 else blur_strength[0],
            blur_strength[1] + 1 if blur_strength[1] % 2 == 0 else blur_strength[1]
        )
        
        # Minimal padding for precise blurring
        self.padding = min(max(padding, 0.0), 0.1)  # Max 10% padding
    
    def _adjust_bbox_for_face_neck(self, x1, y1, x2, y2, image_height, image_width):
        """
        Adjust bounding box to include face + neck region only
        """
        face_height = y2 - y1
        face_width = x2 - x1
        
        # Add minimal horizontal padding (5% each side for face width)
        pad_x = int(face_width * 0.05)
        x1_adjusted = max(0, x1 - pad_x)
        x2_adjusted = min(image_width, x2 + pad_x)
        
        # Vertical adjustment: include neck (extend 30% of face height downward)
        neck_extension = int(face_height * 0.3)
        
        # Minimal top padding (just 5% for hairline)
        pad_y_top = int(face_height * 0.05)
        y1_adjusted = max(0, y1 - pad_y_top)
        
        # Extend downward for neck
        y2_adjusted = min(image_height, y2 + neck_extension)
        
        return x1_adjusted, y1_adjusted, x2_adjusted, y2_adjusted
    
    def blur_faces(self, image, confidence_threshold=None):
        """
        Detect and blur face+neck regions with precision
        """
        if image is None or image.size == 0:
            return image
        
        boxes = self.detector.detect_faces(image, confidence_threshold)
        height, width = image.shape[:2]
        
        blurred_count = 0
        for (x1, y1, x2, y2) in boxes:
            # Adjust bbox for face + neck only
            x1_blur, y1_blur, x2_blur, y2_blur = self._adjust_bbox_for_face_neck(
                x1, y1, x2, y2, height, width
            )
            
            # Extract and blur region
            roi = image[y1_blur:y2_blur, x1_blur:x2_blur]
            if roi.size > 0:
                # Adaptive blur strength based on face size
                face_area = (x2_blur - x1_blur) * (y2_blur - y1_blur)
                sigma = max(15, min(50, int(np.sqrt(face_area) / 10)))
                
                blurred_roi = cv2.GaussianBlur(roi, self.blur_strength, sigma)
                image[y1_blur:y2_blur, x1_blur:x2_blur] = blurred_roi
                blurred_count += 1
                
                logger.debug(f"Blurred face at ({x1_blur},{y1_blur})-({x2_blur},{y2_blur})")
        
        if blurred_count > 0:
            logger.info(f"✓ Blurred {blurred_count} face(s)")
        
        return image
    
    def blur_faces_selective(self, image, confidence_threshold=None, exclude_center=False):
        """
        Selective face blurring - optionally exclude center faces
        """
        if image is None or image.size == 0:
            return image
        
        boxes = self.detector.detect_faces(image, confidence_threshold)
        height, width = image.shape[:2]
        center_x, center_y = width // 2, height // 2
        
        blurred_count = 0
        skipped_count = 0
        
        for (x1, y1, x2, y2) in boxes:
            # Calculate face center
            face_center_x = (x1 + x2) // 2
            face_center_y = (y1 + y2) // 2
            
            # Check if face is in center region (±20% from center)
            is_center_face = (
                abs(face_center_x - center_x) < width * 0.2 and 
                abs(face_center_y - center_y) < height * 0.2
            )
            
            # Skip if excluding center faces
            if exclude_center and is_center_face:
                logger.debug(f"Skipping center face at ({x1},{y1})-({x2},{y2})")
                skipped_count += 1
                continue
            
            # Adjust bbox for face + neck
            x1_blur, y1_blur, x2_blur, y2_blur = self._adjust_bbox_for_face_neck(
                x1, y1, x2, y2, height, width
            )
            
            # Extract and blur region
            roi = image[y1_blur:y2_blur, x1_blur:x2_blur]
            if roi.size > 0:
                face_area = (x2_blur - x1_blur) * (y2_blur - y1_blur)
                sigma = max(15, min(50, int(np.sqrt(face_area) / 10)))
                
                blurred_roi = cv2.GaussianBlur(roi, self.blur_strength, sigma)
                image[y1_blur:y2_blur, x1_blur:x2_blur] = blurred_roi
                blurred_count += 1
                
                logger.debug(f"Blurred face at ({x1_blur},{y1_blur})-({x2_blur},{y2_blur})")
        
        if exclude_center and skipped_count > 0:
            logger.info(f"✓ Blurred {blurred_count} face(s), skipped {skipped_count} center face(s)")
        elif blurred_count > 0:
            logger.info(f"✓ Blurred {blurred_count} face(s)")
        
        return image