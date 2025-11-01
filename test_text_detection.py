import cv2
from social_moderation.detectors.text_detector import TextDetector

# Load the test image
img_path = 'data/input/1000_F_1566391003_PcPMXVvR99sK3Rf9YxJaChFZRhD7XuAx.jpg'
image = cv2.imread(img_path)

print(f"Image size: {image.shape}")

# Create detector
detector = TextDetector(languages=['en'], gpu=False)

# Test with different confidence thresholds
for conf in [0.0, 0.1, 0.3, 0.5, 0.7]:
    print(f"\n{'='*60}")
    print(f"Testing with confidence: {conf}")
    print(f"{'='*60}")
    
    detections = detector.detect_text(image, conf)
    
    print(f"Type: {type(detections)}")
    print(f"Length: {len(detections) if detections else 0}")
    
    if detections:
        print(f"\nDetections found: {len(detections)}")
        for idx, det in enumerate(detections[:5]):  # Show first 5
            print(f"  {idx}: {det}")
    else:
        print("NO DETECTIONS!")
