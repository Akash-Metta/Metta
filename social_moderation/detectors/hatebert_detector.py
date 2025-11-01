# detectors/hatebert_detector.py
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import easyocr
import cv2
import numpy as np

class HateBERTDetector:
    def __init__(self, confidence_threshold=0.7):
        """
        Initialize hate speech detection model.
        :param confidence_threshold: minimum confidence for hate speech classification
        """
        self.confidence_threshold = confidence_threshold
        
        # Load fine-tuned hate speech detection model
        print("Loading hate speech detection model...")
        # Using a properly fine-tuned model for hate speech detection
        self.model_name = "Hate-speech-CNERG/dehatebert-mono-english"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.eval()
        
        # Initialize OCR reader
        print("Loading EasyOCR...")
        self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        print("HateBERT and OCR loaded successfully.")
    
    def detect_text_regions(self, image):
        """
        Detect text regions in an image using OCR.
        :param image: input image (BGR format)
        :return: list of (bbox, text) tuples
        """
        # Convert BGR to RGB for EasyOCR
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect text
        results = self.reader.readtext(rgb_image)
        
        text_regions = []
        for (bbox, text, prob) in results:
            # Convert bbox to [x1, y1, x2, y2] format
            bbox_array = np.array(bbox)
            x1 = int(bbox_array[:, 0].min())
            y1 = int(bbox_array[:, 1].min())
            x2 = int(bbox_array[:, 0].max())
            y2 = int(bbox_array[:, 1].max())
            
            text_regions.append(([x1, y1, x2, y2], text, prob))
        
        return text_regions
    
    def is_hate_speech(self, text):
        """
        Check if text contains hate speech using HateBERT.
        :param text: input text
        :return: (is_hate, confidence_score)
        """
        # Tokenize and predict
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
            
        # Get hate speech probability (assuming class 1 is hate)
        hate_prob = probabilities[0][1].item()
        
        is_hate = hate_prob >= self.confidence_threshold
        return is_hate, hate_prob
    
    def detect_hate_regions(self, image):
        """
        Detect regions containing hate speech in an image.
        :param image: input image (BGR format)
        :return: list of bounding boxes containing hate speech
        """
        # Detect all text regions
        text_regions = self.detect_text_regions(image)
        
        hate_regions = []
        for (bbox, text, ocr_prob) in text_regions:
            # Check if text contains hate speech
            is_hate, hate_prob = self.is_hate_speech(text)
            
            if is_hate:
                print(f"Detected hate speech: '{text}' (confidence: {hate_prob:.2f})")
                hate_regions.append(bbox)
        
        return hate_regions