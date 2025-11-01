# detectors/offensive_word_detector.py
import easyocr
import cv2
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

class OffensiveWordDetector:
    def __init__(self, confidence_threshold=0.6):
        """
        Detector that combines keyword matching with ML model for better detection.
        Works well for both sentences and individual words.
        """
        self.confidence_threshold = confidence_threshold
        
        # Hate/offensive keyword list (common hate speech terms)
        self.offensive_keywords = {
            # Hate-related terms
            'hate', 'hatred', 'hater', 'haters', 'hating',
            # Violence terms
            'kill', 'murder', 'death', 'die', 'violence', 'violent', 'attack', 'attacks',
            # Slurs and discrimination
            'racist', 'racism', 'sexist', 'sexism', 'bigot', 'bigotry', 'slur', 'slurs',
            # Offensive descriptors
            'stupid', 'idiot', 'moron', 'dumb', 'retard', 'retarded',
            # Threats
            'threat', 'threaten', 'terrorize', 'terror',
            # Derogatory terms
            'scum', 'trash', 'garbage', 'worthless', 'inferior',
            # Extremism
            'supremacy', 'nazi', 'fascist',
            # Additional hate indicators
            'discriminate', 'discrimination', 'prejudice', 'prejudicial',
            'misogyny', 'misogynist', 'homophobe', 'homophobia',
            'xenophobe', 'xenophobia', 'islamophobe', 'islamophobia',
            'antisemite', 'antisemitism',
            # Add common slurs (redacted versions)
            # Note: You can expand this list based on your needs
        }
        
        # Load ML model for phrase-level detection
        print("Loading offensive content detection model...")
        self.model_name = "Hate-speech-CNERG/dehatebert-mono-english"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.model.eval()
        
        # Initialize OCR
        print("Loading EasyOCR...")
        self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        print("Model loaded successfully.")
    
    def detect_text_regions(self, image):
        """
        Detect text regions with improved handling for word clouds.
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Use EasyOCR with adjusted parameters for better word cloud detection
        results = self.reader.readtext(
            rgb_image,
            paragraph=False,  # Don't group into paragraphs
            min_size=10,      # Detect smaller text
            text_threshold=0.6  # Lower threshold for varied fonts
        )
        
        text_regions = []
        for (bbox, text, prob) in results:
            bbox_array = np.array(bbox)
            x1 = int(bbox_array[:, 0].min())
            y1 = int(bbox_array[:, 1].min())
            x2 = int(bbox_array[:, 0].max())
            y2 = int(bbox_array[:, 1].max())
            
            text_regions.append(([x1, y1, x2, y2], text, prob))
        
        return text_regions
    
    def is_offensive_keyword(self, text):
        """
        Check if text contains offensive keywords.
        """
        text_lower = text.lower().strip()
        
        # Check exact match
        if text_lower in self.offensive_keywords:
            return True, 1.0
        
        # Check if any keyword is contained in the text
        for keyword in self.offensive_keywords:
            if keyword in text_lower:
                return True, 0.9
        
        return False, 0.0
    
    def is_offensive_ml(self, text):
        """
        Use ML model to detect offensive content in phrases.
        """
        if len(text.strip()) < 3:
            return False, 0.0
        
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, 
                               max_length=512, padding=True)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=-1)
        
        offensive_prob = probabilities[0][1].item()
        is_offensive = offensive_prob >= self.confidence_threshold
        
        return is_offensive, offensive_prob
    
    def is_offensive(self, text):
        """
        Combined detection: keyword matching + ML model.
        Returns True if either method detects offensive content.
        """
        # First check keywords (fast)
        is_keyword_match, keyword_score = self.is_offensive_keyword(text)
        if is_keyword_match:
            return True, keyword_score
        
        # Then use ML model for context-aware detection
        is_ml_offensive, ml_score = self.is_offensive_ml(text)
        if is_ml_offensive:
            return True, ml_score
        
        return False, max(keyword_score, ml_score)
    
    def detect_offensive_regions(self, image, verbose=True):
        """
        Detect all regions containing offensive content.
        """
        text_regions = self.detect_text_regions(image)
        
        if verbose:
            print(f"Detected {len(text_regions)} text region(s)")
        
        offensive_regions = []
        
        for (bbox, text, ocr_prob) in text_regions:
            # Clean the text
            text_cleaned = text.strip()
            
            if len(text_cleaned) < 2:
                continue
            
            # Check if offensive
            is_offensive, confidence = self.is_offensive(text_cleaned)
            
            if is_offensive:
                if verbose:
                    print(f"⚠️  Offensive: '{text_cleaned}' (confidence: {confidence:.2f})")
                offensive_regions.append(bbox)
            elif verbose and len(text_cleaned) > 2:
                print(f"✓  Clean: '{text_cleaned}' (score: {confidence:.2f})")
        
        return offensive_regions