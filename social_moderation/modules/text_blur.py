import cv2
import os
from transformers import pipeline
import torch
import numpy as np
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextBlurrer:
    def __init__(self, blur_strength=(51, 51), padding=0.02, sentiment_threshold=-0.3):
        """
        Context-aware text blurring using sentiment analysis
        First analyzes full sentences, then identifies toxic words in context
        sentiment_threshold: negative sentiment threshold for blurring (-1 to 1, default -0.3)
        """
        from detectors.text_detector import TextDetector
        
        self.detector = TextDetector()
        
        # Ensure blur strength values are odd
        self.blur_strength = (
            blur_strength[0] + 1 if blur_strength[0] % 2 == 0 else blur_strength[0],
            blur_strength[1] + 1 if blur_strength[1] % 2 == 0 else blur_strength[1]
        )
        
        self.padding = min(max(padding, 0.0), 0.05)
        self.sentiment_threshold = sentiment_threshold
        
        # Cache for analysis
        self.sentence_cache = {}
        self.word_impact_cache = {}
        self.cache_size = 1000
        
        # Initialize models
        self._init_sentiment_model()
        self._init_toxicity_model()
        
        # Offensive words as fallback
        self.offensive_words = self._load_offensive_words()
    
    def _load_offensive_words(self):
        """Fallback offensive words list"""
        return {
            'fuck', 'shit', 'bitch', 'bastard', 'damn', 'hell', 'ass', 'crap',
            'dick', 'cock', 'pussy', 'whore', 'slut', 'fag', 'dyke',
            'stupid', 'idiot', 'moron', 'dumb', 'fool', 'loser', 'jerk',
            'ugly', 'fat', 'worthless', 'useless', 'pathetic', 'disgusting',
            'terrible', 'awful', 'horrible', 'trash', 'garbage', 'waste',
            'nigger', 'nigga', 'chink', 'spic', 'kike', 'raghead', 'wetback',
            'gook', 'towelhead', 'cracker', 'honky', 'beaner',
            'hate', 'kill', 'die', 'death', 'murder', 'rape', 'attack',
            'fck', 'fuk', 'sh1t', 'b1tch', 'a$$', 'stfu', 'gtfo', 'kys',
            'retard', 'retarded', 'autistic', 'cancer', 'aids'
        }
    
    def _init_sentiment_model(self):
        """Initialize sentiment analysis model"""
        logger.info("Loading sentiment analysis model...")
        
        device = 0 if torch.cuda.is_available() else -1
        
        try:
            # Use RoBERTa for better sentiment understanding
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=device,
                truncation=True,
                max_length=128
            )
            logger.info("✓ Loaded sentiment model: cardiffnlp/twitter-roberta-base-sentiment-latest")
        except Exception as e:
            logger.warning(f"Failed to load sentiment model: {e}")
            try:
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    device=device,
                    truncation=True
                )
                logger.info("✓ Loaded fallback sentiment model")
            except Exception as e2:
                logger.error(f"Could not load any sentiment model: {e2}")
                self.sentiment_analyzer = None
    
    def _init_toxicity_model(self):
        """Initialize toxicity detection model"""
        logger.info("Loading toxicity detection model...")
        
        device = 0 if torch.cuda.is_available() else -1
        
        model_priority = [
            "unitary/toxic-bert",
            "martin-ha/toxic-comment-model",
            "cardiffnlp/twitter-roberta-base-offensive"
        ]
        
        for model_name in model_priority:
            try:
                self.toxicity_analyzer = pipeline(
                    "text-classification",
                    model=model_name,
                    device=device,
                    truncation=True,
                    max_length=128
                )
                logger.info(f"✓ Loaded toxicity model: {model_name}")
                return
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}")
                continue
        
        logger.warning("Could not load toxicity model, using sentiment only")
        self.toxicity_analyzer = None
    
    def _analyze_sentence_sentiment(self, sentence):
        """
        Analyze the overall sentiment and toxicity of a sentence
        Returns: (sentiment_score, is_toxic, toxicity_score)
        """
        if not sentence or len(sentence.strip()) < 3:
            return 0.0, False, 0.0
        
        sentence_clean = sentence.strip()
        
        # Check cache
        cache_key = sentence_clean.lower()
        if cache_key in self.sentence_cache:
            return self.sentence_cache[cache_key]
        
        sentiment_score = 0.0
        toxicity_score = 0.0
        is_toxic = False
        
        # Analyze sentiment
        if self.sentiment_analyzer:
            try:
                result = self.sentiment_analyzer(sentence_clean)[0]
                label = result['label'].lower()
                score = result['score']
                
                # Map sentiment to -1 to 1 scale
                if 'negative' in label or 'neg' in label:
                    sentiment_score = -score
                elif 'positive' in label or 'pos' in label:
                    sentiment_score = score
                else:  # neutral
                    sentiment_score = 0.0
                    
                logger.debug(f"Sentiment: '{sentence_clean[:50]}...' -> {label} ({score:.2f})")
            except Exception as e:
                logger.debug(f"Sentiment analysis failed: {e}")
        
        # Analyze toxicity
        if self.toxicity_analyzer:
            try:
                result = self.toxicity_analyzer(sentence_clean)[0]
                label = result['label'].lower()
                score = result['score']
                
                toxic_labels = {'toxic', 'offensive', 'hate', 'label_1', '1'}
                if any(tox in label for tox in toxic_labels):
                    toxicity_score = score
                    is_toxic = score > 0.5
                    
                logger.debug(f"Toxicity: '{sentence_clean[:50]}...' -> {label} ({score:.2f})")
            except Exception as e:
                logger.debug(f"Toxicity analysis failed: {e}")
        
        # Cache result
        if len(self.sentence_cache) >= self.cache_size:
            self.sentence_cache.pop(next(iter(self.sentence_cache)))
        self.sentence_cache[cache_key] = (sentiment_score, is_toxic, toxicity_score)
        
        return sentiment_score, is_toxic, toxicity_score
    
    def _calculate_word_impact(self, word, full_sentence):
        """
        Calculate how much a specific word contributes to negative sentiment
        Uses ablation: compares sentiment with and without the word
        Returns: impact_score (higher = more negative impact)
        """
        if not word or not full_sentence:
            return 0.0
        
        word_clean = re.sub(r'[^\w\s]', '', word.lower().strip())
        
        # Check cache
        cache_key = f"{word_clean}|{full_sentence.lower()}"
        if cache_key in self.word_impact_cache:
            return self.word_impact_cache[cache_key]
        
        # Get baseline sentiment
        base_sentiment, base_toxic, base_tox_score = self._analyze_sentence_sentiment(full_sentence)
        
        # Quick check: if word is in offensive list, high impact
        if word_clean in self.offensive_words:
            impact = 0.9
            logger.debug(f"Dictionary hit: '{word_clean}' -> impact {impact:.2f}")
            self.word_impact_cache[cache_key] = impact
            return impact
        
        # Create sentence without the word (ablation)
        sentence_without = re.sub(r'\b' + re.escape(word) + r'\b', '', full_sentence, flags=re.IGNORECASE)
        sentence_without = re.sub(r'\s+', ' ', sentence_without).strip()
        
        # If removing word doesn't change sentence much, skip expensive analysis
        if len(sentence_without) < 3 or len(word_clean) < 2:
            return 0.0
        
        # Analyze without the word
        new_sentiment, new_toxic, new_tox_score = self._analyze_sentence_sentiment(sentence_without)
        
        # Calculate impact: how much did removing the word improve sentiment?
        sentiment_improvement = new_sentiment - base_sentiment
        toxicity_reduction = base_tox_score - new_tox_score
        
        # Combined impact score
        impact = 0.0
        
        if sentiment_improvement > 0.1:  # Sentiment improved
            impact += sentiment_improvement * 0.6
        
        if toxicity_reduction > 0.1:  # Toxicity reduced
            impact += toxicity_reduction * 0.8
        
        # If base sentence was toxic and word removal made it non-toxic
        if base_toxic and not new_toxic:
            impact += 0.5
        
        impact = min(1.0, max(0.0, impact))
        
        logger.debug(f"Word impact: '{word_clean}' -> {impact:.2f} (sent: {sentiment_improvement:.2f}, tox: {toxicity_reduction:.2f})")
        
        # Cache result
        if len(self.word_impact_cache) >= self.cache_size:
            self.word_impact_cache.pop(next(iter(self.word_impact_cache)))
        self.word_impact_cache[cache_key] = impact
        
        return impact
    
    def _should_blur_word(self, word, sentence, impact_threshold=0.3):
        """
        Decide if a word should be blurred based on its impact in context
        """
        impact = self._calculate_word_impact(word, sentence)
        return impact >= impact_threshold
    
    def blur_hate_text(self, image, confidence_threshold=0.5):
        """
        Context-aware blurring: analyze sentences first, then blur problematic words
        """
        if image is None or image.size == 0:
            return image
        
        result_image = image.copy()
        height, width = image.shape[:2]
        
        try:
            # Get word-level detections
            word_detections = self.detector.detect_words_precise(image, confidence_threshold)
            
            if not word_detections:
                logger.info("✓ No text detected")
                return result_image
            
            # Group words into sentences (words close in Y coordinate)
            sentences = self._group_words_into_sentences(word_detections)
            
            blurred_count = 0
            blurred_words = []
            
            for sentence_words in sentences:
                # Reconstruct full sentence
                full_text = ' '.join([w[1] for w in sentence_words])
                
                # Analyze sentence sentiment
                sentiment, is_toxic, tox_score = self._analyze_sentence_sentiment(full_text)
                
                logger.info(f"📝 Sentence: '{full_text[:60]}...' | Sentiment: {sentiment:.2f} | Toxic: {is_toxic} ({tox_score:.2f})")
                
                # Only process if sentence is negative or toxic
                if sentiment < self.sentiment_threshold or is_toxic:
                    logger.info(f"⚠️  Negative/toxic sentence detected, analyzing words...")
                    
                    # Analyze each word's impact
                    for (x1, y1, x2, y2), word, conf in sentence_words:
                        if self._should_blur_word(word, full_text):
                            logger.info(f"🚫 Blurring: '{word}' (contributes to negativity)")
                            blurred_count += 1
                            blurred_words.append(word)
                            
                            # Apply blur
                            box_width = x2 - x1
                            box_height = y2 - y1
                            
                            pad_x = max(1, int(box_width * self.padding))
                            pad_y = max(1, int(box_height * self.padding))
                            
                            x1_blur = max(0, x1 - pad_x)
                            y1_blur = max(0, y1 - pad_y)
                            x2_blur = min(width, x2 + pad_x)
                            y2_blur = min(height, y2 + pad_y)
                            
                            roi = result_image[y1_blur:y2_blur, x1_blur:x2_blur]
                            if roi.size > 0:
                                sigma = max(5, min(30, box_width // 4))
                                blurred_roi = cv2.GaussianBlur(roi, self.blur_strength, sigma)
                                result_image[y1_blur:y2_blur, x1_blur:x2_blur] = blurred_roi
                else:
                    logger.info("✓ Sentence is neutral/positive, no blurring needed")
            
            if blurred_count > 0:
                logger.info(f"✓ Blurred {blurred_count} contextually toxic words: {blurred_words}")
            else:
                logger.info("✓ No problematic words found in context")
                
        except Exception as e:
            logger.error(f"Error in context-aware text blurring: {e}")
        
        return result_image
    
    def _group_words_into_sentences(self, word_detections, y_threshold=20):
        """
        Group detected words into sentences based on vertical proximity
        """
        if not word_detections:
            return []
        
        # Sort by Y coordinate
        sorted_words = sorted(word_detections, key=lambda x: x[0][1])
        
        sentences = []
        current_sentence = [sorted_words[0]]
        
        for i in range(1, len(sorted_words)):
            prev_y = sorted_words[i-1][0][1]
            curr_y = sorted_words[i][0][1]
            
            # If Y difference is small, same sentence
            if abs(curr_y - prev_y) <= y_threshold:
                current_sentence.append(sorted_words[i])
            else:
                # New sentence
                if current_sentence:
                    sentences.append(current_sentence)
                current_sentence = [sorted_words[i]]
        
        if current_sentence:
            sentences.append(current_sentence)
        
        return sentences
    
    def blur_hate_text_simple(self, image, confidence_threshold=0.5):
        """
        Simplified version for video processing (every 5th frame)
        Uses faster heuristics instead of full ablation analysis
        """
        if image is None or image.size == 0:
            return image
        
        result_image = image.copy()
        height, width = image.shape[:2]
        
        try:
            word_detections = self.detector.detect_words_precise(image, confidence_threshold)
            sentences = self._group_words_into_sentences(word_detections)
            
            blurred_count = 0
            
            for sentence_words in sentences:
                full_text = ' '.join([w[1] for w in sentence_words])
                sentiment, is_toxic, _ = self._analyze_sentence_sentiment(full_text)
                
                # Only blur if sentence is negative/toxic
                if sentiment < self.sentiment_threshold or is_toxic:
                    for (x1, y1, x2, y2), word, conf in sentence_words:
                        word_clean = re.sub(r'[^\w\s]', '', word.lower().strip())
                        
                        # Fast check: only dictionary lookup
                        if word_clean in self.offensive_words:
                            blurred_count += 1
                            
                            pad = 1
                            x1_blur = max(0, x1 - pad)
                            y1_blur = max(0, y1 - pad)
                            x2_blur = min(width, x2 + pad)
                            y2_blur = min(height, y2 + pad)
                            
                            roi = result_image[y1_blur:y2_blur, x1_blur:x2_blur]
                            if roi.size > 0:
                                blurred_roi = cv2.GaussianBlur(roi, self.blur_strength, 0)
                                result_image[y1_blur:y2_blur, x1_blur:x2_blur] = blurred_roi
            
            if blurred_count > 0:
                logger.debug(f"Quick-blurred {blurred_count} words")
                
        except Exception as e:
            logger.error(f"Error in simple text blurring: {e}")
        
        return result_image