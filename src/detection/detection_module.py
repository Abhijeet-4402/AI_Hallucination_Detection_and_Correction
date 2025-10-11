"""
Core Detection Logic for AI Hallucination Detection

This module implements the two-step hallucination detection process:
1. Semantic Similarity Check using sentence transformers
2. Natural Language Inference (NLI) for contradiction detection
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging
import torch
from nltk.tokenize import sent_tokenize
import nltk

# Configure logging FIRST, so the logger is available for all subsequent calls.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download the necessary NLTK tokenizer models if they are not already present.
# This prevents LookupError during runtime.
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.info("NLTK 'punkt' resource not found. Downloading...")
    nltk.download('punkt')

# The traceback indicates that 'punkt_tab' is also required by the tokenizer.
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    logger.info("NLTK 'punkt_tab' resource not found. Downloading...")
    nltk.download('punkt_tab')


class DetectionResult:
    """Container for detection results"""
    def __init__(self, is_hallucination: bool, confidence_score: float, 
                 detection_method: str, raw_answer: str, evidence_docs: List[str]):
        self.is_hallucination = is_hallucination
        self.confidence_score = confidence_score
        self.detection_method = detection_method
        self.raw_answer = raw_answer
        self.evidence_docs = evidence_docs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy serialization"""
        return {
            'is_hallucination': self.is_hallucination,
            'confidence_score': self.confidence_score,
            'detection_method': self.detection_method,
            'raw_answer': self.raw_answer,
            'evidence_docs': self.evidence_docs
        }

class HallucinationDetector:
    """Main class for hallucination detection"""
    
    def __init__(self, 
                 similarity_model: str = "all-MiniLM-L6-v2",
                 nli_model: str = "roberta-large-mnli",
                 similarity_threshold: float = 0.5,
                 contradiction_threshold: float = 0.98,
                 entailment_threshold: float = 0.95, # New threshold for positive confirmation
                 device: str = "cpu"):
        """
        Initialize the HallucinationDetector
        
        Args:
            similarity_model: Sentence-transformer model for similarity
            nli_model: NLI model for contradiction detection
            similarity_threshold: Threshold below which an answer is a hallucination
            contradiction_threshold: NLI score above which an answer is a contradiction
            entailment_threshold: NLI score above which an answer is confirmed by evidence
            device: "cpu" or "cuda"
        """
        self.similarity_model_name = similarity_model
        self.nli_model_name = nli_model
        self.similarity_threshold = similarity_threshold
        self.contradiction_threshold = contradiction_threshold
        self.entailment_threshold = entailment_threshold # Store the new threshold
        self.device = device
        self._load_models(similarity_model, nli_model)

    def detect_hallucination(self, answer: str, evidence_docs: List[str]) -> DetectionResult:
        """
        Detects if an answer is a hallucination based on evidence using an entailment-first approach.
        """
        if not evidence_docs:
            logger.warning("No evidence provided. Cannot verify answer.")
            return DetectionResult(True, 1.0, "no_evidence", answer, [])

        # NLI model output order is [contradiction, neutral, entailment]
        ENTAILMENT_INDEX = 2
        CONTRADICTION_INDEX = 0

        # 1. Check for Entailment (Positive Confirmation) first
        for doc in evidence_docs:
            sentences = sent_tokenize(doc)
            for sentence in sentences:
                premise = sentence
                hypothesis = answer
                
                tokenized_input = self.nli_tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, max_length=512).to(self.device)
                with torch.no_grad():
                    logits = self.nli_model(**tokenized_input).logits
                
                probs = torch.softmax(logits, dim=1)[0]
                entailment_prob = probs[ENTAILMENT_INDEX].item()

                if entailment_prob > self.entailment_threshold:
                    logger.info(f"Entailment detected with score {entailment_prob:.3f}. The answer is supported by evidence.")
                    return DetectionResult(False, entailment_prob, "entailment", answer, evidence_docs)

        # 2. If no entailment, check for Contradiction
        for doc in evidence_docs:
            sentences = sent_tokenize(doc)
            for sentence in sentences:
                premise = sentence
                hypothesis = answer
                
                tokenized_input = self.nli_tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, max_length=512).to(self.device)
                with torch.no_grad():
                    logits = self.nli_model(**tokenized_input).logits
                
                probs = torch.softmax(logits, dim=1)[0]
                contradiction_prob = probs[CONTRADICTION_INDEX].item()
                
                if contradiction_prob > self.contradiction_threshold:
                    logger.info(f"Contradiction detected with score {contradiction_prob:.3f}")
                    return DetectionResult(True, contradiction_prob, "contradiction", answer, evidence_docs)

        # 3. If no strong NLI signal, fall back to semantic similarity
        combined_evidence = "\n".join(evidence_docs)
        answer_embedding = self.similarity_model.encode(answer, convert_to_tensor=True)
        evidence_embedding = self.similarity_model.encode(combined_evidence, convert_to_tensor=True)
        
        similarity_score = util.pytorch_cos_sim(answer_embedding, evidence_embedding).item()
        
        if similarity_score < self.similarity_threshold:
            logger.info(f"Low similarity detected. Score: {similarity_score:.3f}")
            return DetectionResult(True, 1 - similarity_score, "low_similarity", answer, evidence_docs)
            
        # 4. If all checks pass, it's not a hallucination
        logger.info(f"No strong hallucination signal found. Similarity score: {similarity_score:.3f}")
        return DetectionResult(False, similarity_score, "no_hallucination", answer, evidence_docs)

    def _load_models(self, similarity_model_name: str, nli_model_name: str):
        """Load the required ML models"""
        try:
            logger.info(f"Loading semantic similarity model: {similarity_model_name}")
            self.similarity_model = SentenceTransformer(similarity_model_name)
            
            logger.info(f"Loading NLI model: {nli_model_name}")
            self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_name)
            self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_name).to(self.device)
            
            logger.info("Models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise

# Global detector instance
_detector = None

def get_detector() -> HallucinationDetector:
    """Get or create the global detector instance"""
    global _detector
    if _detector is None:
        _detector = HallucinationDetector()
    return _detector

def detect_hallucination(raw_answer: str, evidence_docs: List[str]) -> Tuple[bool, float]:
    """
    Convenience function for the main detection logic
    
    Args:
        raw_answer: The generated answer to check
        evidence_docs: List of evidence documents
        
    Returns:
        Tuple of (is_hallucination, confidence_score)
    """
    detector = get_detector()
    result = detector.detect_hallucination(raw_answer, evidence_docs)
    return result.is_hallucination, result.confidence_score
