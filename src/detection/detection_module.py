"""
Core Detection Logic for AI Hallucination Detection

This module implements the two-step hallucination detection process:
1. Semantic Similarity Check using sentence transformers
2. Natural Language Inference (NLI) for contradiction detection
"""

import numpy as np
from typing import List, Tuple, Dict, Any
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    """Main class for detecting hallucinations in AI-generated text"""
    
    def __init__(self, similarity_threshold: float = 0.7):
        """
        Initialize the hallucination detector
        
        Args:
            similarity_threshold: Threshold for semantic similarity (0-1)
        """
        self.similarity_threshold = similarity_threshold
        self.similarity_model = None
        self.nli_pipeline = None
        self._load_models()
    
    def _load_models(self):
        """Load the required ML models"""
        try:
            logger.info("Loading semantic similarity model...")
            self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            logger.info("Loading NLI model for contradiction detection...")
            self.nli_pipeline = pipeline(
                "text-classification",
                model="roberta-large-mnli",
                return_all_scores=True
            )
            logger.info("Models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def check_similarity(self, raw_answer: str, evidence_docs: List[str]) -> float:
        """
        Check semantic similarity between answer and evidence documents
        
        Args:
            raw_answer: The generated answer to check
            evidence_docs: List of evidence documents
            
        Returns:
            Highest cosine similarity score (0-1)
        """
        if not evidence_docs:
            logger.warning("No evidence documents provided for similarity check")
            return 0.0
        
        try:
            # Generate embeddings
            answer_embedding = self.similarity_model.encode([raw_answer])
            evidence_embeddings = self.similarity_model.encode(evidence_docs)
            
            # Calculate cosine similarities
            similarities = []
            for evidence_emb in evidence_embeddings:
                similarity = np.dot(answer_embedding[0], evidence_emb) / (
                    np.linalg.norm(answer_embedding[0]) * np.linalg.norm(evidence_emb)
                )
                similarities.append(similarity)
            
            max_similarity = max(similarities)
            logger.info(f"Maximum similarity score: {max_similarity:.3f}")
            return float(max_similarity)
            
        except Exception as e:
            logger.error(f"Error in similarity check: {e}")
            return 0.0
    
    def check_contradiction(self, raw_answer: str, evidence_docs: List[str]) -> bool:
        """
        Check for contradictions using Natural Language Inference
        
        Args:
            raw_answer: The generated answer to check
            evidence_docs: List of evidence documents
            
        Returns:
            True if contradiction is detected, False otherwise
        """
        if not evidence_docs:
            logger.warning("No evidence documents provided for contradiction check")
            return False
        
        try:
            contradictions_found = 0
            
            for evidence in evidence_docs:
                # Create premise-hypothesis pair for NLI
                premise = evidence
                hypothesis = raw_answer
                
                # Get NLI predictions
                results = self.nli_pipeline(f"{premise} [SEP] {hypothesis}")
                
                # Check for contradiction (label index 2 in roberta-large-mnli)
                for result in results:
                    if result['label'] == 'CONTRADICTION' and result['score'] > 0.5:
                        contradictions_found += 1
                        logger.info(f"Contradiction detected with score: {result['score']:.3f}")
                        break
            
            has_contradiction = contradictions_found > 0
            logger.info(f"Contradiction check result: {has_contradiction}")
            return has_contradiction
            
        except Exception as e:
            logger.error(f"Error in contradiction check: {e}")
            return False
    
    def detect_hallucination(self, raw_answer: str, evidence_docs: List[str]) -> DetectionResult:
        """
        Main detection function that combines both methods
        
        Args:
            raw_answer: The generated answer to check
            evidence_docs: List of evidence documents
            
        Returns:
            DetectionResult object with detection outcome
        """
        logger.info("Starting hallucination detection...")
        logger.info(f"Answer to check: {raw_answer[:100]}...")
        logger.info(f"Number of evidence documents: {len(evidence_docs)}")
        
        # Step 1: Check for contradictions using NLI
        has_contradiction = self.check_contradiction(raw_answer, evidence_docs)
        
        # Step 2: Check semantic similarity
        similarity_score = self.check_similarity(raw_answer, evidence_docs)
        
        # Determine if hallucination is detected
        is_hallucination = has_contradiction or (similarity_score < self.similarity_threshold)
        
        # Determine which method triggered the detection
        if has_contradiction:
            detection_method = "contradiction_detected"
        elif similarity_score < self.similarity_threshold:
            detection_method = "low_similarity"
        else:
            detection_method = "no_hallucination"
        
        result = DetectionResult(
            is_hallucination=is_hallucination,
            confidence_score=similarity_score,
            detection_method=detection_method,
            raw_answer=raw_answer,
            evidence_docs=evidence_docs
        )
        
        logger.info(f"Detection complete. Result: {result.to_dict()}")
        return result

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
