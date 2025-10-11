<<<<<<< HEAD
"""
Test Suite for Member 1 Retrieval Module

This module contains comprehensive tests for the retrieval functionality
including Wikipedia integration, vector database operations, and the
main retrieval pipeline.
"""

import logging
import sys
import os
from typing import List, Dict, Any

# Add the src directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.retrieval.retrieval_module import EvidenceRetriever, retrieve_evidence
from src.retrieval.wikipedia_integration import WikipediaRetriever
from src.retrieval.vector_database import VectorDatabase
from src.retrieval.dataset_loader import TruthfulQALoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_wikipedia_integration():
    """Test Wikipedia API integration"""
    logger.info("Testing Wikipedia integration...")
    
    try:
        retriever = WikipediaRetriever(max_results=3)
        
        # Test keyword extraction
        question = "What is the capital of France?"
        keywords = retriever.extract_keywords(question)
        logger.info(f"Extracted keywords: {keywords}")
        assert len(keywords) > 0, "Should extract keywords from question"
        
        # Test evidence retrieval
        evidence_docs = retriever.retrieve_evidence_documents(question)
        logger.info(f"Retrieved {len(evidence_docs)} evidence documents")
        
        # Check that we got some results
        assert len(evidence_docs) > 0, "Should retrieve at least one evidence document"
        
        # Check document quality
        for doc in evidence_docs:
            assert len(doc) > 50, "Evidence documents should have substantial content"
            assert "france" in doc.lower() or "paris" in doc.lower(), "Should contain relevant information"
        
        logger.info("✓ Wikipedia integration test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Wikipedia integration test failed: {e}")
        return False

def test_vector_database():
    """Test vector database operations"""
    logger.info("Testing vector database...")
    
    try:
        # Create a test database
        db = VectorDatabase(collection_name="test_collection", persist_directory="./test_chroma_db")
        
        # Test adding documents
        test_docs = [
            "Paris is the capital of France.",
            "London is the capital of England.",
            "Berlin is the capital of Germany."
        ]
        
        doc_ids = db.add_documents(test_docs)
        assert len(doc_ids) == 3, "Should add all documents"
        
        # Test searching
        results = db.search_similar("What is the capital of France?", n_results=2)
        assert len(results) > 0, "Should find similar documents"
        
        # Test collection stats
        stats = db.get_collection_stats()
        assert stats['total_documents'] == 3, "Should have 3 documents"
        
        # Clean up
        db.clear_collection()
        
        logger.info("✓ Vector database test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Vector database test failed: {e}")
        return False

def test_evidence_retriever():
    """Test the main evidence retriever"""
    logger.info("Testing evidence retriever...")
    
    try:
        retriever = EvidenceRetriever(max_evidence_docs=3, similarity_threshold=0.5)
        
        # Test evidence retrieval
        question = "What is the population of Tokyo?"
        evidence_docs = retriever.retrieve_evidence(question)
        
        logger.info(f"Retrieved {len(evidence_docs)} evidence documents")
        assert len(evidence_docs) > 0, "Should retrieve evidence documents"
        
        # Test evidence with scores
        evidence_with_scores = retriever.get_evidence_with_scores(question)
        assert len(evidence_with_scores) > 0, "Should get evidence with scores"
        
        for result in evidence_with_scores:
            assert 'document' in result, "Should have document field"
            assert 'similarity_score' in result, "Should have similarity score"
            assert 'is_relevant' in result, "Should have relevance flag"
        
        # Test cache stats
        stats = retriever.get_cache_stats()
        assert 'total_documents' in stats, "Should have cache statistics"
        
        logger.info("✓ Evidence retriever test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Evidence retriever test failed: {e}")
        return False

def test_convenience_function():
    """Test the main convenience function"""
    logger.info("Testing convenience function...")
    
    try:
        question = "Who wrote Romeo and Juliet?"
        evidence_docs = retrieve_evidence(question)
        
        assert isinstance(evidence_docs, list), "Should return a list"
        assert len(evidence_docs) > 0, "Should retrieve evidence documents"
        
        # Check that documents contain relevant information
        combined_text = " ".join(evidence_docs).lower()
        assert "shakespeare" in combined_text, "Should contain relevant information about Shakespeare"
        
        logger.info("✓ Convenience function test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Convenience function test failed: {e}")
        return False

def test_truthfulqa_loader():
    """Test TruthfulQA dataset loader"""
    logger.info("Testing TruthfulQA dataset loader...")
    
    try:
        loader = TruthfulQALoader()
        
        # Test getting sample questions
        samples = loader.get_sample_questions(num_samples=5)
        assert len(samples) > 0, "Should get sample questions"
        
        # Check sample structure
        for sample in samples:
            assert 'question' in sample, "Should have question field"
            assert 'best_answer' in sample, "Should have best_answer field"
            assert 'category' in sample, "Should have category field"
        
        # Test getting categories
        categories = loader.get_all_categories()
        assert len(categories) > 0, "Should get categories"
        
        # Test dataset info
        info = loader.get_dataset_info()
        assert 'total_samples' in info, "Should have total samples info"
        
        logger.info("✓ TruthfulQA loader test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ TruthfulQA loader test failed: {e}")
        return False

def test_integration_with_detection():
    """Test integration with Member 2's detection module"""
    logger.info("Testing integration with detection module...")
    
    try:
        # Import detection module
        from src.detection.detection_module import detect_hallucination
        
        # Test question and answer
        question = "What is the largest planet in our solar system?"
        raw_answer = "Jupiter is the largest planet in our solar system."
        
        # Get evidence using Member 1's retrieval
        evidence_docs = retrieve_evidence(question)
        assert len(evidence_docs) > 0, "Should retrieve evidence documents"
        
        # Test detection using detection module
        is_hallucination, confidence_score = detect_hallucination(raw_answer, evidence_docs)
        
        assert isinstance(is_hallucination, bool), "Should return boolean for hallucination"
        assert isinstance(confidence_score, float), "Should return float for confidence"
        assert 0.0 <= confidence_score <= 1.0, "Confidence score should be between 0 and 1"
        
        logger.info(f"Detection result: hallucination={is_hallucination}, confidence={confidence_score:.3f}")
        logger.info("✓ Integration test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Integration test failed: {e}")
        return False

def run_all_tests():
    """Run all tests and report results"""
    logger.info("Starting comprehensive test suite for Retrieval Module...")
    
    tests = [
        ("Wikipedia Integration", test_wikipedia_integration),
        ("Vector Database", test_vector_database),
        ("Evidence Retriever", test_evidence_retriever),
        ("Convenience Function", test_convenience_function),
        ("TruthfulQA Loader", test_truthfulqa_loader),
        ("Integration with Detection", test_integration_with_detection)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running test: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            if test_func():
                passed += 1
                logger.info(f"✓ {test_name} PASSED")
            else:
                logger.error(f"✗ {test_name} FAILED")
        except Exception as e:
            logger.error(f"✗ {test_name} FAILED with exception: {e}")
    
    logger.info(f"\n{'='*50}")
    logger.info(f"TEST RESULTS: {passed}/{total} tests passed")
    logger.info(f"{'='*50}")
    
    if passed == total:
        logger.info("🎉 All tests passed! Retrieval Module is working correctly.")
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
=======
"""
Test Suite for Member 1 Retrieval Module

This module contains comprehensive tests for the retrieval functionality
including Wikipedia integration, vector database operations, and the
main retrieval pipeline.
"""

import logging
import sys
import os
from typing import List, Dict, Any

# Add the src directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.retrieval.retrieval_module import EvidenceRetriever, retrieve_evidence
from src.retrieval.wikipedia_integration import WikipediaRetriever
from src.retrieval.vector_database import VectorDatabase
from src.retrieval.dataset_loader import TruthfulQALoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_wikipedia_integration():
    """Test Wikipedia API integration"""
    logger.info("Testing Wikipedia integration...")
    
    try:
        retriever = WikipediaRetriever(max_results=3)
        
        # Test keyword extraction
        question = "What is the capital of France?"
        keywords = retriever.extract_keywords(question)
        logger.info(f"Extracted keywords: {keywords}")
        assert len(keywords) > 0, "Should extract keywords from question"
        
        # Test evidence retrieval
        evidence_docs = retriever.retrieve_evidence_documents(question)
        logger.info(f"Retrieved {len(evidence_docs)} evidence documents")
        
        # Check that we got some results
        assert len(evidence_docs) > 0, "Should retrieve at least one evidence document"
        
        # Check document quality
        for doc in evidence_docs:
            assert len(doc) > 50, "Evidence documents should have substantial content"
            assert "france" in doc.lower() or "paris" in doc.lower(), "Should contain relevant information"
        
        logger.info("✓ Wikipedia integration test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Wikipedia integration test failed: {e}")
        return False

def test_vector_database():
    """Test vector database operations"""
    logger.info("Testing vector database...")
    
    try:
        # Create a test database
        db = VectorDatabase(collection_name="test_collection", persist_directory="./test_chroma_db")
        
        # Test adding documents
        test_docs = [
            "Paris is the capital of France.",
            "London is the capital of England.",
            "Berlin is the capital of Germany."
        ]
        
        doc_ids = db.add_documents(test_docs)
        assert len(doc_ids) == 3, "Should add all documents"
        
        # Test searching
        results = db.search_similar("What is the capital of France?", n_results=2)
        assert len(results) > 0, "Should find similar documents"
        
        # Test collection stats
        stats = db.get_collection_stats()
        assert stats['total_documents'] == 3, "Should have 3 documents"
        
        # Clean up
        db.clear_collection()
        
        logger.info("✓ Vector database test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Vector database test failed: {e}")
        return False

def test_evidence_retriever():
    """Test the main evidence retriever"""
    logger.info("Testing evidence retriever...")
    
    try:
        retriever = EvidenceRetriever(max_evidence_docs=3, similarity_threshold=0.5)
        
        # Test evidence retrieval
        question = "What is the population of Tokyo?"
        evidence_docs = retriever.retrieve_evidence(question)
        
        logger.info(f"Retrieved {len(evidence_docs)} evidence documents")
        assert len(evidence_docs) > 0, "Should retrieve evidence documents"
        
        # Test evidence with scores
        evidence_with_scores = retriever.get_evidence_with_scores(question)
        assert len(evidence_with_scores) > 0, "Should get evidence with scores"
        
        for result in evidence_with_scores:
            assert 'document' in result, "Should have document field"
            assert 'similarity_score' in result, "Should have similarity score"
            assert 'is_relevant' in result, "Should have relevance flag"
        
        # Test cache stats
        stats = retriever.get_cache_stats()
        assert 'total_documents' in stats, "Should have cache statistics"
        
        logger.info("✓ Evidence retriever test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Evidence retriever test failed: {e}")
        return False

def test_convenience_function():
    """Test the main convenience function"""
    logger.info("Testing convenience function...")
    
    try:
        question = "Who wrote Romeo and Juliet?"
        evidence_docs = retrieve_evidence(question)
        
        assert isinstance(evidence_docs, list), "Should return a list"
        assert len(evidence_docs) > 0, "Should retrieve evidence documents"
        
        # Check that documents contain relevant information
        combined_text = " ".join(evidence_docs).lower()
        assert "shakespeare" in combined_text, "Should contain relevant information about Shakespeare"
        
        logger.info("✓ Convenience function test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Convenience function test failed: {e}")
        return False

def test_truthfulqa_loader():
    """Test TruthfulQA dataset loader"""
    logger.info("Testing TruthfulQA dataset loader...")
    
    try:
        loader = TruthfulQALoader()
        
        # Test getting sample questions
        samples = loader.get_sample_questions(num_samples=5)
        assert len(samples) > 0, "Should get sample questions"
        
        # Check sample structure
        for sample in samples:
            assert 'question' in sample, "Should have question field"
            assert 'best_answer' in sample, "Should have best_answer field"
            assert 'category' in sample, "Should have category field"
        
        # Test getting categories
        categories = loader.get_all_categories()
        assert len(categories) > 0, "Should get categories"
        
        # Test dataset info
        info = loader.get_dataset_info()
        assert 'total_samples' in info, "Should have total samples info"
        
        logger.info("✓ TruthfulQA loader test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ TruthfulQA loader test failed: {e}")
        return False

def test_integration_with_detection():
    """Test integration with Member 2's detection module"""
    logger.info("Testing integration with detection module...")
    
    try:
        # Import detection module
        from src.detection.detection_module import detect_hallucination
        
        # Test question and answer
        question = "What is the largest planet in our solar system?"
        raw_answer = "Jupiter is the largest planet in our solar system."
        
        # Get evidence using Member 1's retrieval
        evidence_docs = retrieve_evidence(question)
        assert len(evidence_docs) > 0, "Should retrieve evidence documents"
        
        # Test detection using detection module
        is_hallucination, confidence_score = detect_hallucination(raw_answer, evidence_docs)
        
        assert isinstance(is_hallucination, bool), "Should return boolean for hallucination"
        assert isinstance(confidence_score, float), "Should return float for confidence"
        assert 0.0 <= confidence_score <= 1.0, "Confidence score should be between 0 and 1"
        
        logger.info(f"Detection result: hallucination={is_hallucination}, confidence={confidence_score:.3f}")
        logger.info("✓ Integration test passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Integration test failed: {e}")
        return False

def run_all_tests():
    """Run all tests and report results"""
    logger.info("Starting comprehensive test suite for Retrieval Module...")
    
    tests = [
        ("Wikipedia Integration", test_wikipedia_integration),
        ("Vector Database", test_vector_database),
        ("Evidence Retriever", test_evidence_retriever),
        ("Convenience Function", test_convenience_function),
        ("TruthfulQA Loader", test_truthfulqa_loader),
        ("Integration with Detection", test_integration_with_detection)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"Running test: {test_name}")
        logger.info(f"{'='*50}")
        
        try:
            if test_func():
                passed += 1
                logger.info(f"✓ {test_name} PASSED")
            else:
                logger.error(f"✗ {test_name} FAILED")
        except Exception as e:
            logger.error(f"✗ {test_name} FAILED with exception: {e}")
    
    logger.info(f"\n{'='*50}")
    logger.info(f"TEST RESULTS: {passed}/{total} tests passed")
    logger.info(f"{'='*50}")
    
    if passed == total:
        logger.info("🎉 All tests passed! Retrieval Module is working correctly.")
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
>>>>>>> fb3451155c14f135a09046a366815aba1850f393
