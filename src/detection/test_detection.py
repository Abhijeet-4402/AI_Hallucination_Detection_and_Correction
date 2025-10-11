<<<<<<< HEAD
"""
Test Suite for Member 2 Detection Module

This script tests the hallucination detection functionality including:
1. Gemini Pro integration
2. Semantic similarity checking
3. Contradiction detection
4. End-to-end detection pipeline
"""

import os
import sys
from typing import List
from dotenv import load_dotenv

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.detection.detection_module import HallucinationDetector, DetectionResult
from src.detection.gemini_integration import GeminiLLM

# Load environment variables
load_dotenv()

def test_gemini_connection():
    """Test Gemini Pro API connection"""
    print("=" * 50)
    print("Testing Gemini Pro API Connection")
    print("=" * 50)
    
    try:
        llm = GeminiLLM()
        success = llm.test_connection()
        
        if success:
            print("✅ Gemini Pro API connection successful!")
            return True
        else:
            print("❌ Gemini Pro API connection failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Gemini connection: {e}")
        return False

def test_answer_generation():
    """Test answer generation with Gemini Pro"""
    print("\n" + "=" * 50)
    print("Testing Answer Generation")
    print("=" * 50)
    
    try:
        llm = GeminiLLM()
        
        test_questions = [
            "What is the capital of France?",
            "Who wrote Romeo and Juliet?",
            "What is photosynthesis?"
        ]
        
        for question in test_questions:
            print(f"\nQuestion: {question}")
            answer = llm.generate_answer(question)
            print(f"Answer: {answer}")
            print("-" * 30)
        
        print("✅ Answer generation test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error in answer generation test: {e}")
        return False

def test_semantic_similarity():
    """Test semantic similarity detection"""
    print("\n" + "=" * 50)
    print("Testing Semantic Similarity Detection")
    print("=" * 50)
    
    try:
        detector = HallucinationDetector()
        
        # Test cases with different similarity levels
        test_cases = [
            {
                "answer": "Paris is the capital of France.",
                "evidence": ["Paris is the capital and largest city of France.", "France's capital city is Paris."],
                "expected_high_similarity": True
            },
            {
                "answer": "The sky is green and the grass is blue.",
                "evidence": ["The sky is blue and the grass is green.", "Blue is the color of the sky."],
                "expected_high_similarity": False
            },
            {
                "answer": "Shakespeare wrote many famous plays including Romeo and Juliet.",
                "evidence": ["William Shakespeare was an English playwright who wrote Romeo and Juliet.", "Romeo and Juliet is a tragedy by Shakespeare."],
                "expected_high_similarity": True
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\nTest Case {i}:")
            print(f"Answer: {test_case['answer']}")
            print(f"Evidence: {test_case['evidence']}")
            
            similarity_score = detector.check_similarity(
                test_case['answer'], 
                test_case['evidence']
            )
            
            print(f"Similarity Score: {similarity_score:.3f}")
            
            if test_case['expected_high_similarity']:
                if similarity_score > 0.7:
                    print("✅ High similarity detected as expected")
                else:
                    print("⚠️  Lower similarity than expected")
            else:
                if similarity_score < 0.7:
                    print("✅ Low similarity detected as expected")
                else:
                    print("⚠️  Higher similarity than expected")
        
        print("\n✅ Semantic similarity test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error in semantic similarity test: {e}")
        return False

def test_contradiction_detection():
    """Test contradiction detection using NLI"""
    print("\n" + "=" * 50)
    print("Testing Contradiction Detection")
    print("=" * 50)
    
    try:
        detector = HallucinationDetector()
        
        # Test cases for contradiction detection
        test_cases = [
            {
                "answer": "The Earth is flat.",
                "evidence": ["The Earth is round and spherical in shape.", "Scientific evidence shows Earth is a sphere."],
                "expected_contradiction": True
            },
            {
                "answer": "Water boils at 100 degrees Celsius at sea level.",
                "evidence": ["Water boils at 100°C at standard atmospheric pressure.", "The boiling point of water is 100 degrees Celsius."],
                "expected_contradiction": False
            },
            {
                "answer": "The sun rises in the west.",
                "evidence": ["The sun rises in the east and sets in the west.", "Sunrise occurs in the eastern direction."],
                "expected_contradiction": True
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\nTest Case {i}:")
            print(f"Answer: {test_case['answer']}")
            print(f"Evidence: {test_case['evidence']}")
            
            has_contradiction = detector.check_contradiction(
                test_case['answer'], 
                test_case['evidence']
            )
            
            print(f"Contradiction Detected: {has_contradiction}")
            
            if has_contradiction == test_case['expected_contradiction']:
                print("✅ Contradiction detection result as expected")
            else:
                print("⚠️  Contradiction detection result unexpected")
        
        print("\n✅ Contradiction detection test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error in contradiction detection test: {e}")
        return False

def test_end_to_end_detection():
    """Test the complete detection pipeline"""
    print("\n" + "=" * 50)
    print("Testing End-to-End Detection Pipeline")
    print("=" * 50)
    
    try:
        detector = HallucinationDetector()
        
        # Test cases for complete detection
        test_cases = [
            {
                "answer": "Paris is the capital of France and has the Eiffel Tower.",
                "evidence": ["Paris is the capital of France.", "The Eiffel Tower is located in Paris."],
                "expected_hallucination": False,
                "description": "Factual answer with supporting evidence"
            },
            {
                "answer": "The capital of France is London and it's located in England.",
                "evidence": ["Paris is the capital of France.", "London is the capital of England."],
                "expected_hallucination": True,
                "description": "Factual error - wrong capital"
            },
            {
                "answer": "Shakespeare wrote Romeo and Juliet in 1595.",
                "evidence": ["William Shakespeare wrote Romeo and Juliet.", "Romeo and Juliet was written by Shakespeare in the 1590s."],
                "expected_hallucination": False,
                "description": "Mostly accurate with minor details"
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\nTest Case {i}: {test_case['description']}")
            print(f"Answer: {test_case['answer']}")
            print(f"Evidence: {test_case['evidence']}")
            
            result = detector.detect_hallucination(
                test_case['answer'], 
                test_case['evidence']
            )
            
            print(f"Detection Result:")
            print(f"  - Is Hallucination: {result.is_hallucination}")
            print(f"  - Confidence Score: {result.confidence_score:.3f}")
            print(f"  - Detection Method: {result.detection_method}")
            
            if result.is_hallucination == test_case['expected_hallucination']:
                print("✅ Detection result as expected")
            else:
                print("⚠️  Detection result unexpected")
        
        print("\n✅ End-to-end detection test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error in end-to-end detection test: {e}")
        return False

def run_all_tests():
    """Run all tests and provide a summary"""
    print("🚀 Starting Member 2 Detection Module Tests")
    print("=" * 60)
    
    tests = [
        ("Gemini Connection", test_gemini_connection),
        ("Answer Generation", test_answer_generation),
        ("Semantic Similarity", test_semantic_similarity),
        ("Contradiction Detection", test_contradiction_detection),
        ("End-to-End Detection", test_end_to_end_detection)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with error: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your detection module is ready!")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    # Check if API key is available
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ GEMINI_API_KEY not found in environment variables.")
        print("Please create a .env file with your Gemini API key.")
        print("Example: GEMINI_API_KEY=your_api_key_here")
        sys.exit(1)
    
    # Run all tests
    success = run_all_tests()
    sys.exit(0 if success else 1)
=======
"""
Test Suite for Member 2 Detection Module

This script tests the hallucination detection functionality including:
1. Gemini Pro integration
2. Semantic similarity checking
3. Contradiction detection
4. End-to-end detection pipeline
"""

import os
import sys
from typing import List
from dotenv import load_dotenv

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.member2_detection.detection_module import HallucinationDetector, DetectionResult
from src.member2_detection.gemini_integration import GeminiLLM

# Load environment variables
load_dotenv()

def test_gemini_connection():
    """Test Gemini Pro API connection"""
    print("=" * 50)
    print("Testing Gemini Pro API Connection")
    print("=" * 50)
    
    try:
        llm = GeminiLLM()
        success = llm.test_connection()
        
        if success:
            print("✅ Gemini Pro API connection successful!")
            return True
        else:
            print("❌ Gemini Pro API connection failed!")
            return False
            
    except Exception as e:
        print(f"❌ Error testing Gemini connection: {e}")
        return False

def test_answer_generation():
    """Test answer generation with Gemini Pro"""
    print("\n" + "=" * 50)
    print("Testing Answer Generation")
    print("=" * 50)
    
    try:
        llm = GeminiLLM()
        
        test_questions = [
            "What is the capital of France?",
            "Who wrote Romeo and Juliet?",
            "What is photosynthesis?"
        ]
        
        for question in test_questions:
            print(f"\nQuestion: {question}")
            answer = llm.generate_answer(question)
            print(f"Answer: {answer}")
            print("-" * 30)
        
        print("✅ Answer generation test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error in answer generation test: {e}")
        return False

def test_semantic_similarity():
    """Test semantic similarity detection"""
    print("\n" + "=" * 50)
    print("Testing Semantic Similarity Detection")
    print("=" * 50)
    
    try:
        detector = HallucinationDetector()
        
        # Test cases with different similarity levels
        test_cases = [
            {
                "answer": "Paris is the capital of France.",
                "evidence": ["Paris is the capital and largest city of France.", "France's capital city is Paris."],
                "expected_high_similarity": True
            },
            {
                "answer": "The sky is green and the grass is blue.",
                "evidence": ["The sky is blue and the grass is green.", "Blue is the color of the sky."],
                "expected_high_similarity": False
            },
            {
                "answer": "Shakespeare wrote many famous plays including Romeo and Juliet.",
                "evidence": ["William Shakespeare was an English playwright who wrote Romeo and Juliet.", "Romeo and Juliet is a tragedy by Shakespeare."],
                "expected_high_similarity": True
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\nTest Case {i}:")
            print(f"Answer: {test_case['answer']}")
            print(f"Evidence: {test_case['evidence']}")
            
            similarity_score = detector.check_similarity(
                test_case['answer'], 
                test_case['evidence']
            )
            
            print(f"Similarity Score: {similarity_score:.3f}")
            
            if test_case['expected_high_similarity']:
                if similarity_score > 0.7:
                    print("✅ High similarity detected as expected")
                else:
                    print("⚠️  Lower similarity than expected")
            else:
                if similarity_score < 0.7:
                    print("✅ Low similarity detected as expected")
                else:
                    print("⚠️  Higher similarity than expected")
        
        print("\n✅ Semantic similarity test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error in semantic similarity test: {e}")
        return False

def test_contradiction_detection():
    """Test contradiction detection using NLI"""
    print("\n" + "=" * 50)
    print("Testing Contradiction Detection")
    print("=" * 50)
    
    try:
        detector = HallucinationDetector()
        
        # Test cases for contradiction detection
        test_cases = [
            {
                "answer": "The Earth is flat.",
                "evidence": ["The Earth is round and spherical in shape.", "Scientific evidence shows Earth is a sphere."],
                "expected_contradiction": True
            },
            {
                "answer": "Water boils at 100 degrees Celsius at sea level.",
                "evidence": ["Water boils at 100°C at standard atmospheric pressure.", "The boiling point of water is 100 degrees Celsius."],
                "expected_contradiction": False
            },
            {
                "answer": "The sun rises in the west.",
                "evidence": ["The sun rises in the east and sets in the west.", "Sunrise occurs in the eastern direction."],
                "expected_contradiction": True
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\nTest Case {i}:")
            print(f"Answer: {test_case['answer']}")
            print(f"Evidence: {test_case['evidence']}")
            
            has_contradiction = detector.check_contradiction(
                test_case['answer'], 
                test_case['evidence']
            )
            
            print(f"Contradiction Detected: {has_contradiction}")
            
            if has_contradiction == test_case['expected_contradiction']:
                print("✅ Contradiction detection result as expected")
            else:
                print("⚠️  Contradiction detection result unexpected")
        
        print("\n✅ Contradiction detection test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error in contradiction detection test: {e}")
        return False

def test_end_to_end_detection():
    """Test the complete detection pipeline"""
    print("\n" + "=" * 50)
    print("Testing End-to-End Detection Pipeline")
    print("=" * 50)
    
    try:
        detector = HallucinationDetector()
        
        # Test cases for complete detection
        test_cases = [
            {
                "answer": "Paris is the capital of France and has the Eiffel Tower.",
                "evidence": ["Paris is the capital of France.", "The Eiffel Tower is located in Paris."],
                "expected_hallucination": False,
                "description": "Factual answer with supporting evidence"
            },
            {
                "answer": "The capital of France is London and it's located in England.",
                "evidence": ["Paris is the capital of France.", "London is the capital of England."],
                "expected_hallucination": True,
                "description": "Factual error - wrong capital"
            },
            {
                "answer": "Shakespeare wrote Romeo and Juliet in 1595.",
                "evidence": ["William Shakespeare wrote Romeo and Juliet.", "Romeo and Juliet was written by Shakespeare in the 1590s."],
                "expected_hallucination": False,
                "description": "Mostly accurate with minor details"
            }
        ]
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\nTest Case {i}: {test_case['description']}")
            print(f"Answer: {test_case['answer']}")
            print(f"Evidence: {test_case['evidence']}")
            
            result = detector.detect_hallucination(
                test_case['answer'], 
                test_case['evidence']
            )
            
            print(f"Detection Result:")
            print(f"  - Is Hallucination: {result.is_hallucination}")
            print(f"  - Confidence Score: {result.confidence_score:.3f}")
            print(f"  - Detection Method: {result.detection_method}")
            
            if result.is_hallucination == test_case['expected_hallucination']:
                print("✅ Detection result as expected")
            else:
                print("⚠️  Detection result unexpected")
        
        print("\n✅ End-to-end detection test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error in end-to-end detection test: {e}")
        return False

def run_all_tests():
    """Run all tests and provide a summary"""
    print("🚀 Starting Member 2 Detection Module Tests")
    print("=" * 60)
    
    tests = [
        ("Gemini Connection", test_gemini_connection),
        ("Answer Generation", test_answer_generation),
        ("Semantic Similarity", test_semantic_similarity),
        ("Contradiction Detection", test_contradiction_detection),
        ("End-to-End Detection", test_end_to_end_detection)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test failed with error: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your detection module is ready!")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    # Check if API key is available
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ GEMINI_API_KEY not found in environment variables.")
        print("Please create a .env file with your Gemini API key.")
        print("Example: GEMINI_API_KEY=your_api_key_here")
        sys.exit(1)
    
    # Run all tests
    success = run_all_tests()
    sys.exit(0 if success else 1)
>>>>>>> fb3451155c14f135a09046a366815aba1850f393
