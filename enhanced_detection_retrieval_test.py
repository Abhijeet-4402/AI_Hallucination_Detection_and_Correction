"""
Enhanced Integration Test: Detection Module with Retrieval Module

This script demonstrates the complete pipeline working together:
1. Question -> Evidence Retrieval (Member 1)
2. Evidence -> Hallucination Detection (Member 2)
3. Complete integration testing with real scenarios
"""

import os
import sys
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.retrieval.retrieval_module import EvidenceRetriever
from src.detection.detection_module import HallucinationDetector

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_complete_pipeline() -> bool:
    """Test the complete pipeline against a predefined set of scenarios."""
    logger.info("🚀 Running Predefined Test Scenarios...")
    logger.info("=" * 60)
    
    all_scenarios_passed = True
    results = []

    try:
        # Initialize modules
        retriever = EvidenceRetriever()
        detector = HallucinationDetector()
        
        # Expanded test scenarios for more robust testing
        test_scenarios = [
            # --- Original Scenarios ---
            {"name": "Correct factual (France)", "question": "What is the capital of France?", "answer": "Paris is the capital of France.", "expected_hallucination": False},
            {"name": "Incorrect factual (France)", "question": "What is the capital of France?", "answer": "Lyon is the capital of France.", "expected_hallucination": True},
            {"name": "Correct literature (Moby Dick)", "question": "Who wrote Moby Dick?", "answer": "Moby Dick was written by Herman Melville.", "expected_hallucination": False},
            {"name": "Correct astronomy (Mars)", "question": "Which planet is known as the Red Planet?", "answer": "Mars is known as the Red Planet.", "expected_hallucination": False},
            {"name": "Incorrect factual (Australia)", "question": "What is the capital of Australia?", "answer": "Sydney is the capital of Australia.", "expected_hallucination": True},
            
            # --- New, More Challenging Scenarios ---
            {"name": "Subtle factual error (Date)", "question": "When did World War II end?", "answer": "World War II ended in 1946.", "expected_hallucination": True},
            {"name": "Correct but obscure fact", "question": "What is the national animal of Scotland?", "answer": "The national animal of Scotland is the unicorn.", "expected_hallucination": False},
            {"name": "Incorrect attribution", "question": "Who painted the Mona Lisa?", "answer": "The Mona Lisa was painted by Vincent van Gogh.", "expected_hallucination": True},
            {"name": "Correct number", "question": "How many planets are in our solar system?", "answer": "There are eight planets in our solar system.", "expected_hallucination": False},
            {"name": "Non-existent concept", "question": "What are the health benefits of eating glass?", "answer": "Eating glass provides essential minerals for bone health.", "expected_hallucination": True},
        ]
        
        for i, scenario in enumerate(test_scenarios):
            logger.info(f"\n--- Scenario {i+1}: {scenario['name']} ---")
            logger.info(f"Question: {scenario['question']}")
            
            # Step 1: Retrieve evidence
            logger.info("Step 1: Retrieving evidence...")
            evidence = retriever.retrieve_evidence(scenario['question'])
            logger.info(f"Retrieved {len(evidence)} evidence documents.")
            
            # Step 2: Detecting hallucinations...
            result = detector.detect_hallucination(scenario['answer'], evidence)
            
            # Step 3: Check result
            # Convert the DetectionResult object to a dictionary for logging/summary
            result_dict = {
                "is_hallucination": result.is_hallucination,
                "confidence_score": result.confidence_score,
                "detection_method": result.detection_method,
                "raw_answer": result.raw_answer
            }
            scenario_result = {**scenario, **result_dict}

            if result.is_hallucination != scenario['expected_hallucination']:
                logger.warning(f"⚠️ FAILED: Result doesn't match expectation (Expected: {scenario['expected_hallucination']}, Got: {result.is_hallucination})")
                scenario_result['success'] = False
                all_scenarios_passed = False
            else:
                logger.info(f"✅ PASSED: Result matches expectation.")
                scenario_result['success'] = True
            
            results.append(scenario_result)

    except Exception as e:
        logger.error(f"An error occurred during the pipeline test: {e}", exc_info=True)
        all_scenarios_passed = False
        
    print_summary(results)
    return all_scenarios_passed

def run_interactive_mode():
    """Allows for real-time testing with user-provided questions and answers."""
    logger.info("\n" + "=" * 60)
    logger.info("🚀 Entering Interactive Testing Mode")
    logger.info("Type 'exit' or 'quit' at any prompt to end.")
    logger.info("=" * 60)

    try:
        retriever = EvidenceRetriever()
        detector = HallucinationDetector()

        while True:
            question = input("\nEnter your question: ")
            if question.lower() in ['exit', 'quit']:
                break

            answer = input("Enter the answer to check: ")
            if answer.lower() in ['exit', 'quit']:
                break

            logger.info("\n--- Running Pipeline ---")
            logger.info("Step 1: Retrieving evidence...")
            evidence = retriever.retrieve_evidence(question)
            logger.info(f"Retrieved {len(evidence)} evidence documents.")
            
            logger.info("\nStep 2: Detecting hallucinations...")
            result = detector.detect_hallucination(answer, evidence)

            logger.info("\n--- Result ---")
            if result.is_hallucination:
                logger.warning(f"⚠️  Result: HALLUCINATION DETECTED")
            else:
                logger.info(f"✅  Result: NO HALLUCINATION DETECTED")
            
            logger.info(f"   - Method: {result.detection_method}")
            logger.info(f"   - Confidence Score: {result.confidence_score:.3f}")
            logger.info("----------------\n")

    except Exception as e:
        logger.error(f"An error occurred during interactive mode: {e}", exc_info=True)
    
    logger.info("Exiting interactive mode.")


def print_summary(results: List[Dict[str, Any]]):
    """Prints a summary of the test results."""
    logger.info("\n" + "=" * 60)
    logger.info("COMPREHENSIVE TEST SUMMARY")
    logger.info("=" * 60)

    if not results:
        logger.warning("No tests were run.")
        return

    successful_tests = [r for r in results if r.get('success')]
    failed_tests = [r for r in results if not r.get('success')]

    logger.info(f"Total Tests: {len(results)}")
    logger.info(f"Successful: {len(successful_tests)}")
    logger.info(f"Failed: {len(failed_tests)}")

    if failed_tests:
        logger.warning("\n--- FAILED TESTS ---")
        for r in failed_tests:
            logger.warning(f"  - Scenario: {r['name']}")
            logger.warning(f"    - Answer: '{r['raw_answer']}'")
            logger.warning(f"    - Expected: Hallucination={r['expected_hallucination']}")
            logger.warning(f"    - Actual: Hallucination={r['is_hallucination']} (Method: {r['detection_method']}, Score: {r['confidence_score']:.3f})")

def test_individual_components():
    """Placeholder for individual component tests."""
    logger.info("\n" + "=" * 60)
    logger.info("Individual component tests are not implemented in this script.")
    logger.info("=" * 60)

def main():
    """Run all tests and exit with a status code."""
    # Check for command-line arguments to select mode
    if 'interactive' in sys.argv:
        run_interactive_mode()
        sys.exit(0)
    else:
        logger.info("Starting integration test suite...")
        logger.info("To run in interactive mode, use: python enhanced_detection_retrieval_test.py interactive")
        
        pipeline_success = test_complete_pipeline()
        
        if pipeline_success:
            logger.info("\n🎉 All predefined tests passed!")
        else:
            logger.error("\n❌ Some predefined tests failed.")
            
        sys.exit(0 if pipeline_success else 1)

if __name__ == "__main__":
    main()
