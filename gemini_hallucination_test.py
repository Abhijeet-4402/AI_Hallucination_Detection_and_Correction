"""
End-to-End Test: Gemini API + Hallucination Detection

This script performs a full, real-world test of the system by leveraging
the robust GeminiLLM class for API interaction.
"""
import os
import sys
import logging
from dotenv import load_dotenv

# Add the src directory to the path to import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.retrieval.retrieval_module import EvidenceRetriever
from src.detection.detection_module import HallucinationDetector
from src.detection.gemini_integration import GeminiLLM  # Using the new robust class

# --- Configuration ---
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# --- Main Functions ---

def run_full_test(questions_filepath: str):
    """
    Runs the full pipeline: gets answers from Gemini and checks for hallucinations.
    """
    # 1. Initialize our detection pipeline and the robust Gemini LLM
    logger.info("Initializing Evidence Retriever, Hallucination Detector, and Gemini LLM...")
    try:
        retriever = EvidenceRetriever()
        detector = HallucinationDetector()
        gemini_llm = GeminiLLM()  # Initialize the new class
    except Exception as e:
        logger.error(f"Fatal error during module initialization: {e}", exc_info=True)
        return

    # Check if the Gemini API is actually working after initialization
    if not gemini_llm.api_working:
        logger.error("Gemini API is not working. The test cannot proceed with live API calls.")
        logger.error("Please check your GEMINI_API_KEY, network connection, and API permissions.")
        return
    
    logger.info("Initialization complete. Gemini API is responsive.")

    # 2. Read questions from the file
    try:
        with open(questions_filepath, 'r') as f:
            questions = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    except FileNotFoundError:
        logger.error(f"Questions file not found at: {questions_filepath}")
        return

    # 3. Process each question
    logger.info("\n" + "="*60)
    logger.info(f"Starting end-to-end test with {len(questions)} questions...")
    logger.info("="*60 + "\n")

    for i, question in enumerate(questions):
        print(f"--- Processing Question {i+1}/{len(questions)}: {question} ---")

        # Step A: Get live answer from Gemini using the robust class
        gemini_answer = gemini_llm.generate_answer(question)
        
        # The new class returns specific strings on failure
        if "error" in gemini_answer.lower() or "unable to access" in gemini_answer.lower():
            logger.error(f"Failed to get a valid answer from Gemini: {gemini_answer}")
            print(f"Skipping due to API error.\n")
            continue
        
        print(f"Gemini's Answer: {gemini_answer}")

        # Step B: Retrieve evidence for the question
        evidence = retriever.retrieve_evidence(question)

        # Step C: Detect hallucination in Gemini's answer
        result = detector.detect_hallucination(gemini_answer, evidence)

        # Step D: Report the result
        print("\n--- Detection Result ---")
        if result.is_hallucination:
            print(f"🔴 STATUS: HALLUCINATION DETECTED")
        else:
            print(f"🟢 STATUS: NO HALLUCINATION DETECTED")
        
        print(f"   - Method: {result.detection_method}")
        print(f"   - Confidence: {result.confidence_score:.3f}\n")
        print("="*60 + "\n")

def main():
    """Main entry point of the script."""
    questions_file = 'data/sample_questions.txt'
    run_full_test(questions_file)

if __name__ == "__main__":
    main()