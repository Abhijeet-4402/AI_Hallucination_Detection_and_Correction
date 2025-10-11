"""
Main Integration Script for Member 2 Detection Module

This script demonstrates how to use the detection module and integrates
with other parts of the system.
"""

import os
import sys
from typing import List, Dict, Any
from dotenv import load_dotenv

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.detection.detection_module import HallucinationDetector, DetectionResult
from src.detection.gemini_integration import GeminiLLM

# Load environment variables
load_dotenv()

class Member2DetectionPipeline:
    """Main pipeline class for Member 2's detection module"""
    
    def __init__(self):
        """Initialize the detection pipeline"""
        self.gemini_llm = None
        self.detector = None
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize Gemini LLM and detection components"""
        try:
            print("Initializing Member 2 Detection Pipeline...")
            
            # Initialize Gemini LLM
            self.gemini_llm = GeminiLLM()
            print("✅ Gemini LLM initialized")
            
            # Initialize Hallucination Detector
            self.detector = HallucinationDetector()
            print("✅ Hallucination Detector initialized")
            
            print("🎉 Member 2 Detection Pipeline ready!")
            
        except Exception as e:
            print(f"❌ Error initializing pipeline: {e}")
            raise
    
    def generate_and_detect(self, question: str, evidence_docs: List[str]) -> Dict[str, Any]:
        """
        Complete pipeline: Generate answer and detect hallucinations
        
        Args:
            question: The question to answer
            evidence_docs: List of evidence documents from Member 1
            
        Returns:
            Dictionary with complete detection results
        """
        try:
            print(f"\n🔍 Processing question: {question}")
            print(f"📚 Evidence documents: {len(evidence_docs)}")
            
            # Step 1: Generate answer using Gemini Pro
            print("🤖 Generating answer with Gemini Pro...")
            raw_answer = self.gemini_llm.generate_answer(question)
            print(f"📝 Generated answer: {raw_answer}")
            
            # Step 2: Detect hallucinations
            print("🔍 Detecting hallucinations...")
            detection_result = self.detector.detect_hallucination(raw_answer, evidence_docs)
            
            # Step 3: Prepare results for other modules
            result = {
                'question': question,
                'raw_answer': raw_answer,
                'evidence_docs': evidence_docs,
                'is_hallucination': detection_result.is_hallucination,
                'confidence_score': detection_result.confidence_score,
                'detection_method': detection_result.detection_method,
                'detection_details': detection_result.to_dict()
            }
            
            # Print results
            print(f"\n📊 Detection Results:")
            print(f"   - Hallucination Detected: {detection_result.is_hallucination}")
            print(f"   - Confidence Score: {detection_result.confidence_score:.3f}")
            print(f"   - Detection Method: {detection_result.detection_method}")
            
            if detection_result.is_hallucination:
                print("⚠️  Hallucination detected! Answer should be corrected.")
            else:
                print("✅ No hallucination detected. Answer appears reliable.")
            
            return result
            
        except Exception as e:
            print(f"❌ Error in generate_and_detect: {e}")
            raise
    
    def process_question(self, question: str, evidence_docs: List[str]) -> tuple:
        """
        Simplified interface for integration with other modules
        
        Args:
            question: The question to answer
            evidence_docs: List of evidence documents
            
        Returns:
            Tuple of (is_hallucination, confidence_score, raw_answer)
        """
        result = self.generate_and_detect(question, evidence_docs)
        return (
            result['is_hallucination'],
            result['confidence_score'],
            result['raw_answer']
        )

def demo_with_sample_data():
    """Demonstrate the detection module with sample data"""
    print("🎯 Member 2 Detection Module Demo")
    print("=" * 50)
    
    # Sample questions and evidence
    sample_cases = [
        {
            "question": "What is the capital of France?",
            "evidence": [
                "Paris is the capital and largest city of France.",
                "France's capital city is Paris, located in the north-central part of the country."
            ]
        },
        {
            "question": "Who wrote Romeo and Juliet?",
            "evidence": [
                "William Shakespeare wrote Romeo and Juliet.",
                "Romeo and Juliet is a tragedy written by William Shakespeare early in his career."
            ]
        },
        {
            "question": "What is the capital of Australia?",
            "evidence": [
                "Canberra is the capital city of Australia.",
                "Australia's capital is Canberra, not Sydney or Melbourne."
            ]
        }
    ]
    
    try:
        # Initialize pipeline
        pipeline = Member2DetectionPipeline()
        
        # Process each sample case
        for i, case in enumerate(sample_cases, 1):
            print(f"\n{'='*60}")
            print(f"Sample Case {i}")
            print(f"{'='*60}")
            
            result = pipeline.generate_and_detect(
                case["question"], 
                case["evidence"]
            )
            
            print(f"\n📋 Complete Result:")
            print(f"   Question: {result['question']}")
            print(f"   Raw Answer: {result['raw_answer']}")
            print(f"   Is Hallucination: {result['is_hallucination']}")
            print(f"   Confidence Score: {result['confidence_score']:.3f}")
            print(f"   Detection Method: {result['detection_method']}")
        
        print(f"\n🎉 Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")

def main():
    """Main function"""
    # Check if API key is available
    if not os.getenv("GEMINI_API_KEY"):
        print("❌ GEMINI_API_KEY not found in environment variables.")
        print("Please create a .env file with your Gemini API key.")
        print("Example: GEMINI_API_KEY=your_api_key_here")
        return
    
    # Run demo
    demo_with_sample_data()

if __name__ == "__main__":
    main()
