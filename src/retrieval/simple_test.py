"""
Simple Integration Test for Retrieval Module

This is a lightweight test that can be run to verify the basic functionality
of the retrieval module without requiring all dependencies to be installed.
"""

import sys
import os

# Add the src directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

def test_basic_imports():
    """Test that all modules can be imported"""
    print("Testing basic imports...")
    
    try:
        from src.retrieval.retrieval_module import retrieve_evidence, EvidenceRetriever
        print("[OK] retrieval_module imported successfully")
    except ImportError as e:
        print(f"[ERROR] Failed to import retrieval_module: {e}")
        return False
    
    try:
        from src.retrieval.wikipedia_integration import WikipediaRetriever
        print("[OK] wikipedia_integration imported successfully")
    except ImportError as e:
        print(f"[ERROR] Failed to import wikipedia_integration: {e}")
        return False
    
    try:
        from src.retrieval.vector_database import VectorDatabase
        print("[OK] vector_database imported successfully")
    except ImportError as e:
        print(f"[ERROR] Failed to import vector_database: {e}")
        return False
    
    try:
        from src.retrieval.dataset_loader import TruthfulQALoader
        print("[OK] dataset_loader imported successfully")
    except ImportError as e:
        print(f"[ERROR] Failed to import dataset_loader: {e}")
        return False
    
    return True

def test_detection_integration():
    """Test integration with detection module"""
    print("\nTesting integration with detection module...")
    
    try:
        # Import detection module
        from src.detection.detection_module import detect_hallucination
        print("[OK] detection module imported successfully")
        
        # Test the integration
        from src.retrieval.retrieval_module import retrieve_evidence
        
        # Simple test case
        question = "What is the capital of France?"
        raw_answer = "Paris is the capital of France."
        
        print(f"Testing with question: {question}")
        print(f"Testing with answer: {raw_answer}")
        
        # This would normally retrieve evidence, but we'll simulate it for the test
        evidence_docs = ["Paris is the capital and largest city of France."]
        
        # Test detection
        is_hallucination, confidence_score = detect_hallucination(raw_answer, evidence_docs)
        
        print(f"Detection result: hallucination={is_hallucination}, confidence={confidence_score:.3f}")
        print("[OK] Integration with detection module successful")
        
        return True
        
    except ImportError as e:
        print(f"[ERROR] Failed to import detection module: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Integration test failed: {e}")
        return False

def test_file_structure():
    """Test that all required files exist"""
    print("\nTesting file structure...")
    
    required_files = [
        "src/retrieval/__init__.py",
        "src/retrieval/retrieval_module.py",
        "src/retrieval/wikipedia_integration.py",
        "src/retrieval/vector_database.py",
        "src/retrieval/dataset_loader.py",
        "src/retrieval/test_retrieval.py",
        "src/retrieval/main.py",
        "src/retrieval/setup_guide.md",
        "requirements.txt"
    ]
    
    all_exist = True
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"[OK] {file_path} exists")
        else:
            print(f"[MISSING] {file_path} missing")
            all_exist = False
    
    return all_exist

def main():
    """Run all simple tests"""
    print("Retrieval Module - Simple Integration Test")
    print("=" * 60)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Basic Imports", test_basic_imports),
        ("Detection Integration", test_detection_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'-' * 40}")
        print(f"Running: {test_name}")
        print(f"{'-' * 40}")
        
        try:
            if test_func():
                passed += 1
                print(f"[PASSED] {test_name}")
            else:
                print(f"[FAILED] {test_name}")
        except Exception as e:
            print(f"[FAILED] {test_name} with exception: {e}")
    
    print(f"\n{'=' * 60}")
    print(f"RESULTS: {passed}/{total} tests passed")
    print(f"{'=' * 60}")
    
    if passed == total:
        print("[SUCCESS] All simple tests passed!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run full test suite: python src/retrieval/test_retrieval.py")
        print("3. Try the demo: python src/retrieval/main.py --mode demo")
        return True
    else:
        print(f"[ERROR] {total - passed} tests failed.")
        print("Please check the implementation and file structure.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
