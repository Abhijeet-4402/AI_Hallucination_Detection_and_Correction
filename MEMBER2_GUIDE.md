# Member 2: LLM & Detection Module - Complete Guide

## 🎯 Your Role in the Project

As **Member 2**, you are the **LLM & Detection Engineer** responsible for:
1. **Answer Generation** - Using Gemini Pro to generate initial answers
2. **Hallucination Detection** - Implementing two detection methods to identify false information
3. **System Integration** - Providing detection results to other team members

## 📁 Your Module Structure

```
src/member2_detection/
├── __init__.py              # Module initialization
├── detection_module.py      # Core detection logic
├── gemini_integration.py    # Gemini Pro API integration
├── test_detection.py        # Comprehensive test suite
└── main.py                  # Integration and demo script
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Key

```bash
# Copy environment template
copy .env.example .env

# Edit .env file and add your Gemini API key
GEMINI_API_KEY=your_actual_api_key_here
```

### 3. Test Your Module

```bash
# Run comprehensive tests
python src/member2_detection/test_detection.py

# Run demo
python src/member2_detection/main.py
```

## 🔧 How Your Detection Works

### Two-Step Detection Process:

1. **Semantic Similarity Check**
   - Uses `all-MiniLM-L6-v2` model to generate embeddings
   - Compares answer with evidence documents
   - Returns similarity score (0-1)

2. **Contradiction Detection**
   - Uses `roberta-large-mnli` model for Natural Language Inference
   - Detects if answer contradicts evidence
   - Returns boolean result

### Detection Logic:
```python
# If contradiction found OR similarity < threshold → Hallucination detected
is_hallucination = has_contradiction or (similarity_score < 0.7)
```

## 📊 Your Module's Output

Your detection module returns:
```python
{
    "is_hallucination": bool,      # True if hallucination detected
    "confidence_score": float,     # Similarity score (0-1)
    "detection_method": str,       # Which method triggered detection
    "raw_answer": str,             # Original answer from Gemini
    "evidence_docs": list          # Retrieved evidence documents
}
```

## 🤝 Integration with Other Members

### Input from Member 1 (Retrieval):
- `evidence_docs`: List of retrieved documents from knowledge base

### Output to Member 3 (Correction):
- `is_hallucination`: Boolean flag for correction trigger
- `confidence_score`: Score for logging and analysis
- `raw_answer`: Original answer to be corrected

### Integration Code Example:
```python
from src.member2_detection import detect_hallucination, GeminiLLM

# Generate answer
llm = GeminiLLM()
raw_answer = llm.generate_answer(question)

# Detect hallucinations
is_hallucination, confidence_score = detect_hallucination(raw_answer, evidence_docs)

# Pass to Member 3 if hallucination detected
if is_hallucination:
    corrected_data = member3_module.correct_answer(raw_answer, evidence_docs)
```

## 🧪 Testing Your Module

### Run All Tests:
```bash
python src/member2_detection/test_detection.py
```

### Test Categories:
1. **Gemini Connection Test** - Verifies API connectivity
2. **Answer Generation Test** - Tests answer generation
3. **Semantic Similarity Test** - Tests similarity detection
4. **Contradiction Detection Test** - Tests NLI contradiction detection
5. **End-to-End Test** - Tests complete pipeline

### Sample Test Cases:
- ✅ **Factual Answer**: "Paris is the capital of France" + Evidence about Paris
- ❌ **Hallucination**: "London is the capital of France" + Evidence about Paris
- ⚠️ **Edge Case**: "Shakespeare wrote Romeo and Juliet in 1595" + Evidence about Shakespeare

## 🔧 Configuration Options

### Environment Variables (.env):
```bash
GEMINI_API_KEY=your_api_key
SIMILARITY_THRESHOLD=0.7
CONTRADICTION_THRESHOLD=0.8
LOG_LEVEL=INFO
```

### Customizable Parameters:
- `similarity_threshold`: Threshold for similarity detection (default: 0.7)
- `temperature`: Gemini response creativity (default: 0.7)
- `max_output_tokens`: Maximum response length (default: 1024)

## 🐛 Troubleshooting

### Common Issues:

1. **API Key Error**:
   ```
   ❌ GEMINI_API_KEY not found
   ```
   **Solution**: Add your API key to `.env` file

2. **Model Loading Error**:
   ```
   ❌ Error loading models
   ```
   **Solution**: Ensure sufficient RAM (8GB+) and internet connection

3. **Import Error**:
   ```
   ❌ ModuleNotFoundError
   ```
   **Solution**: Install dependencies with `pip install -r requirements.txt`

### Performance Tips:
- First run downloads models (~2GB) - be patient
- Use GPU if available for faster inference
- Adjust similarity threshold based on your use case

## 📈 Performance Metrics

### Expected Performance:
- **Similarity Detection**: ~95% accuracy on factual questions
- **Contradiction Detection**: ~90% accuracy on clear contradictions
- **Processing Time**: ~2-5 seconds per question (depending on hardware)

### Optimization:
- Cache model loading for multiple questions
- Batch process multiple questions together
- Use smaller models for faster inference if needed

## 🎯 Next Steps

1. **Test Your Module**: Run the test suite to ensure everything works
2. **Get API Key**: Obtain Gemini Pro API key from Google AI Studio
3. **Coordinate with Team**: Share your module interface with other members
4. **Prepare Integration**: Ensure your output format matches Member 3's expectations

## 📞 Support

### For Your Module:
- Check `test_detection.py` for usage examples
- Review `main.py` for integration patterns
- See `About/Member-2-Task.md` for detailed requirements

### Team Coordination:
- Share your module's input/output format with other members
- Test integration points before final submission
- Document any changes to the interface

## 🎉 Success Criteria

Your module is complete when:
- ✅ All tests pass
- ✅ Gemini Pro integration works
- ✅ Detection accuracy is >85%
- ✅ Integration interface is ready for other members
- ✅ Documentation is complete

**You're ready to help build an amazing AI hallucination detection system!** 🚀
