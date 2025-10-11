
# Retrieval Module Setup Guide

## Prerequisites

Before running the Retrieval Module, ensure you have the following installed:

### 1. Python Installation
- **Python 3.8 or higher** is required
- Download from [python.org](https://www.python.org/downloads/) or install via Microsoft Store
- Verify installation: `python --version` or `python3 --version`

### 2. Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt

# Or install retrieval module specific dependencies:
pip install datasets>=2.14.0 wikipedia>=1.4.0 chromadb>=0.4.0 sentence-transformers>=2.2.0
```

## Quick Start

### 1. Basic Usage
```python
from src.retrieval.retrieval_module import retrieve_evidence

# Retrieve evidence for a question
question = "What is the capital of France?"
evidence_docs = retrieve_evidence(question)
print(f"Retrieved {len(evidence_docs)} evidence documents")
```

### 2. Advanced Usage
```python
from src.retrieval.retrieval_module import EvidenceRetriever

# Create retriever with custom settings
retriever = EvidenceRetriever(
    max_evidence_docs=5,
    similarity_threshold=0.7
)

# Get evidence with similarity scores
evidence_with_scores = retriever.get_evidence_with_scores(question)
for result in evidence_with_scores:
    print(f"Score: {result['similarity_score']:.3f}")
    print(f"Document: {result['document'][:100]}...")
```

### 3. Command Line Interface
```bash
# Run demonstration
python src/retrieval/main.py --mode demo

# Interactive mode
python src/retrieval/main.py --mode interactive

# Test single question
python src/retrieval/main.py --mode demo --question "What is the capital of Japan?"

# Run all tests
python src/retrieval/main.py --mode test
```

## Testing

### Run Comprehensive Tests
```bash
python src/retrieval/test_retrieval.py
```

### Test Individual Components
```python
# Test Wikipedia integration
from src.retrieval.wikipedia_integration import WikipediaRetriever
retriever = WikipediaRetriever()
docs = retriever.retrieve_evidence_documents("What is photosynthesis?")

# Test vector database
from src.retrieval.vector_database import VectorDatabase
db = VectorDatabase()
db.add_documents(["Test document"])
results = db.search_similar("test query")

# Test TruthfulQA dataset
from src.retrieval.dataset_loader import TruthfulQALoader
loader = TruthfulQALoader()
samples = loader.get_sample_questions(5)
```

## Integration with Other Modules

### With Detection Module
```python
from src.retrieval.retrieval_module import retrieve_evidence
from src.detection.detection_module import detect_hallucination

# Get evidence
question = "What is the largest planet?"
evidence_docs = retrieve_evidence(question)

# Generate answer (using Member 2's LLM)
raw_answer = "Jupiter is the largest planet in our solar system."

# Detect hallucination
is_hallucination, confidence = detect_hallucination(raw_answer, evidence_docs)
```

### With Correction Module
```python
# Evidence documents are passed directly to correction module
evidence_docs = retrieve_evidence(question)
# Correction module will use these documents for correction
```

## Configuration

### Environment Variables
Create a `.env` file in the project root:
```
# Optional: Custom ChromaDB settings
CHROMA_PERSIST_DIRECTORY=./chroma_db
CHROMA_COLLECTION_NAME=evidence_documents

# Optional: Wikipedia settings
WIKIPEDIA_LANGUAGE=en
WIKIPEDIA_MAX_RESULTS=5
```

### Custom Settings
```python
# Custom embedding model
retriever = EvidenceRetriever(embedding_model="all-mpnet-base-v2")

# Custom similarity threshold
retriever = EvidenceRetriever(similarity_threshold=0.8)

# Custom vector database location
from src.retrieval.vector_database import VectorDatabase
db = VectorDatabase(
    collection_name="my_collection",
    persist_directory="./my_chroma_db"
)
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure you're in the project root directory
   - Check that all dependencies are installed
   - Verify Python path includes the project directory

2. **Wikipedia API Errors**
   - Check internet connection
   - Wikipedia API may be temporarily unavailable
   - Rate limiting may apply (handled automatically)

3. **ChromaDB Issues**
   - Ensure write permissions in the project directory
   - Clear the database: `retriever.clear_cache()`

4. **Model Download Issues**
   - First run downloads models (may take time)
   - Ensure sufficient disk space
   - Check internet connection for model downloads

### Performance Tips

1. **Caching**: Evidence documents are cached in ChromaDB for faster retrieval
2. **Batch Processing**: Process multiple questions together for efficiency
3. **Model Loading**: Models are loaded once and reused
4. **Memory Usage**: Large models require 4-8GB RAM

## File Structure

```
src/retrieval/
├── __init__.py              # Module initialization
├── retrieval_module.py      # Main retrieval logic
├── wikipedia_integration.py # Wikipedia API integration
├── vector_database.py       # ChromaDB operations
├── dataset_loader.py        # TruthfulQA dataset handling
├── test_retrieval.py        # Comprehensive tests
├── main.py                  # Command-line interface
└── setup_guide.md          # This file
```

## API Reference

### Main Functions

- `retrieve_evidence(question: str) -> List[str]`: Main retrieval function
- `EvidenceRetriever`: Main class for evidence retrieval
- `WikipediaRetriever`: Wikipedia API integration
- `VectorDatabase`: ChromaDB vector storage
- `TruthfulQALoader`: Dataset loading utilities

### Key Methods

- `retrieve_evidence()`: Get evidence documents for a question
- `get_evidence_with_scores()`: Get evidence with similarity scores
- `clear_cache()`: Clear the vector database cache
- `get_cache_stats()`: Get cache statistics

## Support

For issues or questions:
1. Check this setup guide
2. Run the test suite: `python src/retrieval/test_retrieval.py`
3. Check the logs for detailed error messages
4. Verify all dependencies are installed correctly
