<<<<<<< HEAD
# Member 1: Retrieval Module - Complete Guide

## 🎯 Your Role in the Project

As **Member 1**, you are the **Data & Retrieval Engineer** responsible for:
1. **Evidence Retrieval** - Fetching relevant documents from Wikipedia and knowledge bases
2. **Vector Database Management** - Storing and searching evidence using ChromaDB
3. **Semantic Search** - Using embeddings to find the most relevant evidence
4. **System Foundation** - Providing the data layer for the entire hallucination detection system

## 📁 Your Module Structure

```
src/retrieval/
├── __init__.py                    # Module initialization
├── retrieval_module.py            # Core retrieval logic and main functions
├── wikipedia_integration.py       # Wikipedia API integration
├── vector_database.py             # ChromaDB vector storage operations
├── dataset_loader.py              # TruthfulQA dataset handling
├── test_retrieval.py              # Comprehensive test suite
├── simple_test.py                 # Lightweight integration test
├── main.py                        # Command-line interface and demos
├── setup_guide.md                 # Detailed setup and usage guide
└── IMPLEMENTATION_SUMMARY.md      # Complete implementation summary
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

### 2. Test Your Module

```bash
# Run simple integration test
python src/retrieval/simple_test.py

# Run comprehensive tests
python src/retrieval/test_retrieval.py

# Run demo
python src/retrieval/main.py --mode demo
```

### 3. Try Interactive Mode

```bash
# Interactive question-answering
python src/retrieval/main.py --mode interactive
```

## 🔧 How Your Retrieval Works

### Three-Stage Retrieval Process:

1. **Wikipedia API Integration**
   - Extracts keywords from user questions
   - Searches Wikipedia for relevant pages
   - Retrieves and cleans page content
   - Handles disambiguation and errors gracefully

2. **Semantic Embedding**
   - Uses `all-MiniLM-L6-v2` model to generate embeddings
   - Creates vector representations of questions and documents
   - Enables semantic similarity search

3. **Vector Database Storage**
   - Stores embeddings in ChromaDB for fast retrieval
   - Caches results for improved performance
   - Provides similarity-based search capabilities

### Retrieval Logic:
```python
# Main retrieval function
evidence_docs = retrieve_evidence(question)

# Process:
# 1. Extract keywords from question
# 2. Search Wikipedia for relevant pages
# 3. Generate embeddings for documents
# 4. Store in vector database
# 5. Return top-k most relevant documents
```

## 📊 Your Module's Output

Your retrieval module returns:
```python
# Main function output
evidence_docs = [
    "Paris is the capital and largest city of France...",
    "France is a country in Western Europe...",
    "The French Republic is a unitary state..."
]

# With similarity scores (optional)
evidence_with_scores = [
    {
        'document': 'Paris is the capital...',
        'similarity_score': 0.85,
        'is_relevant': True
    }
]
```

## 🤝 Integration with Other Members

### Input to Your Module:
- `question`: User's question from the main pipeline

### Output to Member 2 (Detection):
- `evidence_docs`: List of relevant evidence documents
- Used for semantic similarity and contradiction detection

### Output to Member 3 (Correction):
- `evidence_docs`: Same evidence documents for answer correction
- Provides factual basis for generating corrected answers

### Integration Code Example:
```python
from src.retrieval import retrieve_evidence, EvidenceRetriever

# Simple usage
question = "What is the capital of France?"
evidence_docs = retrieve_evidence(question)

# Advanced usage with custom settings
retriever = EvidenceRetriever(
    max_evidence_docs=5,
    similarity_threshold=0.7
)
evidence_docs = retriever.retrieve_evidence(question)

# Pass to detection module
from src.detection import detect_hallucination
is_hallucination, confidence = detect_hallucination(raw_answer, evidence_docs)
```

## 🧪 Testing Your Module

### Run All Tests:
```bash
python src/retrieval/test_retrieval.py
```

### Test Categories:
1. **Wikipedia Integration Test** - Verifies API connectivity and content retrieval
2. **Vector Database Test** - Tests ChromaDB storage and search
3. **Evidence Retriever Test** - Tests main retrieval pipeline
4. **Convenience Function Test** - Tests simple interface
5. **TruthfulQA Loader Test** - Tests dataset loading capabilities
6. **Integration Test** - Tests integration with detection module

### Sample Test Cases:
- ✅ **Geography**: "What is the capital of France?" → Paris-related documents
- ✅ **Science**: "What is photosynthesis?" → Biology and chemistry documents
- ✅ **History**: "When did World War II end?" → Historical documents
- ✅ **Literature**: "Who wrote Romeo and Juliet?" → Shakespeare documents

## 🔧 Configuration Options

### Environment Variables (.env):
```bash
# Optional: Custom ChromaDB settings
CHROMA_PERSIST_DIRECTORY=./chroma_db
CHROMA_COLLECTION_NAME=evidence_documents

# Optional: Wikipedia settings
WIKIPEDIA_LANGUAGE=en
WIKIPEDIA_MAX_RESULTS=5
```

### Customizable Parameters:
- `max_evidence_docs`: Maximum number of documents to return (default: 5)
- `similarity_threshold`: Minimum similarity for relevance (default: 0.7)
- `embedding_model`: Sentence transformer model (default: "all-MiniLM-L6-v2")

## 🐛 Troubleshooting

### Common Issues:

1. **Wikipedia API Error**:
   ```
   ❌ Error retrieving from Wikipedia
   ```
   **Solution**: Check internet connection and Wikipedia API status

2. **ChromaDB Error**:
   ```
   ❌ Error with vector database
   ```
   **Solution**: Ensure write permissions in project directory

3. **Model Download Error**:
   ```
   ❌ Error loading embedding model
   ```
   **Solution**: Ensure sufficient disk space and internet connection

4. **Import Error**:
   ```
   ❌ ModuleNotFoundError
   ```
   **Solution**: Install dependencies with `pip install -r requirements.txt`

### Performance Tips:
- First run downloads models (~500MB) - be patient
- Evidence documents are cached for faster future retrieval
- Adjust similarity threshold based on your use case
- Use batch processing for multiple questions

## 📈 Performance Metrics

### Expected Performance:
- **Retrieval Accuracy**: ~90% relevant documents for factual questions
- **Response Time**: ~3-8 seconds for first retrieval, ~1-2 seconds for cached
- **Cache Hit Rate**: ~70% for similar questions
- **Wikipedia Coverage**: Comprehensive coverage for general knowledge topics

### Optimization:
- Cache frequently accessed evidence in ChromaDB
- Use keyword extraction for better search results
- Batch process multiple questions together
- Monitor and adjust similarity thresholds

## 🎯 Key Features

### 1. Wikipedia Integration
- Real-time content retrieval from Wikipedia
- Intelligent keyword extraction from questions
- Handles disambiguation and missing pages
- Content cleaning and preprocessing

### 2. Vector Database
- Persistent storage with ChromaDB
- Semantic search capabilities
- Metadata management
- Collection statistics and monitoring

### 3. TruthfulQA Dataset
- Benchmark dataset loading
- Category-based filtering
- Sample question extraction
- CSV export functionality

### 4. Flexible Interface
- Simple `retrieve_evidence()` function
- Advanced `EvidenceRetriever` class
- Command-line interface
- Comprehensive test suite

## 🎯 Next Steps

1. **Test Your Module**: Run the test suite to ensure everything works
2. **Try Different Questions**: Test with various question types and topics
3. **Coordinate with Team**: Share your module interface with other members
4. **Prepare Integration**: Ensure your output format matches Member 2's expectations
5. **Monitor Performance**: Track retrieval accuracy and response times

## 📞 Support

### For Your Module:
- Check `test_retrieval.py` for usage examples
- Review `main.py` for command-line options
- See `setup_guide.md` for detailed setup instructions
- Read `IMPLEMENTATION_SUMMARY.md` for complete overview

### Team Coordination:
- Share your module's input/output format with other members
- Test integration points before final submission
- Document any changes to the interface
- Provide sample evidence documents for testing

## 🎉 Success Criteria

Your module is complete when:
- ✅ All tests pass
- ✅ Wikipedia API integration works reliably
- ✅ Vector database stores and retrieves documents correctly
- ✅ Evidence retrieval accuracy is >85%
- ✅ Integration interface is ready for other members
- ✅ Documentation is complete and clear

## 🚀 Advanced Usage

### Custom Retrieval Settings:
```python
from src.retrieval import EvidenceRetriever

# Custom configuration
retriever = EvidenceRetriever(
    max_evidence_docs=10,
    similarity_threshold=0.8,
    embedding_model="all-mpnet-base-v2"
)

# Get evidence with similarity scores
evidence_with_scores = retriever.get_evidence_with_scores(question)
for result in evidence_with_scores:
    print(f"Score: {result['similarity_score']:.3f}")
    print(f"Relevant: {result['is_relevant']}")
```

### TruthfulQA Dataset Usage:
```python
from src.retrieval import TruthfulQALoader

# Load dataset
loader = TruthfulQALoader()

# Get sample questions
samples = loader.get_sample_questions(10)

# Get questions by category
history_questions = loader.get_questions_by_category("History", 5)

# Export to CSV
loader.export_to_csv("sample_questions.csv", 100)
```

### Cache Management:
```python
# Get cache statistics
stats = retriever.get_cache_stats()
print(f"Total documents: {stats['total_documents']}")

# Clear cache if needed
retriever.clear_cache()
```

**You're ready to provide the foundational data layer for an amazing AI hallucination detection system!** 🚀
=======
# Member 1: Retrieval Module - Complete Guide

## 🎯 Your Role in the Project

As **Member 1**, you are the **Data & Retrieval Engineer** responsible for:
1. **Evidence Retrieval** - Fetching relevant documents from Wikipedia and knowledge bases
2. **Vector Database Management** - Storing and searching evidence using ChromaDB
3. **Semantic Search** - Using embeddings to find the most relevant evidence
4. **System Foundation** - Providing the data layer for the entire hallucination detection system

## 📁 Your Module Structure

```
src/retrieval/
├── __init__.py                    # Module initialization
├── retrieval_module.py            # Core retrieval logic and main functions
├── wikipedia_integration.py       # Wikipedia API integration
├── vector_database.py             # ChromaDB vector storage operations
├── dataset_loader.py              # TruthfulQA dataset handling
├── test_retrieval.py              # Comprehensive test suite
├── simple_test.py                 # Lightweight integration test
├── main.py                        # Command-line interface and demos
├── setup_guide.md                 # Detailed setup and usage guide
└── IMPLEMENTATION_SUMMARY.md      # Complete implementation summary
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

### 2. Test Your Module

```bash
# Run simple integration test
python src/retrieval/simple_test.py

# Run comprehensive tests
python src/retrieval/test_retrieval.py

# Run demo
python src/retrieval/main.py --mode demo
```

### 3. Try Interactive Mode

```bash
# Interactive question-answering
python src/retrieval/main.py --mode interactive
```

## 🔧 How Your Retrieval Works

### Three-Stage Retrieval Process:

1. **Wikipedia API Integration**
   - Extracts keywords from user questions
   - Searches Wikipedia for relevant pages
   - Retrieves and cleans page content
   - Handles disambiguation and errors gracefully

2. **Semantic Embedding**
   - Uses `all-MiniLM-L6-v2` model to generate embeddings
   - Creates vector representations of questions and documents
   - Enables semantic similarity search

3. **Vector Database Storage**
   - Stores embeddings in ChromaDB for fast retrieval
   - Caches results for improved performance
   - Provides similarity-based search capabilities

### Retrieval Logic:
```python
# Main retrieval function
evidence_docs = retrieve_evidence(question)

# Process:
# 1. Extract keywords from question
# 2. Search Wikipedia for relevant pages
# 3. Generate embeddings for documents
# 4. Store in vector database
# 5. Return top-k most relevant documents
```

## 📊 Your Module's Output

Your retrieval module returns:
```python
# Main function output
evidence_docs = [
    "Paris is the capital and largest city of France...",
    "France is a country in Western Europe...",
    "The French Republic is a unitary state..."
]

# With similarity scores (optional)
evidence_with_scores = [
    {
        'document': 'Paris is the capital...',
        'similarity_score': 0.85,
        'is_relevant': True
    }
]
```

## 🤝 Integration with Other Members

### Input to Your Module:
- `question`: User's question from the main pipeline

### Output to Member 2 (Detection):
- `evidence_docs`: List of relevant evidence documents
- Used for semantic similarity and contradiction detection

### Output to Member 3 (Correction):
- `evidence_docs`: Same evidence documents for answer correction
- Provides factual basis for generating corrected answers

### Integration Code Example:
```python
from src.retrieval import retrieve_evidence, EvidenceRetriever

# Simple usage
question = "What is the capital of France?"
evidence_docs = retrieve_evidence(question)

# Advanced usage with custom settings
retriever = EvidenceRetriever(
    max_evidence_docs=5,
    similarity_threshold=0.7
)
evidence_docs = retriever.retrieve_evidence(question)

# Pass to detection module
from src.detection import detect_hallucination
is_hallucination, confidence = detect_hallucination(raw_answer, evidence_docs)
```

## 🧪 Testing Your Module

### Run All Tests:
```bash
python src/retrieval/test_retrieval.py
```

### Test Categories:
1. **Wikipedia Integration Test** - Verifies API connectivity and content retrieval
2. **Vector Database Test** - Tests ChromaDB storage and search
3. **Evidence Retriever Test** - Tests main retrieval pipeline
4. **Convenience Function Test** - Tests simple interface
5. **TruthfulQA Loader Test** - Tests dataset loading capabilities
6. **Integration Test** - Tests integration with detection module

### Sample Test Cases:
- ✅ **Geography**: "What is the capital of France?" → Paris-related documents
- ✅ **Science**: "What is photosynthesis?" → Biology and chemistry documents
- ✅ **History**: "When did World War II end?" → Historical documents
- ✅ **Literature**: "Who wrote Romeo and Juliet?" → Shakespeare documents

## 🔧 Configuration Options

### Environment Variables (.env):
```bash
# Optional: Custom ChromaDB settings
CHROMA_PERSIST_DIRECTORY=./chroma_db
CHROMA_COLLECTION_NAME=evidence_documents

# Optional: Wikipedia settings
WIKIPEDIA_LANGUAGE=en
WIKIPEDIA_MAX_RESULTS=5
```

### Customizable Parameters:
- `max_evidence_docs`: Maximum number of documents to return (default: 5)
- `similarity_threshold`: Minimum similarity for relevance (default: 0.7)
- `embedding_model`: Sentence transformer model (default: "all-MiniLM-L6-v2")

## 🐛 Troubleshooting

### Common Issues:

1. **Wikipedia API Error**:
   ```
   ❌ Error retrieving from Wikipedia
   ```
   **Solution**: Check internet connection and Wikipedia API status

2. **ChromaDB Error**:
   ```
   ❌ Error with vector database
   ```
   **Solution**: Ensure write permissions in project directory

3. **Model Download Error**:
   ```
   ❌ Error loading embedding model
   ```
   **Solution**: Ensure sufficient disk space and internet connection

4. **Import Error**:
   ```
   ❌ ModuleNotFoundError
   ```
   **Solution**: Install dependencies with `pip install -r requirements.txt`

### Performance Tips:
- First run downloads models (~500MB) - be patient
- Evidence documents are cached for faster future retrieval
- Adjust similarity threshold based on your use case
- Use batch processing for multiple questions

## 📈 Performance Metrics

### Expected Performance:
- **Retrieval Accuracy**: ~90% relevant documents for factual questions
- **Response Time**: ~3-8 seconds for first retrieval, ~1-2 seconds for cached
- **Cache Hit Rate**: ~70% for similar questions
- **Wikipedia Coverage**: Comprehensive coverage for general knowledge topics

### Optimization:
- Cache frequently accessed evidence in ChromaDB
- Use keyword extraction for better search results
- Batch process multiple questions together
- Monitor and adjust similarity thresholds

## 🎯 Key Features

### 1. Wikipedia Integration
- Real-time content retrieval from Wikipedia
- Intelligent keyword extraction from questions
- Handles disambiguation and missing pages
- Content cleaning and preprocessing

### 2. Vector Database
- Persistent storage with ChromaDB
- Semantic search capabilities
- Metadata management
- Collection statistics and monitoring

### 3. TruthfulQA Dataset
- Benchmark dataset loading
- Category-based filtering
- Sample question extraction
- CSV export functionality

### 4. Flexible Interface
- Simple `retrieve_evidence()` function
- Advanced `EvidenceRetriever` class
- Command-line interface
- Comprehensive test suite

## 🎯 Next Steps

1. **Test Your Module**: Run the test suite to ensure everything works
2. **Try Different Questions**: Test with various question types and topics
3. **Coordinate with Team**: Share your module interface with other members
4. **Prepare Integration**: Ensure your output format matches Member 2's expectations
5. **Monitor Performance**: Track retrieval accuracy and response times

## 📞 Support

### For Your Module:
- Check `test_retrieval.py` for usage examples
- Review `main.py` for command-line options
- See `setup_guide.md` for detailed setup instructions
- Read `IMPLEMENTATION_SUMMARY.md` for complete overview

### Team Coordination:
- Share your module's input/output format with other members
- Test integration points before final submission
- Document any changes to the interface
- Provide sample evidence documents for testing

## 🎉 Success Criteria

Your module is complete when:
- ✅ All tests pass
- ✅ Wikipedia API integration works reliably
- ✅ Vector database stores and retrieves documents correctly
- ✅ Evidence retrieval accuracy is >85%
- ✅ Integration interface is ready for other members
- ✅ Documentation is complete and clear

## 🚀 Advanced Usage

### Custom Retrieval Settings:
```python
from src.retrieval import EvidenceRetriever

# Custom configuration
retriever = EvidenceRetriever(
    max_evidence_docs=10,
    similarity_threshold=0.8,
    embedding_model="all-mpnet-base-v2"
)

# Get evidence with similarity scores
evidence_with_scores = retriever.get_evidence_with_scores(question)
for result in evidence_with_scores:
    print(f"Score: {result['similarity_score']:.3f}")
    print(f"Relevant: {result['is_relevant']}")
```

### TruthfulQA Dataset Usage:
```python
from src.retrieval import TruthfulQALoader

# Load dataset
loader = TruthfulQALoader()

# Get sample questions
samples = loader.get_sample_questions(10)

# Get questions by category
history_questions = loader.get_questions_by_category("History", 5)

# Export to CSV
loader.export_to_csv("sample_questions.csv", 100)
```

### Cache Management:
```python
# Get cache statistics
stats = retriever.get_cache_stats()
print(f"Total documents: {stats['total_documents']}")

# Clear cache if needed
retriever.clear_cache()
```

**You're ready to provide the foundational data layer for an amazing AI hallucination detection system!** 🚀
>>>>>>> fb3451155c14f135a09046a366815aba1850f393
