# AI Hallucination Detection and Correction System

## 🎯 Project Overview

This project implements a comprehensive AI hallucination detection and correction system that identifies when AI-generated responses contain factual errors and automatically corrects them using retrieved evidence. The system is designed as a collaborative project with multiple specialized modules working together.

## 🏗️ System Architecture

The system consists of four main modules:

1. **Retrieval Module** - Retrieves relevant evidence from knowledge bases
2. **Detection Module** - Generates answers and detects hallucinations
3. **Correction Module** - Corrects detected hallucinations using evidence
4. **Frontend Module** - Provides user interface and API endpoints

## 🔧 Technology Stack

- **Python 3.8+**
- **Google Gemini Pro** - Primary language model
- **LangChain** - Framework for LLM applications
- **Sentence Transformers** - Semantic similarity calculations
- **Transformers (Hugging Face)** - Natural Language Inference models
- **SQLite** - Logging and data storage
- **Flask** - Backend API
- **Streamlit** - Frontend interface

## 📁 Project Structure

```
DA_Project/
├── About/                          # Project documentation and member tasks
├── src/                           # Source code
│   ├── retrieval/                 # Retrieval Module
│   ├── detection/                 # Detection Module
│   ├── correction/                # Correction Module
│   ├── frontend/                  # Frontend Module
│   └── shared/                    # Shared utilities and configurations
├── tests/                         # Test files
├── data/                          # Sample data and datasets
├── logs/                          # System logs and database
├── requirements.txt               # Python dependencies
├── .env.example                  # Environment variables template
└── README.md                     # This file
```

## 🚀 Quick Start Guide

### 1. Environment Setup

```bash
# Clone or navigate to the project directory
cd DA_Project

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env file and add your Gemini API key
# GEMINI_API_KEY=your_actual_api_key_here
```

### 3. Running the System

```bash
# Test individual modules
python src/detection/test_detection.py

# Run the complete system
python src/main.py
```

## 📋 Detection Module Tasks

The **Detection Module** is responsible for:

### Core Functions:
1. **Answer Generation** - Use Gemini Pro to generate initial answers
2. **Hallucination Detection** - Implement two detection methods:
   - Semantic similarity checking
   - Natural Language Inference (NLI) for contradiction detection
3. **Integration** - Provide detection results to other modules

### Key Files:
- `src/detection/detection_module.py` - Main detection logic
- `src/detection/gemini_integration.py` - Gemini Pro integration
- `src/detection/test_detection.py` - Testing and validation

## 🔍 How Detection Works

The detection system uses a two-step approach:

1. **Semantic Similarity Check**: Compares the generated answer with retrieved evidence using sentence embeddings
2. **Contradiction Detection**: Uses NLI models to detect if the answer contradicts the evidence

If either method indicates a potential hallucination, the system flags it for correction.

## 📊 Output Format

The detection module returns:
```python
{
    "is_hallucination": bool,      # True if hallucination detected
    "confidence_score": float,     # Similarity score (0-1)
    "detection_method": str,       # Which method triggered detection
    "raw_answer": str,             # Original answer from Gemini
    "evidence_docs": list          # Retrieved evidence documents
}
```

## 🧪 Testing

Run tests for the detection module:
```bash
python src/detection/test_detection.py
```

## 🤝 Integration with Other Modules

The detection module integrates with:
- **Retrieval Module**: Receives `evidence_docs` from retrieval module
- **Correction Module**: Sends detection results and confidence scores
- **Frontend Module**: Provides final results for user display

## 📚 Additional Resources

- Check `About/` folder for detailed member-specific tasks
- Review integration guides in the About folder
- See system architecture diagram: `About/AI_Hallucination_System_Architecture.png`

## 🐛 Troubleshooting

Common issues and solutions:

1. **API Key Issues**: Ensure your Gemini API key is correctly set in `.env`
2. **Model Loading**: First run may take time to download models
3. **Memory Issues**: Large models require sufficient RAM (8GB+ recommended)

## 📞 Support

For questions about the detection module, refer to:
- `About/Member-2-Task.md` - Detailed task breakdown
- `About/AI_Hallucination_Detection_Project.docx` - Complete project documentation
