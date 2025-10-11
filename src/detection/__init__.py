<<<<<<< HEAD
"""
Detection Module
AI Hallucination Detection System

This module is responsible for:
1. Generating answers using Gemini Pro
2. Detecting hallucinations using semantic similarity and NLI
3. Providing detection results to other modules
"""

from .detection_module import detect_hallucination, DetectionResult
from .gemini_integration import GeminiLLM

__all__ = ['detect_hallucination', 'DetectionResult', 'GeminiLLM']
=======
"""
Detection Module
AI Hallucination Detection System

This module is responsible for:
1. Generating answers using Gemini Pro
2. Detecting hallucinations using semantic similarity and NLI
3. Providing detection results to other modules
"""

from .detection_module import detect_hallucination, DetectionResult
from .gemini_integration import GeminiLLM

__all__ = ['detect_hallucination', 'DetectionResult', 'GeminiLLM']
>>>>>>> fb3451155c14f135a09046a366815aba1850f393
