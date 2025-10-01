"""
Retrieval Module

This module handles the retrieval of evidence documents from various sources
including Wikipedia and other knowledge bases. It provides the foundational
data layer for the AI hallucination detection system.
"""

from .retrieval_module import retrieve_evidence, EvidenceRetriever
from .wikipedia_integration import WikipediaRetriever
from .vector_database import VectorDatabase

__all__ = [
    'retrieve_evidence',
    'EvidenceRetriever', 
    'WikipediaRetriever',
    'VectorDatabase'
]
