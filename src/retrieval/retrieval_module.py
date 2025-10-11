"""
Main Retrieval Module for AI Hallucination Detection

This module implements the core retrieval functionality that combines
Wikipedia integration with vector database storage for efficient
evidence retrieval and semantic search.
"""

import logging
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer, util
import numpy as np

from .wikipedia_integration import WikipediaRetriever
from .vector_database import VectorDatabase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EvidenceRetriever:
    """Main class for evidence retrieval and management"""
    
    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 max_evidence_docs: int = 5,
                 similarity_threshold: float = 0.7):
        """
        Initialize the EvidenceRetriever
        
        Args:
            embedding_model: Name of the sentence-transformer model
            max_evidence_docs: Max number of documents to return
            similarity_threshold: Threshold for filtering documents
        """
        self.max_evidence_docs = max_evidence_docs
        self.similarity_threshold = similarity_threshold
        
        # Initialize components
        self.wikipedia_retriever = WikipediaRetriever(max_results=max_evidence_docs)
        self.vector_db = VectorDatabase()
        self.embedding_model = None
        
        # Load the embedding model
        self._load_embedding_model(embedding_model)
    
    def _load_embedding_model(self, model_name: str):
        """Load the sentence embedding model"""
        try:
            logger.info(f"Loading embedding model: {model_name}")
            self.embedding_model = SentenceTransformer(model_name)
            logger.info("Embedding model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise
    
    def _calculate_similarity(self, query: str, documents: List[str]) -> List[float]:
        """
        Calculate similarity scores between query and documents
        
        Args:
            query: Search query
            documents: List of documents to compare
            
        Returns:
            List of similarity scores
        """
        if not documents or not self.embedding_model:
            return []
        
        try:
            # Generate embeddings
            query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)
            doc_embeddings = self.embedding_model.encode(documents, convert_to_tensor=True)
            
            # Calculate cosine similarities using the optimized util function
            cosine_scores = util.cos_sim(query_embedding, doc_embeddings)
            
            # Return as a flat list of floats
            return cosine_scores.cpu().numpy().flatten().tolist()
        except Exception as e:
            logger.error(f"Error calculating similarities: {e}")
            return [0.0] * len(documents)
    
    def _filter_by_similarity(self, documents: List[str], similarities: List[float]) -> List[str]:
        """
        Filter documents by similarity threshold.
        
        Args:
            documents: List of documents
            similarities: List of similarity scores
            
        Returns:
            Filtered list of documents, sorted by similarity.
        """
        if not documents or not similarities:
            return []
        
        # Pair documents with their similarities
        doc_sim_pairs = list(zip(documents, similarities))
        
        # Filter based on the fixed similarity threshold
        filtered_pairs = [(doc, sim) for doc, sim in doc_sim_pairs if sim >= self.similarity_threshold]
        
        # Sort by similarity (descending) and return documents
        filtered_pairs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, sim in filtered_pairs]
    
    def retrieve_evidence(self, question: str, use_cached: bool = True) -> List[str]:
        """
        Main function to retrieve evidence documents for a question.
        This simplified logic prioritizes fresh, relevant results efficiently.
        
        Args:
            question: User's question
            use_cached: Whether to use cached results from vector database
            
        Returns:
            List of relevant evidence documents
        """
        logger.info(f"Starting evidence retrieval for question: {question}")
        
        all_docs = {}  # Use a dictionary to handle duplicates automatically

        # 1. Fetch fresh evidence from Wikipedia
        try:
            wikipedia_docs = self.wikipedia_retriever.retrieve_evidence_documents(question)
            for doc in wikipedia_docs:
                all_docs[doc] = "fresh" # Store unique docs, marking them as fresh
            
            # Store new documents in vector database for future use
            if wikipedia_docs:
                self.vector_db.add_documents(
                    wikipedia_docs,
                    metadatas=[{"source": "wikipedia", "question": question} for _ in wikipedia_docs]
                )
                logger.info(f"Stored {len(wikipedia_docs)} new documents in vector database")

        except Exception as e:
            logger.error(f"Error retrieving from Wikipedia: {e}")

        # 2. If enabled, get cached results from vector database
        if use_cached:
            try:
                cached_results = self.vector_db.search_similar(question, n_results=self.max_evidence_docs)
                if cached_results:
                    for result in cached_results:
                        # Add to dict; will not overwrite fresh docs if they are the same
                        all_docs.setdefault(result['document'], "cached")
                    logger.info(f"Retrieved {len(cached_results)} potential cached evidence documents")
            except Exception as e:
                logger.warning(f"Error retrieving cached results: {e}")
        
        # 3. Rank all unique documents and return the best ones
        unique_docs = list(all_docs.keys())
        if not unique_docs:
            logger.warning("No evidence documents found.")
            return []

        similarities = self._calculate_similarity(question, unique_docs)
        
        # Pair documents with scores, sort, and take the top N
        doc_sim_pairs = sorted(zip(unique_docs, similarities), key=lambda x: x[1], reverse=True)
        
        # Return only the document text of the top results
        final_evidence = [doc for doc, sim in doc_sim_pairs[:self.max_evidence_docs]]
        
        logger.info(f"Retrieved {len(final_evidence)} final evidence documents after ranking.")
        return final_evidence
    
    def get_evidence_with_scores(self, question: str) -> List[Dict[str, Any]]:
        """
        Retrieve evidence and their similarity scores
        
        Args:
            question: User's question
            
        Returns:
            List of dictionaries with 'document' and 'score'
        """
        evidence_docs = self.retrieve_evidence(question, use_cached=True)
        
        if not evidence_docs:
            return []
        
        similarities = self._calculate_similarity(question, evidence_docs)
        
        return [
            {"document": doc, "score": score}
            for doc, score in zip(evidence_docs, similarities)
        ]
    
    def clear_cache(self) -> bool:
        """Clear the vector database cache"""
        logger.info("Clearing vector database cache...")
        return self.vector_db.clear_collection()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector database cache"""
        logger.info("Getting cache statistics...")
        return self.vector_db.get_collection_stats()

# Global retriever instance
_retriever = None

def get_retriever() -> EvidenceRetriever:
    """Get or create the global retriever instance"""
    global _retriever
    if _retriever is None:
        _retriever = EvidenceRetriever()
    return _retriever

def retrieve_evidence(question: str) -> List[str]:
    """
    Convenience function to retrieve evidence
    
    Args:
        question: The question to find evidence for
        
    Returns:
        List of evidence documents
    """
    retriever = get_retriever()
    return retriever.retrieve_evidence(question)
