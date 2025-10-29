"""
Main Retrieval Module for AI Hallucination Detection

This module implements the core retrieval functionality. It now uses a two-stage
process:
1. Retrieve relevant full documents from Wikipedia and a vector cache.
2. Chunk these documents into passages using a sliding sentence window and
   rerank them to find the most semantically relevant passages.
"""

import logging
from typing import List
from sentence_transformers import SentenceTransformer, util
from nltk.tokenize import sent_tokenize
import nltk

from .wikipedia_integration import WikipediaRetriever
from .vector_database import VectorDatabase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure NLTK sentence tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    logger.info("NLTK 'punkt' resource not found. Downloading...")
    nltk.download('punkt')


class EvidenceRetriever:
    """Main class for evidence retrieval and management"""

    def __init__(self, 
                 embedding_model: str = "all-MiniLM-L6-v2",
                 max_evidence_docs: int = 5,
                 similarity_threshold: float = 0.5):
        self.max_evidence_docs = max_evidence_docs
        self.similarity_threshold = similarity_threshold

        self.wikipedia_retriever = WikipediaRetriever(max_results=max_evidence_docs)
        self.vector_db = VectorDatabase()
        self.embedding_model = None

        self._load_embedding_model(embedding_model)

    def _load_embedding_model(self, model_name: str):
        try:
            logger.info(f"Loading embedding model: {model_name}")
            self.embedding_model = SentenceTransformer(model_name)
            logger.info("Embedding model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise

    def _calculate_similarity(self, query: str, documents: List[str]) -> List[float]:
        if not documents or not self.embedding_model:
            return []
        try:
            query_embedding = self.embedding_model.encode(query, convert_to_tensor=True)
            doc_embeddings = self.embedding_model.encode(documents, convert_to_tensor=True)
            cosine_scores = util.cos_sim(query_embedding, doc_embeddings)
            return cosine_scores.cpu().numpy().flatten().tolist()
        except Exception as e:
            logger.error(f"Error calculating similarities: {e}")
            return [0.0] * len(documents)

    def _chunk_document_sliding_window(self, doc: str, sentences_per_chunk: int = 4, overlap: int = 1) -> List[str]:
        """Chunks a document using a sliding window of sentences."""
        sentences = sent_tokenize(doc)
        if not sentences:
            return []

        chunks = []
        step = sentences_per_chunk - overlap
        for i in range(0, len(sentences), step):
            chunk = sentences[i:i + sentences_per_chunk]
            if chunk:
                chunks.append(" ".join(chunk))
        return chunks

    def retrieve_evidence(self, question: str, use_cached: bool = True) -> List[str]:
        """
        Retrieves evidence by fetching full docs, chunking them into passages,
        and reranking the passages to find the most relevant snippets.
        """
        logger.info(f"Starting evidence retrieval for question: {question}")

        all_docs = {}

        # Step 1: Gather Full Candidate Documents
        try:
            wikipedia_docs = self.wikipedia_retriever.retrieve_evidence_documents(question)
            for doc in wikipedia_docs:
                all_docs[doc] = "fresh"
        except Exception as e:
            logger.error(f"Error retrieving from Wikipedia: {e}")

        if use_cached:
            # Placeholder for caching logic
            pass

        unique_docs = list(all_docs.keys())
        if not unique_docs:
            return []

        # Step 2: Chunk Documents into Overlapping Passages
        all_passages = []
        for doc in unique_docs:
            passages = self._chunk_document_sliding_window(doc)
            all_passages.extend(passages)

        if not all_passages:
            logger.warning("Could not extract any valid passages from the retrieved documents.")
            return []

        logger.info(f"Extracted {len(all_passages)} passages for reranking.")

        # Step 3: Rerank Passages and Select the Best
        similarities = self._calculate_similarity(question, all_passages)
        passage_sim_pairs = list(zip(all_passages, similarities))

        relevant_pairs = [(p, s) for p, s in passage_sim_pairs if s >= self.similarity_threshold]

        if not relevant_pairs:
            logger.warning(f"No passages met the similarity threshold of {self.similarity_threshold}")
            return []

        relevant_pairs.sort(key=lambda x: x[1], reverse=True)
        final_evidence = [p for p, _ in relevant_pairs[:self.max_evidence_docs]]

        logger.info(f"Retrieved {len(final_evidence)} final evidence passages after reranking.")
        return final_evidence


# ✅ Separate convenience function (OUTSIDE the class)
def retrieve_evidence(question: str):
    """
    Convenience function to quickly fetch evidence documents
    without manually instantiating the EvidenceRetriever.
    """
    retriever = EvidenceRetriever(max_evidence_docs=3, similarity_threshold=0.5)
    return retriever.retrieve_evidence(question)
