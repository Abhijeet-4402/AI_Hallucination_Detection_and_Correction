import sqlite3
import os
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv

load_dotenv()


DATABASE_NAME = "hallucination_log.db"

def initialize_database():
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            Question TEXT NOT NULL,
            RawAnswer TEXT NOT NULL,
            CorrectedAnswer TEXT,
            Citations TEXT,
            ConfidenceScore REAL
        )
    """)
    conn.commit()
    conn.close()

def calculate_confidence_score(corrected_answer: str, source_documents: list) -> float:
    if not source_documents:
        return 0.0

    model = SentenceTransformer('all-MiniLM-L6-v2')
    answer_embedding = model.encode(corrected_answer, convert_to_tensor=True)

    # Combine all source document content into a single string for embedding
    # This assumes that the combined semantic meaning of all sources is relevant
    # for confidence against the corrected answer.
    source_contents = [doc.page_content for doc in source_documents]
    if not source_contents: # Handle case where page_content might be empty
        return 0.0
    
    source_embeddings = model.encode(source_contents, convert_to_tensor=True)

    # Calculate cosine similarity between the answer and each source document
    cosine_scores = util.cos_sim(answer_embedding, source_embeddings)
    
    # Return the maximum similarity score as the confidence
    return round(float(cosine_scores.max()), 4)

def log_hallucination_data(question: str, raw_answer: str, corrected_answer: str, citations: list, confidence_score: float):
    conn = sqlite3.connect(DATABASE_NAME)
    cursor = conn.cursor()
    try:
        # Convert citations list to a string for storage
        citations_str = "; ".join(citations)
        cursor.execute("""
            INSERT INTO logs (Question, RawAnswer, CorrectedAnswer, Citations, ConfidenceScore)
            VALUES (?, ?, ?, ?, ?)
        """, (question, raw_answer, corrected_answer, citations_str, confidence_score))
        conn.commit()
        print("Successfully logged data to database.")
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        conn.close()

def correct_and_regenerate(question: str, raw_answer: str, evidence: list) -> dict:
    # Lazy imports to avoid hard dependency during module import and allow tests to mock
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except Exception:
        ChatGoogleGenerativeAI = None

    try:
        from langchain.chains import RetrievalQA
    except Exception:
        RetrievalQA = None

    try:
        from langchain_core.retrievers import BaseRetriever
        from langchain.docstore.document import Document
    except Exception:
        # Minimal fallbacks to allow tests to run without langchain installed
        class BaseRetriever:  # type: ignore
            pass
        class Document:  # type: ignore
            def __init__(self, page_content: str, metadata: dict | None = None):
                self.page_content = page_content
                self.metadata = metadata or {}

    # Initialize Google Generative AI model if available; otherwise create a stub object
    if ChatGoogleGenerativeAI is not None:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)
    else:
        class _StubLLM:  # type: ignore
            pass
        llm = _StubLLM()

    class MockRetriever(BaseRetriever):
        def _get_relevant_documents(self, query):  # type: ignore
            return [Document(page_content=doc) for doc in evidence]

    mock_retriever = MockRetriever()

    if RetrievalQA is not None:
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=mock_retriever,
            return_source_documents=True
        )
        response = qa_chain.invoke({"query": question})
    else:
        # Minimal fallback behavior to keep function usable in tests without langchain
        response = {
            "result": raw_answer,
            "source_documents": [Document(page_content=e) for e in evidence],
        }
    corrected_answer = response["result"]
    source_documents = response["source_documents"]

    # Extracting citations from source_documents
    citations_list = []
    for doc in source_documents:
        if doc.metadata and doc.metadata.get("source"):
            citations_list.append(doc.metadata.get("source"))
        else:
            # If no explicit source, use a truncated version of the content as a citation
            citations_list.append(doc.page_content[:50] + "...")
    citations = list(set(citations_list)) # Ensure unique citations
    
    # Calculate confidence score
    confidence_score = calculate_confidence_score(corrected_answer, source_documents)

    # Log the data to the database
    log_hallucination_data(question, raw_answer, corrected_answer, citations, confidence_score)

    print(f"Received question: {question}")
    print(f"Received raw answer: {raw_answer}")
    print(f"Received evidence: {evidence}")
    return {
        "CorrectedAnswer": corrected_answer,
        "Citations": citations,
        "ConfidenceScore": confidence_score
    }

if __name__ == "__main__":
    # This module is now designed to be imported and used by other modules.
    # For testing, please run the test.py file in this directory.
    print("Correction module loaded. Use test.py to run tests.")
