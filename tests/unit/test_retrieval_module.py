import pytest

from retrieval.retrieval_module import EvidenceRetriever


@pytest.mark.unit
def test_retrieve_evidence_basic(fake_sentence_transformer, fake_util_cosine, patch_wikipedia, patch_nltk_tokenize):
    er = EvidenceRetriever(max_evidence_docs=3, similarity_threshold=0.2)

    # Patch internal wikipedia retriever to return deterministic documents
    er.wikipedia_retriever.retrieve_evidence_documents = lambda q: [
        "Doc A. Sentence A1. Sentence A2.",
        "Doc B. Sentence B1. Sentence B2."
    ]

    passages = er.retrieve_evidence("What is Doc A?")
    assert 0 < len(passages) <= 3
    assert all(isinstance(p, str) for p in passages)


@pytest.mark.unit
def test_retrieve_evidence_no_docs(fake_sentence_transformer, fake_util_cosine, patch_nltk_tokenize):
    er = EvidenceRetriever(max_evidence_docs=3)
    er.wikipedia_retriever.retrieve_evidence_documents = lambda q: []
    assert er.retrieve_evidence("question") == []


