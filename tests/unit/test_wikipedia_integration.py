import pytest

from retrieval.wikipedia_integration import WikipediaRetriever


@pytest.mark.unit
def test_extract_keywords_basic():
    wr = WikipediaRetriever()
    keywords = wr._extract_keywords("What is the Capital of France in Europe?")
    assert "Capital" in keywords
    assert "France" in keywords
    assert len(keywords) <= 5


@pytest.mark.unit
def test_generate_search_queries():
    wr = WikipediaRetriever()
    keywords = ["France", "Capital"]
    queries = wr._generate_search_queries("Capital of France", keywords)
    assert "France Capital" in queries[0]
    assert "Capital of France" in queries
    assert "France" in queries and "Capital" in queries


@pytest.mark.unit
def test_retrieve_evidence_documents_happy_path(patch_wikipedia):
    wr = WikipediaRetriever(max_results=2)
    docs = wr.retrieve_evidence_documents("What is Alpha?")
    assert len(docs) <= 2
    assert all(isinstance(d, str) and d for d in docs)


@pytest.mark.unit
def test_retrieve_evidence_documents_handles_search_failure(mocker):
    wr = WikipediaRetriever(max_results=2)
    mocker.patch("retrieval.wikipedia_integration.wikipedia.search", side_effect=Exception("boom"))
    mocker.patch("retrieval.wikipedia_integration.wikipedia.page", side_effect=Exception("nope"))
    docs = wr.retrieve_evidence_documents("Query")
    assert docs == []


