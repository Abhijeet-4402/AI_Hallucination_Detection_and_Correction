import pytest

from retrieval.vector_database import VectorDatabase


@pytest.mark.unit
def test_add_and_query_documents(patch_chromadb, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = VectorDatabase(persist_directory=str(tmp_path / "db"))
    ids = db.add_documents(["doc one", "doc two"], metadatas=[{"source": "s1"}, {"source": "s2"}])
    assert len(ids) == 2

    results = db.search_similar("query", n_results=1)
    assert len(results) == 1
    assert "document" in results[0]


@pytest.mark.unit
def test_get_and_update_document(patch_chromadb, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = VectorDatabase(persist_directory=str(tmp_path / "db"))
    ids = db.add_documents(["original"], metadatas=[{"source": "s"}])
    doc_id = ids[0]

    got = db.get_document_by_id(doc_id)
    assert got and got["document"] == "original"

    ok = db.update_document(doc_id, "updated", {"source": "s2"})
    assert ok is True
    got2 = db.get_document_by_id(doc_id)
    assert got2 and got2["document"] == "updated"


@pytest.mark.unit
def test_stats_and_clear(patch_chromadb, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = VectorDatabase(persist_directory=str(tmp_path / "db"))
    db.add_documents(["a", "b", "c"])
    stats = db.get_collection_stats()
    assert stats["collection_name"]
    assert stats["total_documents"] == 3

    assert db.clear_collection() is True
    assert db.get_collection_stats()["total_documents"] == 0


