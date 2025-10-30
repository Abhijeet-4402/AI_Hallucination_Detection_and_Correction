import pytest
import sys

from correction import correction_module as cm


@pytest.mark.unit
def test_initialize_database_works(tmp_path, monkeypatch):
    db = tmp_path / "test.db"
    monkeypatch.setattr(cm, "DATABASE_NAME", str(db))
    cm.initialize_database()
    assert db.exists()


@pytest.mark.unit
def test_log_and_confidence_with_mocks(tmp_path, monkeypatch, mocker):
    db = tmp_path / "log.db"
    monkeypatch.setattr(cm, "DATABASE_NAME", str(db))
    cm.initialize_database()

    # Patch SentenceTransformer and util.cos_sim to deterministic value
    class FakeST:
        def __init__(self, *_, **__):
            pass
        def encode(self, x, convert_to_tensor=False):
            import torch
            return torch.ones((1,))

    mocker.patch("correction.correction_module.SentenceTransformer", FakeST)

    class Doc:
        def __init__(self, content):
            self.page_content = content
            self.metadata = {}

    mocker.patch("correction.correction_module.util.cos_sim", return_value=__import__('torch').tensor([[0.42]]))

    score = cm.calculate_confidence_score("answer", [Doc("evidence")])
    assert score == pytest.approx(0.42, rel=1e-3)

    cm.log_hallucination_data("Q", "R", "C", ["cit"], 0.1)
    # Ensure row inserted
    import sqlite3
    conn = sqlite3.connect(str(db))
    cur = conn.cursor()
    cur.execute("select count(*) from logs")
    n = cur.fetchone()[0]
    conn.close()
    assert n == 1


@pytest.mark.unit
def test_correct_and_regenerate_with_mocks(tmp_path, monkeypatch, mocker):
    db = tmp_path / "rag.db"
    monkeypatch.setattr(cm, "DATABASE_NAME", str(db))
    cm.initialize_database()

    # Patch correction module attributes directly
    class FakeLLM:
        pass
    mocker.patch("correction.correction_module.ChatGoogleGenerativeAI", return_value=FakeLLM())

    class FakeChain:
        def __init__(self, *_, **__):
            pass
        def invoke(self, inputs):
            return {
                "result": "Corrected text",
                "source_documents": [type("D", (), {"page_content": "doc1", "metadata": {}})()],
            }
    mocker.patch("correction.correction_module.RetrievalQA.from_chain_type", return_value=FakeChain())

    out = cm.correct_and_regenerate("Q", "RAW", ["Ev1", "Ev2"])
    assert out["CorrectedAnswer"] == "Corrected text"
    assert isinstance(out["Citations"], list)
    assert isinstance(out["ConfidenceScore"], float)


