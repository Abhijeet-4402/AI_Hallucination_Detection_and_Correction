import pytest

from retrieval.retrieval_module import EvidenceRetriever
from detection.detection_module import HallucinationDetector
from correction import correction_module as cm


@pytest.mark.e2e
def test_full_flow_html_report(fake_sentence_transformer, fake_util_cosine, patch_nltk_tokenize, mocker, tmp_path, monkeypatch):
    # Mock external services to ensure hermetic test
    er = EvidenceRetriever(max_evidence_docs=2, similarity_threshold=0.2)
    er.wikipedia_retriever.retrieve_evidence_documents = lambda q: [
        "Python is a programming language created by Guido van Rossum in 1991.",
        "It emphasizes code readability."
    ]
    evidence = er.retrieve_evidence("Who created Python?")
    assert evidence

    import torch, types
    class DummyBatch(dict):
        def to(self, device):
            return self
    mocker.patch(
        "detection.detection_module.AutoTokenizer.from_pretrained",
        return_value=lambda pairs, **_: DummyBatch({"input_ids": torch.zeros((len(pairs), 10), dtype=torch.long)})
    )
    class FakeNLI:
        def to(self, device):
            return self
        def eval(self):
            return self
        def __call__(self, **_):
            logits = torch.tensor([[0.01, 0.01, 0.995] for _ in range(5)])
            return types.SimpleNamespace(logits=logits)
    mocker.patch("detection.detection_module.AutoModelForSequenceClassification.from_pretrained", return_value=FakeNLI())

    det = HallucinationDetector(device="cpu")
    result = det.detect_hallucination("Python was created by Guido van Rossum.", evidence)
    assert result.is_hallucination is False

    # Correction still runs but should return consistent structure
    db = tmp_path / "e2e.db"
    monkeypatch.setattr(cm, "DATABASE_NAME", str(db))
    cm.initialize_database()

    class FakeLLM:
        pass
    mocker.patch("correction.correction_module.ChatGoogleGenerativeAI", return_value=FakeLLM())

    class FakeChain:
        def __init__(self, *_, **__):
            pass
        def invoke(self, inputs):
            return {
                "result": "Python was created by Guido van Rossum.",
                "source_documents": [type("D", (), {"page_content": evidence[0], "metadata": {}})()],
            }
    mocker.patch("correction.correction_module.RetrievalQA.from_chain_type", return_value=FakeChain())

    out = cm.correct_and_regenerate("Who created Python?", "Python was created by Guido van Rossum.", evidence)
    assert "Guido" in out["CorrectedAnswer"]


