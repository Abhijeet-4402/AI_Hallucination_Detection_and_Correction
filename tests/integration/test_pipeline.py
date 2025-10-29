import pytest

from retrieval.retrieval_module import EvidenceRetriever
from detection.detection_module import HallucinationDetector
from correction import correction_module as cm


@pytest.mark.integration
def test_pipeline_detection_then_correction(fake_sentence_transformer, fake_util_cosine, patch_nltk_tokenize, mocker, tmp_path, monkeypatch):
    # Retrieval mocked to deterministic docs
    er = EvidenceRetriever(max_evidence_docs=2, similarity_threshold=0.2)
    er.wikipedia_retriever.retrieve_evidence_documents = lambda q: [
        "Apple is a company. It was founded in 1976.",
        "Orange is a fruit."
    ]
    evidence = er.retrieve_evidence("Tell me about Apple")
    assert evidence

    # Detection: force contradiction for one claim
    import torch, types
    mocker.patch(
        "detection.detection_module.AutoTokenizer.from_pretrained",
        return_value=lambda pairs, **_: {"input_ids": torch.zeros((len(pairs), 10), dtype=torch.long)}
    )
    class FakeNLI:
        def to(self, device):
            return self
        def eval(self):
            return self
        def __call__(self, **_):
            logits = torch.tensor([[0.999, 0.001, 0.001] for _ in range(5)])
            return types.SimpleNamespace(logits=logits)
    mocker.patch("detection.detection_module.AutoModelForSequenceClassification.from_pretrained", return_value=FakeNLI())

    det = HallucinationDetector(device="cpu")
    detection = det.detect_hallucination("Apple was founded in 1876.", evidence)
    assert detection.is_hallucination is True

    # Correction: mock chain to return corrected answer
    db = tmp_path / "int.db"
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
                "result": "Apple was founded in 1976.",
                "source_documents": [type("D", (), {"page_content": evidence[0], "metadata": {}})()],
            }
    mocker.patch("correction.correction_module.RetrievalQA.from_chain_type", return_value=FakeChain())

    out = cm.correct_and_regenerate("When was Apple founded?", "Apple was founded in 1876.", evidence)
    assert "1976" in out["CorrectedAnswer"]


