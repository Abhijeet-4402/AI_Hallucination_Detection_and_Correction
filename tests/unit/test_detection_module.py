import pytest
import torch

from detection.detection_module import HallucinationDetector


def _patch_nli_mock(mocker, entail: float, contra: float):
    # Tokenizer returns simple dict of tensors
    mocker.patch(
        "detection.detection_module.AutoTokenizer.from_pretrained",
        return_value=lambda pairs, **_: {"input_ids": torch.zeros((len(pairs), 10), dtype=torch.long)}
    )

    class FakeNLI:
        def to(self, device):
            return self

        def eval(self):
            return self

        @property
        def logits(self):
            return None

        def __call__(self, **_):
            # 3-class: [contradiction, neutral, entailment]
            logits = torch.tensor([[contra, 0.01, entail] for _ in range(5)])
            return types.SimpleNamespace(logits=logits)

    import types
    mocker.patch(
        "detection.detection_module.AutoModelForSequenceClassification.from_pretrained",
        return_value=FakeNLI(),
    )


@pytest.mark.unit
def test_detect_empty_answer(fake_sentence_transformer, fake_util_cosine, patch_nltk_tokenize, mocker):
    _patch_nli_mock(mocker, entail=0.1, contra=0.1)
    det = HallucinationDetector(device="cpu")
    res = det.detect_hallucination("   ", ["evidence."])
    assert res.is_hallucination is False
    assert res.detection_method == "empty_answer"


@pytest.mark.unit
def test_detect_no_evidence(fake_sentence_transformer, fake_util_cosine, patch_nltk_tokenize, mocker):
    _patch_nli_mock(mocker, entail=0.1, contra=0.1)
    det = HallucinationDetector(device="cpu")
    res = det.detect_hallucination("Some answer.", [])
    assert res.is_hallucination is True
    assert res.detection_method == "no_evidence"


@pytest.mark.unit
def test_detect_contradiction(fake_sentence_transformer, fake_util_cosine, patch_nltk_tokenize, mocker):
    _patch_nli_mock(mocker, entail=0.1, contra=0.999)
    det = HallucinationDetector(contradiction_threshold=0.98, entailment_threshold=0.99, device="cpu")
    res = det.detect_hallucination("Claim A.", ["E1.", "E2."])
    assert res.is_hallucination is True
    assert res.detection_method == "contradiction"


@pytest.mark.unit
def test_detect_low_similarity(fake_sentence_transformer, fake_util_cosine, patch_nltk_tokenize, mocker):
    _patch_nli_mock(mocker, entail=0.2, contra=0.2)
    det = HallucinationDetector(similarity_threshold=0.95, device="cpu")
    res = det.detect_hallucination("Claim.", ["Ev A.", "Ev B."])
    assert res.is_hallucination is True
    assert res.detection_method == "low_similarity"


@pytest.mark.unit
def test_detect_verified(fake_sentence_transformer, patch_nltk_tokenize, mocker):
    # Force entailment above threshold
    import types

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
            logits = torch.tensor([[0.01, 0.01, 0.999] for _ in range(5)])
            return types.SimpleNamespace(logits=logits)

    mocker.patch(
        "detection.detection_module.AutoModelForSequenceClassification.from_pretrained",
        return_value=FakeNLI(),
    )

    det = HallucinationDetector(entailment_threshold=0.9, device="cpu")
    res = det.detect_hallucination("True claim.", ["Evidence supports."])
    assert res.is_hallucination is False
    assert res.detection_method == "verified"


