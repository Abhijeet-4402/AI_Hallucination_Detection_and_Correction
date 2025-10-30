import os
import sys
from typing import List
import types
import pytest


# Ensure src path is importable
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.join(os.path.dirname(PROJECT_ROOT), "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, tmp_path):
    """Isolate env-dependent state for tests.
    - Redirect HOME to temp to avoid pollution
    - Ensure no external API keys are required
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)


@pytest.fixture()
def fake_sentence_transformer(mocker):
    """Patch SentenceTransformer to a lightweight fake with deterministic outputs."""
    class FakeST:
        def __init__(self, *args, **kwargs):
            pass

        def encode(self, inputs, convert_to_tensor=False):
            # Deterministic numeric encoding: length of string(s)
            import torch
            if isinstance(inputs, list):
                vals = [float(len(str(x))) for x in inputs]
            else:
                vals = [float(len(str(inputs)))]
            tensor = torch.tensor(vals).unsqueeze(0) if not isinstance(inputs, list) else torch.tensor(vals)
            return tensor if convert_to_tensor else tensor.numpy()

    return mocker.patch("sentence_transformers.SentenceTransformer", FakeST)


@pytest.fixture()
def fake_util_cosine(mocker):
    """Patch sentence_transformers.util cosine functions to simple deterministic versions."""
    import torch

    def fake_cos_sim(a, b):
        # Return increasing similarity across columns
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)
        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b)
        cols = b.shape[0] if b.dim() == 1 else b.shape[-1] if b.shape[0] == 1 else b.shape[0]
        return torch.linspace(0.1, 0.9, steps=cols).unsqueeze(0)

    def fake_pytorch_cos_sim(a, b):
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)
        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b)
        # Produce decreasing similarity to trigger thresholds as needed in tests
        cols = b.shape[0] if b.dim() == 1 else b.shape[0]
        return torch.linspace(0.9, 0.1, steps=cols).unsqueeze(0)

    mocker.patch("sentence_transformers.util.cos_sim", side_effect=fake_cos_sim)
    mocker.patch("sentence_transformers.util.pytorch_cos_sim", side_effect=fake_pytorch_cos_sim)


@pytest.fixture()
def patch_wikipedia(mocker):
    """Patch wikipedia.search and wikipedia.page."""
    search_results = ["Alpha", "Beta", "Gamma"]
    mocker.patch("retrieval.wikipedia_integration.wikipedia.search", return_value=search_results)

    class Page:
        def __init__(self, title):
            self.title = title
            self.content = f"Content for {title}. Sentence one. Sentence two. Sentence three."

    mocker.patch("retrieval.wikipedia_integration.wikipedia.page", side_effect=lambda title, **_: Page(title))
    return search_results


@pytest.fixture()
def patch_chromadb(mocker):
    """Patch chromadb PersistentClient and collection to an in-memory fake."""
    class FakeCollection:
        def __init__(self):
            self._docs = {}
            self._metas = {}

        def upsert(self, documents: List[str], metadatas: List[dict], ids: List[str]):
            for doc, meta, id_ in zip(documents, metadatas, ids):
                self._docs[id_] = doc
                self._metas[id_] = meta

        def count(self):
            return len(self._docs)

        def query(self, query_texts: List[str], n_results: int):
            # Return up to n_results arbitrary docs with fake distances
            ids = list(self._docs.keys())[:n_results]
            docs = [self._docs[i] for i in ids]
            metas = [self._metas[i] for i in ids]
            distances = [0.1 * i for i in range(1, len(ids) + 1)]
            return {
                "documents": [docs],
                "distances": [distances],
                "metadatas": [metas],
                "ids": [ids],
            }

        def get(self, ids: List[str]):
            out_docs = [self._docs.get(ids[0])] if ids else []
            out_metas = [self._metas.get(ids[0], {})] if ids else []
            return {"documents": out_docs, "metadatas": out_metas}

        def update(self, ids: List[str], documents: List[str], metadatas: List[dict] = None):
            for i, d in zip(ids, documents):
                self._docs[i] = d
            if metadatas:
                for i, m in zip(ids, metadatas):
                    self._metas[i] = m

    class FakeClient:
        def __init__(self, *_, **__):
            self._collections = {}

        def get_collection(self, name: str):
            if name not in self._collections:
                raise Exception("not found")
            return self._collections[name]

        def create_collection(self, name: str, metadata: dict):
            coll = FakeCollection()
            self._collections[name] = coll
            return coll

        def delete_collection(self, name: str):
            self._collections.pop(name, None)

    mocker.patch("retrieval.vector_database.chromadb.PersistentClient", FakeClient)
    class DummySettings:
        def __init__(self, **kwargs):
            self._settings = kwargs
    mocker.patch("retrieval.vector_database.Settings", DummySettings)


@pytest.fixture()
def patch_nltk_tokenize(mocker):
    mocker.patch("nltk.tokenize.sent_tokenize", side_effect=lambda s: [seg.strip() for seg in s.split(".") if seg.strip()])


