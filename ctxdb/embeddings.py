"""Embedding providers, swappable per collection.

A `spec` is a string stored on the collection row that selects the engine:

    none                                  -> no vectors; search is BM25 only
    local:intfloat/multilingual-e5-small  -> sentence-transformers on your machine
    voyage:voyage-3.5                     -> Voyage API (requires VOYAGE_API_KEY)

Storing the spec on the collection row is what makes the hybrid setup possible:
one collection can be local and another can use the API, inside the same .db
file. Vectors from different models are never compared, because they live in
tables separated by dimension and every item records which model indexed it.
"""

from __future__ import annotations

import os
from typing import Literal, Protocol

Mode = Literal["query", "document"]

DEFAULT_LOCAL_MODEL = "intfloat/multilingual-e5-small"  # 384 dims, multilingual
DEFAULT_VOYAGE_MODEL = "voyage-3.5"  # 1024 dims

_VOYAGE_DIMS = {
    "voyage-3.5": 1024,
    "voyage-3.5-lite": 1024,
    "voyage-3-large": 1024,
    "voyage-code-3": 1024,
    "voyage-law-2": 1024,
    "voyage-finance-2": 1024,
}


class Embedder(Protocol):
    spec: str
    dim: int

    def embed(self, texts: list[str], mode: Mode = "document") -> list[list[float]]: ...


class NullEmbedder:
    """A collection without vectors. `dim = 0` disables the semantic branch."""

    spec = "none"
    dim = 0

    def embed(self, texts: list[str], mode: Mode = "document") -> list[list[float]]:
        return []


class LocalEmbedder:
    """sentence-transformers. The model downloads on first use (~120 MB)."""

    def __init__(self, model_name: str = DEFAULT_LOCAL_MODEL):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "Local embeddings are missing. Install them with "
                "`uv pip install sentence-transformers`, or use embed_spec='none'."
            ) from exc
        self.spec = f"local:{model_name}"
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())
        # E5 models were trained with these prefixes; omitting them degrades
        # retrieval noticeably.
        self._e5 = "e5" in model_name.lower()

    def embed(self, texts: list[str], mode: Mode = "document") -> list[list[float]]:
        if not texts:
            return []
        if self._e5:
            prefix = "query: " if mode == "query" else "passage: "
            texts = [prefix + t for t in texts]
        vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [[float(x) for x in v] for v in vectors]


class VoyageEmbedder:
    """The Voyage API, the embedding provider Anthropic recommends with Claude."""

    def __init__(self, model_name: str = DEFAULT_VOYAGE_MODEL):
        try:
            import voyageai
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "The Voyage client is missing. Install it with `uv pip install voyageai`."
            ) from exc
        if not os.environ.get("VOYAGE_API_KEY"):
            raise RuntimeError("Set VOYAGE_API_KEY to use Voyage embeddings.")
        self.spec = f"voyage:{model_name}"
        self._model_name = model_name
        self._client = voyageai.Client()
        self.dim = _VOYAGE_DIMS.get(model_name, 1024)

    def embed(self, texts: list[str], mode: Mode = "document") -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        # The API caps batch size; 128 is comfortable for any ingest.
        for start in range(0, len(texts), 128):
            batch = texts[start : start + 128]
            result = self._client.embed(batch, model=self._model_name, input_type=mode)
            vectors.extend(result.embeddings)
        return vectors


_CACHE: dict[str, Embedder] = {}


def get_embedder(spec: str) -> Embedder:
    """Return the embedder for a spec, reusing an already-loaded model."""
    spec = (spec or "none").strip()
    if spec in _CACHE:
        return _CACHE[spec]

    provider, _, model = spec.partition(":")
    provider = provider.lower()

    embedder: Embedder
    if provider in ("none", ""):
        embedder = NullEmbedder()
    elif provider == "local":
        embedder = LocalEmbedder(model or DEFAULT_LOCAL_MODEL)
    elif provider == "voyage":
        embedder = VoyageEmbedder(model or DEFAULT_VOYAGE_MODEL)
    else:
        raise ValueError(f"unknown embedding provider: {provider!r}")

    _CACHE[spec] = embedder
    return embedder
