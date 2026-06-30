from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from app.config import settings


class BaseEmbedder(ABC):
    @abstractmethod
    def embed_documents(self, texts: list[str]) -> tuple[list, list]:
        """Returns (dense_vecs: list[list[float]], sparse_dicts: list[dict[str, float]])"""

    @abstractmethod
    def embed_query(self, text: str) -> tuple[list, dict]:
        """Returns (dense_vec: list[float], sparse_dict: dict[str, float])"""


class _BGE_M3_Singleton:
    """Lazy singleton for the BGE-M3 model — loaded at most once per process."""
    _instance = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            from FlagEmbedding import BGEM3FlagModel
            print("[embedder] Loading BAAI/bge-m3 (first use)...")
            cls._instance = BGEM3FlagModel(
                settings.bge_model_name,
                use_fp16=False,  # fp16 unsupported on CPU with this transformers version
                devices=["cpu"],
            )
            print("[embedder] BAAI/bge-m3 ready.")
        return cls._instance

    @classmethod
    def encode_sparse(cls, texts: list[str]) -> list[dict]:
        model = cls.get()
        out = model.encode(
            texts,
            batch_size=32,
            max_length=512,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        return out["lexical_weights"]  # list of {str(token_id): float}

    @classmethod
    def encode_both(cls, texts: list[str]) -> tuple[list, list]:
        model = cls.get()
        out = model.encode(
            texts,
            batch_size=32,
            max_length=512,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        return out["dense_vecs"].tolist(), out["lexical_weights"]


class VoyageEmbedder(BaseEmbedder):
    """Dense vectors from Voyage AI, sparse from BGE-M3 (hybrid search requires both)."""

    def __init__(self, api_key: str, model: str = "voyage-code-3") -> None:
        import voyageai
        self.voyage = voyageai.Client(api_key=api_key)
        self.model = model

    def embed_documents(self, texts: list[str]) -> tuple[list, list]:
        result = self.voyage.embed(texts, model=self.model, input_type="document")
        dense = result.embeddings
        sparse = _BGE_M3_Singleton.encode_sparse(texts)
        return dense, sparse

    def embed_query(self, text: str) -> tuple[list, dict]:
        result = self.voyage.embed([text], model=self.model, input_type="query")
        dense = result.embeddings[0]
        sparse = _BGE_M3_Singleton.encode_sparse([text])[0]
        return dense, sparse


class LocalEmbedder(BaseEmbedder):
    """Both dense and sparse from BGE-M3. Used when no Voyage API key is configured."""

    def embed_documents(self, texts: list[str]) -> tuple[list, list]:
        return _BGE_M3_Singleton.encode_both(texts)

    def embed_query(self, text: str) -> tuple[list, dict]:
        dense_list, sparse_list = _BGE_M3_Singleton.encode_both([text])
        return dense_list[0], sparse_list[0]


def create_embedder() -> BaseEmbedder:
    if settings.use_voyage:
        print("[embedder] Using Voyage AI voyage-code-3 for dense embeddings.")
        return VoyageEmbedder(api_key=settings.voyage_api_key, model=settings.voyage_model)
    print("[embedder] No VOYAGE_API_KEY — using local BAAI/bge-m3 for all embeddings.")
    return LocalEmbedder()
