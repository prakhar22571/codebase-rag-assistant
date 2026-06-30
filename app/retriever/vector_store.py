from __future__ import annotations

import time
import uuid
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FusionQuery,
    MatchValue,
    PayloadSchemaType,
    Prefetch,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
    Fusion,
)

from app.indexer.code_parser import Chunk

_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def _make_point_id(chunk: Chunk) -> str:
    raw = f"{chunk.repo_name}:{chunk.file_path}:{chunk.start_line}:{chunk.chunk_index}"
    return str(uuid.uuid5(_NAMESPACE, raw))


def _sparse_dict_to_qdrant(weights: dict) -> SparseVector:
    """Convert FlagEmbedding sparse dict (str keys) to Qdrant SparseVector (int indices)."""
    pairs = sorted((int(k), float(v)) for k, v in weights.items())
    return SparseVector(
        indices=[p[0] for p in pairs],
        values=[p[1] for p in pairs],
    )


class VectorStore:
    def __init__(
        self,
        url: str,
        api_key: str,
        collection: str,
        dense_dim: int = 1024,
    ) -> None:
        self.client = QdrantClient(url=url, api_key=api_key, timeout=30)
        self.collection = collection
        self.dense_dim = dense_dim

    def ensure_collection(self) -> None:
        """Create the Qdrant collection and required payload indexes if they don't exist."""
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config={
                    "dense": VectorParams(
                        size=self.dense_dim,
                        distance=Distance.COSINE,
                        on_disk=False,
                    )
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        index=SparseIndexParams(on_disk=False)
                    )
                },
            )
            print(f"[vector_store] Created Qdrant collection '{self.collection}'.")
        else:
            print(f"[vector_store] Collection '{self.collection}' already exists.")

        # Payload index on repo_name is required for delete_by_repo filtering.
        # create_payload_index is idempotent — safe to call even if index already exists.
        self.client.create_payload_index(
            collection_name=self.collection,
            field_name="repo_name",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        print(f"[vector_store] Payload index on 'repo_name' ensured.")

    def upsert_batch(
        self,
        chunks: list[Chunk],
        dense_vecs: list,
        sparse_dicts: list,
    ) -> None:
        points: list[PointStruct] = []
        for chunk, dense, sparse_d in zip(chunks, dense_vecs, sparse_dicts):
            sparse_qdrant = _sparse_dict_to_qdrant(sparse_d)
            point_id = _make_point_id(chunk) if chunk.file_path else str(uuid.uuid4())
            points.append(PointStruct(
                id=point_id,
                vector={"dense": dense, "sparse": sparse_qdrant},
                payload=chunk.to_payload(),
            ))
        self.client.upsert(collection_name=self.collection, points=points, wait=True)

    def hybrid_search(
        self,
        dense_vec: list,
        sparse_vec: SparseVector,
        top_k: int = 8,
    ) -> list:
        """Reciprocal Rank Fusion over dense + sparse prefetch results."""
        results = self.client.query_points(
            collection_name=self.collection,
            prefetch=[
                Prefetch(query=dense_vec, using="dense", limit=top_k * 3),
                Prefetch(query=sparse_vec, using="sparse", limit=top_k * 3),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )
        return results.points

    def delete_by_repo(self, repo_name: str) -> None:
        """Remove all chunks belonging to a repository (for clean re-indexing)."""
        try:
            self.client.delete(
                collection_name=self.collection,
                points_selector=Filter(
                    must=[FieldCondition(key="repo_name", match=MatchValue(value=repo_name))]
                ),
            )
        except Exception as exc:
            print(f"[vector_store] delete_by_repo error (non-fatal): {exc}")

    def ping(self, retries: int = 3, base_delay: float = 2.0) -> bool:
        """Health ping — resets Qdrant free-tier inactivity timer. Retries on wakeup delay."""
        for attempt in range(retries):
            try:
                self.client.get_collections()
                return True
            except Exception as exc:
                if attempt < retries - 1:
                    time.sleep(base_delay * (attempt + 1))
                else:
                    print(f"[vector_store] ping failed after {retries} attempts: {exc}")
        return False
