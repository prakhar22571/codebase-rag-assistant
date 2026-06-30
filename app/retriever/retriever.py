from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.retriever.vector_store import VectorStore, _sparse_dict_to_qdrant


@dataclass
class SearchResult:
    content: str
    chunk_type: str
    score: float
    repo_name: str
    file_path: Optional[str] = None
    language: Optional[str] = None
    node_type: Optional[str] = None
    name: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    parent_class: Optional[str] = None
    heading: Optional[str] = None
    issue_number: Optional[int] = None
    issue_title: Optional[str] = None
    issue_url: Optional[str] = None

    @property
    def citation(self) -> str:
        if self.chunk_type in ("issue_body", "issue_comment"):
            return f"Issue #{self.issue_number}: {self.issue_title}"
        if self.file_path and self.start_line and self.end_line:
            return f"{self.file_path}:{self.start_line}-{self.end_line}"
        return self.file_path or "(unknown source)"


class Retriever:
    def __init__(self, vector_store: VectorStore, embedder) -> None:
        self.store = vector_store
        self.embedder = embedder

    def search(self, query: str, top_k: int = 8) -> list[SearchResult]:
        dense_vec, sparse_dict = self.embedder.embed_query(query)
        sparse_qdrant = _sparse_dict_to_qdrant(sparse_dict)
        raw_points = self.store.hybrid_search(dense_vec, sparse_qdrant, top_k)
        return [self._to_result(p) for p in raw_points]

    def _to_result(self, point) -> SearchResult:
        p = point.payload
        return SearchResult(
            content=p.get("content", ""),
            chunk_type=p.get("chunk_type", "code"),
            score=point.score,
            repo_name=p.get("repo_name", ""),
            file_path=p.get("file_path"),
            language=p.get("language"),
            node_type=p.get("node_type"),
            name=p.get("name"),
            start_line=p.get("start_line"),
            end_line=p.get("end_line"),
            parent_class=p.get("parent_class"),
            heading=p.get("heading"),
            issue_number=p.get("issue_number"),
            issue_title=p.get("issue_title"),
            issue_url=p.get("issue_url"),
        )

    def format_context_for_llm(self, results: list[SearchResult]) -> str:
        parts: list[str] = []
        for i, r in enumerate(results, 1):
            if r.chunk_type in ("issue_body", "issue_comment"):
                header = f"[{i}] {r.citation}"
                parts.append(f"{header}\n{r.content}")
            else:
                lang_tag = r.language or ""
                header = f"[{i}] {r.citation}"
                if r.parent_class:
                    header += f"  (in class {r.parent_class})"
                parts.append(f"{header}\n```{lang_tag}\n{r.content}\n```")
        return "\n\n".join(parts)

    def format_sources_markdown(self, results: list[SearchResult]) -> str:
        lines = ["**Sources:**"]
        for i, r in enumerate(results, 1):
            if r.chunk_type in ("issue_body", "issue_comment") and r.issue_url:
                lines.append(f"{i}. [{r.citation}]({r.issue_url})")
            else:
                lines.append(f"{i}. `{r.citation}`")
        return "\n".join(lines)
