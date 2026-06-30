from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Iterator

from git import Repo as GitRepo
from git.exc import GitCommandError

from app.config import settings
from app.indexer.code_parser import (
    CODE_EXTENSIONS,
    SKIP_DIRS,
    Chunk,
    CodeParser,
    iter_code_files,
)
from app.indexer.doc_parser import DOC_EXTENSIONS, DocParser, iter_doc_files
from app.indexer.issue_fetcher import IssueFetcher


class IndexingPipeline:
    def __init__(self, embedder, vector_store) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.code_parser = CodeParser(
            max_tokens=settings.max_tokens_per_chunk,
            overlap_tokens=settings.chunk_overlap_tokens,
        )
        self.doc_parser = DocParser(max_tokens=settings.max_tokens_per_chunk)

    def run(self, repo_url: str, include_issues: bool = True) -> Iterator[str]:
        """Generator — yields progress log lines for Gradio streaming display."""
        repo_name = self._normalize_repo_url(repo_url)
        clone_url = f"https://github.com/{repo_name}.git"
        tmpdir = tempfile.mkdtemp(prefix="rag_clone_")

        try:
            yield f"Cloning {repo_name} (shallow clone)...\n"
            try:
                GitRepo.clone_from(clone_url, tmpdir, depth=1)
            except GitCommandError as exc:
                yield f"ERROR: Clone failed — {exc}\n"
                return

            yield "Clone complete.\n"

            # Delete existing chunks for this repo so re-indexing is clean
            yield f"Clearing existing index for {repo_name}...\n"
            self.vector_store.delete_by_repo(repo_name)

            repo_root = Path(tmpdir)
            all_chunks: list[Chunk] = []

            # ---- Code files ----
            code_files = list(iter_code_files(repo_root))
            yield f"Found {len(code_files)} code files to parse.\n"
            for i, fpath in enumerate(code_files):
                chunks = self.code_parser.parse_file(fpath, repo_root, repo_name)
                all_chunks.extend(chunks)
                if (i + 1) % 20 == 0 or (i + 1) == len(code_files):
                    yield f"  Code: {i + 1}/{len(code_files)} files → {len(all_chunks)} chunks so far...\n"

            # ---- Doc files ----
            doc_files = list(iter_doc_files(repo_root))
            yield f"Found {len(doc_files)} doc files to parse.\n"
            doc_chunks: list[Chunk] = []
            for fpath in doc_files:
                doc_chunks.extend(self.doc_parser.parse_file(fpath, repo_root, repo_name))
            all_chunks.extend(doc_chunks)
            yield f"  Docs: {len(doc_chunks)} chunks from {len(doc_files)} files.\n"

            yield f"Total code + doc chunks: {len(all_chunks)}\n"

            # ---- Embed + upsert ----
            yield "Embedding and uploading to Qdrant...\n"
            self._embed_and_upsert(all_chunks, "code+doc", yield_fn=lambda s: s)
            for progress in self._embed_and_upsert_stream(all_chunks, "code+doc"):
                yield progress

            # ---- Issues ----
            if include_issues:
                yield "Fetching GitHub issues...\n"
                fetcher = IssueFetcher(settings.github_token, repo_name)
                issue_chunks = fetcher.fetch_all()
                yield f"  {len(issue_chunks)} issue chunks fetched.\n"
                for progress in self._embed_and_upsert_stream(issue_chunks, "issues"):
                    yield progress

            yield "\nIndexing complete!\n"

        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def _embed_and_upsert_stream(
        self,
        chunks: list[Chunk],
        label: str,
        batch_size: int = 32,
    ) -> Iterator[str]:
        if not chunks:
            return
        total = len(chunks)
        for start in range(0, total, batch_size):
            batch = chunks[start : start + batch_size]
            texts = [c.content for c in batch]
            dense_vecs, sparse_dicts = self.embedder.embed_documents(texts)
            self.vector_store.upsert_batch(batch, dense_vecs, sparse_dicts)
            done = min(start + batch_size, total)
            yield f"  [{label}] Uploaded {done}/{total} chunks\n"

    def _embed_and_upsert(self, chunks, label, yield_fn):
        # Kept for interface compat; streaming version used in practice
        pass

    @staticmethod
    def _normalize_repo_url(url: str) -> str:
        url = url.strip()
        if url.startswith("https://github.com/"):
            url = url[len("https://github.com/"):]
        url = url.rstrip("/")
        if url.endswith(".git"):
            url = url[:-4]
        return url  # "owner/repo"
