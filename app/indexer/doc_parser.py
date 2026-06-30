from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator, Optional

from app.indexer.code_parser import Chunk, count_tokens

DOC_EXTENSIONS = {".md", ".rst", ".txt"}

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_PARAGRAPH_SEP = re.compile(r"\n{2,}")

SKIP_DIRS = {
    "__pycache__", "node_modules", ".git", "dist", "build", "target",
    "venv", ".venv", "vendor",
}


class DocParser:
    def __init__(self, max_tokens: int = 512) -> None:
        self.max_tokens = max_tokens

    def parse_file(self, file_path: Path, repo_root: Path, repo_name: str) -> list[Chunk]:
        suffix = file_path.suffix.lower()
        if suffix not in DOC_EXTENSIONS:
            return []
        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []

        relative_path = str(file_path.relative_to(repo_root)).replace("\\", "/")

        if suffix in (".md", ".txt"):
            return self._parse_markdown(content, relative_path, repo_name)
        if suffix == ".rst":
            return self._parse_rst(content, relative_path, repo_name)
        return []

    # ------------------------------------------------------------------
    # Markdown
    # ------------------------------------------------------------------

    def _parse_markdown(self, content: str, file_path: str, repo_name: str) -> list[Chunk]:
        sections = self._split_by_headings(content)
        chunks: list[Chunk] = []
        for section in sections:
            chunks.extend(self._chunk_section(
                text=section["content"],
                heading=section["heading"],
                file_path=file_path,
                repo_name=repo_name,
                start_line=section["start_line"],
            ))
        return chunks

    def _split_by_headings(self, content: str) -> list[dict]:
        lines = content.split("\n")
        heading_positions: list[tuple[int, int, str]] = []

        for i, line in enumerate(lines):
            m = re.match(r"^(#{1,6})\s+(.+)$", line)
            if m:
                heading_positions.append((i, len(m.group(1)), m.group(2)))

        if not heading_positions:
            return [{"heading": "(root)", "content": content, "start_line": 1}]

        sections: list[dict] = []

        # Content before first heading
        if heading_positions[0][0] > 0:
            intro = "\n".join(lines[: heading_positions[0][0]])
            if intro.strip():
                sections.append({"heading": "(introduction)", "content": intro, "start_line": 1})

        for idx, (line_num, _level, heading) in enumerate(heading_positions):
            if idx + 1 < len(heading_positions):
                next_line = heading_positions[idx + 1][0]
                section_text = "\n".join(lines[line_num:next_line])
            else:
                section_text = "\n".join(lines[line_num:])
            sections.append({
                "heading": heading,
                "content": section_text,
                "start_line": line_num + 1,
            })

        return sections

    def _chunk_section(
        self,
        text: str,
        heading: str,
        file_path: str,
        repo_name: str,
        start_line: int,
    ) -> list[Chunk]:
        if count_tokens(text) <= self.max_tokens:
            return [Chunk(
                content=text,
                chunk_type="doc",
                repo_name=repo_name,
                file_path=file_path,
                heading=heading,
                node_type="section",
                name=heading,
                start_line=start_line,
                end_line=start_line + text.count("\n"),
            )]

        paragraphs = _PARAGRAPH_SEP.split(text)
        chunks: list[Chunk] = []
        current_batch: list[str] = []
        current_tokens = 0
        batch_start_line = start_line

        def flush():
            nonlocal current_batch, current_tokens, batch_start_line
            batch_text = "\n\n".join(current_batch)
            chunks.append(Chunk(
                content=batch_text,
                chunk_type="doc",
                repo_name=repo_name,
                file_path=file_path,
                heading=heading,
                node_type="section",
                name=heading,
                start_line=batch_start_line,
                end_line=batch_start_line + batch_text.count("\n"),
                chunk_index=len(chunks),
            ))
            batch_start_line += batch_text.count("\n") + 2
            current_batch = []
            current_tokens = 0

        for para in paragraphs:
            para_tokens = count_tokens(para)
            if current_tokens + para_tokens > self.max_tokens and current_batch:
                flush()
            current_batch.append(para)
            current_tokens += para_tokens

        if current_batch:
            flush()

        for c in chunks:
            c.total_chunks = len(chunks)

        return chunks

    # ------------------------------------------------------------------
    # RST
    # ------------------------------------------------------------------

    def _parse_rst(self, content: str, file_path: str, repo_name: str) -> list[Chunk]:
        lines = content.split("\n")
        sections: list[dict] = []
        section_start = 0
        current_heading = "(root)"

        for i in range(len(lines) - 1):
            next_line = lines[i + 1]
            if (
                next_line
                and all(c == next_line[0] for c in next_line)
                and next_line[0] in "=-~^\"'`#*+"
                and len(next_line) >= len(lines[i])
                and lines[i].strip()
            ):
                if i > section_start:
                    section_text = "\n".join(lines[section_start:i])
                    if section_text.strip():
                        sections.append({
                            "heading": current_heading,
                            "content": section_text,
                            "start_line": section_start + 1,
                        })
                current_heading = lines[i].strip()
                section_start = i

        remaining = "\n".join(lines[section_start:])
        if remaining.strip():
            sections.append({
                "heading": current_heading,
                "content": remaining,
                "start_line": section_start + 1,
            })

        chunks: list[Chunk] = []
        for s in sections:
            chunks.extend(self._chunk_section(
                text=s["content"],
                heading=s["heading"],
                file_path=file_path,
                repo_name=repo_name,
                start_line=s["start_line"],
            ))
        return chunks


def iter_doc_files(repo_root: Path) -> Iterator[Path]:
    for path in repo_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in DOC_EXTENSIONS:
            if not any(part in SKIP_DIRS for part in path.parts):
                yield path
