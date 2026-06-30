from __future__ import annotations

import uuid
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import tiktoken

with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    from tree_sitter_languages import get_parser

# ---------------------------------------------------------------------------
# Token counting (shared encoder)
# ---------------------------------------------------------------------------

_ENCODER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


# ---------------------------------------------------------------------------
# Unified Chunk data model (used by all parsers)
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    content: str
    chunk_type: str        # "code" | "doc" | "issue_body" | "issue_comment"
    repo_name: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    # Code fields
    file_path: Optional[str] = None
    language: Optional[str] = None
    node_type: Optional[str] = None    # "function" | "method" | "class"
    name: Optional[str] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    parent_class: Optional[str] = None

    # Doc fields
    heading: Optional[str] = None

    # Issue fields
    issue_number: Optional[int] = None
    issue_title: Optional[str] = None
    issue_state: Optional[str] = None
    issue_url: Optional[str] = None
    issue_labels: Optional[list] = None

    # Window fields (for split large functions)
    chunk_index: int = 0
    total_chunks: int = 1

    def to_payload(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "id" and v is not None}


# ---------------------------------------------------------------------------
# Language / extension mappings
# ---------------------------------------------------------------------------

CODE_EXTENSIONS = {".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java", ".rb"}

EXT_TO_LANG: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".rb": "ruby",
}

FUNCTION_TYPES: dict[str, set[str]] = {
    "python": {"function_definition", "decorated_definition"},
    "javascript": {"function_declaration", "function_expression", "arrow_function", "method_definition"},
    "typescript": {"function_declaration", "function_expression", "arrow_function", "method_definition"},
    "go": {"function_declaration", "method_declaration"},
    "rust": {"function_item"},
    "java": {"method_declaration", "constructor_declaration"},
    "ruby": {"method", "singleton_method"},
}

CLASS_TYPES: dict[str, set[str]] = {
    "python": {"class_definition"},
    "javascript": {"class_declaration", "class_expression"},
    "typescript": {"class_declaration", "class_expression"},
    "go": {"type_declaration"},
    "rust": {"impl_item", "struct_item", "enum_item", "trait_item"},
    "java": {"class_declaration", "interface_declaration", "enum_declaration"},
    "ruby": {"class", "module"},
}

SKIP_DIRS: set[str] = {
    "__pycache__", "node_modules", ".git", "dist", "build", "target",
    ".pytest_cache", "venv", ".venv", "env", ".tox", "coverage",
    "vendor", "third_party", ".mypy_cache", ".ruff_cache",
}


# ---------------------------------------------------------------------------
# CodeParser
# ---------------------------------------------------------------------------

class CodeParser:
    def __init__(self, max_tokens: int = 512, overlap_tokens: int = 64) -> None:
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self._parsers: dict[str, object] = {}

    def _get_parser(self, lang: str):
        if lang not in self._parsers:
            self._parsers[lang] = get_parser(lang)
        return self._parsers[lang]

    def parse_file(self, file_path: Path, repo_root: Path, repo_name: str) -> list[Chunk]:
        lang = EXT_TO_LANG.get(file_path.suffix.lower())
        if lang is None:
            return []

        try:
            source_bytes = file_path.read_bytes()
            source_str = source_bytes.decode("utf-8", errors="replace")
        except (OSError, PermissionError):
            return []

        relative_path = str(file_path.relative_to(repo_root)).replace("\\", "/")
        parser = self._get_parser(lang)
        tree = parser.parse(source_bytes)

        raw_chunks: list[Chunk] = []
        self._traverse(
            node=tree.root_node,
            source_str=source_str,
            file_path=relative_path,
            lang=lang,
            repo_name=repo_name,
            parent_class=None,
            chunks=raw_chunks,
        )

        final_chunks: list[Chunk] = []
        for chunk in raw_chunks:
            if count_tokens(chunk.content) > self.max_tokens:
                final_chunks.extend(self._sliding_window(chunk))
            else:
                final_chunks.append(chunk)

        return final_chunks

    def _traverse(
        self,
        node,
        source_str: str,
        file_path: str,
        lang: str,
        repo_name: str,
        parent_class: Optional[str],
        chunks: list[Chunk],
    ) -> None:
        func_types = FUNCTION_TYPES.get(lang, set())
        class_types = CLASS_TYPES.get(lang, set())
        node_type = node.type

        if node_type in class_types:
            class_name = self._get_node_name(node, source_str)
            class_text = source_str[node.start_byte:node.end_byte]
            chunks.append(Chunk(
                content=class_text,
                chunk_type="code",
                repo_name=repo_name,
                file_path=file_path,
                language=lang,
                node_type="class",
                name=class_name or node_type,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                parent_class=parent_class,
            ))
            for child in node.children:
                self._traverse(child, source_str, file_path, lang, repo_name,
                                parent_class=class_name, chunks=chunks)
            return

        if node_type in func_types:
            func_name = self._get_node_name(node, source_str)
            func_text = source_str[node.start_byte:node.end_byte]
            determined_type = "method" if parent_class else "function"
            chunks.append(Chunk(
                content=func_text,
                chunk_type="code",
                repo_name=repo_name,
                file_path=file_path,
                language=lang,
                node_type=determined_type,
                name=func_name or "anonymous",
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                parent_class=parent_class,
            ))
            # Stop recursion here — nested functions would produce overlapping chunks
            return

        for child in node.children:
            self._traverse(child, source_str, file_path, lang, repo_name,
                           parent_class=parent_class, chunks=chunks)

    def _get_node_name(self, node, source_str: str) -> Optional[str]:
        name_node = node.child_by_field_name("name")
        if name_node:
            return source_str[name_node.start_byte:name_node.end_byte]
        for child in node.children:
            if child.type in (
                "identifier", "property_identifier", "field_identifier",
                "type_identifier", "constant",
            ):
                return source_str[child.start_byte:child.end_byte]
        return None

    def _sliding_window(self, chunk: Chunk) -> list[Chunk]:
        lines = chunk.content.split("\n")
        sig_lines = self._extract_signature_lines(lines)
        sig_text = "\n".join(sig_lines)
        body_lines = lines[len(sig_lines):]

        windows: list[Chunk] = []
        start = 0

        while start < len(body_lines):
            end = start
            while end < len(body_lines):
                candidate = sig_text + "\n" + "\n".join(body_lines[start:end + 1])
                if count_tokens(candidate) > self.max_tokens:
                    break
                end += 1

            if end == start:
                end = start + 1

            window_text = sig_text + "\n" + "\n".join(body_lines[start:end])
            window_start_line = (chunk.start_line or 1) + len(sig_lines) + start
            window_end_line = (chunk.start_line or 1) + len(sig_lines) + end - 1

            windows.append(Chunk(
                content=window_text,
                chunk_type=chunk.chunk_type,
                repo_name=chunk.repo_name,
                file_path=chunk.file_path,
                language=chunk.language,
                node_type=chunk.node_type,
                name=chunk.name,
                start_line=window_start_line,
                end_line=window_end_line,
                parent_class=chunk.parent_class,
                chunk_index=len(windows),
            ))

            # Calculate overlap: back up from end by overlap_tokens worth of lines
            overlap_counted = 0
            new_start = end
            for i in range(end - 1, start - 1, -1):
                overlap_counted += count_tokens(body_lines[i])
                if overlap_counted >= self.overlap_tokens:
                    new_start = i
                    break

            start = new_start if new_start < end else end

        for w in windows:
            w.total_chunks = len(windows)

        return windows

    def _extract_signature_lines(self, lines: list[str]) -> list[str]:
        sig: list[str] = []
        for line in lines[:5]:
            sig.append(line)
            stripped = line.strip()
            if stripped.endswith(":") or stripped.endswith("{"):
                break
        return sig


# ---------------------------------------------------------------------------
# File iterator for the pipeline
# ---------------------------------------------------------------------------

def iter_code_files(repo_root: Path) -> Iterator[Path]:
    for path in repo_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in CODE_EXTENSIONS:
            if not any(part in SKIP_DIRS for part in path.parts):
                yield path
