from __future__ import annotations

from typing import Iterator

from groq import Groq

from app.config import settings

_SYSTEM_PROMPT = """\
You are an expert software engineer helping developers understand a GitHub codebase.

You are given numbered code snippets, documentation sections, and GitHub issue discussions \
as context. Each source is labelled [1], [2], etc.

CITATION FORMAT — use exactly these patterns:
- Code:   `[path/to/file.py:10-45]`
- Issues: `[Issue #123: Issue Title]`

RULES:
1. Cite sources inline every time you reference specific code or logic.
2. End your response with a "## Sources" section listing all citations used.
3. For "where is X implemented?" → lead with the file path and function/class name.
4. For "how does X work?" → walk through it step-by-step with inline citations.
5. If the context is insufficient, say so clearly — do not invent implementations.
6. Use markdown code blocks (with language tag) when showing example code.
7. Keep answers focused and precise.\
"""


class LLMGenerator:
    def __init__(self, api_key: str, model: str = "llama-3.1-70b-versatile") -> None:
        self.client = Groq(api_key=api_key)
        self.model = model

    def stream(
        self,
        query: str,
        context_text: str,
        history: list[dict] | None = None,
    ) -> Iterator[str]:
        """Yield streaming response tokens."""
        messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]

        # Include the last 4 conversation turns (8 messages) for follow-up context
        if history:
            messages.extend(history[-8:])

        user_content = (
            f"## Retrieved Context\n\n{context_text}\n\n"
            f"## Question\n\n{query}"
        )
        messages.append({"role": "user", "content": user_content})

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            max_tokens=settings.groq_max_tokens,
            temperature=settings.groq_temperature,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
