from __future__ import annotations

import time
from typing import Optional

from github import Github, GithubException

from app.indexer.code_parser import Chunk, count_tokens


class IssueFetcher:
    def __init__(
        self,
        token: Optional[str],
        repo_name: str,
        max_issues: int = 500,
    ) -> None:
        self.gh = Github(token) if token else Github()
        self.repo_name = repo_name
        self.max_issues = max_issues

    def fetch_all(self) -> list[Chunk]:
        try:
            repo = self.gh.get_repo(self.repo_name)
        except GithubException as exc:
            print(f"[issue_fetcher] GitHub error: {exc}")
            return []

        chunks: list[Chunk] = []
        count = 0

        for state in ("open", "closed"):
            for issue in repo.get_issues(state=state):
                if issue.pull_request:
                    continue
                if count >= self.max_issues:
                    break

                chunks.extend(self._issue_to_chunks(issue))
                count += 1

                if count % 50 == 0:
                    time.sleep(1)

        return chunks

    def _issue_to_chunks(self, issue) -> list[Chunk]:
        labels = [label.name for label in issue.labels]
        body = issue.body or "(no description)"
        content = (
            f"Issue #{issue.number}: {issue.title}\n"
            f"State: {issue.state}\n"
            f"Labels: {', '.join(labels) if labels else 'none'}\n\n"
            f"{body}"
        )

        chunks = [
            Chunk(
                content=content,
                chunk_type="issue_body",
                repo_name=self.repo_name,
                issue_number=issue.number,
                issue_title=issue.title,
                issue_state=issue.state,
                issue_url=issue.html_url,
                issue_labels=labels,
                name=f"Issue #{issue.number}: {issue.title}",
            )
        ]

        try:
            comments = list(issue.get_comments())
        except GithubException:
            return chunks

        if not comments:
            return chunks

        batch_texts: list[str] = []
        batch_tokens = 0
        batch_idx = 0

        for comment in comments:
            comment_text = f"@{comment.user.login}:\n{comment.body}"
            t = count_tokens(comment_text)

            if batch_tokens + t > 400 and batch_texts:
                self._flush_comment_batch(chunks, batch_texts, batch_idx, issue, labels)
                batch_texts = [comment_text]
                batch_tokens = t
                batch_idx += 1
            else:
                batch_texts.append(comment_text)
                batch_tokens += t

        if batch_texts:
            self._flush_comment_batch(chunks, batch_texts, batch_idx, issue, labels)

        return chunks

    def _flush_comment_batch(
        self,
        chunks: list[Chunk],
        batch_texts: list[str],
        batch_idx: int,
        issue,
        labels: list[str],
    ) -> None:
        content = (
            f"Comments on Issue #{issue.number}: {issue.title} "
            f"(part {batch_idx + 1})\n\n"
            + "\n\n---\n\n".join(batch_texts)
        )
        chunks.append(Chunk(
            content=content,
            chunk_type="issue_comment",
            repo_name=self.repo_name,
            issue_number=issue.number,
            issue_title=issue.title,
            issue_state=issue.state,
            issue_url=issue.html_url,
            issue_labels=labels,
            name=f"Issue #{issue.number} comments (part {batch_idx + 1})",
            chunk_index=batch_idx,
        ))
