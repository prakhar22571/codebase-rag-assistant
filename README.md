# Codebase RAG Assistant

Ask natural-language questions about any public GitHub repository. The app clones and indexes the repo (code, docs, and issues), then answers questions with cited sources using hybrid (dense + sparse) retrieval and a Groq-hosted LLM.

## How it works

1. **Index** — Enter a GitHub repo (`owner/repo` or full URL). The repo is shallow-cloned, parsed into chunks (code via tree-sitter, docs via mistune, issues via the GitHub API), embedded, and upserted into Qdrant.
2. **Retrieve** — Each question is embedded and matched against the indexed chunks using hybrid dense + sparse search.
3. **Generate** — Retrieved chunks are passed as context to a Groq LLM, which streams an answer with source citations.

## Stack

- **UI / API**: Gradio chat UI mounted on FastAPI (`/health` endpoint for uptime monitoring)
- **LLM**: Groq (`llama-3.1-70b-versatile` by default)
- **Vector store**: Qdrant (hybrid dense + sparse search)
- **Embeddings**: [Voyage AI](https://www.voyageai.com/) `voyage-code-3` for dense vectors if `VOYAGE_API_KEY` is set, otherwise local `BAAI/bge-m3` for both dense and sparse
- **Code parsing**: tree-sitter (Python, JavaScript, TypeScript, Go, Rust, Java, Ruby)
- **Doc parsing**: mistune (Markdown)
- **Issue ingestion**: PyGithub

## Setup

### Prerequisites

- Python 3.11
- A [Groq](https://console.groq.com/) API key
- A [Qdrant](https://cloud.qdrant.io/) instance (cloud free tier works) and API key

### Local install

```bash
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

### Configuration

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes | Groq API key for the LLM |
| `QDRANT_URL` | Yes | Qdrant cluster URL |
| `QDRANT_API_KEY` | Yes | Qdrant API key |
| `VOYAGE_API_KEY` | No | Voyage AI key for higher-quality dense embeddings; omit to fall back to local `BAAI/bge-m3` |
| `GITHUB_TOKEN` | No | GitHub token for issue fetching (5,000 req/hr vs 60/hr unauthenticated) |
| `GROQ_MODEL` | No | Overrides the default Groq model |
| `QDRANT_COLLECTION` | No | Overrides the default Qdrant collection name |

**Never commit `.env`** — it holds live credentials and is excluded via `.gitignore`.

### Run

```bash
python app/main.py
```

The app serves on `http://localhost:7860`. Use the **Index Repository** tab to index a repo, then ask questions in the **Chat** tab.

## Docker

```bash
docker build -t codebase-rag-assistant .
docker run -p 7860:7860 --env-file .env codebase-rag-assistant
```

The image is built for Hugging Face Spaces (non-root UID 1000, CPU-only torch, pre-downloaded `BAAI/bge-m3` weights and tree-sitter grammars to avoid cold-start delays).
