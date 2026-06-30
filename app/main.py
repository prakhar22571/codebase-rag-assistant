"""
Entry point for the Codebase RAG Assistant.

Starts a FastAPI app with a /health endpoint (used by UptimeRobot to keep the
HF Space and Qdrant free cluster alive) and a Gradio chat UI mounted at /.
"""
from __future__ import annotations

import gradio as gr
import uvicorn
from fastapi import FastAPI

from app.config import settings
from app.generator.llm import LLMGenerator
from app.indexer.pipeline import IndexingPipeline
from app.retriever.embedder import create_embedder
from app.retriever.retriever import Retriever
from app.retriever.vector_store import VectorStore

# ---------------------------------------------------------------------------
# Global singletons — initialised once at startup
# ---------------------------------------------------------------------------

print("[main] Initialising embedder (may take ~30s on first run)...")
embedder = create_embedder()

print("[main] Connecting to Qdrant...")
vector_store = VectorStore(
    url=settings.qdrant_url,
    api_key=settings.qdrant_api_key,
    collection=settings.qdrant_collection,
    dense_dim=settings.embedding_dim,
)
vector_store.ensure_collection()

retriever = Retriever(vector_store=vector_store, embedder=embedder)
llm = LLMGenerator(api_key=settings.groq_api_key, model=settings.groq_model)
pipeline = IndexingPipeline(embedder=embedder, vector_store=vector_store)

print("[main] All components ready.")

# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------

fastapi_app = FastAPI(title="Codebase RAG API")


@fastapi_app.get("/health")
async def health():
    qdrant_ok = vector_store.ping()
    return {
        "status": "ok",
        "qdrant": "alive" if qdrant_ok else "unreachable",
        "embedder": type(embedder).__name__,
        "model": settings.groq_model,
    }


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def _bot_stream(history: list[dict], _sources_state: str):
    """Stream assistant tokens for the most recent user message."""
    if not history:
        yield history, ""
        return

    last_user_msg = history[-1]["content"]
    results = retriever.search(last_user_msg, top_k=settings.top_k)

    if not results:
        history.append({
            "role": "assistant",
            "content": "No relevant code found in the indexed repositories. "
                       "Please index a repository first using the **Index Repository** tab.",
        })
        yield history, ""
        return

    context_text = retriever.format_context_for_llm(results)
    sources_md = retriever.format_sources_markdown(results)

    # Build Groq-compatible history from previous turns (exclude the latest user msg)
    groq_history = [
        {"role": t["role"], "content": t["content"]}
        for t in history[:-1][-8:]
    ]

    history.append({"role": "assistant", "content": ""})

    for token in llm.stream(last_user_msg, context_text, groq_history):
        history[-1]["content"] += token
        yield history, sources_md


def _user_submit(message: str, history: list[dict]):
    history = history or []
    history.append({"role": "user", "content": message})
    return "", history


def _index_repo(repo_url: str, include_issues: bool):
    """Generator for the Index tab — accumulates and yields the full log each iteration."""
    log = ""
    for line in pipeline.run(repo_url, include_issues):
        log += line
        yield log


with gr.Blocks(
    title="Codebase RAG Assistant",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(
        "# Codebase RAG Assistant\n"
        "Ask questions about any indexed GitHub repository. "
        "Use the **Index Repository** tab to add a new repo."
    )

    with gr.Tabs():

        # ----------------------------------------------------------------
        # Chat tab
        # ----------------------------------------------------------------
        with gr.Tab("Chat"):
            chatbot = gr.Chatbot(
                height=500,
                type="messages",
                show_copy_button=True,
                placeholder=(
                    "Index a repository first, then ask:\n\n"
                    "- *How does Session.send() work?*\n"
                    "- *Where is the authentication middleware implemented?*\n"
                    "- *What does the tokenizer do with special tokens?*"
                ),
            )

            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Ask a question about the codebase…",
                    container=False,
                    scale=8,
                    autofocus=True,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
                clear_btn = gr.Button("Clear", scale=1)

            with gr.Accordion("Source Citations", open=False):
                sources_display = gr.Markdown(value="")

            sources_state = gr.State("")

            # Wire up submission — first add user msg, then stream bot response
            (
                msg_input.submit(
                    fn=_user_submit,
                    inputs=[msg_input, chatbot],
                    outputs=[msg_input, chatbot],
                ).then(
                    fn=_bot_stream,
                    inputs=[chatbot, sources_state],
                    outputs=[chatbot, sources_display],
                )
            )
            (
                send_btn.click(
                    fn=_user_submit,
                    inputs=[msg_input, chatbot],
                    outputs=[msg_input, chatbot],
                ).then(
                    fn=_bot_stream,
                    inputs=[chatbot, sources_state],
                    outputs=[chatbot, sources_display],
                )
            )
            clear_btn.click(
                fn=lambda: ([], "", ""),
                outputs=[chatbot, sources_display, sources_state],
            )

        # ----------------------------------------------------------------
        # Index tab
        # ----------------------------------------------------------------
        with gr.Tab("Index Repository"):
            gr.Markdown(
                "Enter a public GitHub repository to index. "
                "Indexing typically takes **2–10 minutes** depending on repo size. "
                "Re-indexing a repo replaces its previous index."
            )

            with gr.Row():
                repo_input = gr.Textbox(
                    label="GitHub Repository",
                    placeholder="e.g. psf/requests  or  https://github.com/huggingface/transformers",
                    scale=4,
                )
                include_issues_cb = gr.Checkbox(
                    label="Include Issues",
                    value=True,
                    scale=1,
                )

            index_btn = gr.Button("Start Indexing", variant="primary")

            index_log = gr.Textbox(
                label="Indexing Progress",
                interactive=False,
                lines=18,
                max_lines=40,
            )

            index_btn.click(
                fn=_index_repo,
                inputs=[repo_input, include_issues_cb],
                outputs=[index_log],
            )


# Mount Gradio on FastAPI so /health and the UI coexist on the same port
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")
