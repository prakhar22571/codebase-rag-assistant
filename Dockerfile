FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces requires non-root user with UID 1000
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
ENV HOME="/home/user"
ENV HF_HOME="/home/user/.cache/huggingface"

WORKDIR /app

COPY --chown=user requirements.txt .

# Install torch CPU-only first to avoid pulling CUDA (~700MB vs ~3GB)
RUN pip install --no-cache-dir torch==2.6.0 \
    --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Pre-compile tree-sitter grammars (avoids ~5s parse on first request)
RUN python -c "\
from tree_sitter_languages import get_parser; \
langs = ['python', 'javascript', 'typescript', 'go', 'rust', 'java', 'ruby']; \
[get_parser(l) for l in langs]; print('grammars compiled')"

# Pre-download BGE-M3 weights into the image layer (~2.2GB).
# This adds build time once but eliminates the cold-start download on every deploy.
RUN python -c "\
from FlagEmbedding import BGEM3FlagModel; \
BGEM3FlagModel('BAAI/bge-m3', use_fp16=False, devices=['cpu']); \
print('bge-m3 ready')"

# Copy application code last so model layers stay cached across code changes
COPY --chown=user . .

EXPOSE 7860

CMD ["python", "app/main.py"]
