# Carryia app image (P0-9) -- the Streamlit coach the reviewer runs.
#
# Runtime only: requirements.txt (not requirements-dev.txt) and the runtime slice of
# the package + the committed data it reads. The pipeline/eval code and data/raw stay
# out (.dockerignore keeps raw out of the
# build context). No model weights are baked in -- the local embedding model downloads
# once at first startup, and the LLM is a hosted API.
FROM python:3.12-slim

WORKDIR /app

# Runtime dependencies first, so the layer caches across code edits.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# The package and the three committed files the app reads at runtime: the coaching
# corpus (index source), the subject snapshot, and the cohort benchmark.
COPY pyproject.toml ./
COPY carryia ./carryia
COPY assets ./assets
COPY data/corpus.jsonl ./data/corpus.jsonl
COPY data/snapshot ./data/snapshot
COPY data/benchmark ./data/benchmark
RUN pip install --no-cache-dir --no-deps -e .

EXPOSE 8501

# The vector + keyword indexes are built in-memory from the committed corpus at startup
# (keyless, local embeddings), so there is no separate ingest step.
CMD ["streamlit", "run", "carryia/serve/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
