"""Carryia — a support/bot-lane League of Legends post-game coach.

One installed package (`uv pip install -e .`) so tests, notebooks, CLI
(`python -m carryia.…`), and the Docker image all resolve imports the same way.

Subpackages:
  pipeline/  — build-time corpus pipeline (① Acquire → ⑤ Ingest)
  serve/     — the RAG answer-runtime (retrieval + generation)
  personal/  — the personal plane (Riot match data)
  eval/      — offline retrieval (P0-5) + answer (P0-6) evals
Top-level shared modules: schema (corpus record), phases (game phases), paths.
"""
