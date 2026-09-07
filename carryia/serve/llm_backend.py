"""LLM backend selection -- one place that decides which client + model the answer
generator (rag_helper.RAGBase) and the P0-6 judge (eval_answer) run on.

Two backends, chosen by the CARRYIA_LLM_BACKEND env var:
  "anthropic" (default) -- the direct Claude API; reviewers bring ANTHROPIC_API_KEY.
                           This is the reproducibility path, left
                           as the default so a reviewer's run is unchanged.
  "bedrock"             -- Amazon Bedrock, billed to AWS. The builder's switch for
                           running the P0-6 eval on AWS credits.
                           Auth is a Bedrock API key (bearer token): the SDK reads it
                           from AWS_BEARER_TOKEN_BEDROCK automatically.

Both clients satisfy evaluation_utils.llm_structured's `"anthropic" in module` branch
(AnthropicBedrock lives in the anthropic package), so the generate + judge code is
identical across backends -- only construction and the model id differ.
"""

from __future__ import annotations

import os

# Bedrock needs a prefixed inference-profile id (the bare model id fails on-demand with
# HTTP 400); "global." carries no regional pricing premium.
_DIRECT_MODEL = "claude-haiku-4-5"
_BEDROCK_MODEL = "global.anthropic.claude-haiku-4-5-20251001-v1:0"

# The P0-6 eval fires a burst of sequential calls; Bedrock's default on-demand quota
# throttles that with 429s. The SDK retries 429/5xx with exponential backoff -- lift its
# budget (default 2) so a batch run rides out per-minute throttling. Overridable via env.
_MAX_RETRIES = int(os.environ.get("CARRYIA_MAX_RETRIES", "10"))


def backend() -> str:
    """The selected backend name, normalised. Defaults to the direct Anthropic API.
    `or` (not a get-default) so an env var set to "" -- as docker-compose does with
    `${CARRYIA_LLM_BACKEND:-anthropic}` when it's unset in .env -- falls back too."""
    return (os.environ.get("CARRYIA_LLM_BACKEND") or "anthropic").lower()


def make_client():
    """Return an Anthropic-compatible client for the selected backend.

    Bedrock: AnthropicBedrock reads the bearer token from AWS_BEARER_TOKEN_BEDROCK and
    the region from AWS_REGION (default us-west-2). Direct: Anthropic reads
    ANTHROPIC_API_KEY. `anthropic` is imported lazily so importing this module stays
    dependency-free for the tests.
    """
    import anthropic

    if backend() == "bedrock":
        return anthropic.AnthropicBedrock(
            aws_region=os.environ.get("AWS_REGION", "us-west-2"),
            max_retries=_MAX_RETRIES,
        )
    return anthropic.Anthropic(max_retries=_MAX_RETRIES)


def model_id() -> str:
    """The model id for the selected backend (Haiku 4.5 either way), overridable via
    CARRYIA_MODEL / CARRYIA_BEDROCK_MODEL for a quick model swap without code change.
    `or` (not a get-default) so an override set to "" -- as docker-compose does with
    `${CARRYIA_BEDROCK_MODEL:-}` when it's unset in .env -- falls back to the pinned id
    instead of sending an empty model to the API."""
    if backend() == "bedrock":
        return os.environ.get("CARRYIA_BEDROCK_MODEL") or _BEDROCK_MODEL
    return os.environ.get("CARRYIA_MODEL") or _DIRECT_MODEL
