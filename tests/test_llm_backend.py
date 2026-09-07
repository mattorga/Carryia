"""Spec for carryia/serve/llm_backend.py -- backend + model-id selection.

GREEN: pure env-var logic, no client construction. The load-bearing case is the
empty-string one: docker-compose passes optional overrides as "" (via
`${VAR:-}`), and `os.environ.get(key, default)` returns "" for a set-but-empty key
-- which once sent an empty model id to Bedrock. These pin the `or`-fallback fix.
"""

from __future__ import annotations

from carryia.serve import llm_backend
from carryia.serve.llm_backend import _BEDROCK_MODEL, _DIRECT_MODEL


# --- backend() ---------------------------------------------------------------

def test_backend_defaults_to_anthropic_when_unset(monkeypatch):
    monkeypatch.delenv("CARRYIA_LLM_BACKEND", raising=False)
    assert llm_backend.backend() == "anthropic"


def test_backend_empty_string_falls_back_to_anthropic(monkeypatch):
    monkeypatch.setenv("CARRYIA_LLM_BACKEND", "")   # as compose sets it when unset in .env
    assert llm_backend.backend() == "anthropic"


def test_backend_selects_bedrock(monkeypatch):
    monkeypatch.setenv("CARRYIA_LLM_BACKEND", "bedrock")
    assert llm_backend.backend() == "bedrock"


# --- model_id() --------------------------------------------------------------

def test_model_id_defaults_per_backend(monkeypatch):
    monkeypatch.delenv("CARRYIA_MODEL", raising=False)
    monkeypatch.delenv("CARRYIA_BEDROCK_MODEL", raising=False)
    monkeypatch.setenv("CARRYIA_LLM_BACKEND", "anthropic")
    assert llm_backend.model_id() == _DIRECT_MODEL
    monkeypatch.setenv("CARRYIA_LLM_BACKEND", "bedrock")
    assert llm_backend.model_id() == _BEDROCK_MODEL


def test_model_id_ignores_empty_override_and_uses_the_pinned_id(monkeypatch):
    # The compose-set "" must NOT become the model id (an empty model breaks the call).
    monkeypatch.setenv("CARRYIA_LLM_BACKEND", "bedrock")
    monkeypatch.setenv("CARRYIA_BEDROCK_MODEL", "")
    assert llm_backend.model_id() == _BEDROCK_MODEL
    monkeypatch.setenv("CARRYIA_LLM_BACKEND", "anthropic")
    monkeypatch.setenv("CARRYIA_MODEL", "")
    assert llm_backend.model_id() == _DIRECT_MODEL


def test_model_id_honours_a_real_override(monkeypatch):
    monkeypatch.setenv("CARRYIA_LLM_BACKEND", "bedrock")
    monkeypatch.setenv("CARRYIA_BEDROCK_MODEL", "some.other.model-v1:0")
    assert llm_backend.model_id() == "some.other.model-v1:0"
