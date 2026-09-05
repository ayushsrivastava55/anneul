"""Tests for anneal.llm: model resolution and cost math (offline).

Also a live smoke test, skipped without a key.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from anneal import llm
from anneal.llm import Usage, cost, get_client, resolve_model

FIXTURE_YAML = """
tiers:
  frontier: {provider: frontier, model: big-model, price_in: 10.0, price_out: 30.0}
  mid: {provider: tensormux, model: mid-model, price_in: 1.0, price_out: 2.0}
  cheap: {provider: tensormux, model: small-model, price_in: 0.1, price_out: 0.2}
downshift_order: [frontier, mid, cheap]
providers:
  tensormux: {base_url_env: T_BASE, api_key_env: T_KEY}
  frontier: {base_url_env: F_BASE, api_key_env: F_KEY}
"""


@pytest.fixture
def models_path(tmp_path: Path) -> Path:
    path = tmp_path / "models.yaml"
    path.write_text(FIXTURE_YAML)
    return path


def test_resolve_model_returns_tier_model(models_path: Path) -> None:
    assert resolve_model("mid", models_path) == "mid-model"
    assert resolve_model("cheap", models_path) == "small-model"


def test_resolve_model_unknown_tier_raises(models_path: Path) -> None:
    with pytest.raises(KeyError):
        resolve_model("nope", models_path)


def test_cost_uses_requested_model_when_no_backend(models_path: Path) -> None:
    usage = Usage(tokens_in=1_000_000, tokens_out=500_000, backend=None, model="mid-model")
    assert cost(usage, models_path) == pytest.approx(1.0 + 1.0)


def test_cost_prefers_backend_when_it_is_a_known_model(models_path: Path) -> None:
    usage = Usage(tokens_in=1_000_000, tokens_out=0, backend="small-model", model="mid-model")
    assert cost(usage, models_path) == pytest.approx(0.1)


def test_cost_falls_back_to_model_when_backend_unknown(models_path: Path) -> None:
    usage = Usage(tokens_in=0, tokens_out=1_000_000, backend="gpu-node-7", model="big-model")
    assert cost(usage, models_path) == pytest.approx(30.0)


def test_cost_unknown_model_raises(models_path: Path) -> None:
    usage = Usage(tokens_in=1, tokens_out=1, backend=None, model="ghost")
    with pytest.raises(KeyError):
        cost(usage, models_path)


def test_get_client_reads_provider_env(models_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("T_BASE", "http://localhost:1/v1")
    monkeypatch.setenv("T_KEY", "sk-test")
    client = get_client("mid", models_path)
    assert str(client.base_url).startswith("http://localhost:1/v1")
    assert client.api_key == "sk-test"


def test_get_client_missing_key_raises(models_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("T_BASE", "http://localhost:1/v1")
    monkeypatch.delenv("T_KEY", raising=False)
    with pytest.raises(RuntimeError, match="T_KEY"):
        get_client("mid", models_path)


def test_real_models_yaml_parses() -> None:
    models = llm.load_models()
    assert set(models["tiers"]) >= {"frontier", "mid", "cheap"}
    assert models["downshift_order"][0] == "frontier"


@pytest.mark.skipif(
    not os.getenv("TENSORMUX_API_KEY"), reason="ready pending key: TENSORMUX_API_KEY unset"
)
def test_live_hello_world_is_traced_and_costed() -> None:
    messages = [{"role": "user", "content": "Say hello in one word."}]
    text, usage = llm.chat("mid", messages, max_tokens=16)
    assert text.strip()
    assert usage.tokens_out > 0
    assert usage.latency_ms > 0
    assert cost(usage) >= 0.0
