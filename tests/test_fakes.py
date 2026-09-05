"""Tests for tests.fakes.FakeClient and anneal.llm.complete() driven by it (offline)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from anneal.llm import Usage, chat, complete, cost
from tests.fakes import FakeClient

FIXTURE_YAML = """
tiers:
  mid: {provider: tensormux, model: mid-model, price_in: 1.0, price_out: 2.0}
downshift_order: [mid]
providers:
  tensormux: {base_url_env: T_BASE, api_key_env: T_KEY}
"""


@pytest.fixture
def models_path(tmp_path: Path) -> Path:
    path = tmp_path / "models.yaml"
    path.write_text(FIXTURE_YAML)
    return path


def test_fake_client_text_turn_via_plain_create() -> None:
    client = FakeClient(turns=["hi there"])
    response = client.chat.completions.create(
        model="m", messages=[{"role": "user", "content": "x"}]
    )
    assert response.choices[0].message.content == "hi there"
    assert response.choices[0].message.tool_calls is None
    assert response.usage.prompt_tokens > 0 and response.usage.completion_tokens > 0
    assert client.calls[0]["model"] == "m"


def test_fake_client_raw_response_has_backend_header() -> None:
    client = FakeClient(turns=["ok"])
    raw = client.chat.completions.with_raw_response.create(model="m", messages=[])
    assert raw.headers["x-tensormux-backend"] == "fake-backend"
    assert raw.parse().choices[0].message.content == "ok"


def test_fake_client_tool_call_turn() -> None:
    client = FakeClient(turns=[[{"name": "lookup", "arguments": '{"id": 7}'}]])
    response = client.chat.completions.create(model="m", messages=[])
    call = response.choices[0].message.tool_calls[0]
    assert call.function.name == "lookup"
    assert json.loads(call.function.arguments) == {"id": 7}
    assert response.choices[0].finish_reason == "tool_calls"


def test_fake_client_dict_turn_overrides_usage_and_backend() -> None:
    client = FakeClient(turns=[{"content": "x", "tokens_in": 3, "tokens_out": 4, "backend": "b2"}])
    raw = client.chat.completions.with_raw_response.create(model="m", messages=[])
    assert raw.headers["x-tensormux-backend"] == "b2"
    assert (raw.parse().usage.prompt_tokens, raw.parse().usage.completion_tokens) == (3, 4)


def test_fake_client_exhausted_turns_raise() -> None:
    client = FakeClient(turns=["only one"])
    client.chat.completions.create(model="m", messages=[])
    with pytest.raises(RuntimeError, match="no scripted turns"):
        client.chat.completions.create(model="m", messages=[])


def test_complete_returns_plain_message_with_tool_calls(models_path: Path) -> None:
    client = FakeClient(turns=[[{"name": "lookup", "arguments": '{"id": 7}'}]])
    message, usage = complete(
        "mid", [{"role": "user", "content": "x"}], tools=[], client=client, path=models_path
    )
    assert isinstance(message, dict)
    assert message["role"] == "assistant"
    assert message["tool_calls"][0]["function"] == {"name": "lookup", "arguments": '{"id": 7}'}
    assert message["tool_calls"][0]["type"] == "function"
    assert isinstance(usage, Usage)
    assert usage.model == "mid-model" and usage.backend == "fake-backend"
    assert "tools" in client.calls[0]


def test_complete_text_turn_and_cost(models_path: Path) -> None:
    client = FakeClient(turns=[{"content": "hello", "tokens_in": 1_000_000, "tokens_out": 500_000}])
    message, usage = complete("mid", [], client=client, path=models_path)
    assert message["content"] == "hello"
    assert message.get("tool_calls") is None
    assert cost(usage, models_path) == pytest.approx(2.0)
    assert usage.latency_ms >= 0


def test_chat_shares_path_with_complete(models_path: Path) -> None:
    client = FakeClient(turns=["plain text"])
    text, usage = chat("mid", [], client=client, path=models_path)
    assert text == "plain text"
    assert usage.tokens_out > 0
