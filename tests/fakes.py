"""Offline stand-ins for the OpenAI client returned by ``anneal.llm.get_client``.

``FakeClient(turns=[...])`` replays one scripted assistant turn per request. Each turn is:

- a ``str`` -> plain text reply;
- a ``list`` of ``{"name": ..., "arguments": <JSON string>}`` -> tool calls;
- a ``dict`` with optional ``content``, ``tool_calls``, ``tokens_in``, ``tokens_out``,
  ``backend`` for full control.

Responses are real ``openai.types.chat`` objects, so ``model_dump`` and attribute access
behave exactly as they do against the live gateway. Both ``chat.completions.create`` and
``chat.completions.with_raw_response.create`` are supported; the raw form exposes a
``headers`` dict carrying ``x-tensormux-backend``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageFunctionToolCall,
    Function,
)
from openai.types.completion_usage import CompletionUsage

Turn = str | list[dict[str, str]] | dict[str, Any]

DEFAULT_TOKENS_IN = 10
DEFAULT_TOKENS_OUT = 5
DEFAULT_BACKEND = "fake-backend"


def _normalise(turn: Turn) -> dict[str, Any]:
    if isinstance(turn, str):
        return {"content": turn}
    if isinstance(turn, list):
        return {"tool_calls": turn}
    return dict(turn)


def _tool_calls(specs: list[dict[str, str]]) -> list[ChatCompletionMessageFunctionToolCall]:
    return [
        ChatCompletionMessageFunctionToolCall(
            id=f"call_{i}",
            type="function",
            function=Function(name=spec["name"], arguments=spec["arguments"]),
        )
        for i, spec in enumerate(specs)
    ]


def build_completion(turn: Turn, model: str) -> ChatCompletion:
    """Build a ChatCompletion for one scripted turn."""
    spec = _normalise(turn)
    calls = _tool_calls(spec["tool_calls"]) if spec.get("tool_calls") else None
    message = ChatCompletionMessage(role="assistant", content=spec.get("content"), tool_calls=calls)
    tokens_in = int(spec.get("tokens_in", DEFAULT_TOKENS_IN))
    tokens_out = int(spec.get("tokens_out", DEFAULT_TOKENS_OUT))
    return ChatCompletion(
        id="chatcmpl-fake",
        object="chat.completion",
        created=0,
        model=model,
        choices=[
            Choice(
                index=0,
                message=message,
                finish_reason="tool_calls" if calls else "stop",
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=tokens_in,
            completion_tokens=tokens_out,
            total_tokens=tokens_in + tokens_out,
        ),
    )


@dataclass
class FakeRawResponse:
    """Mimics ``openai`` ``LegacyAPIResponse``: ``headers`` mapping plus ``parse()``."""

    completion: ChatCompletion
    headers: dict[str, str]

    def parse(self) -> ChatCompletion:
        return self.completion


class _Completions:
    def __init__(self, client: FakeClient) -> None:
        self._client = client
        self.with_raw_response = _RawCompletions(client)

    def create(self, **kw: Any) -> ChatCompletion:
        return self._client._next(**kw).completion


class _RawCompletions:
    def __init__(self, client: FakeClient) -> None:
        self._client = client

    def create(self, **kw: Any) -> FakeRawResponse:
        return self._client._next(**kw)


class _Chat:
    def __init__(self, client: FakeClient) -> None:
        self.completions = _Completions(client)


@dataclass
class FakeClient:
    """Scripted OpenAI-compatible client. ``calls`` records every request's kwargs."""

    turns: list[Turn] = field(default_factory=list)
    backend: str = DEFAULT_BACKEND
    calls: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._queue = list(self.turns)
        self.chat = _Chat(self)

    def _next(self, **kw: Any) -> FakeRawResponse:
        if not self._queue:
            raise RuntimeError("FakeClient has no scripted turns left")
        turn = self._queue.pop(0)
        self.calls.append(kw)
        backend = _normalise(turn).get("backend", self.backend)
        completion = build_completion(turn, model=str(kw.get("model", "fake-model")))
        return FakeRawResponse(completion, {"x-tensormux-backend": backend})
