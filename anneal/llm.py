"""LLM gateway: every model call in Anneal goes through here.

Model ids and prices live only in ``specs/models.yaml``. Providers are resolved to
``base_url``/``api_key`` via environment variables named in that file (loaded from ``.env``).
The ``x-tensormux-backend`` response header, when present, attributes cost to the backend
that actually served the request.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from openai import OpenAI

try:  # anneal.tracing is task 0.4; until it lands, spans are a no-op seam.
    from anneal.tracing import span
except ImportError:  # pragma: no cover - exercised only before tracing merges

    def span(name: str, **tags: Any) -> contextlib.AbstractContextManager[None]:
        return contextlib.nullcontext()


load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
MODELS_PATH = ROOT / "specs" / "models.yaml"
BACKEND_HEADER = "x-tensormux-backend"


@dataclass(frozen=True)
class Usage:
    """Token accounting for one chat call."""

    tokens_in: int
    tokens_out: int
    backend: str | None
    model: str
    latency_ms: float = 0.0


@lru_cache(maxsize=8)
def load_models(path: Path | str | None = None) -> dict[str, Any]:
    """Parse specs/models.yaml (or ``path``). Cached per path."""
    with open(Path(path) if path else MODELS_PATH, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict) or "tiers" not in data or "providers" not in data:
        raise ValueError(f"{path or MODELS_PATH}: expected top-level 'tiers' and 'providers'")
    return data


def _tier(tier: str, path: Path | str | None = None) -> dict[str, Any]:
    tiers = load_models(path)["tiers"]
    if tier not in tiers:
        raise KeyError(f"unknown model tier {tier!r}; known: {sorted(tiers)}")
    return tiers[tier]


def resolve_model(tier: str, path: Path | str | None = None) -> str:
    """Return the model id configured for ``tier``."""
    return str(_tier(tier, path)["model"])


def _env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"environment variable {name} is unset or empty (see .env.example)")
    return value


def get_client(tier: str = "mid", path: Path | str | None = None) -> OpenAI:
    """OpenAI-compatible client for the provider backing ``tier`` (TensorMux by default)."""
    provider_name = _tier(tier, path)["provider"]
    providers = load_models(path)["providers"]
    if provider_name not in providers:
        raise KeyError(f"tier {tier!r} names unknown provider {provider_name!r}")
    provider = providers[provider_name]
    return OpenAI(base_url=_env(provider["base_url_env"]), api_key=_env(provider["api_key_env"]))


def _price_table(path: Path | str | None = None) -> dict[str, tuple[float, float]]:
    """model id -> (price_in, price_out) USD per 1M tokens; first tier wins on duplicates."""
    table: dict[str, tuple[float, float]] = {}
    for spec in load_models(path)["tiers"].values():
        table.setdefault(str(spec["model"]), (float(spec["price_in"]), float(spec["price_out"])))
    return table


def cost(usage: Usage, path: Path | str | None = None) -> float:
    """USD cost of ``usage``, priced by backend if it is a known model, else by requested model."""
    table = _price_table(path)
    key = usage.backend if usage.backend in table else usage.model
    if key not in table:
        raise KeyError(f"no price for model {key!r} in models.yaml")
    price_in, price_out = table[key]
    return usage.tokens_in / 1e6 * price_in + usage.tokens_out / 1e6 * price_out


def chat(tier: str, messages: list[dict[str, Any]], **kw: Any) -> tuple[str, Usage]:
    """One chat completion via the gateway. Returns (text, Usage) inside an ``llm.chat`` span."""
    model = resolve_model(tier)
    client = get_client(tier)
    with span("llm.chat", tier=tier, model=model):
        start = time.perf_counter()
        raw = client.chat.completions.with_raw_response.create(
            model=model, messages=messages, **kw
        )
        latency_ms = (time.perf_counter() - start) * 1000
    response = raw.parse()
    usage = Usage(
        tokens_in=response.usage.prompt_tokens if response.usage else 0,
        tokens_out=response.usage.completion_tokens if response.usage else 0,
        backend=raw.headers.get(BACKEND_HEADER),
        model=model,
        latency_ms=round(latency_ms, 1),
    )
    return response.choices[0].message.content or "", usage


def main(argv: list[str] | None = None) -> int:
    """CLI smoke: ``python -m anneal.llm --tier mid "hello"``."""
    parser = argparse.ArgumentParser(description="Send one prompt through the Anneal LLM gateway.")
    parser.add_argument("prompt")
    parser.add_argument("--tier", default="mid")
    args = parser.parse_args(argv)
    try:
        text, usage = chat(args.tier, [{"role": "user", "content": args.prompt}])
    except RuntimeError as exc:
        print(f"ready pending key: {exc}", file=sys.stderr)
        return 2
    print(text)
    print(json.dumps({**asdict(usage), "cost_usd": cost(usage)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
