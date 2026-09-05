"""Tests for anneal.prompts: local versioning round-trip, offline Neatlogs gating."""

from __future__ import annotations

from pathlib import Path

import pytest

from anneal import prompts


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NEATLOGS_API_KEY", raising=False)


def test_parse_ref_round_trip() -> None:
    assert prompts.parse_ref("anneal/airline/executor@v3") == ("anneal/airline/executor", 3)
    assert prompts.make_ref("x/y", 7) == "x/y@v7"


@pytest.mark.parametrize("bad", ["noversion", "a@1", "a@vx", "a b@v1", ""])
def test_parse_ref_rejects_malformed(bad: str) -> None:
    with pytest.raises(ValueError):
        prompts.parse_ref(bad)


def test_save_and_get_round_trip(tmp_path: Path) -> None:
    ref = prompts.save_version("anneal/demo/executor", "be helpful", root=tmp_path)
    assert ref == "anneal/demo/executor@v1"
    assert (tmp_path / "anneal/demo/executor/v1.md").is_file()
    assert prompts.get_prompt(ref, root=tmp_path) == "be helpful"


def test_versions_increment(tmp_path: Path) -> None:
    prompts.save_version("n", "one", root=tmp_path)
    ref2 = prompts.save_version("n", "two", root=tmp_path)
    assert ref2 == "n@v2"
    assert prompts.latest_version("n", root=tmp_path) == 2
    assert prompts.get_prompt("n@v1", root=tmp_path) == "one"
    assert prompts.get_prompt("n@v2", root=tmp_path) == "two"


def test_get_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        prompts.get_prompt("ghost@v1", root=tmp_path)
    assert prompts.latest_version("ghost", root=tmp_path) == 0


def test_no_neatlogs_call_without_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import neatlogs

    def boom(*a: object, **k: object) -> None:
        raise AssertionError("neatlogs must not be called offline")

    monkeypatch.setattr(neatlogs, "get_prompt", boom)
    monkeypatch.setattr(neatlogs, "create_prompt", boom)
    monkeypatch.setattr(neatlogs, "save_as_version", boom)
    assert prompts.save_version("n", "t", root=tmp_path) == "n@v1"
    prompts.set_label("n@v1", "production")


def test_sync_failure_is_swallowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import neatlogs

    monkeypatch.setenv("NEATLOGS_API_KEY", "nl-test")
    calls: list[str] = []

    def get_prompt(name: str, **k: object) -> None:
        raise neatlogs.PromptNotFoundError(name)

    def create_prompt(**k: object) -> None:
        calls.append("create")
        raise RuntimeError("network down")

    monkeypatch.setattr(neatlogs, "get_prompt", get_prompt)
    monkeypatch.setattr(neatlogs, "create_prompt", create_prompt)
    with caplog.at_level("WARNING", logger="anneal.prompts"):
        assert prompts.save_version("n", "t", root=tmp_path) == "n@v1"
    assert calls == ["create"]
    assert any("sync failed" in r.getMessage() for r in caplog.records)
    assert prompts.get_prompt("n@v1", root=tmp_path) == "t"
