"""config.env returns the environment value or the supplied default."""

from anneal import config


def test_env_returns_default_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("ANNEAL_TEST_MISSING", raising=False)
    assert config.env("ANNEAL_TEST_MISSING", "fallback") == "fallback"


def test_env_returns_value_when_set(monkeypatch) -> None:
    monkeypatch.setenv("ANNEAL_TEST_SET", "present")
    assert config.env("ANNEAL_TEST_SET", "fallback") == "present"
