"""config.env returns the environment value or the supplied default."""

from anneal import config


def test_env_returns_default_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("ANNEAL_TEST_MISSING", raising=False)
    assert config.env("ANNEAL_TEST_MISSING", "fallback") == "fallback"


def test_env_returns_value_when_set(monkeypatch) -> None:
    monkeypatch.setenv("ANNEAL_TEST_SET", "present")
    assert config.env("ANNEAL_TEST_SET", "fallback") == "present"


def test_blank_env_counts_as_unset(monkeypatch) -> None:
    """An exported-but-empty var must not read as a value. See config._load."""
    monkeypatch.setenv("ANNEAL_TEST_BLANK", "")
    assert config.env("ANNEAL_TEST_BLANK", "fallback") == "fallback"
    monkeypatch.setenv("ANNEAL_TEST_BLANK", "   ")
    assert config.env("ANNEAL_TEST_BLANK", "fallback") == "fallback"


def test_load_lets_dotenv_win_over_a_blank_exported_var(monkeypatch, tmp_path) -> None:
    """The bug this guards: a blank export shadowed a filled-in .env, and the error said
    the var was unset while the file plainly set it."""
    dotenv = tmp_path / ".env"
    dotenv.write_text("ANNEAL_TEST_SHADOW=from_file\nANNEAL_TEST_REAL=from_file\n")
    monkeypatch.setenv("ANNEAL_TEST_SHADOW", "")  # blank export: must lose
    monkeypatch.setenv("ANNEAL_TEST_REAL", "from_shell")  # real export: must win
    monkeypatch.chdir(tmp_path)
    config._load()
    assert config.env("ANNEAL_TEST_SHADOW") == "from_file"
    assert config.env("ANNEAL_TEST_REAL") == "from_shell"
