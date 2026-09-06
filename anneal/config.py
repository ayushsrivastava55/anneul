"""Environment configuration. Loads `.env` once; never prints values."""

from __future__ import annotations

import os

from dotenv import find_dotenv, load_dotenv


def _load() -> None:
    """Load `.env`. A real exported value wins, but only if it is not blank.

    An exported-but-empty var must not shadow a filled-in .env. You get one from
    `set -a; source .env; set +a` against a half-filled .env, or from a stale shell
    profile, and the symptom is "TENSORMUX_BASE_URL is unset" printed while the file
    in front of you plainly sets it. So blanks are dropped from the environment first,
    and only for keys the .env actually defines -- a blank var we do not own may well
    be meaningful to something else.
    """
    path = find_dotenv(usecwd=True) or find_dotenv()
    if path:
        from dotenv import dotenv_values

        for key in dotenv_values(path):
            if key in os.environ and not os.environ[key].strip():
                del os.environ[key]
    load_dotenv(path or None, override=False)


_load()


def env(name: str, default: str | None = None) -> str | None:
    """Return the environment variable `name`, or `default` when unset or blank.

    Blank counts as unset. `load_dotenv(override=False)` lets the real environment win, so
    an exported-but-empty var (what you get from `set -a; source .env` against a half-filled
    .env, or a stale shell profile) would otherwise shadow a perfectly good value in .env and
    surface as "key is unset" while the file plainly shows it set. Nobody debugs that quickly.
    """
    value = os.environ.get(name)
    return default if value is None or not value.strip() else value
