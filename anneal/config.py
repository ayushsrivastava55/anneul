"""Environment configuration. Loads `.env` once; never prints values."""

from __future__ import annotations

import os

from dotenv import load_dotenv

# Real environment wins over .env; a missing .env is a no-op.
load_dotenv(override=False)


def env(name: str, default: str | None = None) -> str | None:
    """Return the environment variable `name`, or `default` when unset."""
    return os.environ.get(name, default)
