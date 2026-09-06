"""Shared pytest configuration.

Registers the ``integration`` marker used by tests that need a real external process
(currently ``tests/test_mcp.py``, which launches a third-party MCP server over stdio).
Those tests skip themselves when the binary is unavailable; the marker exists so a run can
also deselect them up front with ``-m "not integration"``.
"""

from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "integration: needs a real external process (skips when unavailable)"
    )
