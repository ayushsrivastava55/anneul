"""Smoke test: the anneal package imports and declares a version."""

import anneal


def test_package_exposes_version() -> None:
    assert isinstance(anneal.__version__, str)
    assert anneal.__version__
