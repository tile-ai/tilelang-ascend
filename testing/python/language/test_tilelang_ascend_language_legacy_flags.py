"""Regression tests for removal of legacy string-based flag helpers."""

import tilelang.language as T


def test_legacy_flag_helpers_are_not_exported():
    assert not hasattr(T, "init_flag")
    assert not hasattr(T, "clear_flag")
