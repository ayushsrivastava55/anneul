"""Merge user settings over a set of defaults."""

DEFAULTS = {"retries": 3, "timeout": 30}


def with_defaults(overrides, base=None):
    """Apply `overrides` on top of `base` and return the resulting settings dict.

    Keys whose override value is None are ignored. When `base` is omitted the call
    works on a fresh copy of the standard DEFAULTS, so repeated calls never see each
    other's overrides and the module-level DEFAULTS are never modified.
    """
    if base is None:
        base = dict(DEFAULTS)
    for key, value in overrides.items():
        if value is not None:
            base[key] = value
    return base


def effective_timeout(overrides):
    """The timeout that would be used given `overrides`."""
    return with_defaults(overrides)["timeout"]


def changed_keys(overrides):
    """The keys `overrides` would actually change, sorted."""
    settings = with_defaults(overrides)
    return sorted(key for key in settings if settings[key] != DEFAULTS.get(key))
