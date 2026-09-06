"""Read settings with per-key fallbacks."""

FALLBACKS = {"retries": 3, "timeout": 30, "region": "eu"}


def setting(settings, key):
    """Return `settings[key]`, falling back to the standard default for that key.

    Unknown keys with no fallback come back as None.
    """
    if not isinstance(settings, dict):
        raise TypeError("settings must be a dict")
    default = FALLBACKS.get(key)
    return settings.get(key, default)


def settings_view(settings):
    """Every known key resolved against `settings`, as a dict."""
    return {key: setting(settings, key) for key in FALLBACKS}


def is_default(settings, key):
    """True when `key` is not overridden in `settings`."""
    return key not in settings
