from settings_lookup import is_default, setting, settings_view


def test_override_wins():
    assert setting({"retries": 9}, "retries") == 9


def test_fallback_is_used():
    assert setting({}, "timeout") == 30


def test_unknown_key_is_none():
    assert setting({}, "nope") is None


def test_settings_view():
    assert settings_view({"region": "us"}) == {"retries": 3, "timeout": 30, "region": "us"}


def test_is_default():
    assert is_default({}, "timeout") is True
