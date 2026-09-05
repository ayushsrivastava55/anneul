from password_strength import is_strong, weak_reasons


def test_all_rules_met():
    assert is_strong("Longenough1") is True


def test_missing_digit_is_not_strong():
    assert is_strong("Longenoughx") is False


def test_missing_uppercase_is_not_strong():
    assert is_strong("longenough1") is False


def test_too_short_is_not_strong():
    assert is_strong("Short1") is False


def test_weak_reasons():
    assert weak_reasons("short") == ["digit", "length", "uppercase"]
