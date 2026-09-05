from phone import digits_only, is_valid_phone, valid_phones


def test_canonical_number():
    assert is_valid_phone("(555) 123-4567") is True


def test_trailing_junk_is_rejected():
    assert is_valid_phone("(555) 123-4567 ext 9") is False


def test_wrong_shape_is_rejected():
    assert is_valid_phone("555-123-4567") is False


def test_valid_phones():
    assert valid_phones(["(555) 123-4567", "nope"]) == ["(555) 123-4567"]


def test_digits_only():
    assert digits_only("(555) 123-4567") == "5551234567"
