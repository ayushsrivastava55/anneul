from to_float import to_float, to_floats


def test_plain_number_is_a_float():
    value = to_float("12")
    assert value == 12.0
    assert isinstance(value, float)


def test_thousands_separator():
    assert to_float("1,250.5") == 1250.5


def test_percent():
    assert to_float("50%") == 0.5


def test_junk_returns_the_default():
    assert to_float("n/a") == 0.0


def test_to_floats():
    assert to_floats(["1", "x"]) == [1.0, 0.0]
