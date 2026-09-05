from safe_divide import percent, ratios, safe_divide


def test_ordinary_division():
    assert safe_divide(10, 4) == 2.5


def test_zero_denominator_is_none():
    assert safe_divide(10, 0) is None


def test_zero_numerator():
    assert safe_divide(0, 4) == 0.0


def test_ratios_with_a_zero():
    assert ratios([(4, 2), (1, 0)]) == [2.0, None]


def test_percent_of_nothing():
    assert percent(3, 0) is None
