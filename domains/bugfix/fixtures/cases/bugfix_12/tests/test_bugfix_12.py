from leap_year import days_in_year, is_leap_year


def test_ordinary_leap_year():
    assert is_leap_year(2024) is True


def test_ordinary_common_year():
    assert is_leap_year(2023) is False


def test_century_divisible_by_400_is_a_leap_year():
    assert is_leap_year(2000) is True


def test_century_not_divisible_by_400_is_common():
    assert is_leap_year(1900) is False


def test_days_in_year():
    assert days_in_year(2000) == 366
    assert days_in_year(1900) == 365
