from peak import peak_index, peak_value


def test_peak_of_mixed_readings():
    assert peak_value([3, 9, 4, 1]) == 9


def test_peak_of_a_single_reading():
    assert peak_value([7]) == 7


def test_peak_of_nothing():
    assert peak_value([]) is None


def test_negative_readings():
    assert peak_value([-8, -2, -5]) == -2


def test_peak_index():
    assert peak_index([3, 9, 4, 9]) == 1
