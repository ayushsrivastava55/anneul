from clamp_report import clamp, clamp_all, clamp_reading


def test_value_inside_the_band():
    assert clamp_reading(5, (0, 10)) == 5


def test_value_below_the_band():
    assert clamp_reading(-3, (0, 10)) == 0


def test_value_above_the_band():
    assert clamp_reading(42, (0, 10)) == 10


def test_clamp_all():
    assert clamp_all([-1, 5, 11], (0, 10)) == [0, 5, 10]


def test_clamp_directly():
    assert clamp(7, 0, 5) == 5
