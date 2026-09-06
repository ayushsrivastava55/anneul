from find_versions import find_versions, latest_version


def test_single_digit_version():
    assert find_versions("shipped v1.2.3 today") == [(1, 2, 3)]


def test_multi_digit_components():
    assert find_versions("v10.20.30") == [(10, 20, 30)]


def test_several_versions():
    assert find_versions("v1.0.0 then v1.0.11") == [(1, 0, 0), (1, 0, 11)]


def test_no_versions():
    assert find_versions("nothing here") == []


def test_latest_version():
    assert latest_version("v1.0.0 and v1.0.11") == (1, 0, 11)
