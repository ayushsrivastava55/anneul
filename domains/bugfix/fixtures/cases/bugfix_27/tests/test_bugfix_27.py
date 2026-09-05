from parse_flags import has_flag, parse_flags


def test_returns_a_tuple_of_two_lists():
    result = parse_flags(["-v", "file.txt"])
    assert isinstance(result, tuple)
    assert result == (["v"], ["file.txt"])


def test_long_flags():
    assert parse_flags(["--verbose"]) == (["verbose"], [])


def test_lone_dash_is_positional():
    assert parse_flags(["-"]) == ([], ["-"])


def test_has_flag():
    assert has_flag(["--verbose", "x"], "verbose") is True
