from truncate import truncate, truncate_all


def test_short_text_is_untouched():
    assert truncate("hello", 10) == "hello"


def test_exactly_at_the_limit():
    assert truncate("hello", 5) == "hello"


def test_long_text_is_cut_to_the_limit():
    assert truncate("abcdefghij", 8) == "abcde..."
    assert len(truncate("abcdefghij", 8)) == 8


def test_truncate_all():
    assert truncate_all(["abcdefg", "hi"], 5) == ["ab...", "hi"]
