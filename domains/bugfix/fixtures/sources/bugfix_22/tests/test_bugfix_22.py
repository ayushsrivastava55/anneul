from title_case import title_case, title_case_all


def test_simple_phrase():
    assert title_case("the lord of the rings") == "The Lord of the Rings"


def test_double_space_is_preserved():
    assert title_case("hello  world") == "Hello  World"


def test_empty_phrase():
    assert title_case("") == ""


def test_title_case_all():
    assert title_case_all(["a b", ""]) == ["A B", ""]
