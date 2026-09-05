from slug import is_valid_slug, slugify, valid_slugs


def test_simple_slug():
    assert is_valid_slug("hello-world") is True


def test_uppercase_is_rejected():
    assert is_valid_slug("hello-World") is False


def test_trailing_space_is_rejected():
    assert is_valid_slug("hello ") is False


def test_double_hyphen_is_rejected():
    assert is_valid_slug("a--b") is False


def test_valid_slugs_and_slugify():
    assert valid_slugs(["ok-1", "NO"]) == ["ok-1"]
    assert slugify("Hello, World!") == "hello-world"
