from hashtags import hashtags, unique_hashtags


def test_simple_tags():
    assert hashtags("love #python and #rust") == ["python", "rust"]


def test_bare_hash_is_not_a_tag():
    assert hashtags("a # b #real") == ["real"]


def test_no_tags():
    assert hashtags("nothing") == []


def test_unique_hashtags():
    assert unique_hashtags("#Py #py #rust") == ["py", "rust"]
