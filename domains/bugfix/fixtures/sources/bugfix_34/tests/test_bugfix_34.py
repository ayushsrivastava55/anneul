from paths import basename, storage_path, storage_paths


def test_simple_join():
    assert storage_path("uploads", "a.txt") == "uploads/a.txt"


def test_trailing_slash_is_not_doubled():
    assert storage_path("uploads/", "a.txt") == "uploads/a.txt"


def test_nested_base():
    assert storage_path("uploads/2024", "a.txt") == "uploads/2024/a.txt"


def test_storage_paths():
    assert storage_paths("u", ["a", "b"]) == ["u/a", "u/b"]


def test_basename():
    assert basename("uploads/a.txt") == "a.txt"
