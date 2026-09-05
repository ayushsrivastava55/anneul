from strip_prefix import common_prefix_count, strip_all, strip_prefix


def test_prefix_is_removed():
    assert strip_prefix("env_HOME", "env_") == "HOME"


def test_name_without_the_prefix_is_unchanged():
    assert strip_prefix("HOME", "env_") == "HOME"


def test_empty_prefix_is_a_no_op():
    assert strip_prefix("HOME", "") == "HOME"


def test_strip_all_mixed():
    assert strip_all(["env_A", "B"], "env_") == ["A", "B"]


def test_common_prefix_count():
    assert common_prefix_count(["env_A", "B"], "env_") == 1
