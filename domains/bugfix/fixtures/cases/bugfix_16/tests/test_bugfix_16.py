from config_merge import DEFAULTS, changed_keys, effective_timeout, with_defaults


def test_defaults_are_applied():
    assert with_defaults({}) == {"retries": 3, "timeout": 30}


def test_override_wins():
    assert with_defaults({"timeout": 5})["timeout"] == 5


def test_defaults_are_not_polluted_between_calls():
    with_defaults({"timeout": 5})
    assert with_defaults({}) == {"retries": 3, "timeout": 30}
    assert DEFAULTS == {"retries": 3, "timeout": 30}


def test_effective_timeout():
    assert effective_timeout({"timeout": 9}) == 9


def test_changed_keys():
    assert changed_keys({"timeout": 5}) == ["timeout"]
