from redact import MASK, contains_secret, redact, redact_all


def test_secret_is_masked():
    assert redact("token=hunter2 ok", "hunter2") == f"token={MASK} ok"


def test_every_occurrence_is_masked():
    assert redact("a a", "a") == f"{MASK} {MASK}"


def test_line_without_the_secret_is_unchanged():
    assert redact("nothing here", "hunter2") == "nothing here"


def test_empty_secret():
    assert redact("keep me", "") == "keep me"


def test_redact_all_and_contains():
    assert redact_all(["hunter2"], "hunter2") == [MASK]
    assert contains_secret("hunter2", "hunter2") is True
