from logbook import append_entry, last_message, messages_since


def test_new_log():
    assert append_entry("started", 1) == [{"at": 1, "message": "started"}]


def test_two_logs_stay_separate():
    first = append_entry("started", 1)
    second = append_entry("stopped", 2)
    assert len(first) == 1
    assert len(second) == 1
    assert last_message(second) == "stopped"


def test_existing_log_is_extended():
    log = append_entry("started", 1)
    append_entry("stopped", 2, log)
    assert last_message(log) == "stopped"


def test_messages_since():
    log = append_entry("started", 1)
    append_entry("stopped", 5, log)
    assert messages_since(log, 5) == ["stopped"]
