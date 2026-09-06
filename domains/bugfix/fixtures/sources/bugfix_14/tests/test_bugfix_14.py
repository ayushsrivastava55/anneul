from tags import collect_tags, tag_list


def test_flat_record():
    assert collect_tags({"tags": ["a", "b"]}) == {"a", "b"}


def test_nested_children():
    record = {"tags": ["a"], "children": [{"tags": ["b"]}]}
    assert collect_tags(record) == {"a", "b"}


def test_separate_calls_do_not_accumulate():
    collect_tags({"tags": ["a"]})
    assert collect_tags({"tags": ["b"]}) == {"b"}


def test_tag_list_is_sorted():
    assert tag_list({"tags": ["c", "a"]}) == ["a", "c"]
