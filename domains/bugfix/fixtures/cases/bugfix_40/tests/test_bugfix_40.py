from split_csv_line import field_count, split_fields


def test_spaces_around_commas_are_removed():
    assert split_fields("a , b ,c") == ["a", "b", "c"]


def test_no_spaces():
    assert split_fields("a,b") == ["a", "b"]


def test_empty_line():
    assert split_fields("   ") == []


def test_field_count():
    assert field_count("a , b") == 2
