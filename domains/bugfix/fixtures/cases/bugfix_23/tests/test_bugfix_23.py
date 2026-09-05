from parse_row import parse_row, parse_rows

COLUMNS = ["name", "city", "note"]


def test_full_row():
    assert parse_row("ada, london, hi", COLUMNS) == {
        "name": "ada",
        "city": "london",
        "note": "hi",
    }


def test_blank_field_is_none():
    assert parse_row("ada, , hi", COLUMNS)["city"] is None


def test_short_row_is_padded():
    assert parse_row("ada", COLUMNS) == {"name": "ada", "city": None, "note": None}


def test_parse_rows():
    assert parse_rows(["a,b,c", "d"], COLUMNS)[1]["note"] is None
