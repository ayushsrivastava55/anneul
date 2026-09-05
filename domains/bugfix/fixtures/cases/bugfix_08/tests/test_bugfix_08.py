from inventory import needs_restock, restock_list


def test_below_threshold():
    assert needs_restock(2, 5) is True


def test_exactly_at_threshold_still_reorders():
    assert needs_restock(5, 5) is True


def test_above_threshold():
    assert needs_restock(6, 5) is False


def test_restock_list():
    items = [
        {"name": "bolts", "quantity": 5, "threshold": 5},
        {"name": "nuts", "quantity": 9, "threshold": 5},
    ]
    assert restock_list(items) == ["bolts"]
