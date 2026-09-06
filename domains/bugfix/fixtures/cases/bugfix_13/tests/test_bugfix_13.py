from cart import add_item, cart_total


def test_first_cart():
    assert add_item("apple", 2.0) == [{"name": "apple", "price": 2.0}]


def test_two_default_calls_do_not_share_state():
    first = add_item("apple", 2.0)
    second = add_item("pear", 3.0)
    assert second == [{"name": "pear", "price": 3.0}]
    assert len(first) == 1


def test_explicit_cart_is_extended():
    cart = [{"name": "apple", "price": 2.0}]
    assert len(add_item("pear", 3.0, cart)) == 2


def test_cart_total():
    assert cart_total(add_item("apple", 2.5)) == 2.5
