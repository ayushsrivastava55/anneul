"""A tiny shopping cart helper."""


def add_item(name, price, cart=None):
    """Return a cart with {"name", "price"} appended.

    Calling this without a cart starts a brand new one every time: two independent
    calls must not share a list.
    """
    if cart is None:
        cart = []
    if price < 0:
        raise ValueError("price cannot be negative")
    cart.append({"name": name, "price": price})
    return cart


def cart_total(cart):
    """Sum of the prices in `cart`."""
    return sum(entry["price"] for entry in cart)
