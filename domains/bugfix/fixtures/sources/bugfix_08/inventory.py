"""Restock decisions for a warehouse."""


def needs_restock(quantity, threshold):
    """True when stock has fallen to or below the reorder threshold.

    Hitting the threshold exactly already means reorder.
    """
    if quantity < 0:
        raise ValueError("quantity cannot be negative")
    return quantity <= threshold


def restock_list(items):
    """Names of the items that need reordering, in input order.

    `items` is a list of {"name": str, "quantity": int, "threshold": int}.
    """
    names = []
    for item in items:
        if needs_restock(item["quantity"], item["threshold"]):
            names.append(item["name"])
    return names
