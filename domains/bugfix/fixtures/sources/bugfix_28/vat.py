"""VAT amounts on invoice totals."""

RATE = 0.2


def vat_amount(net):
    """Return the VAT due on `net`, rounded to the nearest cent as a float.

    Callers add this to the net total, so fractional cents matter: the result must
    keep its decimals rather than being cut down to whole units.
    """
    if net < 0:
        raise ValueError("net cannot be negative")
    return round(net * RATE, 2)


def gross(net):
    """The net plus its VAT, rounded to the nearest cent."""
    return round(net + vat_amount(net), 2)


def vat_breakdown(nets):
    """A list of {"net", "vat", "gross"} rows for each net amount."""
    return [{"net": net, "vat": vat_amount(net), "gross": gross(net)} for net in nets]
