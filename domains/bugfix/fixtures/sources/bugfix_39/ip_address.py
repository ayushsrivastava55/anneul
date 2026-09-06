"""Spot IPv4 addresses in log lines."""

import re

IP_RE = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")


def find_ips(line):
    """Return every IPv4-shaped address in `line`, in order.

    The dots are literal dots: 1x2x3x4 is not an address.
    """
    return IP_RE.findall(line)


def first_ip(line):
    """The first address in `line`, or None when there is none."""
    found = find_ips(line)
    return found[0] if found else None


def is_private(address):
    """True for addresses in the 10.x and 192.168.x ranges."""
    return address.startswith("10.") or address.startswith("192.168.")
