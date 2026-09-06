"""Iterative binary search over a sorted list."""


def search(values, target):
    """Return the index of `target` in the sorted list `values`, or -1 if absent."""
    low = 0
    high = len(values) - 1
    while low <= high:
        mid = (low + high) // 2
        found = values[mid]
        if found == target:
            return mid
        if found < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1


def contains(values, target):
    """True if `target` is in the sorted list `values`."""
    return search(values, target) != -1
