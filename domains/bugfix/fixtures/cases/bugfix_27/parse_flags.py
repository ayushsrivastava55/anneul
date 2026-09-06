"""Split a command line into flags and positional arguments."""


def parse_flags(argv):
    """Return (flags, positionals) for `argv`.

    Flags are the arguments starting with "-", with the dashes stripped;
    positionals are everything else. The return value is a tuple of two lists.
    """
    flags = []
    positionals = []
    for arg in argv:
        if arg.startswith("-") and len(arg) > 1:
            flags.append(arg.lstrip("-"))
        else:
            positionals.append(arg)
    return [flags, positionals]


def has_flag(argv, name):
    """True when `name` appears as a flag in `argv`."""
    flags, _ = parse_flags(argv)
    return name in flags
