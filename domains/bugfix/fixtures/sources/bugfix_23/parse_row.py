"""Parse one comma-separated row into typed fields."""


def parse_row(row, columns):
    """Return {column: value} for a comma-separated `row`.

    Blank fields become None. A row with fewer fields than columns pads the missing
    trailing columns with None rather than failing.
    """
    fields = [field.strip() for field in row.split(",")]
    if len(fields) < len(columns):
        fields = fields + [""] * (len(columns) - len(fields))
    parsed = {}
    for column, field in zip(columns, fields, strict=False):
        parsed[column] = field if field else None
    return parsed


def parse_rows(rows, columns):
    """parse_row over a list of rows."""
    return [parse_row(row, columns) for row in rows]
