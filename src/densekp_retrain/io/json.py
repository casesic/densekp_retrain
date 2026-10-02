"""Strict JSON report writers; preserve the two historical Unicode policies."""

import json


def write_json(path, data, *, ensure_ascii=False):
    """Write ordered, indented JSON plus newline; reject NaN/Inf, never filter it."""
    path.write_text(json.dumps(data, ensure_ascii=ensure_ascii, indent=2, allow_nan=False) + "\n")


def write_json_ascii(path, data):
    """Historical training-report policy: escape non-ASCII characters."""
    write_json(path, data, ensure_ascii=True)
