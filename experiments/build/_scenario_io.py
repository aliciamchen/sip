"""Shared writer for the scenario source files.

Each `scenarios*.py` defines its `COLUMNS` and `rows` (the scenario data, the
source of truth) and calls `write_scenarios_csv(...)` to validate and emit the
generated CSV — so the validation + DictWriter boilerplate lives in one place
rather than being copy-pasted into every scenario file.
"""

import csv


def write_scenarios_csv(rows, columns, out_path, *, expected_rows=None):
    """Validate `rows` and write them to `out_path` as a CSV.

    Asserts `scenario_label` uniqueness (and the exact row count when
    `expected_rows` is given), then writes `columns` in order with minimal
    quoting. Returns `out_path`.
    """
    if expected_rows is not None and len(rows) != expected_rows:
        raise AssertionError(f"expected {expected_rows} rows, got {len(rows)}")
    labels = [r["scenario_label"] for r in rows]
    if len(set(labels)) != len(labels):
        raise AssertionError(f"duplicate scenario_label: {labels}")

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_path}")
    return out_path
