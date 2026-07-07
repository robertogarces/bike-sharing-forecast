from pathlib import Path

import pandas as pd


def append_monitoring_record(summary: dict, history_path: Path) -> None:
    """
    Append a monitoring summary record to a history CSV.

    Shared by performance_monitoring.py, output_drift_detection.py, and
    drift_detection.py. Unlike append_prediction (predict.py) — one record
    per hour, deduped by timestamp — each monitoring run is its own
    observation: even a rerun in the same window is a legitimate, distinct
    measurement. Records are always appended, never skipped.

    Reads the existing file and rewrites it with the new row concatenated,
    rather than a blind mode="a" append — pd.concat aligns columns by name
    and fills NaN for any column missing on either side, so the schema can
    evolve (e.g. a new column added later) without corrupting older rows.
    """
    df_new = pd.DataFrame([summary])

    if history_path.exists():
        existing = pd.read_csv(history_path)
        updated = pd.concat([existing, df_new], ignore_index=True)
    else:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        updated = df_new

    updated.to_csv(history_path, index=False)
