from pathlib import Path

import pandas as pd


def append_monitoring_record(summary: dict, history_path: Path) -> None:
    """
    Append a monitoring summary record to a history CSV.

    Shared by performance_monitoring.py and output_drift_detection.py.
    Unlike append_prediction (predict.py) — one record per hour, deduped by
    timestamp — each monitoring run is its own observation: even a rerun in
    the same window is a legitimate, distinct measurement. Records are
    always appended, never skipped.
    """
    df_new = pd.DataFrame([summary])

    if history_path.exists():
        df_new.to_csv(history_path, mode="a", header=False, index=False)
    else:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        df_new.to_csv(history_path, index=False)
