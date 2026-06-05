"""
Initialization script — run ONCE before starting the simulation.

    python src/bike_sharing/data/shift_dates.py

This script is NOT part of the DVC pipeline. It generates the initial
hour_past.csv and hour_future.csv files that the pipeline depends on.
To reset the simulation, delete the simulation_state file and re-run.
"""

import json
import logging
from pathlib import Path
from datetime import datetime

import hydra
import pandas as pd
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def check_existing_state(state_path: Path) -> None:
    """
    Abort if a simulation state already exists.

    Prevents accidental overwrite of an ongoing simulation.
    To reset, delete the simulation state file explicitly.
    """
    if state_path.exists():
        with open(state_path) as f:
            state = json.load(f)
        raise RuntimeError(
            f"A simulation is already running (started {state['shift_applied_at']}, "
            f"reference date: {state['reference_date']}, "
            f"future: {state['future_start_date']} → {state['future_end_date']}).\n"
            f"To reset, delete {state_path} explicitly."
        )


def shift_dates(df: pd.DataFrame, reference_date: pd.Timestamp, future_pct: float) -> tuple[pd.DataFrame, dict]:
    """
    Shift all dates in the dataset so that the last `future_pct` of records
    start at `reference_date`.

    The temporal structure of the dataset is fully preserved — only the
    calendar labels change. The shift is computed as:

        shift = reference_date - date_of_first_future_record

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset with a `dteday` column.
    reference_date : pd.Timestamp
        The date from which the future window begins.
    future_pct : float
        Fraction of total records to leave as future (0 < future_pct < 1).

    Returns
    -------
    tuple[pd.DataFrame, dict]
        Shifted DataFrame and simulation state metadata.
    """
    n_future = int(len(df) * future_pct)
    n_past   = len(df) - n_future

    future_start_original = df.iloc[n_past]["dteday"]
    shift = reference_date - future_start_original

    df = df.copy()
    df["dteday"] = pd.to_datetime(df["dteday"]) + shift

    future_start = df.iloc[n_past]["dteday"]
    future_end   = df.iloc[-1]["dteday"]

    state = {
        "reference_date":    reference_date.strftime("%Y-%m-%d"),
        "future_pct":        future_pct,
        "shift_applied_at":  datetime.now().isoformat(),
        "future_start_date": future_start.strftime("%Y-%m-%d"),
        "future_end_date":   future_end.strftime("%Y-%m-%d"),
        "n_future_records":  n_future,
        "n_past_records":    n_past,
    }

    return df, state


@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    raw_dir    = Path(cfg.paths.raw_dir)
    state_path = Path(cfg.paths.simulation_state)

    # ── Safety check ──────────────────────────────────────────────────────────
    check_existing_state(state_path)

    # ── Load ──────────────────────────────────────────────────────────────────
    logger.info("Loading raw dataset")
    df = pd.read_csv(raw_dir / "hour.csv")
    df["dteday"] = pd.to_datetime(df["dteday"])

    reference_date = pd.Timestamp(cfg.simulation.reference_date)
    future_pct     = cfg.simulation.future_pct

    logger.info(
        f"Shifting dates — reference: {reference_date.date()} | "
        f"future: {future_pct:.0%} ({int(len(df) * future_pct):,} records)"
    )

    # ── Shift ─────────────────────────────────────────────────────────────────
    df_shifted, state = shift_dates(df, reference_date, future_pct)

    logger.info(
        f"Future window: {state['future_start_date']} → {state['future_end_date']}"
    )

    # ── Save shifted dataset ──────────────────────────────────────────────────
    output_path = raw_dir / "hour_shifted.csv"
    df_shifted.to_csv(output_path, index=False)
    logger.info(f"Saved shifted dataset to {output_path}")

    # ── Save simulation state ─────────────────────────────────────────────────
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)
    logger.info(f"Saved simulation state to {state_path}")

    # ── Split past / future ───────────────────────────────────────────────────
    logger.info("Splitting into past and future datasets")
    now    = pd.Timestamp(cfg.simulation.reference_date)
    past   = df_shifted[df_shifted["dteday"] <  now].copy()
    future = df_shifted[df_shifted["dteday"] >= now].copy()

    past_path   = raw_dir / cfg.paths.input_file
    future_path = raw_dir / "hour_future.csv"

    past.to_csv(past_path,   index=False)
    future.to_csv(future_path, index=False)

    logger.info(f"Saved past dataset ({len(past):,} records)")
    logger.info(f"Saved future dataset ({len(future):,} records)")
    logger.info(
        f"Done — {state['n_past_records']:,} past records | "
        f"{state['n_future_records']:,} future records"
    )


if __name__ == "__main__":
    main()