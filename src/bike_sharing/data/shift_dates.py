import json
import logging
from pathlib import Path
from datetime import datetime

import hydra
import pandas as pd
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

STATE_PATH = Path("data/simulation_state.json")


def check_existing_state() -> None:
    """
    Abort if a simulation state already exists.

    Prevents accidental overwrite of an ongoing simulation.
    To reset, delete data/simulation_state.json explicitly.
    """
    if STATE_PATH.exists():
        with open(STATE_PATH) as f:
            state = json.load(f)
        raise RuntimeError(
            f"A simulation is already running (started {state['shift_applied_at']}, "
            f"reference date: {state['reference_date']}, "
            f"future: {state['future_start_date']} → {state['future_end_date']}).\n"
            f"To reset, delete {STATE_PATH} explicitly."
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
        "reference_date":   reference_date.strftime("%Y-%m-%d"),
        "future_pct":       future_pct,
        "shift_applied_at": datetime.now().isoformat(),
        "future_start_date": future_start.strftime("%Y-%m-%d"),
        "future_end_date":   future_end.strftime("%Y-%m-%d"),
        "n_future_records":  n_future,
        "n_past_records":    n_past,
    }

    return df, state


@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    raw_dir = Path(cfg.dataset.raw_dir)

    # ── Safety check ──────────────────────────────────────────────────────────
    check_existing_state()

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
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)
    logger.info(f"Saved simulation state to {STATE_PATH}")

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info(
        f"Done — {state['n_past_records']:,} past records | "
        f"{state['n_future_records']:,} future records"
    )


if __name__ == "__main__":
    main()