import json
import logging
from pathlib import Path
from datetime import datetime

import hydra
import pandas as pd
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


def load_simulation_state(state_path: Path) -> dict:
    """
    Load the simulation state from disk.
    Raises if the simulation has not been initialized.
    """
    if not state_path.exists():
        raise RuntimeError(
            "No simulation state found. "
            "Run shift_dates.py first to initialize the simulation."
        )
    with open(state_path) as f:
        return json.load(f)


def move_revealed_records(
    past_path: Path,
    future_path: Path,
    now: datetime,
) -> tuple[pd.DataFrame, int]:
    """
    Move records from hour_future.csv to hour_past.csv whose datetime
    has already passed relative to `now`.

    Parameters
    ----------
    past_path : Path
        Path to hour_past.csv.
    future_path : Path
        Path to hour_future.csv.
    now : datetime
        Current simulation time.

    Returns
    -------
    tuple[pd.DataFrame, int]
        Updated past DataFrame and number of records moved.
    """
    past   = pd.read_csv(past_path)
    future = pd.read_csv(future_path)

    # Parse dteday consistently in both dataframes
    past["dteday"]   = pd.to_datetime(past["dteday"]).dt.normalize()
    future["dteday"] = pd.to_datetime(future["dteday"]).dt.normalize()

    # Build datetime for comparison
    future["datetime"] = future["dteday"] + pd.to_timedelta(future["hr"], unit="h")
    now_ts = pd.Timestamp(now)

    revealed  = future[future["datetime"] <= now_ts].copy()
    remaining = future[future["datetime"] >  now_ts].copy()

    if len(revealed) == 0:
        logger.info("No new records to reveal")
        return past, 0

    # Drop the temporary datetime column before saving
    revealed  = revealed.drop(columns=["datetime"])
    remaining = remaining.drop(columns=["datetime"])

    # Concat and sort
    updated_past = pd.concat([past, revealed], ignore_index=True)
    updated_past = updated_past.sort_values(["dteday", "hr"]).reset_index(drop=True)

    # Save dteday as date string only
    updated_past["dteday"] = updated_past["dteday"].dt.strftime("%Y-%m-%d")
    remaining["dteday"]    = remaining["dteday"].dt.strftime("%Y-%m-%d")

    updated_past.to_csv(past_path,   index=False)
    remaining.to_csv(future_path,    index=False)

    return updated_past, len(revealed)


@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    raw_dir    = Path(cfg.paths.raw_dir)
    state_path = Path(cfg.paths.simulation_state)

    # ── Load state ────────────────────────────────────────────────────────────
    state = load_simulation_state(state_path)
    logger.info(
        f"Simulation state — started: {state['shift_applied_at'][:10]} | "
        f"future: {state['future_start_date']} → {state['future_end_date']}"
    )

    # ── Check if simulation is exhausted ──────────────────────────────────────
    future_path = raw_dir / "hour_future.csv"
    future      = pd.read_csv(future_path)

    if len(future) == 0:
        logger.warning("Simulation exhausted — no future records remaining.")
        return

    # ── Move revealed records ─────────────────────────────────────────────────
    now = datetime.now()
    logger.info(f"Current time: {now.strftime('%Y-%m-%d %H:%M')}")

    past, n_moved = move_revealed_records(
        past_path=raw_dir / cfg.paths.input_file,
        future_path=future_path,
        now=now,
    )

    if n_moved > 0:
        logger.info(f"Moved {n_moved} revealed records from future to past")
        logger.info(f"Past: {len(past):,} records | Future: {len(future) - n_moved:,} records remaining")

    # ── Summary ───────────────────────────────────────────────────────────────
    total_records = state["n_future_records"] + state["n_past_records"]
    remaining     = len(future) - n_moved
    pct_complete  = (total_records - remaining) / total_records * 100
    logger.info(f"Simulation progress: {pct_complete:.1f}% complete")


if __name__ == "__main__":
    main()