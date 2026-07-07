import logging
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

logger = logging.getLogger(__name__)

MIN_WEEKS = 12
PERCENTILE = 90


def suggest_drift_threshold(
    history_path: Path, min_weeks: int = MIN_WEEKS
) -> tuple[float | None, int]:
    """
    Suggest a drift_threshold from the historical drift_share values in
    drift_history.csv, using the Nth percentile (PERCENTILE) — the value
    that separates ordinary weeks from the unusual tail.

    Returns (suggested, n_weeks). suggested is None if there isn't enough
    history yet (n_weeks < min_weeks) or the file doesn't exist.
    """
    if not history_path.exists():
        return None, 0
    df = pd.read_csv(history_path)
    if len(df) < min_weeks:
        return None, len(df)
    return float(np.percentile(df["drift_share"], PERCENTILE)), len(df)


def suggest_performance_degradation_threshold(
    history_path: Path, min_weeks: int = MIN_WEEKS
) -> tuple[float | None, int]:
    """
    Suggest a performance_degradation_threshold from the week-over-week %
    change in rolling RMSE in performance_history.csv — measures natural
    variation, not the absolute RMSE level.

    Returns (suggested, n_weeks). n_weeks counts % changes, one fewer than
    the number of rows (the first week has no prior week to compare
    against). suggested is None if there isn't enough history yet.
    """
    if not history_path.exists():
        return None, 0
    df = pd.read_csv(history_path)
    pct_change = df["rmse"].pct_change().dropna()
    if len(pct_change) < min_weeks:
        return None, len(pct_change)
    return float(np.percentile(pct_change, PERCENTILE)), len(pct_change)


@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    monitoring_dir = Path(cfg.paths.artifacts_dir) / "monitoring"

    drift_suggestion, n_drift_weeks = suggest_drift_threshold(monitoring_dir / "drift_history.csv")
    if drift_suggestion is None:
        logger.info(f"drift_threshold: not enough history yet ({n_drift_weeks}/{MIN_WEEKS} weeks)")
    else:
        logger.info(
            f"drift_threshold: current={cfg.monitoring.drift_threshold:.2f}, "
            f"suggested={drift_suggestion:.2f} (p{PERCENTILE} over {n_drift_weeks} weeks)"
        )

    perf_suggestion, n_perf_weeks = suggest_performance_degradation_threshold(
        monitoring_dir / "performance_history.csv"
    )
    if perf_suggestion is None:
        logger.info(
            f"performance_degradation_threshold: not enough history yet "
            f"({n_perf_weeks}/{MIN_WEEKS} weeks)"
        )
    else:
        logger.info(
            f"performance_degradation_threshold: "
            f"current={cfg.monitoring.performance_degradation_threshold:.2f}, "
            f"suggested={perf_suggestion:.2f} (p{PERCENTILE} over {n_perf_weeks} weeks)"
        )


if __name__ == "__main__":
    main()
