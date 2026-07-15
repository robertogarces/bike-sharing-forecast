import json
import logging
import os
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from bike_sharing.utils.alerting import create_github_issue, send_email
from bike_sharing.utils.datetime_utils import utc_now

logger = logging.getLogger(__name__)


def _format_drift_section(drift_summary: dict | None) -> str:
    if drift_summary is None:
        return (
            "## Input Data Drift\n\nNot available yet — drift detection has not produced a report."
        )

    status = "DRIFT DETECTED" if drift_summary["drift_detected"] else "No drift detected"
    lines = [
        "## Input Data Drift",
        "",
        f"- Status: {status}",
        f"- Drifted features: {drift_summary['n_drifted']}/{drift_summary['n_features']} "
        f"({drift_summary['drift_share']:.0%}, threshold {drift_summary['threshold']:.0%})",
    ]
    drifted_cols = [
        col for col, info in drift_summary["drift_by_column"].items() if info["drifted"]
    ]
    if drifted_cols:
        lines.append(f"- Drifted: {', '.join(drifted_cols)}")
    return "\n".join(lines)


def _format_retrain_section(retrain_outcome: dict | None) -> str:
    if retrain_outcome is None:
        return "## Retraining\n\nNot available yet — retrain.py has not produced an outcome."

    if not retrain_outcome["retrain_attempted"]:
        lines = [
            "## Retraining",
            "",
            "- Attempted: No",
            f"- Reason: {retrain_outcome['skip_reason']}",
        ]
        _append_performance_line(lines, retrain_outcome)
        return "\n".join(lines)

    lines = [
        "## Retraining",
        "",
        "- Attempted: Yes",
        f"- Data quality check: {'passed' if retrain_outcome['data_quality_passed'] else 'failed'}",
    ]
    if retrain_outcome["data_quality_issues"]:
        lines.append(f"- Data quality issues: {retrain_outcome['data_quality_issues']}")
    if retrain_outcome["new_rmse"] is not None:
        lines.append(f"- New model RMSE: {retrain_outcome['new_rmse']:.4f}")
    if retrain_outcome["prod_rmse"] is not None:
        lines.append(f"- Production model RMSE: {retrain_outcome['prod_rmse']:.4f}")
    if retrain_outcome.get("promoted_registered") is not None:
        lines.append(
            f"- Promoted to production: "
            f"registered={'Yes' if retrain_outcome['promoted_registered'] else 'No'}, "
            f"casual={'Yes' if retrain_outcome['promoted_casual'] else 'No'}"
        )
    _append_performance_line(lines, retrain_outcome)
    return "\n".join(lines)


def _append_performance_line(lines: list[str], retrain_outcome: dict) -> None:
    baseline_rmse = retrain_outcome.get("baseline_rmse")
    live_rmse = retrain_outcome.get("live_rmse")
    if baseline_rmse is not None and live_rmse is not None:
        status = "DEGRADED" if retrain_outcome["performance_degraded"] else "stable"
        lines.append(f"- Live vs. baseline RMSE: {live_rmse:.2f} vs {baseline_rmse:.2f} ({status})")


def _format_performance_section(
    performance_summary: dict | None,
    horizon_curve: list[dict] | None = None,
) -> str:
    if performance_summary is None:
        return "## Live Performance\n\nNot available yet — no resolved predictions to score."

    horizon = performance_summary.get("horizon")
    headline = "## Live Performance"
    if horizon is not None:
        headline += f" (primary horizon h+{int(horizon)})"

    lines = [
        headline,
        "",
        f"- Window: last {int(performance_summary['n_hours'])}h "
        f"({int(performance_summary['n_resolved'])} resolved predictions)",
        f"- RMSE: {performance_summary['rmse']:.2f}",
        f"- MAE: {performance_summary['mae']:.2f}",
        f"- RMSLE: {performance_summary['rmsle']:.4f}",
        f"- R²: {performance_summary['r2']:.4f}",
    ]
    naive_rmse = performance_summary.get("naive_rmse")
    skill = performance_summary.get("skill_vs_naive")
    if pd.notna(naive_rmse) and pd.notna(skill):
        lines.append(
            f"- vs seasonal-naive: {performance_summary['rmse']:.2f} vs {naive_rmse:.2f} "
            f"RMSE ({skill:+.1%} skill)"
        )

    # Per-horizon skill curve — the multi-horizon payoff: shows where along the
    # trajectory the model still beats the naive baseline. Only rendered when
    # more than the primary horizon is present.
    if horizon_curve and len(horizon_curve) > 1:
        lines.append("")
        lines.append("Skill by horizon (latest):")
        for row in horizon_curve:
            row_skill = row.get("skill_vs_naive")
            skill_str = f"{row_skill:+.1%} skill" if pd.notna(row_skill) else "skill n/a"
            lines.append(f"- h+{int(row['horizon'])}: RMSE {row['rmse']:.2f} ({skill_str})")

    return "\n".join(lines)


def build_weekly_digest(
    drift_summary: dict | None,
    retrain_outcome: dict | None,
    performance_summary: dict | None,
    horizon_curve: list[dict] | None = None,
) -> str:
    """
    Build the weekly monitoring digest as markdown. Purely a function of its
    inputs — no I/O — so it can be unit tested without touching disk. Any
    section whose source is None (e.g. the very first run) renders as
    "not available yet" instead of failing.

    performance_summary is the primary-horizon (h+1) headline — the number
    comparable to the retrain baseline. horizon_curve, when given, adds the
    latest RMSE/skill per lead time so the report shows the whole trajectory's
    quality, not just the next hour.
    """
    return "\n\n".join(
        [
            _format_drift_section(drift_summary),
            _format_retrain_section(retrain_outcome),
            _format_performance_section(performance_summary, horizon_curve),
        ]
    )


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _read_performance_history(path: Path) -> pd.DataFrame | None:
    """Load performance_history.csv with the horizon column normalized (legacy
    rows without it read as horizon=1). None if missing or empty."""
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    if "horizon" not in df.columns:
        df["horizon"] = 1
    else:
        df["horizon"] = df["horizon"].fillna(1).astype(int)
    return df


def _load_primary_performance_row(path: Path, primary_horizon: int) -> dict | None:
    """Most recent performance record at primary_horizon — the headline number
    comparable to the retrain baseline."""
    df = _read_performance_history(path)
    if df is None:
        return None
    df = df[df["horizon"] == primary_horizon]
    if df.empty:
        return None
    return df.iloc[-1].to_dict()


def _load_horizon_curve(path: Path) -> list[dict] | None:
    """Latest record per horizon (the current error/skill-vs-horizon curve),
    ordered by horizon. None if no history yet."""
    df = _read_performance_history(path)
    if df is None:
        return None
    latest = df.groupby("horizon").tail(1).sort_values("horizon")
    return latest.to_dict("records")


@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    artifacts_dir = Path(cfg.paths.artifacts_dir)

    drift_summary = _load_json(artifacts_dir / "drift" / "drift_detected.json")
    retrain_outcome = _load_json(artifacts_dir / "monitoring" / "retrain_outcome.json")
    performance_history_path = artifacts_dir / "monitoring" / "performance_history.csv"
    performance_summary = _load_primary_performance_row(
        performance_history_path, cfg.forecast.primary_horizon
    )
    horizon_curve = _load_horizon_curve(performance_history_path)

    body = build_weekly_digest(drift_summary, retrain_outcome, performance_summary, horizon_curve)
    title = f"Weekly Monitoring Report — {utc_now():%Y-%m-%d}"

    created = create_github_issue(
        title, body, [cfg.alerting.weekly_report_label], cfg.alerting.dedup_hours
    )
    if not created:
        return

    to = os.environ.get("ALERT_EMAIL_TO")
    if to:
        send_email(title, body, to)
    else:
        logger.warning("ALERT_EMAIL_TO not set — skipping email")


if __name__ == "__main__":
    main()
