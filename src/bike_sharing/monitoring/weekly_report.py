import json
import logging
import os
from datetime import datetime
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from bike_sharing.utils.alerting import create_github_issue, send_email

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


def _format_performance_section(performance_summary: dict | None) -> str:
    if performance_summary is None:
        return "## Live Performance\n\nNot available yet — no resolved predictions to score."

    lines = [
        "## Live Performance",
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
    return "\n".join(lines)


def build_weekly_digest(
    drift_summary: dict | None,
    retrain_outcome: dict | None,
    performance_summary: dict | None,
) -> str:
    """
    Build the weekly monitoring digest as markdown. Purely a function of its
    inputs — no I/O — so it can be unit tested without touching disk. Any
    section whose source is None (e.g. the very first run) renders as
    "not available yet" instead of failing.
    """
    return "\n\n".join(
        [
            _format_drift_section(drift_summary),
            _format_retrain_section(retrain_outcome),
            _format_performance_section(performance_summary),
        ]
    )


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _load_last_performance_row(path: Path) -> dict | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    return df.iloc[-1].to_dict()


@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    artifacts_dir = Path(cfg.paths.artifacts_dir)

    drift_summary = _load_json(artifacts_dir / "drift" / "drift_detected.json")
    retrain_outcome = _load_json(artifacts_dir / "monitoring" / "retrain_outcome.json")
    performance_summary = _load_last_performance_row(
        artifacts_dir / "monitoring" / "performance_history.csv"
    )

    body = build_weekly_digest(drift_summary, retrain_outcome, performance_summary)
    title = f"Weekly Monitoring Report — {datetime.now():%Y-%m-%d}"

    create_github_issue(title, body, [cfg.alerting.weekly_report_label], cfg.alerting.dedup_hours)

    to = os.environ.get("ALERT_EMAIL_TO")
    if to:
        send_email(title, body, to)
    else:
        logger.warning("ALERT_EMAIL_TO not set — skipping email")


if __name__ == "__main__":
    main()
