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


def build_output_drift_alert(row: dict) -> str:
    drifted_cols = [
        col.replace("drifted_", "")
        for col in ("drifted_pred_total", "drifted_pred_registered", "drifted_pred_casual")
        if row.get(col)
    ]
    return "\n".join(
        [
            "Output drift detected in the model's predictions.",
            "",
            f"- Drifted columns: {row['n_drifted']}/{row['n_features']} "
            f"({row['drift_share']:.0%}, threshold {row['threshold']:.0%})",
            f"- Affected: {', '.join(drifted_cols) if drifted_cols else 'none'}",
        ]
    )


def build_data_quality_alert(validation: dict) -> str:
    return "\n".join(
        [
            "Hourly data validation failed for newly revealed rows.",
            "",
            f"- Rows checked: {validation['n_rows_checked']}",
            f"- Issues: {validation['issues']}",
            "",
            "predict.py served a fallback prediction (168h-ago actuals) for this "
            "hour instead of the model.",
        ]
    )


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _load_last_output_drift_row(path: Path) -> dict | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    return df.iloc[-1].to_dict()


def _dispatch(title: str, body: str, label: str, cfg: DictConfig, email_to: str | None) -> None:
    create_github_issue(title, body, [label], cfg.alerting.dedup_hours)
    if email_to:
        send_email(title, body, email_to)
    else:
        logger.warning("ALERT_EMAIL_TO not set — skipping email")


@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    artifacts_dir = Path(cfg.paths.artifacts_dir)
    output_drift_row = _load_last_output_drift_row(
        artifacts_dir / "monitoring" / "output_drift_history.csv"
    )
    validation = _load_json(artifacts_dir / "validation" / "hourly_validation.json")
    email_to = os.environ.get("ALERT_EMAIL_TO")

    triggered = False
    now = datetime.now()

    if output_drift_row is not None and output_drift_row["drift_detected"]:
        triggered = True
        _dispatch(
            f"Output Drift Alert — {now:%Y-%m-%d %H:%M}",
            build_output_drift_alert(output_drift_row),
            cfg.alerting.output_drift_label,
            cfg,
            email_to,
        )

    if validation is not None and not validation["valid"]:
        triggered = True
        _dispatch(
            f"Data Quality Alert — {now:%Y-%m-%d %H:%M}",
            build_data_quality_alert(validation),
            cfg.alerting.data_quality_label,
            cfg,
            email_to,
        )

    if not triggered:
        logger.info("No alert conditions triggered this hour")


if __name__ == "__main__":
    main()
