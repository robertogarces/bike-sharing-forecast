import pandas as pd
from omegaconf import OmegaConf

from bike_sharing.monitoring import hourly_alert


def _cfg(artifacts_dir):
    return OmegaConf.create(
        {
            "paths": {"artifacts_dir": str(artifacts_dir)},
            "alerting": {
                "dedup_hours": 24,
                "output_drift_label": "output-drift-alert",
                "data_quality_label": "data-quality-alert",
            },
        }
    )


# ── build_output_drift_alert / build_data_quality_alert ────────────────────────


def test_build_output_drift_alert_content():
    row = {
        "n_drifted": 2,
        "n_features": 3,
        "drift_share": 0.67,
        "threshold": 0.3,
        "drifted_pred_total": True,
        "drifted_pred_registered": True,
        "drifted_pred_casual": False,
    }

    body = hourly_alert.build_output_drift_alert(row)

    assert "2/3" in body
    assert "pred_total, pred_registered" in body
    assert "pred_casual" not in body.split("Affected:")[1]


def test_build_data_quality_alert_content():
    validation = {
        "n_rows_checked": 1,
        "issues": ["1 row(s) with 'hum' outside [0.0, 1.0]"],
    }

    body = hourly_alert.build_data_quality_alert(validation)

    assert "Rows checked: 1" in body
    assert "'hum' outside" in body
    assert "fallback prediction" in body


# ── main(): independent triggers ────────────────────────────────────────────────


def test_main_sends_nothing_when_no_triggers(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(hourly_alert, "create_github_issue", lambda *a, **k: calls.append(a))
    monkeypatch.setattr(hourly_alert, "send_email", lambda *a, **k: calls.append(a))

    hourly_alert.main.__wrapped__(_cfg(tmp_path))

    assert calls == []


def test_main_triggers_only_output_drift(monkeypatch, tmp_path):
    history_path = tmp_path / "monitoring" / "output_drift_history.csv"
    history_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "n_drifted": 2,
                "n_features": 3,
                "drift_share": 0.67,
                "drift_detected": True,
                "threshold": 0.3,
                "drifted_pred_total": True,
                "drifted_pred_registered": True,
                "drifted_pred_casual": False,
            }
        ]
    ).to_csv(history_path, index=False)

    issue_calls = []
    monkeypatch.setattr(hourly_alert, "create_github_issue", lambda *a, **k: issue_calls.append(a))
    monkeypatch.setattr(hourly_alert, "send_email", lambda *a, **k: None)

    hourly_alert.main.__wrapped__(_cfg(tmp_path))

    assert len(issue_calls) == 1
    assert issue_calls[0][2] == ["output-drift-alert"]


def test_main_triggers_both_independently_when_both_occur(monkeypatch, tmp_path):
    history_path = tmp_path / "monitoring" / "output_drift_history.csv"
    history_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "n_drifted": 1,
                "n_features": 3,
                "drift_share": 0.33,
                "drift_detected": True,
                "threshold": 0.3,
                "drifted_pred_total": True,
                "drifted_pred_registered": False,
                "drifted_pred_casual": False,
            }
        ]
    ).to_csv(history_path, index=False)

    validation_path = tmp_path / "validation" / "hourly_validation.json"
    validation_path.parent.mkdir(parents=True)
    validation_path.write_text('{"n_rows_checked": 1, "valid": false, "issues": ["bad row"]}')

    issue_calls = []
    monkeypatch.setattr(hourly_alert, "create_github_issue", lambda *a, **k: issue_calls.append(a))
    monkeypatch.setattr(hourly_alert, "send_email", lambda *a, **k: None)

    hourly_alert.main.__wrapped__(_cfg(tmp_path))

    assert len(issue_calls) == 2
    labels = {c[2][0] for c in issue_calls}
    assert labels == {"output-drift-alert", "data-quality-alert"}
