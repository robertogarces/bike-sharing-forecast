import json
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import hydra
from pathlib import Path

from bike_sharing.utils.datetime_utils import reconstruct_datetime
from bike_sharing.utils.mlflow_utils import setup_mlflow

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bike Sharing — Dashboard",
    page_icon="🚲",
    layout="wide",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
# Streamlit (not Hydra) is the entry point here, so we use Hydra's compose API
# instead of @hydra.main — it builds the same config without taking over
# argument parsing or the process's control flow.
ROOT = Path(__file__).parents[3]
with hydra.initialize(config_path="../../../configs", version_base=None):
    cfg = hydra.compose(config_name="config")

PREDICTIONS = ROOT / cfg.paths.predictions_path
PAST = ROOT / cfg.paths.raw_dir / cfg.paths.input_file
METRICS = ROOT / cfg.paths.evaluation_dir / "metrics.json"
STATE = ROOT / cfg.paths.simulation_state
MONITORING_DIR = ROOT / cfg.paths.artifacts_dir / "monitoring"
DRIFT_DIR = ROOT / cfg.paths.artifacts_dir / "drift"
PROJECT = cfg.project

FALLBACK_SOURCE = "fallback_lag168"


# ── Pure helpers (testable — see tests/test_dashboard.py) ─────────────────────
def normalize_retrain_outcome(outcome: dict) -> dict:
    """
    Fill in the fields the dashboard reads, tolerating the legacy schema
    (single "promoted" bool, before promotion became per-model). Older
    retrain_outcome.json snapshots on disk may still use it.
    """
    if "promoted_registered" not in outcome and "promoted" in outcome:
        outcome = {
            **outcome,
            "promoted_registered": outcome["promoted"],
            "promoted_casual": outcome["promoted"],
        }
    return outcome


def compute_gauge_range(actual_total: pd.Series) -> tuple[float, float]:
    """
    Gauge axis max and warning threshold derived from observed demand, instead
    of a hardcoded number that silently goes stale as demand grows (the
    historical max is already 977, past a hardcoded 900 range). Axis tops out
    10% above the historical max, rounded up to the nearest 50; threshold is
    the 90th percentile of historical demand.
    """
    if actual_total.empty:
        return 900.0, 700.0
    axis_max = float(((actual_total.max() * 1.1) // 50 + 1) * 50)
    threshold = float(actual_total.quantile(0.9))
    return axis_max, threshold


# ── Load data — Operations ─────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def load_predictions() -> pd.DataFrame:
    if not PREDICTIONS.exists():
        return pd.DataFrame()
    df = pd.read_csv(PREDICTIONS)
    df["timestamp_predicted"] = pd.to_datetime(df["timestamp_predicted"], format="ISO8601")
    df["predicted_at"] = pd.to_datetime(df["predicted_at"], format="ISO8601")
    if "prediction_source" not in df.columns:
        df["prediction_source"] = "model"
    df["prediction_source"] = df["prediction_source"].fillna("model")
    return df.sort_values("timestamp_predicted")


@st.cache_data(ttl=60)
def load_actuals() -> pd.DataFrame:
    if not PAST.exists():
        return pd.DataFrame()
    df = pd.read_csv(PAST, parse_dates=["dteday"])
    df = reconstruct_datetime(df)
    return df[["datetime", "cnt"]].rename(
        columns={"datetime": "timestamp_predicted", "cnt": "actual_total"}
    )


@st.cache_data(ttl=300)
def load_metrics() -> dict:
    if not METRICS.exists():
        return {}
    with open(METRICS) as f:
        return json.load(f)


@st.cache_data(ttl=300)
def load_state() -> dict:
    if not STATE.exists():
        return {}
    with open(STATE) as f:
        return json.load(f)


# ── Load data — Monitoring ──────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_model_status() -> dict:
    """
    Current production version + when it was trained, for each of
    registered/casual, read live from the MLflow registry — the same
    authoritative source predict.py uses. Returns {} if MLflow is
    unreachable (e.g. no credentials configured locally) rather than
    crashing the whole dashboard.
    """
    try:
        setup_mlflow()
        import mlflow

        client = mlflow.MlflowClient()
        status = {}
        for slot in ("registered", "casual"):
            version = client.get_model_version_by_alias(f"{PROJECT}-{slot}", "production")
            run = client.get_run(version.run_id)
            status[slot] = {
                "version": version.version,
                "trained_at": pd.Timestamp(run.info.start_time, unit="ms"),
                "baseline_rmse": version.tags.get("combined_rmse_baseline"),
            }
        return status
    except Exception:
        return {}


@st.cache_data(ttl=300)
def load_retrain_outcome() -> dict:
    path = MONITORING_DIR / "retrain_outcome.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return normalize_retrain_outcome(json.load(f))


@st.cache_data(ttl=300)
def load_drift_status() -> dict:
    path = DRIFT_DIR / "drift_detected.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


@st.cache_data(ttl=300)
def load_history_csv(filename: str) -> pd.DataFrame:
    """Shared loader for the three accumulating monitoring histories."""
    path = MONITORING_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="ISO8601")
    return df.sort_values("timestamp")


# ── Page: Operations ────────────────────────────────────────────────────────────
def render_operations():
    st.title("🚲 Bike Sharing — Operations Dashboard")
    st.caption("Next-hour demand forecasting system · Predictions update every hour")

    predictions = load_predictions()
    actuals = load_actuals()
    metrics = load_metrics()
    state = load_state()

    if predictions.empty:
        st.warning("No predictions available yet. Run predict.py first.")
        return

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Model Performance")
        st.caption("How accurate is the model?")

        if metrics:
            rmse = metrics.get("rmse", 0)
            rmsle = metrics.get("rmsle", 0)
            r2 = metrics.get("r2", 0)

            st.metric("RMSE", f"{rmse:.1f} bikes")
            st.caption(f"On average, the model is off by **{rmse:.0f} bikes per hour**.")

            st.metric("RMSLE", f"{rmsle:.4f}")
            st.caption(f"The model has an average relative error of **{rmsle * 100:.1f}%**.")

            st.metric("R²", f"{r2:.4f}")
            st.caption(f"The model explains **{r2 * 100:.1f}%** of demand variability.")

        st.divider()
        st.header("Simulation Status")

        if state:
            future_end = pd.Timestamp(state.get("future_end_date", ""))
            total_future = state.get("n_future_records", 0)
            remaining = (
                len(pd.read_csv(ROOT / "data" / "raw" / "hour_future.csv"))
                if (ROOT / "data" / "raw" / "hour_future.csv").exists()
                else 0
            )
            pct_used = (total_future - remaining) / total_future * 100 if total_future > 0 else 0

            st.progress(pct_used / 100)
            st.caption(f"Simulation: {pct_used:.1f}% complete")
            st.caption(f"Data available until: **{future_end.strftime('%b %d, %Y')}**")

    # ── Next hour prediction ──────────────────────────────────────────────────
    latest = predictions.iloc[-1]
    next_hr = latest["timestamp_predicted"]
    pred_total = latest["pred_total"]
    is_fallback = latest["prediction_source"] == FALLBACK_SOURCE

    st.subheader(f"Next Hour Forecast — {next_hr.strftime('%A %b %d, %H:%M')}")
    if is_fallback:
        st.warning(
            "⚠️ This is a **fallback prediction** (168h-lag), not live model output — "
            "hourly data validation failed for this hour."
        )

    col1, col2, col3 = st.columns([2, 1, 1])

    gauge_max, gauge_threshold = compute_gauge_range(actuals["actual_total"])

    with col1:
        # Gauge
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=pred_total,
                title={"text": "Predicted Bikes Needed", "font": {"size": 18}},
                gauge={
                    "axis": {"range": [0, gauge_max]},
                    "bar": {"color": "#1F77B4" if not is_fallback else "#B0B0B0"},
                    "steps": [
                        {"range": [0, gauge_max * 0.22], "color": "#EAF4FB"},
                        {"range": [gauge_max * 0.22, gauge_max * 0.55], "color": "#AED6F1"},
                        {"range": [gauge_max * 0.55, gauge_max], "color": "#2E86C1"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 3},
                        "thickness": 0.75,
                        "value": gauge_threshold,
                    },
                },
                number={"suffix": " bikes", "font": {"size": 36}},
            )
        )
        fig_gauge.update_layout(height=300, margin=dict(t=40, b=0, l=20, r=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col2:
        st.metric("Registered riders", f"{latest['pred_registered']:.0f}")
        st.caption("Commuters and subscribers")

    with col3:
        st.metric("Casual riders", f"{latest['pred_casual']:.0f}")
        st.caption("Tourists and occasional users")

    st.divider()

    # ── Last 24 hours ─────────────────────────────────────────────────────────
    st.subheader("Last 24 Hours — Prediction History")

    last_24 = predictions.tail(24)
    last_24_with_actuals = last_24.merge(actuals, on="timestamp_predicted", how="left")
    model_rows = last_24_with_actuals[last_24_with_actuals["prediction_source"] != FALLBACK_SOURCE]
    fallback_rows = last_24_with_actuals[
        last_24_with_actuals["prediction_source"] == FALLBACK_SOURCE
    ]

    fig_line = go.Figure()
    fig_line.add_trace(
        go.Scatter(
            x=model_rows["timestamp_predicted"],
            y=model_rows["pred_total"],
            mode="lines+markers",
            name="Predicted",
            line=dict(color="#1F77B4", width=2),
            marker=dict(size=6),
        )
    )
    if not fallback_rows.empty:
        fig_line.add_trace(
            go.Scatter(
                x=fallback_rows["timestamp_predicted"],
                y=fallback_rows["pred_total"],
                mode="markers",
                name="Fallback (168h-lag)",
                marker=dict(size=9, symbol="x", color="#B0B0B0"),
            )
        )
    if last_24_with_actuals["actual_total"].notna().any():
        actuals_known = last_24_with_actuals.dropna(subset=["actual_total"])
        fig_line.add_trace(
            go.Scatter(
                x=actuals_known["timestamp_predicted"],
                y=actuals_known["actual_total"],
                mode="lines+markers",
                name="Actual",
                line=dict(color="#E74C3C", width=2, dash="dot"),
                marker=dict(size=6),
            )
        )
    fig_line.update_layout(
        xaxis_title="Hour",
        yaxis_title="Bikes",
        height=300,
        margin=dict(t=20, b=40, l=40, r=20),
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig_line, use_container_width=True)

    # ── Registered vs Casual breakdown ───────────────────────────────────────
    st.subheader("Demand Breakdown — Registered vs Casual")

    fig_bar = go.Figure()
    fig_bar.add_trace(
        go.Bar(
            x=last_24["timestamp_predicted"],
            y=last_24["pred_registered"],
            name="Registered",
            marker_color="#1F77B4",
        )
    )
    fig_bar.add_trace(
        go.Bar(
            x=last_24["timestamp_predicted"],
            y=last_24["pred_casual"],
            name="Casual",
            marker_color="#AED6F1",
        )
    )
    fig_bar.update_layout(
        barmode="stack",
        xaxis_title="Hour",
        yaxis_title="Bikes",
        height=300,
        margin=dict(t=20, b=40, l=40, r=20),
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Recent predictions table ──────────────────────────────────────────────
    st.subheader("Recent Predictions")

    table = predictions.tail(10).copy()
    table["timestamp_predicted"] = table["timestamp_predicted"].dt.strftime("%Y-%m-%d %H:%M")
    table["predicted_at"] = table["predicted_at"].dt.strftime("%Y-%m-%d %H:%M")
    table["Source"] = table["prediction_source"].map(
        {FALLBACK_SOURCE: "Fallback (168h-lag)", "model": "Model"}
    )
    if "model_version_registered" in table.columns:
        table["Model (Reg/Cas)"] = (
            table["model_version_registered"].fillna("?").astype(str)
            + " / "
            + table["model_version_casual"].fillna("?").astype(str)
        )
    else:
        table["Model (Reg/Cas)"] = "?"
    table = table[
        [
            "timestamp_predicted",
            "pred_total",
            "pred_registered",
            "pred_casual",
            "temp",
            "hum",
            "weathersit",
            "Source",
            "Model (Reg/Cas)",
        ]
    ].rename(
        columns={
            "timestamp_predicted": "Hour",
            "pred_total": "Total",
            "pred_registered": "Registered",
            "pred_casual": "Casual",
            "temp": "Temp (norm)",
            "hum": "Humidity (norm)",
            "weathersit": "Weather",
        }
    )
    table = table.sort_values("Hour", ascending=False).reset_index(drop=True)
    st.dataframe(table, use_container_width=True)


# ── Page: Monitoring ────────────────────────────────────────────────────────────
def render_monitoring():
    st.title("📈 Bike Sharing — Monitoring Dashboard")
    st.caption("Model status, retraining, drift, and live performance over time")

    # ── Model status ──────────────────────────────────────────────────────────
    st.subheader("Model Status")
    status = load_model_status()
    if not status:
        st.info("Could not reach the MLflow registry — model status unavailable.")
    else:
        col1, col2 = st.columns(2)
        for col, slot in zip([col1, col2], ["registered", "casual"]):
            with col:
                s = status[slot]
                st.metric(f"{slot.capitalize()} — production version", f"v{s['version']}")
                st.caption(f"Trained: {s['trained_at'].strftime('%Y-%m-%d %H:%M')}")
                if s["baseline_rmse"]:
                    st.caption(f"Baseline RMSE at promotion: {float(s['baseline_rmse']):.2f}")

    st.divider()

    # ── Retrain gate ──────────────────────────────────────────────────────────
    st.subheader("Retrain Gate — Last Run")
    outcome = load_retrain_outcome()
    if not outcome:
        st.info("No retrain outcome recorded yet.")
    else:
        st.caption(f"As of {outcome.get('timestamp', 'unknown')}")
        if not outcome.get("retrain_attempted"):
            st.write(f"**Attempted:** No — {outcome.get('skip_reason', 'unknown reason')}")
        else:
            col1, col2, col3 = st.columns(3)
            col1.metric(
                "Data quality", "✅ Passed" if outcome.get("data_quality_passed") else "❌ Failed"
            )
            col2.metric(
                "Promoted — registered", "Yes" if outcome.get("promoted_registered") else "No"
            )
            col3.metric("Promoted — casual", "Yes" if outcome.get("promoted_casual") else "No")
            if outcome.get("new_rmse") is not None and outcome.get("prod_rmse") is not None:
                st.caption(
                    f"New pair RMSE: {outcome['new_rmse']:.2f} | "
                    f"Production pair RMSE: {outcome['prod_rmse']:.2f}"
                )
        if outcome.get("baseline_rmse") is not None and outcome.get("live_rmse") is not None:
            status_word = "⚠️ DEGRADED" if outcome.get("performance_degraded") else "✅ stable"
            st.caption(
                f"Live vs. baseline RMSE: {outcome['live_rmse']:.2f} vs "
                f"{outcome['baseline_rmse']:.2f} ({status_word})"
            )

    st.divider()

    # ── Live performance over time ────────────────────────────────────────────
    st.subheader("Live Performance Over Time")
    perf = load_history_csv("performance_history.csv")
    if perf.empty:
        st.info("No performance history yet — accumulates weekly.")
    else:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=perf["timestamp"], y=perf["rmse"], mode="lines+markers", name="Model RMSE")
        )
        if "naive_rmse" in perf.columns and perf["naive_rmse"].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=perf["timestamp"],
                    y=perf["naive_rmse"],
                    mode="lines+markers",
                    name="Seasonal-naive RMSE",
                    line=dict(dash="dot"),
                )
            )
        fig.update_layout(
            xaxis_title="Week", yaxis_title="RMSE", height=280, margin=dict(t=20, b=40, l=40, r=20)
        )
        st.plotly_chart(fig, use_container_width=True)
        if "skill_vs_naive" in perf.columns and perf["skill_vs_naive"].notna().any():
            latest_skill = perf["skill_vs_naive"].dropna().iloc[-1]
            st.caption(f"Latest skill vs. seasonal-naive: **{latest_skill:+.1%}**")

    st.divider()

    # ── Input drift ───────────────────────────────────────────────────────────
    st.subheader("Input Drift")
    drift = load_drift_status()
    if not drift:
        st.info("No drift report yet.")
    else:
        col1, col2 = st.columns(2)
        col1.metric("Status", "⚠️ DRIFT DETECTED" if drift.get("drift_detected") else "✅ No drift")
        col2.metric(
            "Drifted features",
            f"{drift.get('n_drifted', 0)}/{drift.get('n_features', 0)} "
            f"({drift.get('drift_share', 0):.0%})",
        )
        drifted_cols = [
            col for col, info in drift.get("drift_by_column", {}).items() if info.get("drifted")
        ]
        if drifted_cols:
            st.caption(f"Drifted: {', '.join(drifted_cols)}")

    drift_hist = load_history_csv("drift_history.csv")
    if not drift_hist.empty:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=drift_hist["timestamp"], y=drift_hist["drift_share"], mode="lines+markers")
        )
        fig.update_layout(
            xaxis_title="Week",
            yaxis_title="Drift share",
            height=220,
            margin=dict(t=10, b=40, l=40, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("Input drift history: not enough weeks accumulated yet.")

    st.divider()

    # ── Output drift over time ────────────────────────────────────────────────
    st.subheader("Output Drift Over Time")
    output_hist = load_history_csv("output_drift_history.csv")
    if output_hist.empty:
        st.info("No output drift history yet — accumulates hourly once enough predictions exist.")
    else:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=output_hist["timestamp"], y=output_hist["drift_share"], mode="lines+markers"
            )
        )
        fig.update_layout(
            xaxis_title="Hour",
            yaxis_title="Drift share",
            height=220,
            margin=dict(t=10, b=40, l=40, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)


# ── Navigation ──────────────────────────────────────────────────────────────────
def main():
    pg = st.navigation(
        [
            st.Page(render_operations, title="Operations", icon="🚲", default=True),
            st.Page(render_monitoring, title="Monitoring", icon="📈"),
        ]
    )
    pg.run()


if __name__ == "__main__":
    main()
