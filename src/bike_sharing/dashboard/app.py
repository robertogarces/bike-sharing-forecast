import json
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import hydra
from pathlib import Path

from bike_sharing.models.train import compute_metrics
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
MONITORING_DIR = ROOT / cfg.paths.artifacts_dir / "monitoring"
DRIFT_DIR = ROOT / cfg.paths.artifacts_dir / "drift"
PROJECT = cfg.project
PRIMARY_HORIZON = cfg.forecast.primary_horizon
N_HOURS = cfg.monitoring.n_hours

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


def ensure_horizon(df: pd.DataFrame) -> pd.DataFrame:
    """
    Guarantee a horizon column. Rows written before multi-horizon serving have
    none — they were all next-hour predictions, so they read as horizon=1.
    """
    df = df.copy()
    if "horizon" not in df.columns:
        df["horizon"] = 1
    else:
        df["horizon"] = df["horizon"].fillna(1).astype(int)
    return df


def latest_trajectory(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    The most recent origin's full forecast (h+1..h+K), sorted by horizon — the
    current demand profile shown as the trajectory.

    A row's origin is timestamp_predicted - horizon hours; every row of one
    run's rollout shares that origin, so the max origin selects the latest run
    (whether it was live model output or a fallback trajectory).
    """
    if predictions.empty:
        return predictions
    df = ensure_horizon(predictions)
    df["origin"] = df["timestamp_predicted"] - pd.to_timedelta(df["horizon"], unit="h")
    latest_origin = df["origin"].max()
    return df[df["origin"] == latest_origin].sort_values("horizon").reset_index(drop=True)


def filter_to_horizon(predictions: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Rows at a single lead time — one prediction per target hour. The series used
    for the history-vs-actual view and the performance-over-time chart, where
    mixing lead times would double-count hours and blur the signal.
    """
    if predictions.empty:
        return predictions
    df = ensure_horizon(predictions)
    return df[df["horizon"] == horizon].reset_index(drop=True)


def latest_per_horizon(history: pd.DataFrame) -> pd.DataFrame:
    """
    Most recent record per horizon — the current error/skill-vs-horizon curve
    from performance_history.csv (which now holds one row per (run, horizon)).
    """
    if history.empty:
        return history
    df = ensure_horizon(history)
    return df.groupby("horizon").tail(1).sort_values("horizon").reset_index(drop=True)


def live_model_metrics(
    predictions: pd.DataFrame, actuals: pd.DataFrame, n_hours: int
) -> dict | None:
    """
    RMSE/RMSLE/MAE/R² for the combined total and each sub-model (registered,
    casual), over the most recent n_hours of resolved primary-horizon
    predictions — the same window and method performance_monitoring uses, so
    the combined figure matches the pipeline's methodology.

    performance_history.csv only stores the combined metric; the per-model
    figures are computed here from pred_registered/pred_casual against their
    actuals (available in hour_past.csv). Returns None until there is at least
    one resolved model prediction (fallback rows excluded).
    """
    if predictions.empty or actuals.empty:
        return None
    h1 = filter_to_horizon(predictions, PRIMARY_HORIZON)
    h1 = h1[h1["prediction_source"] != FALLBACK_SOURCE]
    joined = h1.merge(actuals, on="timestamp_predicted", how="inner")
    joined = joined.sort_values("timestamp_predicted").tail(n_hours)
    if joined.empty:
        return None
    return {
        "Combined": compute_metrics(joined["actual_total"].values, joined["pred_total"].values),
        "Registered": compute_metrics(
            joined["actual_registered"].values, joined["pred_registered"].values
        ),
        "Casual": compute_metrics(joined["actual_casual"].values, joined["pred_casual"].values),
    }


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
    return df[["datetime", "cnt", "registered", "casual"]].rename(
        columns={
            "datetime": "timestamp_predicted",
            "cnt": "actual_total",
            "registered": "actual_registered",
            "casual": "actual_casual",
        }
    )


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


def _inject_styles(subtitle: str, live_ring: bool = False) -> None:
    """
    Shared page chrome so Operations and Monitoring read as one system: a
    prominent subtitle (same size/color on both), a tighter top margin so the
    title starts high, and enlarged KPI metrics scoped to the first metric row
    on the page. live_ring adds the spinning "live" indicator before the first
    KPI's label (used only on Operations' "Now" metric).
    """
    first_row = '[data-testid="stHorizontalBlock"]:first-of-type'
    ring = (
        f"{first_row} > div:first-child "
        '[data-testid="stMetricLabel"] p::before{content:"";display:inline-block;'
        "width:10px;height:10px;border:2px solid rgba(46,204,113,0.30);"
        "border-top-color:#2ECC71;border-radius:50%;margin-right:7px;"
        "vertical-align:middle;animation:live-spin 2.2s linear infinite;}"
        "@keyframes live-spin{to{transform:rotate(360deg);}}"
        if live_ring
        else ""
    )
    st.markdown(
        f"<div style='font-size:1.6rem;font-weight:600;color:#4a5b6b;margin-top:-6px;"
        f"margin-bottom:2px'>{subtitle}</div>"
        "<style>"
        ".block-container{padding-top:2rem;padding-bottom:1rem;}"
        f'{first_row} [data-testid="stMetricValue"]{{font-size:2.4rem;}}'
        f'{first_row} [data-testid="stMetricLabel"] p{{font-size:1.15rem;font-weight:600;}}'
        f'{first_row} [data-testid="stMetricDelta"]{{font-size:1.05rem;}}'
        f"{ring}"
        "</style>",
        unsafe_allow_html=True,
    )


# ── Page: Operations ────────────────────────────────────────────────────────────
def render_operations():
    st.title("🚲 Bike Sharing — Operations Dashboard")

    predictions = load_predictions()
    actuals = load_actuals()

    if predictions.empty:
        st.warning("No predictions available yet. Run predict.py first.")
        return

    # With multi-horizon serving, the latest run emits h+1..h+K from one origin.
    # The next-hour headline is that run's primary-horizon row (not
    # predictions.iloc[-1], which is now the farthest-out h+K target).
    trajectory = latest_trajectory(predictions)
    primary_rows = trajectory[trajectory["horizon"] == PRIMARY_HORIZON]
    next_row = primary_rows.iloc[0] if not primary_rows.empty else trajectory.iloc[0]
    next_hr = next_row["timestamp_predicted"]
    is_fallback = next_row["prediction_source"] == FALLBACK_SOURCE

    _inject_styles(f"Next {len(trajectory)} hours forecast", live_ring=True)

    if is_fallback:
        st.warning(
            "⚠️ The next-hour value is a **fallback** (168h-lag), not live model output — "
            "hourly data validation failed for this hour."
        )

    # ── Headline KPIs — what the operator acts on ─────────────────────────────
    peak = trajectory.loc[trajectory["pred_total"].idxmax()]
    low = trajectory.loc[trajectory["pred_total"].idxmin()]
    current_demand = float(actuals["actual_total"].iloc[-1]) if not actuals.empty else None
    current_ts = actuals["timestamp_predicted"].iloc[-1] if not actuals.empty else None

    def _vs_now(value: float) -> str | None:
        return f"{value - current_demand:+.0f} vs now" if current_demand is not None else None

    k0, k1, k2, k3 = st.columns(4)
    with k0:
        now_label = f"Now · {current_ts:%H:%M}" if current_ts is not None else "Now"
        st.metric(now_label, f"{current_demand:.0f} bikes" if current_demand is not None else "—")
    with k1:
        st.metric(
            f"Next hour · {next_hr:%H:%M}",
            f"{next_row['pred_total']:.0f} bikes",
            delta=_vs_now(next_row["pred_total"]),
        )
    with k2:
        st.metric(
            f"Next peak · {peak['timestamp_predicted']:%H:%M}",
            f"{peak['pred_total']:.0f} bikes",
            delta=_vs_now(peak["pred_total"]),
        )
    with k3:
        st.metric(
            f"Next quietest · {low['timestamp_predicted']:%H:%M}",
            f"{low['pred_total']:.0f} bikes",
            delta=_vs_now(low["pred_total"]),
        )

    st.divider()

    # ── Hourly forecast strip ─────────────────────────────────────────────────
    st.subheader("Hourly Forecast")

    cols = st.columns(len(trajectory))
    for col, (_, row) in zip(cols, trajectory.iterrows()):
        with col:
            st.markdown(
                f"<div style='text-align:center;padding:6px 0'>"
                f"<div style='font-size:0.8rem;color:gray'>{row['timestamp_predicted']:%H:%M}</div>"
                f"<div style='font-size:1.45rem;font-weight:600'>{row['pred_total']:.0f}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # Breathing room between the hourly strip and the trend chart.
    st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)

    # Smooth trend view of the same trajectory.
    fig_traj = go.Figure()
    fig_traj.add_trace(
        go.Scatter(
            x=trajectory["timestamp_predicted"],
            y=trajectory["pred_total"],
            customdata=trajectory["horizon"],
            mode="lines+markers",
            line=dict(color="#1F77B4", width=2),
            marker=dict(size=7),
            fill="tozeroy",
            fillcolor="rgba(31,119,180,0.12)",
            hovertemplate="+%{customdata}h · %{x|%a %H:%M}: %{y:.0f} bikes<extra></extra>",
        )
    )
    fig_traj.update_layout(
        xaxis_title="Hour",
        yaxis_title="Bikes",
        height=340,
        margin=dict(t=10, b=30, l=40, r=20),
        showlegend=False,
    )
    st.plotly_chart(fig_traj, use_container_width=True)


# ── Page: Monitoring ────────────────────────────────────────────────────────────
def render_monitoring():
    st.title("📈 Bike Sharing — Monitoring Dashboard")
    _inject_styles("Model health & live performance")

    # ── Model metrics — live performance at the primary horizon ───────────────
    st.subheader("Model Metrics")
    metrics_by_model = live_model_metrics(load_predictions(), load_actuals(), N_HOURS)
    if metrics_by_model is None:
        st.info("No resolved model predictions yet — metrics appear once actuals arrive.")
    else:
        # The combined row is the first metric block on the page, so the shared
        # style enlarges it (headline); registered/casual read as supporting rows.
        for model_name, m in metrics_by_model.items():
            st.markdown(f"**{model_name}**")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("RMSE", f"{m['rmse']:.1f} bikes")
            c2.metric("RMSLE", f"{m['rmsle']:.3f}")
            c3.metric("MAE", f"{m['mae']:.1f} bikes")
            c4.metric("R²", f"{m['r2']:.3f}")

    st.divider()

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
        ts = outcome.get("timestamp")
        when = pd.to_datetime(ts).strftime("%Y-%m-%d %H:%M") if ts else "unknown"
        st.caption(f"As of {when}")
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

    # ── Live performance over time (primary horizon) ──────────────────────────
    st.subheader(f"Live Performance Over Time — h+{PRIMARY_HORIZON}")
    perf = load_history_csv("performance_history.csv")
    perf_primary = filter_to_horizon(perf, PRIMARY_HORIZON) if not perf.empty else perf
    if perf_primary.empty:
        st.info("No performance history yet — accumulates weekly.")
    else:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=perf_primary["timestamp"],
                y=perf_primary["rmse"],
                mode="lines+markers",
                name="Model RMSE",
            )
        )
        if "naive_rmse" in perf_primary.columns and perf_primary["naive_rmse"].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=perf_primary["timestamp"],
                    y=perf_primary["naive_rmse"],
                    mode="lines+markers",
                    name="Seasonal-naive RMSE",
                    line=dict(dash="dot"),
                )
            )
        fig.update_layout(
            xaxis_title="Week", yaxis_title="RMSE", height=280, margin=dict(t=20, b=40, l=40, r=20)
        )
        st.plotly_chart(fig, use_container_width=True)
        if (
            "skill_vs_naive" in perf_primary.columns
            and perf_primary["skill_vs_naive"].notna().any()
        ):
            latest_skill = perf_primary["skill_vs_naive"].dropna().iloc[-1]
            st.caption(f"Latest skill vs. seasonal-naive: **{latest_skill:+.1%}**")

    st.divider()

    # ── Performance by horizon (latest) ───────────────────────────────────────
    # The multi-horizon payoff: how model accuracy and its edge over the naive
    # baseline decay as the forecast reaches further ahead.
    st.subheader("Performance by Horizon — Latest")
    curve = latest_per_horizon(perf) if not perf.empty else perf
    if curve.empty or len(curve) <= 1:
        st.info("Per-horizon curve appears once multi-horizon predictions have been scored.")
    else:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=curve["horizon"], y=curve["rmse"], mode="lines+markers", name="Model RMSE")
        )
        if "naive_rmse" in curve.columns and curve["naive_rmse"].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=curve["horizon"],
                    y=curve["naive_rmse"],
                    mode="lines+markers",
                    name="Seasonal-naive RMSE",
                    line=dict(dash="dot"),
                )
            )
        fig.update_layout(
            xaxis_title="Horizon (hours ahead)",
            yaxis_title="RMSE",
            height=280,
            margin=dict(t=20, b=40, l=40, r=20),
        )
        st.plotly_chart(fig, use_container_width=True)
        if "skill_vs_naive" in curve.columns and curve["skill_vs_naive"].notna().any():
            worst = curve.dropna(subset=["skill_vs_naive"])
            crossover = worst[worst["skill_vs_naive"] <= 0]
            if not crossover.empty:
                st.caption(
                    f"Model falls to parity with seasonal-naive by "
                    f"**h+{int(crossover['horizon'].iloc[0])}**."
                )
            else:
                st.caption("Model beats seasonal-naive across all served horizons.")

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
