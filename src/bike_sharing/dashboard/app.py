import json
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import hydra
from pathlib import Path

from bike_sharing.utils.datetime_utils import reconstruct_datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bike Sharing — Operations Dashboard",
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


# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def load_predictions() -> pd.DataFrame:
    if not PREDICTIONS.exists():
        return pd.DataFrame()
    df = pd.read_csv(PREDICTIONS)
    df["timestamp_predicted"] = pd.to_datetime(df["timestamp_predicted"], format="ISO8601")
    df["predicted_at"] = pd.to_datetime(df["predicted_at"], format="ISO8601")
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


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
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

    st.subheader(f"Next Hour Forecast — {next_hr.strftime('%A %b %d, %H:%M')}")

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        # Gauge
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=pred_total,
                title={"text": "Predicted Bikes Needed", "font": {"size": 18}},
                gauge={
                    "axis": {"range": [0, 900]},
                    "bar": {"color": "#1F77B4"},
                    "steps": [
                        {"range": [0, 200], "color": "#EAF4FB"},
                        {"range": [200, 500], "color": "#AED6F1"},
                        {"range": [500, 900], "color": "#2E86C1"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 3},
                        "thickness": 0.75,
                        "value": 700,
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

    fig_line = go.Figure()
    fig_line.add_trace(
        go.Scatter(
            x=last_24_with_actuals["timestamp_predicted"],
            y=last_24_with_actuals["pred_total"],
            mode="lines+markers",
            name="Predicted",
            line=dict(color="#1F77B4", width=2),
            marker=dict(size=6),
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
    table = table[
        [
            "timestamp_predicted",
            "pred_total",
            "pred_registered",
            "pred_casual",
            "temp",
            "hum",
            "weathersit",
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


if __name__ == "__main__":
    main()
