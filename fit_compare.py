"""
FIT Compare — a Streamlit app to overlay data from two or more .fit files.

Useful for comparing:
  - Indoor trainer vs outdoor power meter
  - Dual-sided power meter vs single-sided
  - Crank-based vs pedal-based vs wheel-on
  - Two head units recording the same ride

Run with:
    streamlit run fit_compare.py
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from fitparse import FitFile
from plotly.subplots import make_subplots
from streamlit_folium import st_folium

# ---------------------------------------------------------------------------
# Page config & styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="FIT Compare",
    page_icon="📊",
    layout="wide",
)

# Minimal CSS polish — keeps Streamlit defaults but tightens things up
st.markdown(
    """
    <style>
      .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px; }
      h1 { font-weight: 700; letter-spacing: -0.01em; }
      .stMetric { background: #f8f9fb; border: 1px solid #e6e8ee; border-radius: 4px; padding: 10px 14px; }
      [data-testid="stMetricValue"] { font-size: 1.4rem; font-weight: 600; }
      [data-testid="stMetricDelta"] { font-size: 0.85rem; }
      .stDataFrame { border: 1px solid #e6e8ee; border-radius: 4px; }
      .caption { color: #6b7280; font-size: 0.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Colours for overlaid traces — teal + magenta match the DC Rainmaker default,
# plus two more for when people throw a third or fourth file in.
COLORS = ["#3EC5C9", "#C94FB8", "#F5A623", "#7ED957"]

# Which channels we try to plot. Order matters — power first, it's the reason
# this app exists.
CHANNELS = [
    {"key": "power",     "label": "Power",      "unit": "W",    "y_zero": True},
    {"key": "heart_rate","label": "Heart Rate", "unit": "bpm",  "y_zero": False},
    {"key": "cadence",   "label": "Cadence",    "unit": "rpm",  "y_zero": True},
    {"key": "speed",     "label": "Speed",      "unit": "km/h", "y_zero": True},
    {"key": "altitude",  "label": "Elevation",  "unit": "m",    "y_zero": False},
]

# Standard durations for the mean max power curve (in seconds).
# Log-spaced from 1s to 1h — matches the convention used by TrainingPeaks,
# Golden Cheetah, etc.
MMP_DURATIONS = [
    1, 2, 3, 5, 7, 10, 15, 20, 30, 45,
    60, 90, 120, 180, 240, 300, 420, 600,
    900, 1200, 1800, 2400, 3000, 3600,
]

# Semicircles → degrees conversion for GPS coordinates in FIT files
SEMICIRCLE_TO_DEG = 180.0 / (2 ** 31)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

@dataclass
class Ride:
    name: str
    color: str
    df: pd.DataFrame
    stats: dict[str, Any]
    sensors: list[dict[str, Any]] = field(default_factory=list)


@st.cache_data(show_spinner=False)
def parse_fit(file_bytes: bytes, filename: str) -> tuple[pd.DataFrame, list[dict]]:
    """Parse a .fit file's record and device_info messages.

    Returns (records_df, sensors_list). Cached on the raw bytes so re-runs
    (changing smoothing, toggling zeros) don't re-parse.
    """
    fit = FitFile(io.BytesIO(file_bytes))
    fit.parse()

    # --- Records ---------------------------------------------------------
    rows = []
    for record in fit.get_messages("record"):
        row: dict[str, Any] = {}
        for fld in record:
            name, val = fld.name, fld.value

            # fitparse returns speed in m/s by default — convert to km/h.
            # Prefer enhanced_* fields when present (higher precision).
            if name == "speed" and val is not None:
                row["speed"] = val * 3.6
            elif name == "enhanced_speed" and val is not None:
                row["speed"] = val * 3.6
            elif name == "enhanced_altitude" and val is not None:
                row["altitude"] = val
            elif name == "altitude" and "altitude" not in row:
                row["altitude"] = val
            # GPS coords arrive as semicircles — convert to degrees
            elif name == "position_lat" and val is not None:
                row["lat"] = val * SEMICIRCLE_TO_DEG
            elif name == "position_long" and val is not None:
                row["lon"] = val * SEMICIRCLE_TO_DEG
            else:
                row[name] = val
        if row:
            rows.append(row)

    df = pd.DataFrame(rows) if rows else pd.DataFrame()

    if not df.empty:
        # Elapsed seconds from first record — our overlay axis
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["elapsed_s"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()
        else:
            df["elapsed_s"] = range(len(df))

        # Derive left/right power from left_right_balance + total power.
        # left_right_balance in FIT files is percent right (0-100). Some meters
        # set a high bit to indicate which side the percentage refers to; we
        # strip it by masking to the low 7 bits.
        if "left_right_balance" in df.columns and "power" in df.columns:
            lrb = pd.to_numeric(df["left_right_balance"], errors="coerce")
            # Mask to 7 bits if values look like they have the right-flag bit set
            lrb = lrb.where(lrb <= 100, lrb.astype("Int64") & 0x7F)
            pct_right = lrb / 100.0
            pwr = pd.to_numeric(df["power"], errors="coerce")
            df["right_power"] = (pwr * pct_right).round()
            df["left_power"] = (pwr * (1 - pct_right)).round()

        # Ensure every channel column exists so downstream code can stop worrying
        for ch in CHANNELS:
            if ch["key"] not in df.columns:
                df[ch["key"]] = None

    # --- Device info (sensors) ------------------------------------------
    sensors = []
    seen = set()
    for dev in fit.get_messages("device_info"):
        info: dict[str, Any] = {}
        for fld in dev:
            info[fld.name] = fld.value
        # De-dupe by (manufacturer, product, serial_number) — devices announce
        # themselves multiple times over a ride.
        key = (
            str(info.get("manufacturer", "")),
            str(info.get("product", "")),
            str(info.get("serial_number", "")),
            str(info.get("device_type", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        sensors.append(info)

    return df, sensors


def compute_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Compute summary stats for a single ride."""
    if df.empty:
        return {}

    def safe_mean(series, exclude_zero=False):
        s = pd.to_numeric(series, errors="coerce").dropna()
        if exclude_zero:
            s = s[s > 0]
        return float(s.mean()) if len(s) else None

    def safe_max(series):
        s = pd.to_numeric(series, errors="coerce").dropna()
        return float(s.max()) if len(s) else None

    power = pd.to_numeric(df.get("power"), errors="coerce").dropna()

    # Normalised Power approximation — 30s rolling mean, ^4, mean, ^0.25.
    # Matches TrainingPeaks' NP closely enough for comparison purposes.
    np_value = None
    if len(power) >= 30:
        smoothed = power.rolling(window=30, min_periods=30).mean().dropna()
        np_value = float((smoothed ** 4).mean() ** 0.25)

    # Total ascent — sum of positive elevation changes
    alt = pd.to_numeric(df.get("altitude"), errors="coerce").dropna()
    ascent = float(alt.diff().clip(lower=0).sum()) if len(alt) > 1 else None

    # Distance — take the last non-null value
    dist = pd.to_numeric(df.get("distance"), errors="coerce").dropna()
    distance_km = float(dist.iloc[-1] / 1000) if len(dist) else None

    duration = float(df["elapsed_s"].iloc[-1]) if "elapsed_s" in df.columns else None

    return {
        "duration_s": duration,
        "avg_power": safe_mean(df.get("power")),
        "np": np_value,
        "max_power": safe_max(df.get("power")),
        "avg_hr": safe_mean(df.get("heart_rate")),
        "max_hr": safe_max(df.get("heart_rate")),
        "avg_cadence": safe_mean(df.get("cadence"), exclude_zero=True),
        "avg_speed": safe_mean(df.get("speed"), exclude_zero=True),
        "distance_km": distance_km,
        "ascent": ascent,
        "avg_left_power": safe_mean(df.get("left_power"), exclude_zero=True),
        "avg_right_power": safe_mean(df.get("right_power"), exclude_zero=True),
    }


def mean_max_power(power_series: pd.Series, durations: list[int]) -> dict[int, float]:
    """Compute mean max power for each duration (seconds).

    For each duration d, finds the highest d-second rolling average in the ride.
    This is the standard "power curve" used by TrainingPeaks, Golden Cheetah, etc.
    """
    power = pd.to_numeric(power_series, errors="coerce").fillna(0)
    total_len = len(power)
    results = {}
    for d in durations:
        if d > total_len:
            break
        rolling = power.rolling(window=d, min_periods=d).mean()
        if rolling.notna().any():
            results[d] = float(rolling.max())
    return results


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_duration(seconds: float | None) -> str:
    if seconds is None or pd.isna(seconds):
        return "—"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"


def fmt_mmp_label(seconds: int) -> str:
    """Format a duration for MMP axis labels: 1s, 15s, 1m, 5m, 20m, 1h."""
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        m = seconds // 60
        s = seconds % 60
        return f"{m}m" if s == 0 else f"{m}m{s}s"
    h = seconds // 3600
    m = (seconds % 3600) // 60
    return f"{h}h" if m == 0 else f"{h}h{m}m"


def fmt(value: float | None, unit: str = "", dp: int = 0) -> str:
    if value is None or pd.isna(value):
        return "—"
    return f"{value:,.{dp}f} {unit}".strip()


# ---------------------------------------------------------------------------
# UI — header & uploader
# ---------------------------------------------------------------------------

st.title("FIT Compare")
st.markdown(
    '<p class="caption">Overlay two or more .fit files to compare power meters, trainers, '
    "or head units. Files are parsed in memory — nothing is stored.</p>",
    unsafe_allow_html=True,
)

uploaded = st.file_uploader(
    "Drop .fit files here",
    type=["fit"],
    accept_multiple_files=True,
    help="Upload 2 or more .fit files. They'll be aligned by elapsed time from the first record.",
)

# Sidebar: options that affect display only (no re-parse needed)
with st.sidebar:
    st.header("Options")
    smoothing = st.slider(
        "Smoothing (seconds)",
        min_value=0, max_value=30, value=3, step=1,
        help="Rolling mean applied to all traces. 0 = raw data.",
    )
    show_zeros = st.checkbox(
        "Show zero values",
        value=True,
        help="When off, zero power/cadence points are hidden (useful to mask coasting dropouts).",
    )
    st.markdown("---")
    st.caption(
        "Built for comparing indoor trainers against outdoor power meters, "
        "dual-sided vs single-sided meters, and similar validation work."
    )

if not uploaded:
    st.info("Upload at least two .fit files to begin.")
    st.stop()

if len(uploaded) < 2:
    st.warning("Add at least one more file to create an overlay.")
    st.stop()

# ---------------------------------------------------------------------------
# Parse files
# ---------------------------------------------------------------------------

rides: list[Ride] = []
with st.spinner(f"Parsing {len(uploaded)} files…"):
    for i, uf in enumerate(uploaded):
        try:
            df, sensors = parse_fit(uf.getvalue(), uf.name)
            if df.empty:
                st.warning(f"`{uf.name}` contains no record messages — skipping.")
                continue
            stats = compute_stats(df)
            rides.append(Ride(
                name=uf.name,
                color=COLORS[i % len(COLORS)],
                df=df,
                stats=stats,
                sensors=sensors,
            ))
        except Exception as e:
            st.error(f"Couldn't parse `{uf.name}`: {e}")

if len(rides) < 2:
    st.error("Need at least 2 successfully parsed files.")
    st.stop()

# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------

st.subheader("Summary")

# Build a wide stats table — one row per ride
stats_rows = []
for r in rides:
    s = r.stats
    stats_rows.append({
        "File": r.name,
        "Duration": fmt_duration(s.get("duration_s")),
        "Avg Power": fmt(s.get("avg_power"), "W", 0),
        "NP*": fmt(s.get("np"), "W", 0),
        "Max Power": fmt(s.get("max_power"), "W", 0),
        "Avg HR": fmt(s.get("avg_hr"), "bpm", 0),
        "Avg Cadence": fmt(s.get("avg_cadence"), "rpm", 0),
        "Avg Speed": fmt(s.get("avg_speed"), "km/h", 1),
        "Distance": fmt(s.get("distance_km"), "km", 2),
        "Ascent": fmt(s.get("ascent"), "m", 0),
    })

stats_df = pd.DataFrame(stats_rows)
st.dataframe(stats_df, hide_index=True, use_container_width=True)

# If exactly two files, show a delta panel — the key "is my trainer reading
# high vs my power meter" question.
if len(rides) == 2:
    a, b = rides[0], rides[1]
    st.markdown(f"**Δ — `{b.name}` vs `{a.name}`**")

    cols = st.columns(5)
    metrics = [
        ("Avg Power", "avg_power", "W", 0),
        ("NP", "np", "W", 0),
        ("Max Power", "max_power", "W", 0),
        ("Avg HR", "avg_hr", "bpm", 0),
        ("Avg Cadence", "avg_cadence", "rpm", 0),
    ]
    for col, (label, key, unit, dp) in zip(cols, metrics):
        av = a.stats.get(key)
        bv = b.stats.get(key)
        if av is None or bv is None:
            col.metric(label, "—")
            continue
        delta = bv - av
        pct = (delta / av * 100) if av else 0
        col.metric(
            label,
            f"{bv:,.{dp}f} {unit}",
            delta=f"{delta:+.{dp}f} {unit} ({pct:+.1f}%)",
            delta_color="normal",
        )

st.caption("\\* NP is a 30s-rolling-mean approximation of Normalised Power — close to TrainingPeaks but not identical.")

# ---------------------------------------------------------------------------
# Overlay charts
# ---------------------------------------------------------------------------

st.subheader("Overlay")

# Only plot channels that at least one ride has real data for
active_channels = []
for ch in CHANNELS:
    has_data = any(
        pd.to_numeric(r.df[ch["key"]], errors="coerce").dropna().shape[0] > 0
        for r in rides
    )
    if has_data:
        active_channels.append(ch)

if not active_channels:
    st.warning("No plottable channels found in these files.")
    st.stop()


def prepare_series(df: pd.DataFrame, key: str, smooth_s: int, show_zeros: bool) -> pd.Series:
    """Pull a channel out of a ride, optionally drop zeros, optionally smooth."""
    s = pd.to_numeric(df[key], errors="coerce")
    if not show_zeros and key in {"power", "cadence"}:
        s = s.where(s != 0)
    if smooth_s > 0:
        # Rolling window in seconds — FIT records are typically 1Hz but not always,
        # so we use the smooth_s as a window size in samples and assume ~1Hz.
        s = s.rolling(window=smooth_s, min_periods=1, center=True).mean()
    return s


# Build one tall figure with a subplot per channel — lets users compare
# power and HR at the same instant by eye, which is the whole point.
fig = make_subplots(
    rows=len(active_channels),
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.035,
    subplot_titles=[f"{ch['label']} ({ch['unit']})" for ch in active_channels],
)

for row_idx, ch in enumerate(active_channels, start=1):
    for r in rides:
        y = prepare_series(r.df, ch["key"], smoothing, show_zeros)
        x = r.df["elapsed_s"]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=r.name,
                legendgroup=r.name,
                showlegend=(row_idx == 1),  # one legend entry per ride, not per channel
                line=dict(color=r.color, width=1.2),
                hovertemplate=(
                    f"<b>{r.name}</b><br>"
                    "t=%{x:.0f}s<br>"
                    f"{ch['label']}=%{{y:.1f}} {ch['unit']}<extra></extra>"
                ),
            ),
            row=row_idx, col=1,
        )
    fig.update_yaxes(
        title_text=ch["unit"],
        row=row_idx, col=1,
        gridcolor="#eef0f4",
        rangemode="tozero" if ch["y_zero"] else "normal",
    )

# Format x-axis ticks as mm:ss / h:mm
max_t = max(r.df["elapsed_s"].max() for r in rides)
if max_t > 3600:
    tick_vals = list(range(0, int(max_t) + 1, 600))  # every 10 min
    tick_text = [f"{v // 3600}:{(v % 3600) // 60:02d}" for v in tick_vals]
else:
    tick_vals = list(range(0, int(max_t) + 1, 120))  # every 2 min
    tick_text = [f"{v // 60}:{v % 60:02d}" for v in tick_vals]

fig.update_xaxes(
    title_text="Elapsed time",
    row=len(active_channels), col=1,
    tickvals=tick_vals,
    ticktext=tick_text,
    gridcolor="#eef0f4",
)
for i in range(1, len(active_channels)):
    fig.update_xaxes(tickvals=tick_vals, ticktext=tick_text, row=i, col=1, gridcolor="#eef0f4")

fig.update_layout(
    height=220 * len(active_channels) + 60,
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    margin=dict(l=60, r=20, t=60, b=40),
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(family="system-ui, -apple-system, sans-serif", size=12),
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Mean Max Power curve
# ---------------------------------------------------------------------------

has_power = any(
    pd.to_numeric(r.df.get("power"), errors="coerce").dropna().shape[0] > 0
    for r in rides
)

if has_power:
    st.subheader("Mean Max Power")
    st.caption(
        "Best rolling-average power for each duration. A systematic gap at short durations "
        "but not long ones (or vice versa) usually points to different smoothing algorithms "
        "between the two meters."
    )

    mmp_fig = go.Figure()
    mmp_data = {}  # for showing the delta below
    for r in rides:
        mmp = mean_max_power(r.df["power"], MMP_DURATIONS)
        mmp_data[r.name] = mmp
        if not mmp:
            continue
        xs = list(mmp.keys())
        ys = list(mmp.values())
        mmp_fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode="lines+markers",
            name=r.name,
            line=dict(color=r.color, width=2),
            marker=dict(size=5),
            hovertemplate=(
                f"<b>{r.name}</b><br>"
                "duration=%{customdata}<br>"
                "power=%{y:.0f} W<extra></extra>"
            ),
            customdata=[fmt_mmp_label(x) for x in xs],
        ))

    # Tick marks at the canonical durations
    tick_durations = [1, 5, 15, 60, 300, 1200, 3600]
    mmp_fig.update_layout(
        height=420,
        xaxis=dict(
            type="log",
            title="Duration",
            tickvals=tick_durations,
            ticktext=[fmt_mmp_label(d) for d in tick_durations],
            gridcolor="#eef0f4",
        ),
        yaxis=dict(title="Power (W)", gridcolor="#eef0f4", rangemode="tozero"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=60, r=20, t=40, b=40),
        font=dict(family="system-ui, -apple-system, sans-serif", size=12),
    )
    st.plotly_chart(mmp_fig, use_container_width=True)

    # If exactly two files, show the delta at a few canonical durations
    if len(rides) == 2 and mmp_data[rides[0].name] and mmp_data[rides[1].name]:
        a_mmp = mmp_data[rides[0].name]
        b_mmp = mmp_data[rides[1].name]
        st.markdown("**MMP Δ at key durations**")
        highlight_durations = [5, 60, 300, 1200]  # 5s, 1m, 5m, 20m
        cols = st.columns(len(highlight_durations))
        for col, d in zip(cols, highlight_durations):
            if d in a_mmp and d in b_mmp:
                av, bv = a_mmp[d], b_mmp[d]
                delta = bv - av
                pct = (delta / av * 100) if av else 0
                col.metric(
                    fmt_mmp_label(d),
                    f"{bv:.0f} W",
                    delta=f"{delta:+.0f} W ({pct:+.1f}%)",
                )
            else:
                col.metric(fmt_mmp_label(d), "—")

# ---------------------------------------------------------------------------
# Left/Right power balance
# ---------------------------------------------------------------------------

has_lr = any(
    "left_power" in r.df.columns
    and pd.to_numeric(r.df["left_power"], errors="coerce").dropna().shape[0] > 0
    for r in rides
)

if has_lr:
    st.subheader("Left / Right Power")
    st.caption(
        "Dual-sided power meter data. Gaps here usually mean one of the files doesn't "
        "have dual-sided data (single-sided meters report 50/50 by convention)."
    )

    # One plot per ride so the left/right traces don't all collide
    lr_rides = [r for r in rides if
                "left_power" in r.df.columns
                and pd.to_numeric(r.df["left_power"], errors="coerce").dropna().shape[0] > 0]

    lr_fig = make_subplots(
        rows=len(lr_rides),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=[r.name for r in lr_rides],
    )

    for row_idx, r in enumerate(lr_rides, start=1):
        for side_key, side_label, dash in [("left_power", "Left", "solid"),
                                           ("right_power", "Right", "dash")]:
            y = prepare_series(r.df, side_key, smoothing, show_zeros=False)
            lr_fig.add_trace(
                go.Scatter(
                    x=r.df["elapsed_s"],
                    y=y,
                    mode="lines",
                    name=f"{r.name} — {side_label}",
                    line=dict(color=r.color, width=1.2, dash=dash),
                    showlegend=True,
                    legendgroup=f"{r.name}-{side_label}",
                    hovertemplate=(
                        f"<b>{r.name} {side_label}</b><br>"
                        "t=%{x:.0f}s<br>"
                        "%{y:.0f} W<extra></extra>"
                    ),
                ),
                row=row_idx, col=1,
            )
        lr_fig.update_yaxes(
            title_text="W", row=row_idx, col=1,
            gridcolor="#eef0f4", rangemode="tozero",
        )

    lr_fig.update_xaxes(
        title_text="Elapsed time",
        row=len(lr_rides), col=1,
        tickvals=tick_vals, ticktext=tick_text,
        gridcolor="#eef0f4",
    )
    for i in range(1, len(lr_rides)):
        lr_fig.update_xaxes(tickvals=tick_vals, ticktext=tick_text, row=i, col=1,
                            gridcolor="#eef0f4")

    lr_fig.update_layout(
        height=220 * len(lr_rides) + 60,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=60, r=20, t=60, b=40),
        font=dict(family="system-ui, -apple-system, sans-serif", size=12),
    )
    st.plotly_chart(lr_fig, use_container_width=True)

    # Balance summary per ride
    cols = st.columns(len(lr_rides))
    for col, r in zip(cols, lr_rides):
        left = r.stats.get("avg_left_power")
        right = r.stats.get("avg_right_power")
        if left is None or right is None or (left + right) == 0:
            col.metric(r.name.replace(".fit", ""), "—")
            continue
        total = left + right
        left_pct = left / total * 100
        right_pct = right / total * 100
        col.metric(
            r.name.replace(".fit", ""),
            f"{left_pct:.0f}% / {right_pct:.0f}%",
            delta=f"L {left:.0f}W · R {right:.0f}W",
            delta_color="off",
        )

# ---------------------------------------------------------------------------
# Power vs Power scatter (only meaningful with exactly two rides)
# ---------------------------------------------------------------------------

if len(rides) == 2:
    a, b = rides[0], rides[1]
    a_pwr = pd.to_numeric(a.df["power"], errors="coerce")
    b_pwr = pd.to_numeric(b.df["power"], errors="coerce")

    # Align by elapsed second — merge_asof handles slightly different record timings
    a_aligned = pd.DataFrame({"t": a.df["elapsed_s"], "a": a_pwr}).dropna()
    b_aligned = pd.DataFrame({"t": b.df["elapsed_s"], "b": b_pwr}).dropna()

    if len(a_aligned) > 10 and len(b_aligned) > 10:
        st.subheader("Power vs Power")
        st.caption(
            "Each point is one matched second. A perfect-agreement line is shown "
            "for reference — points above the line mean the Y-axis meter reads higher."
        )

        merged = pd.merge_asof(
            a_aligned.sort_values("t"),
            b_aligned.sort_values("t"),
            on="t",
            direction="nearest",
            tolerance=2,
        ).dropna()

        if len(merged) > 10:
            max_p = max(merged["a"].max(), merged["b"].max()) * 1.05

            scatter_fig = go.Figure()
            scatter_fig.add_trace(go.Scatter(
                x=merged["a"],
                y=merged["b"],
                mode="markers",
                marker=dict(size=3, color="#3EC5C9", opacity=0.35),
                name="Matched seconds",
                hovertemplate=(
                    f"{a.name}: %{{x:.0f}} W<br>"
                    f"{b.name}: %{{y:.0f}} W<extra></extra>"
                ),
            ))
            scatter_fig.add_trace(go.Scatter(
                x=[0, max_p],
                y=[0, max_p],
                mode="lines",
                line=dict(color="#999", dash="dash", width=1),
                name="Perfect agreement",
                hoverinfo="skip",
            ))

            # Linear regression through origin — slope ~= systematic offset
            x_vals = merged["a"].values
            y_vals = merged["b"].values
            mask = (x_vals > 0) & (y_vals > 0)
            if mask.sum() > 10:
                slope = (x_vals[mask] * y_vals[mask]).sum() / (x_vals[mask] ** 2).sum()
                scatter_fig.add_trace(go.Scatter(
                    x=[0, max_p],
                    y=[0, slope * max_p],
                    mode="lines",
                    line=dict(color="#C94FB8", width=1.5),
                    name=f"Best fit (y = {slope:.3f}·x)",
                    hoverinfo="skip",
                ))
                offset_pct = (slope - 1) * 100
                st.metric(
                    f"`{b.name}` reads…",
                    f"{offset_pct:+.1f}% vs `{a.name}`",
                    help="Slope of the best-fit line through the origin. Positive = Y-axis meter reads higher.",
                )

            scatter_fig.update_layout(
                height=500,
                xaxis_title=f"{a.name} (W)",
                yaxis_title=f"{b.name} (W)",
                xaxis=dict(range=[0, max_p], gridcolor="#eef0f4"),
                yaxis=dict(range=[0, max_p], gridcolor="#eef0f4", scaleanchor="x", scaleratio=1),
                plot_bgcolor="white",
                paper_bgcolor="white",
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                margin=dict(l=60, r=20, t=40, b=40),
            )
            st.plotly_chart(scatter_fig, use_container_width=True)

# ---------------------------------------------------------------------------
# GPS map
# ---------------------------------------------------------------------------

has_gps = any(
    "lat" in r.df.columns
    and pd.to_numeric(r.df["lat"], errors="coerce").dropna().shape[0] > 0
    for r in rides
)

if has_gps:
    st.subheader("GPS")
    st.caption("Overlay of each ride's track. Useful for checking both files are the same ride.")

    # Collect valid GPS points for each ride
    tracks = []
    all_points = []
    for r in rides:
        if "lat" not in r.df.columns:
            continue
        gps = r.df[["lat", "lon"]].dropna()
        gps = gps[(gps["lat"].between(-90, 90)) & (gps["lon"].between(-180, 180))]
        if len(gps) < 2:
            continue
        points = list(zip(gps["lat"].tolist(), gps["lon"].tolist()))
        tracks.append((r, points))
        all_points.extend(points)

    if all_points:
        # Center the map on the mean of all points
        avg_lat = np.mean([p[0] for p in all_points])
        avg_lon = np.mean([p[1] for p in all_points])
        m = folium.Map(
            location=[avg_lat, avg_lon],
            zoom_start=13,
            tiles="OpenStreetMap",
        )
        # Fit bounds to all tracks
        lats = [p[0] for p in all_points]
        lons = [p[1] for p in all_points]
        m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])

        for r, points in tracks:
            folium.PolyLine(
                points,
                color=r.color,
                weight=3,
                opacity=0.85,
                tooltip=r.name,
            ).add_to(m)
            # Start marker
            folium.CircleMarker(
                points[0], radius=6, color=r.color,
                fill=True, fill_color=r.color, fill_opacity=1,
                tooltip=f"{r.name} · start",
            ).add_to(m)

        st_folium(m, width=None, height=450, returned_objects=[])

# ---------------------------------------------------------------------------
# Sensors table
# ---------------------------------------------------------------------------

has_sensors = any(r.sensors for r in rides)

if has_sensors:
    st.subheader("Sensors")
    st.caption("Devices that contributed data to each file (power meters, heart-rate straps, head units).")

    sensor_rows = []
    for r in rides:
        for dev in r.sensors:
            manufacturer = dev.get("manufacturer")
            device_type = dev.get("device_type")
            product = dev.get("product") or dev.get("product_name")
            # Skip entries with no useful identifying info
            if not any([manufacturer, product, device_type]):
                continue
            sensor_rows.append({
                "File": r.name,
                "Device Type": str(device_type) if device_type else "—",
                "Manufacturer": str(manufacturer) if manufacturer else "—",
                "Product": str(product) if product else "—",
                "Serial": str(dev.get("serial_number", "—")),
                "SW Version": str(dev.get("software_version", "—")),
            })

    if sensor_rows:
        sensors_df = pd.DataFrame(sensor_rows)
        st.dataframe(sensors_df, hide_index=True, use_container_width=True)
    else:
        st.caption("No named devices found in either file.")
