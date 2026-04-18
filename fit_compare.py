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
from datetime import datetime
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
# Constants & styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="FIT Compare",
    page_icon="📊",
    layout="wide",
)

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

# Ride colours — teal + magenta match the DC Rainmaker default, plus two more
# for when people throw a third or fourth file in.
COLORS = ["#3EC5C9", "#C94FB8", "#F5A623", "#7ED957"]

# Left/Right power colours — deliberately different from the ride colours so
# they read as a pair. Blue = left leg, red = right leg, irrespective of ride.
LR_COLORS = {"left": "#2563eb", "right": "#dc2626"}

# Chart text colour — darker grey to match Streamlit's body text (#31333F)
# rather than Plotly's default washed-out blue-grey.
CHART_TEXT = "#31333F"
CHART_TITLE_FONT = dict(family="system-ui, -apple-system, sans-serif",
                        size=14, color=CHART_TEXT, weight="bold")
CHART_AXIS_FONT = dict(family="system-ui, -apple-system, sans-serif",
                       size=12, color=CHART_TEXT)

# Main overlay channels
CHANNELS = [
    {"key": "power",     "label": "Power",      "unit": "W",    "y_zero": True},
    {"key": "heart_rate","label": "Heart Rate", "unit": "bpm",  "y_zero": False},
    {"key": "cadence",   "label": "Cadence",    "unit": "rpm",  "y_zero": True},
    {"key": "speed",     "label": "Speed",      "unit": "km/h", "y_zero": True},
    {"key": "altitude",  "label": "Elevation",  "unit": "m",    "y_zero": False},
]

# Standard durations (in seconds) for the mean max power curve
MMP_DURATIONS = [
    1, 2, 3, 5, 7, 10, 15, 20, 30, 45,
    60, 90, 120, 180, 240, 300, 420, 600,
    900, 1200, 1800, 2400, 3000, 3600,
]

# Coggan power zones — % of FTP lower bound, zone label
# Z7 has no upper bound (neuromuscular / sprint)
COGGAN_ZONES = [
    (0.00, "Z1 · Active Recovery",    "#94a3b8"),
    (0.55, "Z2 · Endurance",          "#60a5fa"),
    (0.75, "Z3 · Tempo",              "#34d399"),
    (0.90, "Z4 · Threshold",          "#fbbf24"),
    (1.05, "Z5 · VO2 Max",            "#fb923c"),
    (1.20, "Z6 · Anaerobic",          "#f87171"),
    (1.50, "Z7 · Neuromuscular",      "#c084fc"),
]

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
    sample_rate_hz: float = 1.0  # records per second — usually 1, sometimes 0.25/0.5


@st.cache_data(show_spinner=False)
def parse_fit(file_bytes: bytes, filename: str) -> tuple[pd.DataFrame, list[dict], float]:
    """Parse a .fit file's record and device_info messages.

    Returns (records_df, sensors_list, sample_rate_hz). Cached on the raw bytes
    so re-runs (changing smoothing, toggling zeros) don't re-parse.
    """
    fit = FitFile(io.BytesIO(file_bytes))
    fit.parse()

    # --- Records ---------------------------------------------------------
    rows = []
    for record in fit.get_messages("record"):
        row: dict[str, Any] = {}
        for fld in record:
            name, val = fld.name, fld.value
            # fitparse returns speed in m/s — convert to km/h. Prefer enhanced_*.
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
    sample_rate = 1.0

    if not df.empty:
        # Elapsed seconds + sample-rate detection.
        # Some older devices log every 2s or 4s ("smart recording"); we detect
        # from the median gap between records so downstream calculations
        # (MMP windows, smoothing, NP's 30s window) use the right sample count.
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["elapsed_s"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()
            gaps = df["timestamp"].diff().dt.total_seconds().dropna()
            if len(gaps):
                median_gap = float(gaps.median())
                if median_gap > 0:
                    sample_rate = 1.0 / median_gap
        else:
            df["elapsed_s"] = range(len(df))

        # Derive left/right power from left_right_balance + total power.
        # left_right_balance = percent right (0-100). Some meters set a high
        # bit to flag which side the percentage refers to — mask to 7 bits.
        if "left_right_balance" in df.columns and "power" in df.columns:
            lrb = pd.to_numeric(df["left_right_balance"], errors="coerce")
            lrb = lrb.where(lrb <= 100, lrb.astype("Int64") & 0x7F)
            pct_right = lrb / 100.0
            pwr = pd.to_numeric(df["power"], errors="coerce")
            df["right_power"] = (pwr * pct_right).round()
            df["left_power"] = (pwr * (1 - pct_right)).round()

        for ch in CHANNELS:
            if ch["key"] not in df.columns:
                df[ch["key"]] = None

    # --- Device info ----------------------------------------------------
    sensors = []
    seen = set()
    for dev in fit.get_messages("device_info"):
        info: dict[str, Any] = {}
        for fld in dev:
            info[fld.name] = fld.value
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

    return df, sensors, sample_rate


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def seconds_to_samples(seconds: int, sample_rate_hz: float) -> int:
    """Convert a time window in seconds to the equivalent number of samples."""
    return max(1, round(seconds * sample_rate_hz))


def compute_stats(df: pd.DataFrame, sample_rate_hz: float) -> dict[str, Any]:
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

    # --- Normalised Power (Coggan's spec) -------------------------------
    # 1. Fill missing power with 0 (coasting counts)
    # 2. 30-second moving average
    # 3. Raise each to the 4th power, take the mean
    # 4. Take the 4th root of that mean
    power_filled = pd.to_numeric(df.get("power"), errors="coerce").fillna(0)
    power_nonnull = pd.to_numeric(df.get("power"), errors="coerce").dropna()
    np_value = None
    if len(power_nonnull) >= 30:
        window_samples = seconds_to_samples(30, sample_rate_hz)
        moving_avg = power_filled.rolling(window=window_samples, min_periods=1).mean()
        # Drop the first window so the moving average has "warmed up" — matches TP/WKO
        moving_avg = moving_avg.iloc[window_samples - 1:]
        np_value = float((moving_avg ** 4).mean() ** 0.25)

    # --- Total work in kJ -----------------------------------------------
    # Power (W) × time (s) ÷ 1000 = kJ. At non-1Hz sample rates, each sample
    # represents 1/sample_rate_hz seconds.
    total_kj = None
    if len(power_nonnull):
        seconds_per_sample = 1.0 / sample_rate_hz
        total_kj = float(power_filled.sum() * seconds_per_sample / 1000)

    alt = pd.to_numeric(df.get("altitude"), errors="coerce").dropna()
    ascent = float(alt.diff().clip(lower=0).sum()) if len(alt) > 1 else None

    dist = pd.to_numeric(df.get("distance"), errors="coerce").dropna()
    distance_km = float(dist.iloc[-1] / 1000) if len(dist) else None

    duration = float(df["elapsed_s"].iloc[-1]) if "elapsed_s" in df.columns else None

    return {
        "duration_s": duration,
        "avg_power": safe_mean(df.get("power")),
        "np": np_value,
        "max_power": safe_max(df.get("power")),
        "total_kj": total_kj,
        "avg_hr": safe_mean(df.get("heart_rate")),
        "max_hr": safe_max(df.get("heart_rate")),
        "avg_cadence": safe_mean(df.get("cadence"), exclude_zero=True),
        "avg_speed": safe_mean(df.get("speed"), exclude_zero=True),
        "distance_km": distance_km,
        "ascent": ascent,
        "avg_left_power": safe_mean(df.get("left_power"), exclude_zero=True),
        "avg_right_power": safe_mean(df.get("right_power"), exclude_zero=True),
    }


def mean_max_power(power_series: pd.Series, durations: list[int],
                   sample_rate_hz: float) -> dict[int, float]:
    """Compute mean max power for each duration (in seconds)."""
    power = pd.to_numeric(power_series, errors="coerce").fillna(0)
    total_len = len(power)
    results = {}
    for d_sec in durations:
        window = seconds_to_samples(d_sec, sample_rate_hz)
        if window > total_len:
            break
        rolling = power.rolling(window=window, min_periods=window).mean()
        if rolling.notna().any():
            results[d_sec] = float(rolling.max())
    return results


def time_in_zones(power_series: pd.Series, ftp: float,
                  sample_rate_hz: float) -> list[tuple[str, float, float, str]]:
    """Bin power samples into Coggan zones. Returns (label, seconds, pct, color) list."""
    power = pd.to_numeric(power_series, errors="coerce").fillna(0)
    if not len(power):
        return []

    seconds_per_sample = 1.0 / sample_rate_hz
    thresholds_w = [ftp * z[0] for z in COGGAN_ZONES]

    zone_seconds = []
    for i, (_, label, color) in enumerate(COGGAN_ZONES):
        lower = thresholds_w[i]
        upper = thresholds_w[i + 1] if i + 1 < len(COGGAN_ZONES) else float("inf")
        count = ((power >= lower) & (power < upper)).sum()
        zone_seconds.append((label, count * seconds_per_sample, color))

    total = sum(s for _, s, _ in zone_seconds)
    if not total:
        return [(lab, 0, 0, c) for lab, _, c in zone_seconds]
    return [(lab, s, s / total * 100, c) for lab, s, c in zone_seconds]


def cumulative_work(power_series: pd.Series,
                    sample_rate_hz: float) -> pd.Series:
    """Running total of work in kJ — power × time accumulated across the ride."""
    power = pd.to_numeric(power_series, errors="coerce").fillna(0)
    seconds_per_sample = 1.0 / sample_rate_hz
    return power.cumsum() * seconds_per_sample / 1000


# ---------------------------------------------------------------------------
# Formatting
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


def style_chart(fig: go.Figure) -> go.Figure:
    """Apply consistent styling to every Plotly figure in the app."""
    fig.update_layout(
        font=dict(family="system-ui, -apple-system, sans-serif", size=12,
                  color=CHART_TEXT),
        title_font=CHART_TITLE_FONT,
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    for ann in fig.layout.annotations or []:
        ann.font = dict(family="system-ui, -apple-system, sans-serif",
                        size=13, color=CHART_TEXT)
    for axis_name in list(fig.layout):
        if axis_name.startswith(("xaxis", "yaxis")):
            axis = fig.layout[axis_name]
            axis.title.font = CHART_AXIS_FONT
            axis.tickfont = CHART_AXIS_FONT
    return fig


# ---------------------------------------------------------------------------
# UI — header & inputs
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
    st.subheader("Power zones")
    ftp = st.number_input(
        "FTP (W)",
        min_value=50, max_value=600, value=250, step=5,
        help="Used for Coggan zone boundaries (Z1 to Z7).",
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
            df, sensors, sample_rate = parse_fit(uf.getvalue(), uf.name)
            if df.empty:
                st.warning(f"`{uf.name}` contains no record messages — skipping.")
                continue
            stats = compute_stats(df, sample_rate)
            rides.append(Ride(
                name=uf.name,
                color=COLORS[i % len(COLORS)],
                df=df,
                stats=stats,
                sensors=sensors,
                sample_rate_hz=sample_rate,
            ))
        except Exception as e:
            st.error(f"Couldn't parse `{uf.name}`: {e}")

if len(rides) < 2:
    st.error("Need at least 2 successfully parsed files.")
    st.stop()

# Flag any non-1Hz sample rates so results make sense to the user
unusual_rates = [(r.name, r.sample_rate_hz) for r in rides if abs(r.sample_rate_hz - 1.0) > 0.1]
if unusual_rates:
    msgs = ", ".join(f"`{n}` at {rate:.2f} Hz" for n, rate in unusual_rates)
    st.warning(
        f"Non-1Hz sample rate detected: {msgs}. Calculations have been adjusted — "
        "but be aware that sparse sampling reduces the accuracy of short MMP durations."
    )

# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------

st.subheader("Summary")

stats_rows = []
for r in rides:
    s = r.stats
    stats_rows.append({
        "File": r.name,
        "Duration": fmt_duration(s.get("duration_s")),
        "Avg Power": fmt(s.get("avg_power"), "W", 0),
        "NP": fmt(s.get("np"), "W", 0),
        "Max Power": fmt(s.get("max_power"), "W", 0),
        "Work": fmt(s.get("total_kj"), "kJ", 0),
        "Avg HR": fmt(s.get("avg_hr"), "bpm", 0),
        "Avg Cadence": fmt(s.get("avg_cadence"), "rpm", 0),
        "Avg Speed": fmt(s.get("avg_speed"), "km/h", 1),
        "Distance": fmt(s.get("distance_km"), "km", 2),
        "Ascent": fmt(s.get("ascent"), "m", 0),
    })

stats_df = pd.DataFrame(stats_rows)
st.dataframe(stats_df, hide_index=True, use_container_width=True)

# Delta metrics for two-file case
if len(rides) == 2:
    a, b = rides[0], rides[1]
    st.markdown(f"**Δ — `{b.name}` vs `{a.name}`**")

    cols = st.columns(6)
    metrics = [
        ("Avg Power", "avg_power", "W", 0),
        ("NP", "np", "W", 0),
        ("Max Power", "max_power", "W", 0),
        ("Work", "total_kj", "kJ", 0),
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

st.caption(
    "NP (Normalised Power) uses Coggan's standard formula: 30-second moving average, "
    "raised to the fourth power, averaged, then fourth-root."
)

# ---------------------------------------------------------------------------
# Helper for time-based tick labels (used by multiple charts)
# ---------------------------------------------------------------------------

max_t = max(r.df["elapsed_s"].max() for r in rides)
if max_t > 3600:
    tick_vals = list(range(0, int(max_t) + 1, 600))
    tick_text = [f"{v // 3600}:{(v % 3600) // 60:02d}" for v in tick_vals]
else:
    tick_vals = list(range(0, int(max_t) + 1, 120))
    tick_text = [f"{v // 60}:{v % 60:02d}" for v in tick_vals]


def prepare_series(df: pd.DataFrame, key: str, smooth_s: int,
                   show_zeros: bool, sample_rate_hz: float = 1.0) -> pd.Series:
    s = pd.to_numeric(df[key], errors="coerce")
    if not show_zeros and key in {"power", "cadence"}:
        s = s.where(s != 0)
    if smooth_s > 0:
        window = seconds_to_samples(smooth_s, sample_rate_hz)
        s = s.rolling(window=window, min_periods=1, center=True).mean()
    return s


# ---------------------------------------------------------------------------
# Overlay charts (all share the same x-axis via matches='x')
# ---------------------------------------------------------------------------

st.subheader("Overlay")
st.caption("Zoom any subplot to drill into a section — all subplots zoom together.")

active_channels = [
    ch for ch in CHANNELS
    if any(
        pd.to_numeric(r.df[ch["key"]], errors="coerce").dropna().shape[0] > 0
        for r in rides
    )
]

if not active_channels:
    st.warning("No plottable channels found in these files.")
    st.stop()

fig = make_subplots(
    rows=len(active_channels),
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.035,
    subplot_titles=[f"{ch['label']} ({ch['unit']})" for ch in active_channels],
)

for row_idx, ch in enumerate(active_channels, start=1):
    for r in rides:
        y = prepare_series(r.df, ch["key"], smoothing, show_zeros, r.sample_rate_hz)
        fig.add_trace(
            go.Scatter(
                x=r.df["elapsed_s"],
                y=y,
                mode="lines",
                name=r.name,
                legendgroup=r.name,
                showlegend=(row_idx == 1),
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
        title_text=ch["unit"], row=row_idx, col=1,
        gridcolor="#eef0f4",
        rangemode="tozero" if ch["y_zero"] else "normal",
    )

# Range selector on the top subplot only — gives quick-zoom shortcuts
fig.update_xaxes(
    title_text="Elapsed time",
    row=len(active_channels), col=1,
    tickvals=tick_vals, ticktext=tick_text,
    gridcolor="#eef0f4",
)
for i in range(1, len(active_channels)):
    fig.update_xaxes(tickvals=tick_vals, ticktext=tick_text, row=i, col=1,
                     gridcolor="#eef0f4")

fig.update_layout(
    height=220 * len(active_channels) + 60,
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    margin=dict(l=60, r=20, t=60, b=40),
)

st.plotly_chart(style_chart(fig), use_container_width=True)

# ---------------------------------------------------------------------------
# Cumulative work (kJ)
# ---------------------------------------------------------------------------

has_power = any(
    pd.to_numeric(r.df.get("power"), errors="coerce").dropna().shape[0] > 0
    for r in rides
)

if has_power:
    st.subheader("Cumulative Work")
    st.caption(
        "Running total of work done, in kJ. The two lines should grow at roughly the "
        "same rate. Any persistent divergence means one meter is systematically reading "
        "higher across the whole ride — the slope of the gap is the offset."
    )

    kj_fig = go.Figure()
    for r in rides:
        cum = cumulative_work(r.df["power"], r.sample_rate_hz)
        kj_fig.add_trace(go.Scatter(
            x=r.df["elapsed_s"],
            y=cum,
            mode="lines",
            name=r.name,
            line=dict(color=r.color, width=2),
            hovertemplate=(
                f"<b>{r.name}</b><br>"
                "t=%{x:.0f}s<br>"
                "%{y:.1f} kJ<extra></extra>"
            ),
        ))
    kj_fig.update_layout(
        height=320,
        xaxis=dict(
            title="Elapsed time",
            tickvals=tick_vals, ticktext=tick_text,
            gridcolor="#eef0f4",
        ),
        yaxis=dict(title="Work (kJ)", gridcolor="#eef0f4", rangemode="tozero"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=60, r=20, t=40, b=40),
    )
    st.plotly_chart(style_chart(kj_fig), use_container_width=True)

# ---------------------------------------------------------------------------
# Mean Max Power curve
# ---------------------------------------------------------------------------

mmp_data = {}
if has_power:
    st.subheader("Mean Max Power")
    st.caption(
        "Best rolling-average power for each duration. A systematic gap at short durations "
        "but not long ones (or vice versa) usually points to different smoothing algorithms "
        "between the two meters."
    )

    mmp_fig = go.Figure()
    for r in rides:
        mmp = mean_max_power(r.df["power"], MMP_DURATIONS, r.sample_rate_hz)
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
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=60, r=20, t=40, b=40),
    )
    st.plotly_chart(style_chart(mmp_fig), use_container_width=True)

    if len(rides) == 2 and mmp_data[rides[0].name] and mmp_data[rides[1].name]:
        a_mmp = mmp_data[rides[0].name]
        b_mmp = mmp_data[rides[1].name]
        st.markdown("**MMP Δ at key durations**")
        highlight_durations = [5, 60, 300, 1200]
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
# Time in zones
# ---------------------------------------------------------------------------

if has_power:
    st.subheader("Time in Zones")
    st.caption(
        f"Coggan power zones based on FTP = {ftp} W (change in sidebar). Shown as stacked "
        "bars per ride — any shift in the distribution shows up even when averages agree."
    )

    zone_fig = go.Figure()
    zone_results = {}
    for r in rides:
        zones = time_in_zones(r.df["power"], ftp, r.sample_rate_hz)
        zone_results[r.name] = zones

    # Stacked horizontal bars, one per ride. Each zone is a separate trace so
    # hover shows the right details.
    for zone_idx, (_, label, color) in enumerate(COGGAN_ZONES):
        xs, ys, texts, hovers = [], [], [], []
        for r in rides:
            zones = zone_results[r.name]
            seconds = zones[zone_idx][1]
            pct = zones[zone_idx][2]
            xs.append(pct)
            ys.append(r.name)
            texts.append(f"{pct:.0f}%" if pct >= 3 else "")
            hovers.append(
                f"<b>{r.name}</b><br>"
                f"{label}<br>"
                f"{fmt_duration(seconds)} ({pct:.1f}%)<extra></extra>"
            )
        zone_fig.add_trace(go.Bar(
            x=xs, y=ys,
            orientation="h",
            name=label,
            marker=dict(color=color, line=dict(color="white", width=1)),
            text=texts,
            textposition="inside",
            textfont=dict(color="white", size=11),
            hovertemplate=hovers,
        ))

    zone_fig.update_layout(
        barmode="stack",
        height=90 + 50 * len(rides),
        xaxis=dict(title="% of time", range=[0, 100], gridcolor="#eef0f4",
                   ticksuffix="%"),
        yaxis=dict(autorange="reversed"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(size=11)),
        margin=dict(l=120, r=20, t=60, b=40),
    )
    st.plotly_chart(style_chart(zone_fig), use_container_width=True)

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
        for side_key, side_label, side_color in [
            ("left_power", "Left", LR_COLORS["left"]),
            ("right_power", "Right", LR_COLORS["right"]),
        ]:
            y = prepare_series(r.df, side_key, smoothing, False, r.sample_rate_hz)
            lr_fig.add_trace(
                go.Scatter(
                    x=r.df["elapsed_s"],
                    y=y,
                    mode="lines",
                    name=side_label,
                    line=dict(color=side_color, width=1.2),
                    opacity=0.85,
                    showlegend=(row_idx == 1),
                    legendgroup=side_label,
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
        margin=dict(l=60, r=20, t=60, b=40),
    )
    st.plotly_chart(style_chart(lr_fig), use_container_width=True)

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
# Power vs Power scatter + cadence-binned offset (2 rides only)
# ---------------------------------------------------------------------------

if len(rides) == 2:
    a, b = rides[0], rides[1]
    a_pwr = pd.to_numeric(a.df["power"], errors="coerce")
    b_pwr = pd.to_numeric(b.df["power"], errors="coerce")

    a_aligned = pd.DataFrame({
        "t": a.df["elapsed_s"], "a": a_pwr,
        "cad_a": pd.to_numeric(a.df.get("cadence"), errors="coerce"),
    }).dropna(subset=["t", "a"])
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
            on="t", direction="nearest", tolerance=2,
        ).dropna(subset=["a", "b"])

        if len(merged) > 10:
            max_p = max(merged["a"].max(), merged["b"].max()) * 1.05

            scatter_fig = go.Figure()
            scatter_fig.add_trace(go.Scatter(
                x=merged["a"], y=merged["b"],
                mode="markers",
                marker=dict(size=3, color="#3EC5C9", opacity=0.35),
                name="Matched seconds",
                hovertemplate=(
                    f"{a.name}: %{{x:.0f}} W<br>"
                    f"{b.name}: %{{y:.0f}} W<extra></extra>"
                ),
            ))
            scatter_fig.add_trace(go.Scatter(
                x=[0, max_p], y=[0, max_p],
                mode="lines",
                line=dict(color="#999", dash="dash", width=1),
                name="Perfect agreement",
                hoverinfo="skip",
            ))

            x_vals = merged["a"].values
            y_vals = merged["b"].values
            mask = (x_vals > 0) & (y_vals > 0)
            overall_slope = None
            if mask.sum() > 10:
                overall_slope = (x_vals[mask] * y_vals[mask]).sum() / (x_vals[mask] ** 2).sum()
                scatter_fig.add_trace(go.Scatter(
                    x=[0, max_p],
                    y=[0, overall_slope * max_p],
                    mode="lines",
                    line=dict(color="#C94FB8", width=1.5),
                    name=f"Best fit (y = {overall_slope:.3f}·x)",
                    hoverinfo="skip",
                ))
                offset_pct = (overall_slope - 1) * 100
                st.metric(
                    f"`{b.name}` reads…",
                    f"{offset_pct:+.1f}% vs `{a.name}`",
                    help="Slope of the best-fit line through the origin. "
                         "Positive = Y-axis meter reads higher.",
                )

            scatter_fig.update_layout(
                height=500,
                xaxis=dict(title=f"{a.name} (W)", range=[0, max_p], gridcolor="#eef0f4"),
                yaxis=dict(title=f"{b.name} (W)", range=[0, max_p], gridcolor="#eef0f4",
                           scaleanchor="x", scaleratio=1),
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                margin=dict(l=60, r=20, t=40, b=40),
            )
            st.plotly_chart(style_chart(scatter_fig), use_container_width=True)

            # --- Cadence-binned offset ---------------------------------
            # Shows whether disagreement is constant (calibration issue) or
            # cadence-dependent (mechanical / torque-related).
            if "cad_a" in merged.columns and merged["cad_a"].notna().sum() > 20:
                st.subheader("Offset by Cadence")
                st.caption(
                    "Power-meter offset broken out by pedalling cadence. A flat line means "
                    "the disagreement is a simple calibration offset. A sloped line means "
                    "the gap depends on cadence — often a mechanical or torque-measurement "
                    "issue with wheel-on trainers."
                )

                cad_bins = [50, 60, 70, 80, 90, 100, 110, 120]
                cad_binned = merged.dropna(subset=["cad_a"]).copy()
                cad_binned = cad_binned[cad_binned["cad_a"].between(40, 140)]
                cad_binned = cad_binned[(cad_binned["a"] > 0) & (cad_binned["b"] > 0)]
                cad_binned["bin"] = pd.cut(
                    cad_binned["cad_a"],
                    bins=cad_bins,
                    labels=[f"{lo}-{hi}" for lo, hi in zip(cad_bins[:-1], cad_bins[1:])],
                )

                bin_results = []
                for bin_label, group in cad_binned.groupby("bin", observed=True):
                    if len(group) < 10:
                        continue
                    slope = (group["a"] * group["b"]).sum() / (group["a"] ** 2).sum()
                    bin_results.append({
                        "bin": str(bin_label),
                        "offset_pct": (slope - 1) * 100,
                        "n": len(group),
                    })

                if bin_results:
                    bin_df = pd.DataFrame(bin_results)
                    cad_fig = go.Figure()
                    cad_fig.add_trace(go.Bar(
                        x=bin_df["bin"],
                        y=bin_df["offset_pct"],
                        marker=dict(color=[
                            "#dc2626" if v > 2 else ("#16a34a" if abs(v) <= 2 else "#f59e0b")
                            for v in bin_df["offset_pct"]
                        ]),
                        text=[f"{v:+.1f}%" for v in bin_df["offset_pct"]],
                        textposition="outside",
                        hovertemplate=(
                            "Cadence: %{x} rpm<br>"
                            "Offset: %{y:+.1f}%<br>"
                            "n samples: %{customdata}<extra></extra>"
                        ),
                        customdata=bin_df["n"],
                    ))
                    cad_fig.add_hline(y=0, line_color="#999", line_width=1)
                    if overall_slope is not None:
                        overall_pct = (overall_slope - 1) * 100
                        cad_fig.add_hline(
                            y=overall_pct, line_color="#C94FB8", line_dash="dash",
                            line_width=1,
                            annotation_text=f"Overall: {overall_pct:+.1f}%",
                            annotation_position="top right",
                        )
                    cad_fig.update_layout(
                        height=320,
                        xaxis=dict(title="Cadence (rpm)", gridcolor="#eef0f4"),
                        yaxis=dict(title=f"{b.name} vs {a.name} (%)", gridcolor="#eef0f4",
                                   ticksuffix="%"),
                        margin=dict(l=60, r=20, t=40, b=40),
                        showlegend=False,
                    )
                    st.plotly_chart(style_chart(cad_fig), use_container_width=True)

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
        avg_lat = np.mean([p[0] for p in all_points])
        avg_lon = np.mean([p[1] for p in all_points])
        m = folium.Map(location=[avg_lat, avg_lon], zoom_start=13, tiles="OpenStreetMap")
        lats = [p[0] for p in all_points]
        lons = [p[1] for p in all_points]
        m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])

        for r, points in tracks:
            folium.PolyLine(points, color=r.color, weight=3, opacity=0.85,
                            tooltip=r.name).add_to(m)
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

# ---------------------------------------------------------------------------
# PDF report download
# ---------------------------------------------------------------------------

st.markdown("---")
st.subheader("Export")
st.caption(
    "Download a PDF report with the summary stats and all charts — useful for sharing "
    "with a power-meter manufacturer or filing away with a bike setup log."
)


def build_pdf_report() -> bytes:
    """Render a PDF bundling summary stats + key charts. Imports heavy deps lazily."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak,
    )
    from reportlab.lib import colors as rl_colors

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=15 * mm, rightMargin=15 * mm,
        topMargin=15 * mm, bottomMargin=15 * mm,
    )
    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=18,
                        textColor=rl_colors.HexColor(CHART_TEXT), spaceAfter=8)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=13,
                        textColor=rl_colors.HexColor(CHART_TEXT), spaceAfter=6, spaceBefore=10)
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=9,
                          textColor=rl_colors.HexColor(CHART_TEXT))

    story = []
    story.append(Paragraph("FIT Compare — Report", h1))
    story.append(Paragraph(
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} · "
        f"{len(rides)} files · FTP {ftp} W",
        body,
    ))
    story.append(Spacer(1, 6 * mm))

    # Stats table
    story.append(Paragraph("Summary", h2))
    table_data = [list(stats_df.columns)] + stats_df.values.tolist()
    tbl = Table(table_data, repeatRows=1)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#f1f5f9")),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.HexColor(CHART_TEXT)),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.25, rl_colors.HexColor("#e5e7eb")),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(tbl)

    # Charts — render to PNG via kaleido then embed
    charts_to_export = []
    if has_power:
        charts_to_export.append(("Overlay", fig, 180))
        charts_to_export.append(("Cumulative Work", kj_fig, 90))
        charts_to_export.append(("Mean Max Power", mmp_fig, 110))
        charts_to_export.append(("Time in Zones", zone_fig, 70))
    if has_lr:
        charts_to_export.append(("Left / Right Power", lr_fig, 130))
    if len(rides) == 2 and 'scatter_fig' in globals():
        charts_to_export.append(("Power vs Power", scatter_fig, 130))
        if 'cad_fig' in globals():
            charts_to_export.append(("Offset by Cadence", cad_fig, 90))

    for title, figure, height_mm in charts_to_export:
        story.append(PageBreak())
        story.append(Paragraph(title, h2))
        png = figure.to_image(format="png", width=1400, height=int(height_mm * 5),
                              scale=2, engine="kaleido")
        img = Image(io.BytesIO(png), width=180 * mm, height=height_mm * mm)
        story.append(img)

    doc.build(story)
    buf.seek(0)
    return buf.read()


if st.button("Generate PDF report", type="primary"):
    with st.spinner("Rendering charts and building PDF…"):
        try:
            pdf_bytes = build_pdf_report()
            filename = f"fit-compare-{datetime.now().strftime('%Y%m%d-%H%M')}.pdf"
            st.download_button(
                "Download PDF",
                data=pdf_bytes,
                file_name=filename,
                mime="application/pdf",
            )
        except Exception as e:
            st.error(f"Couldn't build PDF: {e}")
            st.caption(
                "PDF export requires `kaleido` and `reportlab` to be installed. "
                "If you see this on Streamlit Community Cloud, make sure both are in "
                "`requirements.txt` and reboot the app."
            )
