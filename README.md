# FIT Compare

A Streamlit app to overlay data from two or more `.fit` files. Useful for comparing indoor trainers against outdoor power meters, dual-sided vs single-sided meters, crank-based vs pedal-based, or any two head units recording the same ride.

## Setup

```bash
pip install streamlit fitparse pandas plotly
```

## Run

```bash
streamlit run fit_compare.py
```

Then open the URL it prints (usually http://localhost:8501) and drop your `.fit` files in.

## What you get

- **Summary table** — duration, avg power, NP, max power, HR, cadence, speed, distance, ascent for each file
- **Delta metrics** (when exactly 2 files) — how the second file's numbers compare to the first in watts and percent
- **Overlay charts** — Power, Heart Rate, Cadence, Speed, and Elevation on a shared time axis, with unified hover
- **Power vs Power scatter** (when exactly 2 files) — every matched second plotted against the 45° perfect-agreement line, plus a best-fit slope that quantifies any systematic offset between the two meters (e.g. "Plowright reads +2.4% vs Marsman")

## Options (sidebar)

- **Smoothing** — rolling mean window in seconds, 0 = raw
- **Show zeros** — off hides zero power/cadence points (useful to stop coasting dropouts from dragging visual averages down)

## Notes

- NP is a 30s-rolling-mean approximation. Close to TrainingPeaks but not identical.
- Files are aligned by elapsed seconds from their first record. If you start one device a few seconds before the other, traces will appear slightly offset. The power-vs-power scatter uses `merge_asof` with a 2s tolerance so it's robust to this.
- Speed is converted from m/s (FIT native) to km/h. If you want mph, change the `* 3.6` to `* 2.23694` in `parse_fit`.
- Parsing is cached on the file bytes, so changing smoothing/zero options doesn't re-parse.
