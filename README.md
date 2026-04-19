# FIT Compare

A Streamlit app to overlay data from two or more `.fit` files. Useful for comparing indoor trainers against outdoor power meters, dual-sided vs single-sided meters, crank-based vs pedal-based, or comparing efforts between different riders (e.g. a team training session).

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run fit_compare.py
```

Then open the URL it prints (usually http://localhost:8501) and drop your `.fit` files in.

## What you get

- **Summary table** — duration, avg power, NP, max power, work (kJ), HR, cadence, speed, distance, ascent for each file. Each row shows the rider's FTP and weight alongside the file name.
- **Delta metrics** (when exactly 2 files) — how the second file compares to the first in absolute units and percent.
- **Overlay charts** — Power, HR, Cadence, Speed, Elevation on a shared time axis with linked zoom across all subplots.
- **Cumulative work** — running total in kJ per ride. Divergence reveals persistent offset across the whole ride.
- **Mean Max Power curve** — log-scale power-duration curve from 1s to 1h, plus delta metrics at canonical durations (5s, 1m, 5m, 20m). Reveals whether meters disagree more on short efforts vs long ones.
- **Time in Zones** — stacked-bar distribution of Coggan power zones (Z1-Z7) for each ride, using that rider's own FTP so zones are comparable across riders of different ability.
- **Left / Right Power** — dual-sided power split per ride, with distinct blue (left) / red (right) colouring.
- **Power vs Power scatter** (when 2 files) — every matched second plotted with a best-fit slope through origin. Headline metric: *"Meter B reads ±X% vs Meter A"*. Warns if the two files appear to be different riders.
- **Offset by Cadence** (when 2 files) — the power-vs-power offset broken out by cadence bins. Flat = calibration issue; sloped = mechanical/torque issue (common with wheel-on trainers).
- **GPS map** — OpenStreetMap overlay of each ride's track.
- **Sensors table** — manufacturer, product, serial, and SW version of every device that contributed to each file.
- **PDF report** — one-click download bundling summary stats + all charts for sharing or archiving.

## Sidebar options

- **Smoothing** — rolling mean window in seconds (0 = raw). Applied to all time-series charts.
- **Show zeros** — off hides zero power/cadence points so coasting dropouts don't pull visual averages down.
- **Power units** — Watts or W/kg. W/kg divides each rider's power by their own weight.
- **Riders** — per-file FTP and weight (up to 10 riders). Used for zone boundaries and W/kg calculations. Values persist across reruns.

## Notes on methodology

- **Normalised Power** uses Coggan's standard formula: 30-second moving average, raise each to the 4th power, take the mean, take the 4th root. Matches TrainingPeaks/WKO.
- **Time alignment.** Files align by elapsed seconds from each file's first record. If you started one device a few seconds before the other, traces will show a slight offset. The power-vs-power scatter uses `merge_asof` with a 2s tolerance so it's robust to this.
- **Sample rate.** Auto-detected from timestamp gaps. Non-1Hz files (e.g. Garmin "smart recording") get a warning, and MMP/smoothing windows adjust accordingly.
- **Work (kJ)** = integral of power over time ÷ 1000.
- **W/kg conversion** is per-rider — each file uses its own rider's weight from the sidebar.
- **Power zones** use each rider's own FTP (Coggan's 7-zone model). Z3 ('Tempo') means the same relative intensity for every ride, regardless of absolute wattage.
- **Best-fit slope** in the Power vs Power scatter is always computed in watts — it's the physical ratio between the two meters regardless of display units. Warns if the two files look like different riders (FTP or weight differs by more than 10W / 2kg).
- **Speed** is converted from m/s (FIT native) to km/h. Swap `* 3.6` for `* 2.23694` in `parse_fit` for mph.
- Parsing is cached on file bytes — changing smoothing, zone, or zero options doesn't re-parse.

## File cap

Up to 10 files at once. Extra files are dropped with a warning. 10 is roughly a WorldTour team's starting line-up, which seemed like a natural ceiling for "how many riders might you compare in one session?"
