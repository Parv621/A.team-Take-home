# Synthetic CPG media panel

Current generated release:

- 1,642 rows across 42 BU-brand-channel series
- 48 monthly periods, January 2022 through December 2025
- iROAS populated on 84.8% of rows, with approximately 20% relative measurement noise
- 15.5% row-level anomaly prevalence across all 7 cause types
- Deterministic generation with seed 42

## Generated files

| File | Purpose |
|---|---|
| `brief_metrics.csv` | Primary analysis input. Contains the 13 brief-level identifiers and metrics. |
| `panel_data_full.csv` | Observable analysis data with funnel counts, reported revenue, the iROAS observation flag, and the weak `analyst_tag` label. |
| `ground_truth/ground_truth.csv` | Hidden row-level anomaly labels. Evaluation only; never use as model input. |
| `ground_truth/ground_truth_events.csv` | One row per injected event, including scope, onset, and true causes. Evaluation and event lookup only. |

`brief_metrics.csv`, `panel_data_full.csv`, and `ground_truth/ground_truth.csv` use the same keys: `bu`, `brand`, `channel`, and `month`. The event log is event-grain rather than panel-grain.

`panel_data.csv` is not generated because `panel_data_full.csv` contains the same observable columns in analysis order. `ground_truth_full.csv` is not generated because it would only duplicate a join between `panel_data_full.csv` and `ground_truth/ground_truth.csv`.

## Regeneration

Run `generate_synthetic_data.py` to regenerate the four CSVs above. Observable files are written beside the script; evaluation-only files are written to `ground_truth/`.

The generator preserves the bottom-up metric relationships and event-sampling principles documented in `design_decisions.md`. The longer panel adds exposure-scaled event opportunities so anomaly prevalence remains near the target.

`data_dictionary.md` contains metric definitions, cause signatures, and the original calibration history.
