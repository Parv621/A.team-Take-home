# Data dictionary: synthetic CPG media panel (v4)

## Current dataset

- 1,642 rows across 42 BU-brand-channel series.
- 3 BUs, 8 brands, 7 reported channels, and 48 monthly periods from 2022-01 through 2025-12.
- Monthly grain with keys `bu`, `brand`, `channel`, `month`.
- Deterministic generator with seed 42.

## Files

| File | Role |
|---|---|
| `brief_metrics.csv` | Primary analysis input with the 13 columns required by the exercise brief. |
| `panel_data_full.csv` | Enriched observable data: all brief columns plus funnel counts, reported revenue, the iROAS observation flag, and `analyst_tag`. |
| `ground_truth/ground_truth.csv` | Hidden row-level anomaly labels. Evaluation only; never use as model input. |
| `ground_truth/ground_truth_events.csv` | Hidden event-level log with event scope, start, onset type, and true causes. Evaluation only. |
| `generate_synthetic_data.py` | Generates all four CSVs above. |
| `README.md` | Current file-selection and regeneration guidance. |
| `data_dictionary.md` | Current schemas, definitions, assumptions, limitations, and validation results. |
| `design_decisions.md` | Architectural and metric-design rationale. |
| `regrounding_summary.md` | Version history, v4 summary, limitations, and next actions. |

## Observable columns

`brief_metrics.csv` contains the keys and brief metrics. `panel_data_full.csv` contains those same columns plus the six fields marked "full only" below.

| Column | Availability | Meaning |
|---|---|---|
| `bu`, `brand`, `channel`, `month` | both | Panel keys. `month` is the first day of the monthly period. |
| `roas` | both | `reported_revenue / spend`. |
| `rroi` | both | `mas / spend`. |
| `iroas` | both | Noisy measured incremental revenue divided by spend. Populated when `iroas_test_flag=True`; observed on 84.8% of v4 rows and carries approximately 20% relative measurement noise. |
| `mas` | both | Media Attributable Sales = conversions × AOV. |
| `spend` | both | Clicks × CPC. |
| `ctr` | both | Clicks / impressions. |
| `cpc` | both | Spend / clicks, with light auction-pressure coupling to impression share. |
| `picr` | both | Conversions / impressions. This is the locked definition for this dataset. |
| `impression_share` | both | Captured impressions / eligible impressions. |
| `impressions`, `clicks`, `conversions` | full only | Observable funnel counts. `conversions` is post-impression, not gated by clicks. |
| `reported_revenue` | full only | MAS × attribution inflation; creates the ROAS-vs-RROI wedge. |
| `iroas_test_flag` | full only | Whether iROAS was observed for the row. |
| `analyst_tag` | full only | Weak, noisy historical-analyst label; not ground truth. |

## Ground-truth columns

| File | Columns |
|---|---|
| `ground_truth/ground_truth.csv` | Keys plus `is_anomaly`, pipe-separated `cause_types`, and pipe-separated `event_ids`. |
| `ground_truth/ground_truth_events.csv` | `event_id`, `scope`, `bu`, `brand`, `channel`, `start_month`, `duration_type`, `cause_types`. |

## Ground truth cause types (7)

| Type | Mechanism injected | Signature |
|---|---|---|
| `genuine_efficiency_gain` | CTR and PICR up 15-30% | ROAS, iROAS, CTR, and PICR all move together |
| `creative_refresh` | same as above, but decays back to baseline over approximately 3 months | temporary version of genuine gain |
| `spend_reduction_artifact` | `impression_share` cut 20-35%, small PICR composition bump | ROAS up, MAS flat/down, iROAS barely moves |
| `survivorship_bias` | `impression_share` cut 40-60%, larger composition bump | **mechanically near-identical to `spend_reduction_artifact` by design.** Real analysts can't reliably tell these apart from aggregates alone either |
| `mix_shift_artifact` | brand-level: `impression_share` reallocated from a low- to a high-ROAS channel, no per-channel primitive changes | brand ROAS up, no individual channel actually improved |
| `external_demand_spike` | BU-wide: `eligible_impressions` and baseline (organic) PICR both rise together | multiple brands in the same BU move together; iROAS stays flat (lift isn't incremental to ads) |
| `measurement_artifact` | `attribution_inflation` jumps with no change to any real driver | ROAS up, RROI/MAS/iROAS untouched. Pure reporting artifact, not a real story. Not in the original 6 cause types; added as a hard negative for the ROAS-vs-RROI divergence trap |

Brand-channel events draw one, two, or three causes with probabilities 50%, 35%, and 15%. Mix-shift and external-demand events are single-cause. Multi-causal overlap is intentional.

## Key design assumptions (stated per ambiguity, not silently resolved)

1. **PICR := conversions / impressions** (post-impression, not post-click). The source material flagged this as non-standardized; this is the definition locked for this dataset.
2. **CTR/CPC applied uniformly across all channels**, including TV/OOH, as generic engagement-rate/cost-per-engagement rather than literal click metrics. Simplification, stated not hidden.
3. **Missingness is generated independently of the spike ground truth.** Two mechanisms: approximately 3% of rows have one random leading-indicator field nulled (pipeline gap), and approximately 5% of series have a 1-3 month full-row blackout (reporting outage). Neither is conditioned on whether a spike is active. The v4 missingness/anomaly correlation is 0.030, near zero.
4. **Structural absence** (not "missing"): 2 brands launch mid-window (no rows before launch), OOH only reports in Nov/Dec for brands that run it. This is a genuinely unbalanced panel, not NaN-filled.
5. Baseline/organic PICR is used only internally to compute the iROAS counterfactual, is always held below observed PICR by construction, and is **not exposed** in either observable CSV.
6. iROAS is observed on a high share of rows by design for this exercise. `iroas_test_flag` preserves the distinction between observed and missing measurements, and multiplicative measurement noise keeps observed iROAS non-negative.
7. **iROAS alone receives an additional metric-level measurement-error layer.** It estimates an unobserved counterfactual (revenue without advertising), so observed iROAS carries approximately 20% relative measurement noise to represent incrementality-test uncertainty. The other metrics are not noise-free: their primitives contain time-correlated variation, clicks and conversions use binomial sampling, and CPC receives a small auction shock. ROAS and RROI inherit that upstream variation but are not independently perturbed, because they remain exact accounting identities (`roas = reported_revenue / spend`; `rroi = mas / spend`). If more reporting noise is needed later, it should be applied to upstream observed inputs and the ratios recalculated, rather than added directly to ROAS or RROI.

## Known limitations, flagged not fixed

- **Cause representation is still uneven.** v4 row-level counts range from 26 (`survivorship_bias`, `measurement_artifact`) to 71 (`creative_refresh`). This is narrower than a total absence problem, but the causes are not class-balanced because their scopes, durations, and overlaps differ.
- **Independent event diversity is thin for macro causes.** `mix_shift_artifact` has 54 labeled rows and `external_demand_spike` has 70, but each comes from only one underlying event. Those rows are correlated examples, which limits cause-level attribution generalization.
- **Two artifact causes remain difficult to distinguish from aggregates.** `survivorship_bias` and `spend_reduction_artifact` are intentionally near-identical in their aggregate signatures. Reliable separation may require information not present in the panel.
- **iROAS availability is analysis-friendly rather than operationally realistic.** The 84.8% observed coverage supports anomaly analysis but is much denser than the original scheduled-test assumption; it should not be treated as a measurement-cadence benchmark.

## Validation checks passed (current data)

- 1,642 rows, 42 series, 48 months, and no duplicate panel keys.
- ROAS ≡ reported_revenue/spend and RROI ≡ mas/spend hold exactly on all non-missing rows.
- No negative spend/impressions.
- ROAS distribution: mean 6.5, median 5.5, range 1.1-21.5.
- iROAS observed on 1,393 rows, or 84.8% (target 85%).
- Row-level anomaly prevalence: 254 rows, or 15.5% (target approximately 15%).
- Missingness vs. true-anomaly-label correlation: 0.030 (near 0, no leakage).
- `analyst_tag`: 77.0% match rate among tagged anomaly rows, 7.5% miss rate on anomaly rows, and 3.4% false-flag rate on normal rows.
- All 7 cause types represented: `creative_refresh` 71, `external_demand_spike` 70, `mix_shift_artifact` 54, `spend_reduction_artifact` 52, `genuine_efficiency_gain` 34, `survivorship_bias` 26, `measurement_artifact` 26. Counts are row-level and can overlap on multi-causal rows.
- 37 events total: 35 brand-channel, 1 brand-level mix shift, and 1 BU-level external-demand spike.
