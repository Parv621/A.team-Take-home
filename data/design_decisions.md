## Design decisions summary: synthetic CPG media dataset

**Architecture**
- Bottom-up simulation: 8 primitives (Eligible_Impressions, Impression_Share, CTR, CPC, PICR, Baseline_PICR, AOV, Attribution_Inflation) generated with trend + seasonality + AR(1) noise; every outcome metric (Impressions, Clicks, Spend, Conversions, MAS, Revenue, ROAS, RROI, iROAS) derived arithmetically from them, not simulated independently.
- Rejected: top-down correlated-noise simulation. Would have let ROAS≠Revenue/Spend and made spike injection "multiply a column" rather than a mechanistic consequence.

**Key formula locks**
- PICR := Conversions ÷ Impressions (post-impression, not post-click). Flagged by the source doc as non-standard; this is the definition we locked.
- iROAS's counterfactual (Baseline_PICR → Baseline_Conversions → Incremental_Rev) is invented, not doc-specified, but mandatory: no other way to produce iROAS at all.
- Reported_Revenue = MAS × Attribution_Inflation, also invented, mandatory: the only mechanism producing the ROAS-vs-RROI divergence the doc calls out as a signal.
- MAS = Conversions × AOV: kept as-is after evaluating and rejecting the alternative (MAS as its own primitive), since that would decouple PICR from revenue and break the mechanistic causal chain.

**Spike design**
- 7 cause types: the brief's 6 (genuine efficiency gain, spend-reduction artifact, mix-shift artifact, survivorship bias, external demand spike, creative refresh) plus a 7th we added, measurement_artifact, as a hard negative for the ROAS-vs-RROI trap.
- Survivorship bias and spend-reduction artifact deliberately left near-identical in signature (intentional realism, not a bug).
- Co-occurrence: 50% single-cause, 35% two-cause, 15% three-cause events. Onset: 60% sudden, 40% gradual (3-month ramp). Roughly one event per 8-10 active months per series.

**Structure**
- 3 business units, 8 brands (3/3/2 split), 7-channel pool, not all channels active per brand, 30 months.
- iROAS sparse (approximately 20% of rows, simulating scheduled incrementality tests) and noisy (approximately 20% relative measurement error) on top of the true value.
- `analyst_tag`: added weak/noisy label (75-83% match true cause, approximately 10% wrong, approximately 10% missed, approximately 3% false positive on normal rows) for weak-supervision training.
- `planned_spend`: skipped, deferred until/unless Forecasting is pursued.
- Missingness generated independent of the spike ground truth (verified via correlation check ≈0), so it can't leak the anomaly label. Structural absence (launch dates, seasonal-only channels) kept separate from random pipeline-gap missingness.

**Deliverable structure**
- Generation kept as one artifact (`generate_synthetic_data.py` + `data_dictionary.md`).
- Analysis-phase data split into three: `brief_metrics.csv` (only what the exercise brief specifies, `bu` included per your call), `panel_data_full.csv` (adds impressions/clicks/conversions/reported_revenue/iroas_test_flag/analyst_tag), `ground_truth_full.csv` (adds is_anomaly/cause_types/event_ids; oracle only, never model input).

**Known, flagged, not yet acted on**
- Row-level anomaly prevalence (approximately 39%) is higher than real-world spike rarity.
- BU-wide and brand-wide events are thin (3 and 2 occurrences total).
