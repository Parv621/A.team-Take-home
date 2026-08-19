# Anomaly detection

Unsupervised, multi-metric anomaly detection for the paid-media panel. This is Part 1 of the take-home
(attribution, Part 2, lives in `../attribution/`).

## Notebook structure

`anomaly_detection.ipynb` runs top to bottom in 12 sections:

1. **Setup and data load.** Loads `brief_metrics.csv`; declares the 5 primary metrics (`roas`, `rroi`,
   `iroas`, `mas`, `spend`) and 4 leading indicators (`ctr`, `cpc`, `picr`, `impression_share`).
2. **EDA.** Panel shape (1,642 rows, 42 series, 48 months), missingness, and series-length distribution.
   Sets up the tiering rule used in Section 5.
3. **Detector functions.** Naive (mean/std) and robust (median/MAD) z-score tests, both look-ahead safe
   (`.shift(1)` before the rolling window). Smoke-tested before use.
4. **Per-metric business floor.** Each metric's own 75th-percentile move size, not one shared threshold.
5. **Generic tiered scoring pipeline.** One function (`score_metric`) scores all 5 metrics identically
   across 3 history tiers (excluded, under 12 months; sparse, 12-24; full, 24+). Robust-vs-naive baseline
   established once, on ROAS.
6. **Coverage check.** How much of each metric's window actually has a computable z-score (surfaces
   iROAS's coverage gap).
7. **Leading-indicator corroboration.** 0-4 evidence score per primary metric, computed independently per
   metric, not merged across metrics.
8. **Synthetic-injection evaluation.** Manufactures known shocks (sudden and gradual onset, 20/30/40%) on
   copies of real series to compute recall and false-positive rate without ground truth.
9. **Metric selection.** Recall and FPR drive the choice; coverage, floor, corroboration, and tag overlap
   are supporting context, not blended into a score. RROI selected as primary, ROAS retained as
   secondary.
10. **Example flagged series.** Visual sanity check on RROI.
11. **Final outputs.** Writes `flagged_anomalies.csv` (RROI) and one `flagged_<metric>.csv` per
    non-chosen metric.
12. **Limitations and next steps.**

## Key design decisions

- **No ground truth used anywhere.** `ground_truth*.csv` are never read. Evaluation is synthetic
  injection (Section 8) plus a descriptive `analyst_tag` overlap check (Section 9), not label-based.
- **Robust over naive scoring.** Median/MAD stays anchored to typical months even with an outlier
  already in the trailing window; validated once on ROAS (74 vs. 48 level-shift flags), not re-derived
  per metric, since it's a property of the method.
- **Per-metric business floor, not a shared one.** MAS and spend are structurally noisier than ROAS/RROI,
  and iROAS carries its own measurement-noise layer; one shared floor would over- or under-flag most
  metrics.
- **The production metric is chosen from evidence, not assumed.** The brief's example uses ROAS, but the
  detector runs identically on all 5 metrics and RROI wins on recall and FPR (Section 9).
- **No composite score.** Recall and FPR have different units and directions from coverage, floor, and
  corroboration; combining them would need stakeholder-specified weights that don't exist. They're
  reported side by side instead.
- **Look-ahead safety.** Every rolling statistic uses `.shift(1)`, so a month's baseline never includes
  that month's own value.

## Files

- `anomaly_detection.ipynb`: the technical notebook. Method, code, evidence, and validation for every
  claim in the proposal below.
- `proposal_anomaly_detection.md`: the component-level design writeup (method, metric selection,
  business framing, limitations) that the notebook implements.
- `flagged_*.csv`, `metric_selection_recall_fpr.png`: outputs written by the notebook when it runs, not
  source documents.

## Primary deliverable

`../ds excercises.md` is the primary deliverable for the exercise. Its "Model design > 1. Anomaly
detection" section summarizes this component; the structure and decisions above are enough context to
follow that summary. For full derivations, all numbers, and the evaluation methodology, see
`proposal_anomaly_detection.md`.
