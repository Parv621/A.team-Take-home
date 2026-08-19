# Anomaly detection design proposal

The pipeline does not read or tune against the ground-truth files (`data/ground_truth/ground_truth.csv` and `data/ground_truth/ground_truth_events.csv`). In the absence of clean labels, evaluation uses synthetic injection tests and `analyst_tag` overlap as a face-validity check.

## Problem framing

Unsupervised statistical anomaly detection on a BU x brand x channel x month panel (1,642 rows, 42 series, 48 months, January 2022 to December 2025). The output is a flag plus an evidence-strength score, not a cause: purely associational, a flag means a metric moved unusually, not why. Cause attribution is a separate exercise component.

The brief names five primary business metrics (ROAS, RROI, iROAS, MAS, spend). Rather than assume ROAS is the one worth monitoring, the detector is built generically across all five, evaluated identically on each, and the primary metric is chosen from the evidence, not by convention. Result: **RROI**, not ROAS, is primary. See Metric selection below.

## Method (see `anomaly_detection.ipynb`, sections 3 and 5)

- **Primary detector**: robust dual test, generic across metrics. Level-shift test (MAD-based z-score on month-over-month percent change) plus trend-break test (MAD-based z-score on 3-month cumulative percent change), 12-month trailing window (spans a full seasonal cycle), **|z| > 2.5**, fixed across all five metrics.
- **Trivial baseline** (required comparison, established once on ROAS, not re-derived per metric): naive mean/std single-test z-score, same window/threshold/floor. Robust scoring catches **74 level-shift flags versus 48** for the naive baseline on ROAS's full tier (33 robust-only, 7 naive-only) - structural, not a threshold artifact: median/MAD stays anchored to typical months even with an outlier already in the trailing window.
- **"Meaningful" anomaly** requires both a statistical signal (|z| > 2.5) and a business-relevant move, where the floor is each metric's **own** 75th percentile of observed month-over-month noise, not a single shared number: ROAS 15%, RROI 14%, MAS 22%, spend 24%, iROAS 37%.
- **Not selected**: STL decomposition was not tested. The rolling 12-month MAD baseline absorbs recurring variance but does not explicitly remove seasonality, so decomposition remains future work. Isolation forest was rejected because it has no natural business threshold and is harder to explain to an analyst than "which test fired, by how much."
- **Short-history tiering**: **under 12 months**, excluded, flagged "insufficient history" (4 OOH series, report only in November/December by design). **12 to 24 months**, naive test only, 6-month window, wider |z| > 3.0 tolerance - **currently 0 series**, since every series with at least 12 months of history in this panel has 24 or more. Kept in the pipeline, not deleted, in case a shorter panel is used in the future. **24 months and up**, full dual test (38 series).

## Metric selection

Business floors are derived independently per metric (Method, above) - reusing ROAS's 15% for MAS or spend would let much more of their normal noise clear the statistical bar than intended. iROAS's floor (37%) is the highest by far, and it's not a data-sparsity artifact - iROAS has 1,157 valid month-over-month pairs, close to the ~1,593 the other metrics have. It's the direct effect of iROAS's own measurement-noise layer on top of its underlying volatility.

**Selection criterion**: synthetic-injection recall and the empirical alert rate on the unmodified panel, used as a false-positive-rate proxy (same protocol and shock sizes as the trivial-baseline comparison above), run once per metric across all 38 full-tier series. Coverage, corroboration, and `analyst_tag` overlap are reported alongside as context, **not blended into a weighted score** - no stakeholder-given weights exist, and the components have incompatible units and directions (recall wants max, the alert-rate proxy wants min, a noise floor has no inherent "good" direction), so any specific weighting would be an unjustified, hard-to-defend choice.

| metric | recall @ 40% shock (sudden/gradual) | recall @ 20% shock (sudden/gradual) | alert-rate proxy | coverage | mean corrob. | tag overlap |
|---|---|---|---|---|---|---|
| **rroi** | 73.7% / 55.3% | 34.2% / 47.4% | **8.5%** | 75.7% | 0.72 | 24.8% |
| roas | 68.4% / 55.3% | 44.7% / 47.4% | 9.4% | 75.7% | 0.70 | 27.2% |
| spend | 44.7% / 34.2% | 23.7% / 31.6% | 12.6% | 75.7% | 0.60 | 23.6% |
| mas | 34.2% / 44.7% | 31.6% / 36.8% | 13.5% | 75.7% | 0.50 | 28.9% |
| iroas | 15.8% / 15.8% | 15.8% / 23.7% | 4.1% | 31.5% | 0.45 | 24.2% |

**RROI is chosen as the primary metric**: it edges out ROAS on both axes that matter (higher recall, lower alert-rate proxy), and is structurally cleaner - `rroi = mas / spend` is an exact identity, unlike ROAS which inherits `reported_revenue`'s attribution-inflation noise. MAS and spend both trade a meaningful increase in the alert-rate proxy for recall that doesn't clearly beat RROI or ROAS - worse on the driving criterion, not close seconds. ROAS is retained as the natural secondary signal: ROAS-vs-RROI divergence is exactly the data dictionary's `measurement_artifact` signature (`roas / rroi = attribution_inflation`, computable directly from the brief columns already in hand, no need for `reported_revenue`).

**Required caveat**: iROAS's alert-rate proxy (4.1%) looks best of all five, but that's a symptom of a detector that almost never fires: recall is 15.8% at a 40% shock and ranges from 15.8% to 23.7% across the tested shock sizes. This is not evidence of precision. It's a direct consequence of iROAS's added measurement noise and its 31.5% z-score coverage (versus 75.7% for the other four), not proof that iROAS is a poor business metric - it remains the metric closest to true incrementality and stays valuable as corroborating context. Revisit if a future data refresh reduces its noise or improves its coverage further.

## Leading-indicator corroboration

The brief requires the model to use CTR, CPC, PICR, and impression share, not just the headline metric. For each primary metric, the notebook counts how many of these four cross the same threshold in the same month - a 0 to 4 score, computed **independently per metric** (not merged across metrics): mean corroboration is highest for RROI (0.72) and ROAS (0.70), lowest for iROAS (0.45) and MAS (0.50). Of RROI's 137 flags, 70 have zero corroborating moves, 26 have two or more. Not cause attribution, just evidence strength.

## Evaluation without clean labels

**Synthetic injection** (RROI, the chosen metric): known shocks of 20%, 30%, and 40% (matching the brief's own "ROAS jumps 40%" example) injected into copies of real series at runtime, across all 38 full-tier series. Recall for sudden (1-month) shocks: 34.2%, 55.3%, 73.7% at those three shock sizes. Recall for gradual (3-month ramp) shocks: 47.4%, 60.5%, 55.3%. Recall rises with shock size overall, as expected under a precision-leaning threshold. The empirical alert rate on the unmodified panel is **8.5%** (137 of 1,612 scorable rows). Because that panel already contains injected events, this is an FPR proxy rather than a ground-truth false-positive rate.

**Face validity**: `analyst_tag` (from `panel_data_full.csv`, used only here) overlaps with 24.8% of RROI's flags (34 of 137). Expected, not a red flag: the tag simulates historical notes on the true event at any magnitude, while the detector only fires when a move clears both RROI's business floor and the |z| > 2.5 threshold.

## Business framing and deployment

**Stakeholder**: the analyst running the monthly manual spike check, on RROI as the primary business metric. A flag triggers a review, not an automated action. **False-positive cost**: about 2 to 4 analyst-hours of triage on a non-event. **False-negative cost**: a real spike's narrative reaches leadership uncorrected, cost bounded below by the media-reallocation decision it could misinform, plausibly over 1 analyst-day. Given that asymmetry, the threshold stays precision-leaning (z = 2.5), reusing the flags-versus-hours sensitivity table built for ROAS (re-deriving it per metric was judged a separate, larger stakeholder exercise, out of scope here). At that threshold and the ~3 analyst-hours/flag ratio the sensitivity table implies: RROI's 137 flags over the 48-month panel work out to **~2.9 flags/month, ~8.6 analyst-hours/month (~1.1 analyst-days/month)** across the 38 scored series. **Deployment**: monthly batch job feeding the analyst's existing checklist, not real-time, no autonomous action. "Retraining" only means recomputing rolling statistics and business floors each month; there is no model fit to refresh.

## Limitations and future work

- Metric selection is recall/alert-rate-driven on this specific rule-based detector, not a claim that RROI is intrinsically the best-behaved metric in the abstract - the ranking could shift if the detector design itself changed.
- The 4 series under 12 months of history are excluded entirely from every metric's detector, not held to a lower bar; they need a separate short-window method or manual fallback before this covers the full portfolio.
- The 12-24 month sparse tier is currently empty - kept in the pipeline for robustness against a future shorter panel, not because it's exercised today.
- Per-metric business floors and the corroboration count have no confirmed "how many is enough" calibration; both were derived from observed noise distributions, not tuned against labels, and would benefit from an analyst feedback loop.
- With more time: re-derive the flags/month vs. analyst-hours/month sensitivity table per metric instead of reusing ROAS's; walk-forward validation against confirmed real spikes, not just synthetic injection, would sharpen every floor and window used here.
