# The Design Proposal

*Anomaly detection and probabilistic attribution for media-performance spikes*

## Problem framing

I frame the solution as a two-stage, human-in-the-loop decision-support system, not an automated attribution engine.

First, the Part 1 unsupervised detector converts monthly business unit, brand, and channel data into a prioritized queue of unusual performance changes.

For each flagged investigation, I use multi-label binary classification to score each candidate cause independently.

This framing reflects business reality: efficiency gains, creative changes, spend cuts, mix shifts, and reporting artifacts can occur together, whereas multiclass classification would force one answer.

The models use existing performance and leading-indicator data: ROAS, RROI, iROAS, MAS, spend, CTR, CPC, PICR, and impression share—plus anomaly evidence, lags, peer context, and channel mix.

When month-end data arrives, the system ranks drivers, abstains when evidence is weak, and drafts a narrative before report circulation, reducing triage while keeping the analyst responsible for the final explanation and any budget decision.

## Data strategy

I used a bottom-up simulator so anomalies represent business mechanisms, not isolated metric jumps. Eight primitives—eligible impressions, impression share, CTR, CPC, PICR, baseline PICR, average order value, and attribution inflation—receive trend, asymmetric seasonality, and AR(1) noise. Derived metrics preserve accounting identities; a top-down correlated generator might not.

- **Event sampling.** Each brand-channel series draws zero to two events, with an extra opportunity for longer histories. Starts are separated by at least four months. Brand-channel events are 60% sudden and 40% three-month ramps; 50% have one cause, 35% two, and 15% three.
- **Mechanism injection.** Injections change primitives before recalculation: genuine efficiency and creative refresh lift CTR/PICR, with refresh decaying; spend reduction and survivorship cut impression share with different composition effects; mix shift reallocates share across channels; demand lifts exposure and baseline PICR across a business unit; measurement artifacts change attribution inflation only.
- **Label creation.** Injection records event IDs and active causes by business unit, brand, channel, and month. Row labels and an event-grain log are written to evaluation-only `data/ground_truth/`; missingness is generated independently to avoid label leakage.
- **Data contract.** `data/brief_metrics.csv` is the modeling input. `data/panel_data_full.csv` adds observable funnel diagnostics and a noisy `analyst_tag`. `data/generate_synthetic_data.py` rebuilds the outputs, while `data/data_dictionary.md` and `data/design_decisions.md` document assumptions. Limits include simulator-defined causality, aggregate monthly signals, generic engagement metrics for TV/OOH, and limited macro-event diversity.

## Model design

### 1. Anomaly detection

The unsupervised detector scores each business unit–brand–channel monthly series. It returns a flag and evidence score, not a cause.

- **Baseline and noise control.** The naive baseline is a 12-month rolling z-score based on the mean and standard deviation of month-over-month change. The chosen, look-ahead-safe detector uses median and median absolute deviation (MAD) z-scores on one-month and three-month cumulative changes. A flag must pass two gates. I chose **|z| > 2.5** as a precision-leaning cutoff that keeps the monthly review queue manageable while retaining useful recall in injection tests. The **metric-specific 75th-percentile movement floor** removes statistically unusual but commercially small moves and respects each metric’s normal volatility. The **12-month trailing window** spans one seasonal cycle, while median/MAD prevents earlier spikes from shifting the baseline. I rejected STL and isolation forest for limited benefit and weaker explainability.
- **Metric roles and indicators.** All five metrics are scored. RROI is primary based on its balance of recall and false-positive rate; ROAS is secondary; MAS and spend expose scale and denominator effects; iROAS provides corroboration despite noise and lower coverage. Same-month CTR, CPC, PICR, and impression-share movements form a 0–4 evidence count, not a cause label.
- **History and irregular spend.** Series with fewer than 12 months go to manual review; those with 12–24 months use a six-month mean/std level test at |z| > 3.0; those with at least 24 months use the full dual test. Windows need 75% valid observations without imputation. Using spend and MAS beside return ratios keeps irregular-spend denominator effects visible.
- **Tradeoff and onset.** False alerts cost 2–4 analyst hours; misses can support a wrong narrative or budget decision. At the chosen operating point, the detector produces about **2.9 RROI alerts and 8.6 review hours per month**. The one-month arm targets sudden shocks; the cumulative arm targets three-month ramps. Across 20–40% injected shocks, recall was 34–74% sudden and 47–61% gradual, so coverage remains incomplete.

### 2. Probabilistic attribution

Each flagged business unit–brand–channel–month becomes one candidate, even if several metrics fire. Seven independent cause scores support multi-cause events. I use one-vs-rest L2-regularized logistic regression for the five causes with enough independent events, with median imputation and standardization learned inside each training fold. `class_weight='balanced'` addresses 2–10% cause prevalence, and a fixed `C=0.1` limits overfitting across 60 features. Mix shift and external demand are rule-scored because the panel contains only one independent event group for each; fitting them would create a misleading validation result. Scores are for ranking, not calibrated probabilities. A 0.30 threshold creates an explicit `insufficient_evidence` state. Gradient-boosted trees are the next challenger once multiple independently generated training panels make a fair comparison possible.

## Feature engineering

The 60 observable features describe current movement, recent trajectory, disagreement across metrics, and context. Each primary metric contributes a trailing robust residual, one- and three-month log changes, Task 1 trigger status, and signed anomaly scores. CTR, CPC, PICR, and impression share add residuals and changes; CTR and PICR also add one- and two-month lags. Cross-metric features capture the ROAS–RROI wedge, iROAS concordance, and MAS-versus-spend movement. Leave-one-out peer features indicate whether the business unit moved together, while brand-channel spend-share changes capture mix reallocation. Missingness and history features prevent cold start from masquerading as evidence. Every rolling baseline is shifted by one month, and preprocessing is fit within training folds, preventing future information and held-out statistics from entering a score.

## Evaluation

Anomaly thresholds are evaluated against injected shocks across all eligible series. The selected detector favors a manageable queue rather than maximum recall and remains imperfect by design. Attribution uses five event-grouped folds: overlapping months and co-occurring event IDs are merged so a real event cannot appear in both training and validation. PR-AUC is primary because causes are rare; prevalence and deterministic expert signatures are the baselines. Logistic regression beats both on three of five fitted causes and reaches macro PR-AUC 0.468, versus 0.341 for the rules and 0.036 for prevalence, but rules remain better for spend reduction and measurement artifacts. At the 0.30 threshold, 56% of candidates abstain; 88.9% of those abstentions are truly all-zero under the simulator. Strong creative-refresh performance largely reflects an easy injected CTR/PICR signature, not expected real-world accuracy. Before deployment, analysts would review ranked cases and label false alerts, missed explanations, and unknown causes.

## Causal honesty

The simulator provides causal ground truth only because its mechanisms were explicitly injected. The fitted model learns associations between observable patterns and those labels; it does not estimate the effect of spend, creative, or engagement on business outcomes. Even iROAS is only corroborating evidence here. Outputs therefore say that a case is “most consistent with” a cause under the synthetic training distribution, list the strongest supporting signals, and explicitly state that the result is not proof. Several causes are observationally similar at monthly grain, so overlapping scores are expected. Low or conflicting evidence produces `insufficient_evidence` and manual review, never an autonomous budget action.

## Productionisation

After monthly reporting closes, a versioned Task 1 pipeline would emit the investigation queue and the attribution service would return ranked causes, supporting signals, and abstention status. Analysts would confirm multiple causes, reject suggestions, or record an unknown cause in their existing workflow. Retraining should wait for roughly 100 reviewed investigations with reasonable cause coverage, rather than run on a fixed monthly schedule. Monitoring would cover missingness, feature drift, alert volume, cause-score prevalence, abstention rate, coefficient stability, and performance on adjudicated cases. New brands use pooled business-unit and channel context plus explicit history features; series with under 12 months of history fall back to conservative expert rules and a low-confidence manual-review state.

## Limits

Monthly aggregates cannot reliably separate survivorship from spend reduction or creative refresh from a broader efficiency gain at onset. Task 2 also inherits Task 1’s misses, two causes lack enough events for fitted models, and the ranking scores are not calibrated. Novel causes may abstain or resemble a known cause. The highest-value additions are creative launch metadata, placement-level inclusion and dropout logs, planned budgets, measurement-change logs, category-demand controls, and repeated incrementality experiments. With more time, I would generate independent training panels while holding seed 42 untouched, then fit the scarce causes, calibrate scores, add bootstrap intervals and evaluation slices, and compare regularized logistic regression with shallow XGBoost.
