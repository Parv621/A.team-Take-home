# Model-driven anomaly detection and attribution

*Design proposal for explaining media-performance spikes before monthly reporting*

## Executive answer

The evidence supports augmenting the manual process, but not replacing it. After the monthly close and before report circulation, the proposed system can identify statistically and commercially unusual changes, rank likely drivers, distinguish likely genuine gains from spend, mix, or reporting artifacts, and draft an evidence-backed explanation. The analyst remains responsible for confirming the cause, editing the narrative, and approving any budget decision.

The current proof of concept has three important limits. It was validated on synthetic rather than confirmed real-world causes, 58% of true anomaly rows never reach attribution, and several causes produce overlapping signals. The system should therefore launch as an analyst decision-support tool with an explicit abstention path, not as an autonomous attribution engine.

## Business need and operating model

A brand analyst currently investigates each spike with a manual checklist that takes days, often after the monthly report has circulated. The analytics lead owns the investigation queue, while brand and media-planning teams act on the result. The proposed system moves triage and an initial explanation ahead of report circulation without transferring accountability from the analyst.

The primary RROI monitoring view, based on media-attributable sales divided by spend, produces 2.9 alerts per month across 38 scored series, equivalent to about 8.6 analyst-hours. The workload estimate uses the midpoint of the 2-to-4-hour review range: `(137 alerts / 48 months) × 3 hours = 8.56 hours`, rounded to 8.6. Attribution considers flags across all five performance metrics, or 9.7 investigations per month, and abstains on 56% of them.

This design reflects an asymmetric cost. A false alert consumes 2 to 4 analyst-hours, while a miss can support an incorrect narrative or budget decision. The detector therefore favors precision and reports the recall it gives up instead of hiding it.

## Data strategy

The bottom-up simulator creates anomalies from business mechanisms rather than isolated metric jumps. Eight inputs—eligible impressions, impression share, click-through rate (CTR), cost per click (CPC), post-impression conversion rate (PICR), baseline PICR, average order value, and attribution inflation—include trend, seasonality, and autoregressive noise. Derived metrics preserve accounting identities; `data/data_dictionary.md` documents the definitions.

- **Events.** Each series contains zero to two events, starting at least four months apart. Events can be sudden or gradual, and half contain multiple causes. `data/design_decisions.md` contains the parameters.
- **Mechanisms.** Efficiency and creative refresh increase CTR and PICR; spend reduction and survivorship reduce impression share. Mix shift reallocates share, external demand affects a business unit, and measurement artifacts change attribution inflation only.
- **Labels and limits.** The pipeline withholds event labels until evaluation to prevent leakage. The simulator defines causality, signals are monthly aggregates, TV and out-of-home channels reuse generic engagement measures, and macro-event diversity is limited.

## Model design

The system has two stages. Stage 1 returns an anomaly flag and evidence score, but no cause. Stage 2 ranks seven causes independently because several can occur together. The scores do not sum to one and are not calibrated probabilities; an abstention state represents insufficient evidence.

### Stage 1: Anomaly detection

- **Baseline and thresholds.** The selected, look-ahead-safe detector replaces a rolling mean and standard deviation with median and median absolute deviation (MAD) z-scores for one-month and three-month changes. On return on ad spend (ROAS), it flags 74 level shifts, compared with 48 for the naive baseline. A change must exceed both an absolute z-score of 2.5 and its metric's 75th-percentile movement floor: RROI 14%, ROAS 15%, media-attributable sales (MAS) 22%, spend 24%, and incremental ROAS (iROAS) 37%.
- **Seasonality.** Rolling MAD does not remove seasonality; it absorbs seasonal change into the noise estimate. This raises the effective threshold for volatile series. Explicit seasonal decomposition remains untested.
- **Metric roles.** RROI is primary based on recall and false-positive rate. ROAS is secondary, MAS and spend expose denominator effects, and iROAS corroborates despite only 31.5% z-score coverage. CTR, CPC, PICR, and impression-share movements provide supporting evidence, not cause labels.
- **Limited history.** Series with less than 12 months of history go to manual review. Those with 12 to 24 months use a six-month test at an absolute z-score above 3.0; longer series use the dual test. Every window requires at least 75% valid observations, without imputation.
- **Detection performance.** For RROI shocks of 20% to 40%, recall ranges from 34.2% to 73.7% for sudden events and 47.4% to 60.5% for gradual events, against an 8.5% false-positive rate.

### Stage 2: Driver attribution

Formally, this is a multi-label, one-versus-rest binary classification problem over the Stage 1 investigation queue. One example is one deduplicated business unit–brand–channel–month, even when several metrics flag it. For each cause `c`, the target `Y_c` equals 1 when that cause is active and 0 otherwise. A row can therefore have several positive targets or an all-zero target vector. Given the 60 observable features `X`, each scorer ranks evidence for `Y_c = 1`, conditional on Stage 1 having flagged the row. Scores above 0.30 form the proposed cause set; if none qualify, the system abstains.

Five causes use separate L2-regularized logistic regressions. Imputation and standardization are learned within each training fold, `class_weight='balanced'` addresses rarity, and `C=0.1` is fixed in advance. `mix_shift_artifact` and `external_demand_spike` use rules because each has only one independent event.

This hybrid avoids forcing co-occurring causes into one probability distribution. Its outputs are uncalibrated ranking scores, not probabilities. XGBoost is deferred because 23 labeled event groups cannot show whether its extra capacity would generalize or merely memorize events.

## Evidence used for attribution

The pipeline creates 60 observable features before opening either ground-truth file. Features include prior-12-month residuals; one-month and three-month changes; lagged CTR and PICR; peer movement; and spend-share changes. The baseline shifts by one month so that an observation never enters its own baseline.

Grouped permutation tests measure the loss in precision-recall area under the curve (PR-AUC) when a signal block is shuffled in held-out data. Blocks matter more than individual columns because correlated lags can split credit. This ranks evidence after classification; all features remain in the models.

| Cause | Main signal block (PR-AUC loss) | Strongest feature (loss) | Interpretation |
|---|---:|---:|---|
| Creative refresh | PICR (0.455) | `picr_resid_lag2` (0.257) | A positive residual two months earlier identifies the ramp. |
| Survivorship bias | Impression share (0.433) | `impression_share_resid` (0.120) | A sharp current decline suggests that placements disappeared. |
| Spend reduction artifact | Impression share (0.134) | `impression_share_d3` (0.075) | A sustained decline suggests that delivery was cut rather than improved. |
| Genuine efficiency gain | CTR (0.132) | `wedge_roas_rroi` (0.062) | A negative wedge represents efficiency without a reporting disconnect. |
| Measurement artifact | Cross-metric consistency (0.103) | `wedge_roas_rroi` (0.112) | A positive wedge means ROAS moved while RROI did not. |

Each fitted cause relies most on the expected signal block. Genuine efficiency is less conclusive: with only 12 positive rows, the model partly learns the absence of a ROAS-RROI reporting wedge. These values are ranking losses, not causal effect sizes.

## Analyst output

Each investigation produces a short narrative card:

> **{Business unit} / {brand} / {channel} / {month}:** The observed pattern is most consistent with **{top cause} ({score})** under the synthetic training distribution. The largest contributing signals are {top three signed contributions}. The second-ranked cause is **{runner-up} ({score})**. This is diagnostic evidence, not proof of cause.

If no score clears the threshold, the card reports the highest score and routes the investigation to manual review. Signed feature contributions let the analyst check the evidence before accepting or revising the explanation. For example, the `Snacks / CrispBite / display / 2023-01` card ranks survivorship bias first and identifies falling impression share as the dominant signal; its 1.00 score is not 100% confidence.

## Evaluation and decision readiness

Verified real-world cause labels do not exist. The pipeline fixes features and out-of-fold scores before opening the synthetic labels, which tests simulator recovery rather than performance in live campaigns.

- **Validation and baselines.** Grouped five-fold validation has zero event-group leakage. Logistic regression reaches a macro PR-AUC of **0.468**, compared with **0.341** for expert signatures and **0.036** for prevalence alone. It beats the rules on three fitted causes but loses on spend reduction and measurement artifact, supporting a hybrid rather than complete model replacement.
- **Threshold and abstention.** At 0.30, creative refresh finds 27 of 28 cases at 0.519 precision; survivorship reaches 0.650 precision and 0.867 recall. Of abstained investigations, 88.9% contain no injected cause, compared with 37.6% of proposed cases. Raising the threshold from 0.20 to 0.50 reduces the reviewed share from 60% to 25%.
- **Calibration.** Class balancing inflates scores, so the output uses ranking rather than probability language. With only 11 to 28 positive examples per fitted cause, calibration would fit noise. Creative refresh's 0.876 PR-AUC reflects direct CTR and PICR changes in the generator and should not be expected in production.

## Causal limits

The system distinguishes genuine improvements from artifacts only when their observable signatures differ. `creative_refresh` is the only fitted cause that ranks first on its own positive rows. On true `survivorship_bias` rows, spend reduction scores 0.813, compared with 0.799 for survivorship, while `measurement_artifact` ranks fourth of five. Cause-level PR-AUC therefore overstates specificity because the models share a broad “something changed” signal.

The models learn associations from injected causes; they do not estimate the effects of spend, creative, or engagement. No intervention was randomized, and no counterfactual was observed. Only 106 of 254 true anomaly rows reach attribution, and 12 of 37 events never produce a candidate, so every score is conditional on the detector.

The output must therefore describe evidence as “most consistent with” a cause and avoid claims of proof. Weak or conflicting evidence produces `insufficient_evidence` and manual review, never an autonomous budget action.

## Production plan

The workflow runs after the monthly close and before report circulation. Stage 1 creates the queue, Stage 2 returns ranked causes and evidence, and the narrative card gives the analyst a starting point. Analysts retain ownership of the final explanation.

- **Learning from scarce labels.** The scarce unit is an independently confirmed event, not another row from the same event. The panel yields 9.2 events per year, and only 25 of 37 reach Stage 2, so a 100-event validation set would take 11 to 16 years. The model remains synthetically pretrained while analyst decisions build a separate validation and calibration set. Unreviewed alerts remain unlabeled.
- **Review cadence.** The team reviews the system annually and refits it after confirmed drift or changed simulator assumptions. Monthly refitting on 23 labeled event groups would create false precision.
- **Monitoring.** Initial baselines are 9.7 investigations per month, 56% abstention, and 11.8% missingness in the iROAS feature block. Monitoring covers volume, abstention, missingness, cause prevalence, feature distributions, and coefficient signs. Rising abstention can indicate an incomplete cause taxonomy.
- **Cold start.** Four of 42 series have less than 12 months of history. Until a brand has 12 months of history and meets the 75%-valid-observation requirement, it uses pooled baselines, expert rules, and manual review. It can then use the shared model without a separate brand model.

## Limits and next steps

- Monthly data cannot reliably separate survivorship from spend reduction or creative refresh from genuine efficiency at onset. Placement-retention and creative-launch records would provide the missing evidence.
- Mix shift and external demand have only one event each, while measurement artifact has 11 positive rows.
- Attribution inherits the 58% of true anomaly rows missed by Stage 1. An unknown cause might trigger abstention or resemble the wrong known cause.

Next, generate 8 to 10 independent panels, reserve seed 42 as a test set, fit the rule-scored causes, evaluate calibration, and then compare a constrained XGBoost model. Also compare explicit deseasonalization with the trailing-MAD detector and derive movement sensitivity by metric.

Operational data matters more than either modeling step. Placement and creative logs, planned budgets, measurement-change records, and incrementality tests are the clearest path toward causal evidence. Forecasting remains out of scope because the brief prioritizes detection and attribution.

*Time spent: approximately 1.5 working days; see `README.md` for the breakdown.*
