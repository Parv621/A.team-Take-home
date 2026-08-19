# The Design Proposal

*Anomaly detection and probabilistic attribution for media-performance spikes*

## Problem framing

I frame the solution as a two-stage, human-in-the-loop decision-support system, not an automated attribution engine.

First, the [Task 1 anomaly-detection notebook][task-1-notebook] converts monthly business unit, brand, and channel data into a prioritized queue of unusual performance changes.

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

The [Task 2 attribution notebook][task-2-notebook] converts each deduplicated [Task 1 investigation][task-1-notebook] into seven independent cause scores. I use multi-label one-vs-rest classification because several mechanisms can occur together; scores do not need to sum to one. They rank explanations for analyst review and are not calibrated probabilities.

- **Model choice and baselines.** Five causes use separate L2-regularized logistic regressions. Median imputation and standardization are learned inside each training fold; `class_weight='balanced'` addresses rare causes, and fixed `C=0.1` limits overfitting. Mix shift and external demand use expert rules because each has only one independent event. Baselines are cause prevalence and deterministic expert signatures. I rejected multiclass softmax because causes co-occur. XGBoost is deferred because the small event count makes memorization more likely than a reliable comparison.
- **Cause taxonomy.** Genuine efficiency is a sustained CTR/PICR improvement; creative refresh is temporary. Spend reduction and survivorship represent delivery cuts with different severity. Measurement artifact is a ROAS–RROI divergence without real movement. Mix shift reallocates spend toward a stronger channel; external demand moves peers across a business unit. The first five causes are fitted and the last two are rule-scored.
- **Feature families.** ROAS, RROI, iROAS, MAS, and spend describe outcome movement and [Task 1 anomaly evidence][task-1-notebook]. CTR, CPC, PICR, and impression share distinguish engagement, auction cost, conversion, and delivery. Cross-metric features capture the ROAS–RROI wedge, iROAS–RROI agreement, and MAS-versus-spend movement. Business-unit peer movement identifies shared demand; channel spend share captures mix changes. Missingness and history stop sparse iROAS or cold start from becoming false evidence.
- **Classification flow.** Observable features are checked for leakage before seven binary targets are attached. Event IDs define five grouped folds, keeping connected months together. Each pipeline learns only from its training folds and scores the held-out fold. Performance metrics and ground-truth comparisons will be refreshed in Evaluation.

## Feature engineering

The pipeline creates **60 observable features** before opening either ground-truth file. This boundary prevents injected labels from influencing feature availability or construction.

- **Build the panel first.** For every business unit–brand–channel–month, each positive metric receives a robust log residual against its previous 12 months and one- and three-month log changes. The rolling window shifts by one month, so the current observation never enters its own baseline. Only after this step does the pipeline retain [Task 1 investigations][task-1-notebook] and attach their trigger flags and signed anomaly scores.
- **Represent both outcomes and mechanisms.** ROAS, RROI, iROAS, MAS, and spend describe the reported outcome. CTR, CPC, PICR, and impression share distinguish engagement, cost, conversion, and delivery mechanisms. The ROAS–RROI wedge isolates reporting movement; iROAS–RROI agreement tests incremental corroboration; and MAS-versus-spend movement separates genuine growth from a denominator effect.
- **Add time and business context.** Current residuals capture sudden changes, while one- and three-month changes capture direction and persistence. CTR and PICR also use residuals from `t-1` and `t-2` to distinguish a creative ramp from a one-month spike. Leave-one-out business-unit peer movement identifies shared demand, and current plus one- and three-month channel spend-share changes identify mix reallocation. Missingness and months of history tell the model when sparse iROAS or cold start makes the evidence less reliable.

**Signals and lag structure.** The table summarizes the dominant signal family and strongest individual evidence reported for each fitted cause.

| Cause | Dominant signal family | Strongest individual evidence | Timing and interpretation |
|---|---|---|---|
| Genuine efficiency gain | CTR | ROAS–RROI wedge | Current month; a larger wedge lowers the score, so the model looks for efficiency without a reporting disconnect. |
| Spend reduction artifact | Impression share | Three-month impression-share change | `t-2` to `t`; a decline raises the score and indicates delivery was cut. |
| Survivorship bias | Impression share | Current impression-share residual | `t`; a sharp negative residual is the strongest evidence of placements disappearing. |
| Creative refresh | PICR | Two-month-lagged PICR residual | `t-2`; a positive residual captures the ramp preceding the flagged month. |
| Measurement artifact | Cross-metric consistency | ROAS–RROI wedge | `t`; a larger positive wedge raises the score because ROAS moved without matching RROI. |

The two rule-scored causes use designed signals rather than learned importance. Mix shift uses one-month spend-share reallocation without matching CTR or PICR improvement, while external demand uses current leave-one-out peer movement.

## Evaluation

Verified real-world cause labels do not exist, so evaluation separates scoring from label inspection. The pipeline fixes all observable features and out-of-fold scores before opening the hidden synthetic labels. These labels test whether the model recovers mechanisms built into the simulator; they do not prove that the same relationships are causal in live campaigns.

- **Out-of-fold scoring and output.** [Section 6 of the Task 2 attribution notebook][task-2-notebook] fits five L2-regularized logistic regressions in event-grouped folds and scores the two scarce causes with expert rules. No event group appears in both training and validation. Each investigation receives seven independent cause scores, their model-or-rule source, ranked causes above the 0.30 threshold, and an abstention status. Scores do not need to sum to one because causes can co-occur.
- **Metrics and baselines.** PR-AUC is primary because each cause appears in only 2–10% of candidates; it measures how well true cases concentrate near the top of the ranking. ROC-AUC provides secondary ranking context, while Brier score and log loss expose unreliable score magnitudes and confident errors. Precision describes analyst queue quality, and recall describes cause coverage at the operating threshold. Logistic regression reaches macro PR-AUC **0.468**, compared with **0.341** for deterministic expert signatures and **0.036** for a prevalence-only baseline, and beats the rules on three of five fitted causes.
- **Feature-evidence validation.** [Section 8 of the Task 2 attribution notebook][task-2-notebook] keeps all 60 features in the fitted models and ranks evidence after classification; it does not select features and refit. Grouped permutation importance shuffles related blocks in held-out folds and measures the drop in PR-AUC, then repeats the test for individual columns. Standardized coefficients provide direction. Block results carry more weight because correlated lags can split credit across columns. Creative refresh has the strongest PICR evidence: the block reduces held-out PR-AUC by 0.455 when shuffled, led by the positive `t-2` PICR residual at 0.257. Survivorship similarly depends on falling impression share, with block and current-residual drops of 0.433 and 0.120. Measurement artifact uses the expected current ROAS–RROI wedge at 0.112, while spend reduction uses the expected three-month impression-share decline at 0.075. Genuine efficiency is less conclusive: CTR is its leading block at 0.132, but its strongest individual feature is the absence of a ROAS–RROI wedge at 0.062. With only 12 positive rows, the model partly learns what this cause is not. These values measure held-out ranking loss, not causal effect size.
- **Decision threshold and abstention.** At 0.30, creative refresh finds 27 of 28 cases with 0.519 precision, while survivorship bias reaches 0.650 precision and 0.867 recall; mix shift and external demand remain weak because they use sparse rule evidence. The policy abstains on 56.0% of investigations, and 88.9% of those abstentions contain no injected cause. The threshold is therefore a workload control: lowering it finds more causes but creates more reviews, while raising it produces a smaller, less complete queue.
- **Worked case and calibration.** For `Snacks / CrispBite / display / 2023-01`, the system correctly identifies all three injected causes—creative refresh, spend reduction, and survivorship bias—but also proposes genuine efficiency and measurement artifact. The result is three hits, two false alarms, and two correct rejections, illustrating how co-occurring mechanisms create observational overlap. Reliability curves show that class balancing inflates score levels, so the output says **“ranked highest for”**, not **“70% probability.”** With only 11–28 positive examples per fitted cause, post-hoc calibration would be unstable; reviewed analyst labels are required before deployment claims can replace synthetic validation.

## Causal honesty

Only one of the five fitted causes, `creative_refresh`, is the top scorer on its own positive rows. On true `survivorship_bias` rows the `spend_reduction_artifact` model scores higher, 0.813 against 0.799, and `measurement_artifact` ranks fourth of five on its own. Per-cause PR-AUC therefore overstates how much of the signal is cause-specific; a large share is a shared "something moved" component that the [attribution notebook][task-2-notebook] measures directly.

That finding sets the honesty boundary. The simulator supplies causal ground truth only because its mechanisms were injected. The model learns associations between observable patterns and those labels and estimates no effect of spend, creative, or engagement on business outcomes. Nothing was randomized or withheld, so the counterfactual is never observed, and iROAS is corroborating evidence rather than a causal warrant.

Selection compounds this. Only 106 of 254 true anomaly rows survive [Task 1][task-1-notebook] and reach attribution, so 58% of real anomalies are invisible to the model and 12 of 37 events never surface a candidate row. Every score is conditional on the detector version.

Output wording follows. A case is described as most consistent with a cause under the synthetic training distribution, its supporting signals are listed, and no proof is claimed. Overlapping scores are expected. Weak or conflicting evidence produces `insufficient_evidence` and manual review, never an autonomous budget action.

## Productionisation

In practice, the system runs after the monthly reporting close. [Task 1][task-1-notebook] creates the investigation queue, and [Task 2][task-2-notebook] returns ranked causes, supporting signals, and an abstention status. Analysts confirm or reject the suggestions, record an unknown cause where needed, and retain ownership of the final narrative.

- **Retraining with scarce labels.** The scarce unit is an independent confirmed event, not another alert row from the same event. The full synthetic panel contains 37 events across 48 months: `37 ÷ 48 × 12 = 9.2` events per year. The benchmark of 100 confirmed events is not a sample-size calculation; it is a round-number stress test showing how long even a modest three-digit validation set would take to build. Using all 37 events gives 130 months, or about 11 years. That is optimistic because only 25 of the 37 events reach Task 2; at that observed rate, the same benchmark is closer to 16 years. The model therefore remains synthetically pretrained, while analyst decisions build a separate validation and calibration set. Unreviewed alerts remain unlabeled; treating them as negatives would teach the model about analyst capacity rather than causes.
- **Retraining cadence.** The team reviews the model annually and refits it when drift is confirmed or the simulator's assumptions change. Annual review is a governance choice, not an optimized statistical interval. Monthly refitting would add false precision because, after overlapping events are grouped together for validation, the [Task 2 training set contains only 23 independent labeled event groups][task-2-notebook].
- **Monitoring.** The starting baselines come directly from the [Task 2 notebook run][task-2-notebook]: 466 investigations across 48 months gives **9.7 per month**; 261 of 466 investigations abstain, giving **56%**; and the engineered iROAS feature block has **11.8%** mean missingness. Production monitoring compares future alert volume, abstention, missingness, cause-score prevalence, feature distributions, and coefficient signs with these baselines. A sustained rise in abstention is particularly important because it suggests that the seven-cause taxonomy no longer explains what analysts are seeing.
- **Cold start.** Four of the 42 synthetic series have less than 12 months of history. The 12-month threshold comes from the [Task 1 history policy][task-1-notebook], while the robust residual also requires at least 9 valid observations in that trailing window—the 75% coverage rule. Until a new brand clears both conditions, it uses pooled business-unit and channel baselines, expert rules, and a low-confidence manual-review state. Once enough history exists, the shared model can score its normalized behavior and peer context without fitting a separate brand model.

## Limits

The model can prioritize investigations, but it cannot turn monthly correlations into proof.

- **Some causes remain indistinguishable.** Monthly data cannot reliably separate survivorship from spend reduction, or creative refresh from genuine efficiency at onset. These problems need placement-retention records and creative launch dates, not simply more rows.
- **Some causes have too little evidence.** Mix shift and external demand have one event each, while measurement artifact has 11 positive rows. More independent panels would improve stability; reviewed cases are still needed to test whether synthetic patterns transfer to real campaigns.
- **Coverage and confidence remain limited.** Scores rank explanations; they are not probabilities. Attribution also inherits the 58% of true anomaly rows missed by [Task 1][task-1-notebook], while an unseen cause may abstain or resemble the wrong known cause.
- **Better data matters most.** Priorities are placement and creative records, planned budgets, measurement-change logs, demand controls, and incrementality tests. Next steps are more independent panels, fitting the two rule-scored causes, unweighted calibration, and only then comparison with shallow XGBoost.

[task-1-notebook]: anomaly_detection/anomaly_detection.ipynb
[task-2-notebook]: attribution/attribution.ipynb
