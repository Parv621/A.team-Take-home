# Attribution design proposal

*Companion notebook: `attribution.ipynb` (runs end-to-end from a clean kernel). Shared pipeline: `attribution_lib.py`. All numbers below are produced by that notebook, not asserted.*

## Problem framing

**Stakeholder and decision.** A brand analyst on the CPG paid-media team who today spends days per spike on a manual checklist. Task 1 hands them a queue; this system tells them which explanations the evidence resembles so they can triage the queue and draft the narrative. It does not move budget.

**Cost asymmetry.** A false positive costs analyst hours chasing a story that isn't there. A false negative costs a wrong narrative reaching a business stakeholder, which is worse and harder to unwind. The system is therefore built to abstain loudly: 56% of candidates score below the abstention threshold on every cause and are routed to manual review rather than given a guess.

**Deployable?** Yes, as a monthly diagnostic *scoring* service behind a human reviewer. No, as an automated attribution engine, for the identification reasons under Limits.

Attribution runs only after Task 1 flags a row. The unit of analysis is one `BU x brand x channel x month` investigation, even when several primary metrics flag the same row: metric-specific alerts collapse into one candidate carrying trigger indicators and anomaly scores, not several training examples.

The ML problem is **multi-label binary classification**. For each cause `c` the model estimates

$$P(Y_c=1 \mid X,\; \text{Task 1 flagged})$$

with one independent binary classifier per cause. Softmax and multiclass are rejected outright: the generator draws 1-3 causes per event at 50/35/15%, so causes demonstrably co-occur and probabilities must not be forced to sum to one. A full 2^7 joint-label model and classifier chains are also rejected: 37 events do not cover enough label combinations, and chain results depend on label order.

Conditioning on the Task 1 flag is deliberate. Training on flagged rows only reproduces the production population rather than the easier but mismatched `P(Y|X)` over all months.

## Data strategy

**Metric selection.** All five primary metrics (ROAS, RROI, iROAS, MAS, spend) enter the feature matrix. The exercise brief names five primary metrics and does not ask for a subset, and narrowing measurably hurts: restricting to RROI/ROAS/iROAS drops the candidate set from 466 to 245, the labelled candidates from 106 to 59, and roughly halves the positives on every cause (`mix_shift_artifact` falls from 12 rows to 3). Task 1's metric ranking is not reused here, because **detector quality and feature informativeness are different questions**. iROAS is a poor detector (15.8% recall at a 40% shock) but a strong feature: the data dictionary's cause signatures invoke it for three of seven causes. At the feature level its trailing baseline is computable on 79.7% of rows, identical to ROAS, RROI and spend, because the 9-of-12-month tolerance absorbs its 15% missingness. All four leading indicators (CTR, CPC, PICR, impression share) are included.

**Candidate construction.** Re-running the Task 1 detector (held verbatim in `attribution_lib.py`, not read from the possibly-stale flag CSVs) gives 779 metric-specific flags, which outer-union and deduplicate to **466 investigations** at 1.67 flags each. 232 fire on one metric, 175 on two, 59 on three or more. Of those, **106 coincide with an injected anomaly row (candidate precision 0.23)**; the other 360 are retained with an all-zero target vector. Those negatives are not noise to be discarded, they are what teaches the model to score low instead of manufacturing an explanation.

**Label scarcity is the binding constraint**, and it is about events, not rows:

| cause | positive rows | independent events | fitted? |
|---|---|---|---|
| creative_refresh | 28 | 10 | yes |
| spend_reduction_artifact | 18 | 12 | yes |
| survivorship_bias | 15 | 8 | yes |
| genuine_efficiency_gain | 12 | 8 | yes |
| measurement_artifact | 11 | 7 | yes |
| mix_shift_artifact | 12 | **1** | no - rule scored |
| external_demand_spike | 45 | **1** | no - rule scored |

`mix_shift_artifact` and `external_demand_spike` are each one macro event. A classifier fitted on one event is validated against itself, so both are scored by the deterministic expert rule and labelled as such in every output row. Reporting 45 "training examples" for external demand would be reporting one event seven ways.

**What I would do with more time, and it is the single highest-value change:** the generator is deterministic and parameterising it by seed and output directory is a three-line change that runs in 1.9 seconds per panel. Generating 8-10 auxiliary training panels while holding seed-42 untouched as a final test would supply the independent events needed to fit the two rule-scored causes, support calibration, and make the evaluation slices below worth running. That was scoped out of this build, not overlooked.

## Model design

**Chosen: one-vs-rest L2-regularised logistic regression.** Seven independent binary models give a factorised Bernoulli approximation to the joint multi-label distribution, train in seconds, and expose signed per-case contributions an analyst can read. Median imputation and standardisation sit *inside* the sklearn pipeline so their statistics are learned on training folds only.

`C = 0.1` is fixed **a priori** from the feature-to-positive ratio, not tuned on the data. With 60 features and 11-28 positives per cause the design is 2-5 features per positive and materially over-parameterised. A development-time sweep, not shipped in the notebook, confirmed the choice is not on a cliff: macro PR-AUC ran 0.485 / 0.468 / 0.439 / 0.433 at C = 0.03 / 0.1 / 0.3 / 1.0, falling monotonically as regularisation weakens, which is itself the signature of over-parameterisation. `class_weight='balanced'` offsets prevalence of 2-10% without resampling, at the calibration cost quantified below.

**Rejected, with reasons.** *Gradient-boosted trees (XGBoost)* are the natural challenger and would test whether the conjunctions matter (genuine efficiency needs PICR and CTR up *while* spend and impression share hold). They are not implemented here: with 23 labelled event groups, a boosted model's extra capacity would be spent memorising events, and choosing between the families on a 23-group CV would not be a real selection. This is a deferral on evidence, not a claim that logistic regression wins in general. *LightGBM and CatBoost* are credible alternatives to XGBoost, but the inputs are almost entirely numeric so CatBoost's categorical handling adds little, and a three-library bake-off would spend the budget on breadth rather than attribution quality. *Neural networks* are unnecessary on a 466-row structured problem. *Bayesian causal networks* would add strong priors and computation without resolving identification from monthly aggregates.

**Calibration: none, deliberately.** Platt scaling needs an independent calibration sample; with 11-28 positives per cause it would fit noise, and isotonic regression is far worse at this size. Outputs are therefore described as **ranking scores, not calibrated probabilities**, everywhere they appear. The notebook measures the gap: the pooled reliability curve sits below the diagonal at every bin (top bin: mean score 0.56 against a true rate of 0.23), and per-cause mean scores run 1.4x to 4.2x above the actual rate, 2.7x pooled. `class_weight='balanced'` causes this by treating a 3% class as if it were 50%, which lifts the intercept. Calling these probabilities would be the single easiest place to overclaim, so the interface does not.

## Feature engineering

60 features in 10 blocks, all computed from information available at month `t`. Normalisation has two look-ahead-safe stages.

**Stage 1, within-series centering.** For positive metric `x`, a trailing robust residual

$$z^{robust}_{x,t}=\frac{\log(x_t)-\operatorname{median}(\log x_{t-12:t-1})}{1.4826\,MAD(\log x_{t-12:t-1})}$$

removes each series' local baseline while resisting contamination from earlier spikes. The `shift(1)` before the rolling window is what makes it safe, and this is verified rather than asserted: recomputing one series' residual using only months `<= t` reproduces the vectorised value to 6 decimal places. A second check asserts that no ground-truth column can reach the feature list.

**Stage 2, model standardisation.** Mean-centering and scaling happen inside the pipeline, fit on training folds only and applied unchanged to held-out folds.

| Block | Contents | Attribution role |
|---|---|---|
| Primary metric x5 | residual, 1- and 3-month log change, Task 1 trigger, signed level and trend z | which metrics moved, how much, how suddenly |
| Leading indicators x4 | residual, 1- and 3-month log change; plus 1- and 2-month lagged residuals for CTR and PICR | separates a one-month jump from a multi-month ramp |
| Cross-metric | ROAS-RROI wedge, iROAS-RROI concordance, MAS-minus-spend movement, count elevated, count triggered | real movement vs reporting or denominator artifacts |
| Peer context | leave-one-out BU-month median residual, fraction of peers elevated, target-minus-peer | the only evidence for external demand |
| Mix context | channel share of brand spend, 1- and 3-month share change | reallocation without within-channel improvement |
| Reliability | iROAS-missing flag, missing count, months of history | stops missingness or cold start reading as evidence |

The peer block excludes the target row from its own peer aggregate, which is what makes it evidence about the market rather than about the target.

**Measured importance, not asserted.** Grouped permutation importance on out-of-fold data is the primary method: whole blocks are permuted together so correlated lags of one signal cannot split credit. Raw coefficients are reported for *direction only*; at 60 correlated features and C=0.1, coefficient magnitude is not an importance claim. The blocks each cause leans on match the injected mechanisms:

| cause | top block | PR-AUC drop | matches expected signature? |
|---|---|---|---|
| creative_refresh | leading PICR | 0.485 | yes - PICR/CTR ramp |
| genuine_efficiency_gain | leading CTR | 0.455 | yes - CTR/PICR with stable spend |
| survivorship_bias | impression share | 0.409 | yes - sharp impression-share cut |
| measurement_artifact | cross-metric | 0.285 | yes - the ROAS-RROI wedge |
| spend_reduction_artifact | reliability | 0.114 | no - see below |

## Evaluation

Splits are **event-grouped, never random by row**. Overlapping event ids are merged into connected components by union-find, so every month of a multi-causal event stays on one side of the split; the notebook asserts zero group violations across all 5 folds. Out-of-fold predictions from `StratifiedGroupKFold` are what all numbers below report. PR-AUC is primary because prevalence is 2-10%; ROC-AUC, Brier and log loss are context.

**Both baselines are real.** Prevalence-only predicts the base rate (PR-AUC equals prevalence by construction, the floor). The expert signature is a deterministic rule per cause written straight from the data dictionary's mechanism table, and it is a strong baseline, not a strawman.

| cause | prevalence floor | expert signature | logistic L2 | verdict |
|---|---|---|---|---|
| survivorship_bias | 0.032 | 0.819 | **0.891** | model wins |
| creative_refresh | 0.060 | 0.313 | **0.876** | model wins |
| genuine_efficiency_gain | 0.026 | 0.097 | **0.237** | model wins |
| spend_reduction_artifact | 0.039 | **0.269** | 0.246 | **rule wins** |
| measurement_artifact | 0.024 | **0.205** | 0.092 | **rule wins** |
| **macro (5 fitted)** | **0.036** | **0.341** | **0.468** | model wins overall |

The logistic model beats both baselines on three of five causes and macro-averages 0.468 against the rule's 0.341 and the 0.036 floor. **It loses to the hand-written rule on two.** For `measurement_artifact` the model finds the right signal (the ROAS-RROI wedge is its largest coefficient at +0.996) but cannot fit it from 11 positives across 60 features, while the rule encodes it directly. The honest recommendation is a hybrid: ship the rule for `measurement_artifact` and `spend_reduction_artifact` alongside the rule-scored macro causes, and the fitted model for the other three.

**The abstention state is the safety property that matters.** 261 of 466 candidates (56%) fall below threshold on every cause. Of those, **88.9% genuinely have no injected cause**, against 62.4% among candidates where a cause is proposed. The system is therefore concentrating its confidence where the labels actually are.

**Interrogating results that look too good.** `creative_refresh` at PR-AUC 0.876 against a 0.060 floor demanded a leakage check. Two development-time checks, cut from the notebook to keep it minimal, cleared it. It is simulator recovery rather than leakage: the generator multiplies CTR and PICR directly for this cause, and `creative_refresh` has 22 rows uncontaminated by `genuine_efficiency_gain` against the latter's 6. Nothing resembling 0.876 should be expected on real data. An ablation of the `reliability` block, which had ranked suspiciously high for `spend_reduction_artifact`, moved every cause by at most 0.024 PR-AUC with inconsistent sign, so metadata is substitutable rather than load-bearing.

**Cut from this build, and why.** Sudden-vs-gradual, single-vs-multi-cause, seen-vs-unseen-brand and event-magnitude slices, event-grouped bootstrap CIs, and top-2 recall are all specified and all omitted: with 23 labelled event groups, slicing five ways leaves single-digit cells whose intervals would be too wide to inform a decision. They become worthwhile the moment auxiliary panels exist.

## Causal honesty

The model estimates `P(injected cause label | observed evidence, Task 1 flagged)`. It does **not** estimate the causal effect of CTR, spend or creative on performance, and no part of this pipeline could: nothing was randomised or withheld, so the counterfactual is never observed. Synthetic causes are causal by construction *inside the simulator*; a classifier trained to recognise their signatures remains associational when pointed at real data. iROAS is itself an incrementality estimate, but using it as one feature does not make the attribution model causal.

Ground truth here validates recovery of a known data-generating process, not truth in the market.

**Two claims that are usually asserted are measured instead.**

*The selection boundary is 58%.* The panel contains 254 true anomaly rows. Only 106 (42%) survive Task 1 and reach attribution; 148 are invisible to it, and 12 of 37 events never surface a single candidate row. Attribution cannot explain what the detector did not flag, and this is a measured share rather than a caveat.

*The causes are far less separable than per-cause PR-AUC suggests.* Scoring each cause's true positive rows with every model gives:

| true cause (n) | genuine | spend_red | surviv | creative | measure |
|---|---|---|---|---|---|
| genuine_efficiency (12) | 0.465 | **0.504** | 0.219 | 0.499 | 0.166 |
| spend_reduction (18) | 0.393 | 0.420 | 0.311 | **0.467** | 0.333 |
| survivorship (15) | 0.376 | **0.813** | 0.799 | 0.561 | 0.278 |
| creative_refresh (28) | 0.430 | 0.578 | 0.319 | **0.881** | 0.314 |
| measurement_artifact (11) | 0.206 | **0.480** | 0.311 | 0.447 | 0.296 |

**Only `creative_refresh` is the top scorer on its own positive rows.** `survivorship_bias` loses to `spend_reduction_artifact` by 0.813 to 0.799, which is the near-identity in the data dictionary appearing in measured output; `measurement_artifact` ranks fourth of five on its own rows. A meaningful share of each model's apparent discrimination is a shared "something anomalous happened" signal rather than cause-specific evidence. That is the concrete reason the interface presents ranked candidates with an abstention state instead of a single answer.

The analyst-facing language is hedged to match, and is generated directly by the notebook:

> *Snacks / CrispBite / display / 2023-01: the observed pattern is most consistent with survivorship bias (1.00) under the synthetic training distribution. Largest contributing signals: impression_share_resid (+5.55), impression_share_d1 (+3.23), impression_share_d3 (+1.57). Second candidate: spend reduction artifact (1.00). This is diagnostic evidence, not proof of cause.*

Both survivorship and spend reduction score 1.00 there, and the true label carries both plus creative refresh. Overlapping scores are correct behaviour, not indecision. Where evidence is weak the output abstains instead of reaching for the least-bad label:

> *HomeCare / CleanWave / paid_search / 2023-11: insufficient evidence for a known cause (top score measurement artifact 0.03, below the 0.30 abstention threshold). Route to manual review.*

No autonomous budget action is taken.

## Productionisation

The system runs monthly once reporting closes. Task 1 emits a deduplicated candidate table; a versioned feature pipeline applies saved trailing-baseline and standardisation parameters; the attribution service returns cause scores, top supporting signals and an abstention state. Analysts confirm causes, reject suggestions, or mark unknown in the existing investigation workflow.

**Retraining cannot be label-driven, and the arithmetic decides this.** The panel produces 37 events over 48 months across 42 series, or **9.2 events per year**. Waiting for 100 adjudicated investigations would take roughly 130 months. An 11-year retraining cycle is not a policy, so the design has to change rather than the cadence:

- The model is **synthetically pretrained** and refreshed when the generator's assumptions are revised, not when labels accumulate.
- Adjudications accumulate as a slow **validation and calibration** set, not a training set. At ~9 events per year, the first meaningful calibration check lands after roughly five years.
- The retrain **trigger is drift, plus an annual scheduled refit**. Never a monthly refit: with 23 event groups that resamples noise.

**Drift detection is therefore the primary monitoring job**, because it fires years before labels do. Monitored against the baselines this build establishes: Task 1 alert volume (~9.7 candidates/month), abstention rate (56%), per-cause score prevalence, iROAS missingness (11.8%), feature distributions across the 60 columns, and coefficient sign stability between refits. A rising abstention rate is the earliest signal that the taxonomy no longer covers what is happening.

**One trap worth naming: an unreviewed alert is not a negative.** Training on "the analyst never confirmed it" learns analyst workload, not causes. Only explicit adjudications count, and unreviewed rows stay unlabelled.

**Cold start** is concrete rather than hypothetical. Four of 42 series are already unscoreable at under 12 months of history, and a robust residual needs 9 of the trailing 12 months present. A new brand receives pooled BU-channel baselines, expert-rule scores only, and an explicit low-confidence flag until it clears 12 months. It never gets its own model. An unseen brand is still scorable thereafter because the model uses normalised behaviour and peer context rather than brand identity.

## Limits

**More rows fix variance; only more columns fix identification.** This distinction decides what is worth building next:

| cause | limited by | do auxiliary panels help? |
|---|---|---|
| `mix_shift_artifact`, `external_demand_spike` | one event each | **yes** - distinct mechanisms, no data |
| `measurement_artifact` | 11 positives | **yes** - the ROAS-RROI wedge is a distinct signature the model already locates (top coefficient +0.996) |
| `survivorship_bias` vs `spend_reduction_artifact` | identification | **no** - near-identical by construction; needs placement-level retention data |
| `genuine_efficiency_gain` vs `creative_refresh` | both | partly - 12 positives, and separation at onset needs creative launch dates |

**Not calibrated.** These are ranking scores, running 1.4x to 4.2x above the true rate per cause and 2.7x pooled, because `class_weight='balanced'` treats a 3% class as if it were 50%. Calling them probabilities would require both an unweighted refit and far more than 23 independent event groups.

**Not fitted.** `mix_shift_artifact` and `external_demand_spike` are rule-scored on one event each.

**Inherited selection boundary.** 58% of true anomaly rows never reach attribution, and every score is conditional on the detector version. A novel cause outside the taxonomy will either score low or, worse, resemble a known one.

**Two ways to add data, fixing different problems.** Re-running the seed-parameterised generator yields ~10 panels in about 20 seconds with exact labels, fixing variance, calibration and the two one-event causes. Analyst adjudication yields ~9 noisy labels a year with the real-world distribution, fixing simulator-to-real shift. Neither adds a column, so neither resolves identification.

**Most valuable additional data**, in order: creative and campaign launch metadata; placement-level inclusion and dropout logs; planned budgets and optimisation actions; attribution-methodology change logs; category demand and promotion controls; repeated incrementality experiments with uncertainty intervals.

**With more time**, in priority order: parameterise the generator by seed and produce 8-10 auxiliary training panels holding seed-42 as an untouched test, which unlocks the two rule-scored causes, an unweighted-plus-calibrated refit, the evaluation slices, and a fair XGBoost comparison in that order.

Until then this is a transparent, human-in-the-loop diagnostic prioritisation system, not a causal attribution engine.
