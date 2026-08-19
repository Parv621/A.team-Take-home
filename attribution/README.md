# Probabilistic attribution

This folder contains a proof-of-concept workflow that takes the investigations flagged by Task 1 and ranks the possible causes of each anomaly. Causes are scored independently because more than one cause can be present in the same month. The scores are diagnostic ranking signals—not causal estimates or calibrated probabilities.

## Quick start

From the repository root, run `python run_all.py` to regenerate the data and execute both notebooks in order. The attribution notebook reads the observable data from `../data`, uses hidden ground truth only for target construction and evaluation, and rewrites its untracked CSV outputs in this folder.

Install the current dependencies from the repository root if needed:

```powershell
python -m pip install -r requirements.txt
```

## What the notebook does

1. Re-runs the Task 1 detector instead of trusting possibly stale flag files.
2. Unions metric-specific alerts into one `BU × brand × channel × month` investigation.
3. Builds 60 look-ahead-safe features from trailing robust residuals, recent movements, Task 1 trigger evidence, leading indicators, cross-metric consistency, peer context, mix context, and reliability.
4. Creates one binary target per cause; cause scores are not forced to sum to one.
5. Uses event-grouped, five-fold out-of-fold evaluation so overlapping months from the same event remain together.
6. Fits L2-regularized logistic regression for five causes. `mix_shift_artifact` and `external_demand_spike` are rule-scored because the candidate set contains only one independent event group for each.
7. Compares the fitted scores with prevalence and expert-signature baselines, calculates grouped permutation importance, creates local linear explanations, and exports analyst-facing results.

The hidden fields in `data/ground_truth` never enter the feature matrix. They are used only to construct labels, keep related events in the same fold, and evaluate recovery.

## Current data flow

```text
brief_metrics.csv
      │
      ▼
re-run Task 1 detector ──► 779 metric-level flags
      │
      ▼
deduplicate investigations ──► 466 candidates
      │                         106 injected-anomaly rows
      │                         360 all-zero rows
      ▼
engineer 60 features + attach evaluation labels
      │
      ▼
5 logistic models + 2 expert rules
      │
      ├──► attribution_predictions.csv
      ├──► model_comparison.csv
      └──► feature_importance.csv
```

## Output files

| File | Contents |
|---|---|
| `attribution_predictions.csv` | One row per candidate, true labels for evaluation, seven independent cause scores, scorer type, top cause, and abstention state |
| `model_comparison.csv` | Per-cause PR-AUC, ROC-AUC, Brier score, and log loss for the prevalence baseline, expert signature, and fitted/rule scorer |
| `feature_importance.csv` | Event-grouped, held-out permutation importance for the five fitted logistic models |
| `attribution_lib.py` | Shared Task 1 detector, feature engineering, event grouping, and expert rules |
| `classification_flow.png` | Classification-flow diagram displayed in the notebook |
| `render_classification_flow.py` | Standalone script that regenerates the classification-flow diagram |
| `proposal_attribution.md` | Detailed design rationale, results, operating model, and limitations |

## Ground-truth check

The saved outputs reconcile cleanly to `data/ground_truth` at the key and label level:

- All 466 candidate keys exist in `ground_truth.csv`, with no duplicate investigations.
- `true_causes` and all 3,262 binary target cells exactly match the hidden labels.
- Task 1 captures 106 of 254 injected anomaly rows (41.7%) and 23 of 35 independent event components (65.7%). Candidate precision is 22.7%: 106 labelled candidates and 360 detector-selected all-zero candidates.
- At the notebook's 0.30 threshold, 253 of 466 rows have an exact predicted label set. This headline is dominated by 232 correct all-zero abstentions; among the 106 labelled candidates, only 21 (19.8%) have an exact cause set.
- At least one true cause is proposed above threshold for 59 of 106 labelled candidates (55.7%). The top-ranked cause is true for 66 of 106 (62.3%), and the top two ranks contain a true cause for 87 of 106 (82.1%).
- Of the 360 all-zero candidates, 232 (64.4%) abstain correctly and 128 receive at least one false cause proposal.
- The exported scores stay in `[0, 1]` individually and are not softmax-normalized; 102 rows have a total score above 1, as expected for a multi-label design.

Per-cause results on the 466 Task 1 candidates:

| Cause | Positive rows | Independent event groups | Scorer | PR-AUC | Precision @ 0.30 | Recall @ 0.30 |
|---|---:|---:|---|---:|---:|---:|
| Genuine efficiency gain | 12 | 8 | Logistic L2 | 0.237 | 17.4% | 66.7% |
| Spend reduction artifact | 18 | 12 | Logistic L2 | 0.246 | 14.5% | 44.4% |
| Survivorship bias | 15 | 8 | Logistic L2 | 0.891 | 65.0% | 86.7% |
| Creative refresh | 28 | 10 | Logistic L2 | 0.876 | 51.9% | 96.4% |
| Measurement artifact | 11 | 7 | Logistic L2 | 0.092 | 12.8% | 45.5% |
| Mix shift artifact | 12 | 1 | Expert rule | 0.035 | 0.0% | 0.0% |
| External demand spike | 45 | 1 | Expert rule | 0.180 | 17.6% | 51.1% |

The strong survivorship and creative-refresh results reflect easily recoverable simulator signatures. The low precision on several causes, zero recall for mix shift, and weak exact-set recovery show that the current scores do not yet match ground truth reliably enough for automated decisions.

## Scope status and overall plan

The current notebook is an honest single-panel proof of concept, but it does **not** yet implement the full requested design:

| Requested end state | Current notebook | Next step |
|---|---|---|
| Exactly three primary metrics | Uses all five | Restrict candidates and fitted primary-metric features to **RROI, ROAS, and Spend**. Task 1 establishes RROI and ROAS as the strongest detectors; Spend is the defensible third metric because it is needed to diagnose spend reduction and mix reallocation. |
| Logistic regression vs. XGBoost | Logistic only; two scarce causes use rules | Generate sufficient independent training events, then compare one-vs-rest L2 logistic regression with shallow, regularized one-vs-rest XGBoost. |
| Seed-42 panel held out for final testing | Uses event-grouped cross-validation on seed 42 | Parameterize the generator, create roughly 8–10 auxiliary panels, split by complete panel/event component, and open seed 42 only after model selection. |
| Calibrated probabilities | Uncalibrated ranking scores | Use a separate validation panel for Platt calibration if event counts are adequate; otherwise retain the score terminology. |
| Evaluation slices and uncertainty | Core metrics only | Add event-grouped bootstrap intervals, missingness/history slices, sudden vs. gradual, single vs. multi-cause, unseen-brand, magnitude, and abstention coverage/error. |
| Comparable model explanations | Logistic grouped importance and local contributions | Apply grouped held-out permutation importance to both families and add TreeSHAP contributions for XGBoost. |

The production decision should remain human-in-the-loop. A low or conflicting score pattern should return `insufficient_evidence`; no score should be interpreted as proof that a metric caused the observed performance change.
