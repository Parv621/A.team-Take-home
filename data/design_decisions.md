## Design decisions summary: synthetic CPG media dataset

**Current release**

- Version 4 contains 1,642 rows across 42 business unit–brand–channel series and 48 monthly periods, from January 2022 through December 2025. The deterministic generator uses seed 42.
- The release contains 37 injected events: 35 brand-channel events, one brand-level mix shift, and one business-unit-level external-demand event.
- Ground truth marks 254 rows as anomalous, or 15.5% of the panel, against a target of approximately 15%.
- Incremental return on ad spend (iROAS) is observed on 1,393 rows, or 84.8%, against an 85% target. Observed iROAS includes approximately 20% relative measurement noise.

**Architecture**

- The simulator generates eight primitives—eligible impressions, impression share, click-through rate (CTR), cost per click (CPC), post-impression conversion rate (PICR), baseline PICR, average order value (AOV), and attribution inflation—with trend, seasonality, and autoregressive noise. It derives impressions, clicks, spend, conversions, media-attributable sales (MAS), reported revenue, ROAS, RROI, and iROAS arithmetically.
- A top-down correlated-noise design was rejected because it could violate accounting identities such as `roas = reported_revenue / spend` and would make anomalies arbitrary column changes rather than consequences of business mechanisms.

**Key formula locks**

- `picr = conversions / impressions`. PICR is post-impression rather than post-click, following the definition locked for this dataset.
- The iROAS counterfactual follows `baseline_picr → baseline_conversions → incremental_revenue`. The source brief did not specify this chain, but the simulator requires a counterfactual to produce iROAS.
- `reported_revenue = mas × attribution_inflation`. This added reporting layer creates the ROAS-RROI divergence used to identify measurement artifacts.
- `mas = conversions × aov`. Treating MAS as a separate primitive was rejected because it would break the link between conversion behavior and revenue.

**Spike design**

- The dataset contains seven cause types: the brief's six causes plus `measurement_artifact`, which acts as a hard negative for the ROAS-RROI divergence pattern.
- For each eligible brand-channel series, the base event draw is zero, one, or two events with probabilities 40%, 45%, and 15%. Series with more than 30 active months receive one additional exposure-scaled event opportunity. Event starts must be at least four months apart. In the current release, 17 series have no brand-channel event, 15 have one, and 10 have two.
- Brand-channel events draw one, two, or three causes with probabilities 50%, 35%, and 15%. The realized seed-42 panel contains 15 single-cause, 10 two-cause, and 10 three-cause brand-channel events.
- Brand-channel onset is sampled as 60% sudden and 40% gradual, with gradual events ramping over three months. The current panel contains 21 sudden and 14 gradual brand-channel events.
- The current seed produced one brand-level mix-shift event and one business-unit-level external-demand event. These are many labeled rows but only one independent event each.
- Survivorship bias and spend-reduction artifacts deliberately have similar aggregate signatures. This overlap reflects an identification limit in monthly data rather than a generator defect.

**Structure and observation policy**

- The panel contains three business units, eight brands in a 3/3/2 split, and a seven-channel pool. Not every channel is active for every brand. Two brands launch during the observation window, and out-of-home activity is seasonal.
- The iROAS target is 85% observed coverage, rather than the earlier 20% scheduled-test assumption. The `iroas_test_flag` field preserves which values were observed.
- For anomalous rows, the `analyst_tag` generator selects a true cause 75% of the time, an incorrect cause 15% of the time, and no tag 10% of the time. Normal rows receive a false tag with 3% probability. In the current release, 77.0% of tagged anomaly rows match a true cause, 7.5% of anomaly rows have no tag, and 3.4% of normal rows receive a false tag.
- Random missingness is independent of ground truth. The generator nulls one leading indicator on approximately 3% of rows and creates a one-to-three-month reporting outage for approximately 5% of series. In the current release, 62 rows, or 3.8%, have at least one non-iROAS metric missing; the missingness-to-anomaly correlation is 0.030.
- Structural absence, including launch dates and seasonal-only channels, remains separate from random missingness. `planned_spend` remains out of scope unless forecasting is added.

**Deliverable structure**

- `generate_synthetic_data.py` writes four nonredundant files: `brief_metrics.csv`, `panel_data_full.csv`, `ground_truth/ground_truth.csv`, and `ground_truth/ground_truth_events.csv`.
- The two observable files contain analysis inputs. The files under `ground_truth/` contain row- and event-level oracle labels for evaluation only and must never enter the model feature matrix.
- `panel_data.csv` and `ground_truth_full.csv` are not generated because they would duplicate an existing file or a reproducible join.

**Known limitations**

- Cause representation remains uneven. Row-level counts range from 26 for `survivorship_bias` and `measurement_artifact` to 71 for `creative_refresh`.
- Event diversity, rather than labeled-row volume, is the binding limitation for macro causes. `mix_shift_artifact` has 54 labeled rows and `external_demand_spike` has 70, but each comes from one event.
- Monthly aggregates cannot reliably separate survivorship bias from spend reduction without placement-retention or related operational records.
