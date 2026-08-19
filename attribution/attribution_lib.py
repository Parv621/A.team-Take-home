"""
Purpose: shared pipeline for Task 2 (attribution). Holds the Task 1 detector verbatim, the
leakage-safe feature builder, and the evaluation helpers, so that any model notebook that imports
this module sees an identical candidate table, feature matrix and split.

Inputs:  ../data/brief_metrics.csv (features), ../data/ground_truth/*.csv (labels, evaluation only).
Outputs: functions only -- this module writes no files.

Design decisions this module depends on (confirmed before building, see proposal_attribution.md):
  - all five primary metrics (ROAS, RROI, iROAS, MAS, spend) enter the fitted feature matrix.
    The exercise brief names five primary metrics and does not ask for a subset; narrowing to
    three cost 221 candidates and roughly half the positives on every cause, for no gain.
  - unit of analysis = one BU x brand x channel x month investigation (outer union across the
    three selected metrics, deduplicated).
  - one-vs-rest binary target per cause, never softmax; probabilities need not sum to 1.
  - 5 causes are fitted; mix_shift_artifact and external_demand_spike are rule-scored instead
    (1 independent event each on this panel -- see FITTED_CAUSES / RULE_CAUSES below).
  - single-panel proof of concept: event-grouped CV, no Platt calibration. Outputs are model
    scores, not calibrated probabilities.
"""

import math

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------------
ALL_PRIMARY = ["roas", "rroi", "iroas", "mas", "spend"]
SELECTED_PRIMARY = ALL_PRIMARY  # all five; see module docstring
LEADING = ["ctr", "cpc", "picr", "impression_share"]

CAUSES = [
    "genuine_efficiency_gain",
    "spend_reduction_artifact",
    "survivorship_bias",
    "creative_refresh",
    "measurement_artifact",
    "mix_shift_artifact",
    "external_demand_spike",
]
# Fitted vs rule-scored. Split is driven by independent-event count on this panel, not by taste:
# mix_shift_artifact has 3 positive rows from 1 event and external_demand_spike 22 rows from 1
# event, so a fitted classifier for either would be validated against the single event it was
# trained on. Both get a deterministic expert-signature score instead.
FITTED_CAUSES = CAUSES[:5]
RULE_CAUSES = CAUSES[5:]

KEY = ["bu", "brand", "channel", "month"]

# --------------------------------------------------------------------------------------------
# Task 1 detector -- lifted verbatim from anomaly_detection/anomaly_detection.ipynb (sections 3-5).
# Reproduced rather than re-derived so Task 2's candidate set is exactly Task 1's output.
# --------------------------------------------------------------------------------------------
MIN_HIST_EXCLUDE = 12  # series shorter than this: excluded, insufficient history
MIN_HIST_FULL = 24     # series with at least this many months: full dual-test tier
SPARSE_WIN = 6         # trailing window (months), 12-24mo tier
FULL_WIN = 12          # trailing window (months), 24mo+ tier -- spans one seasonal year
SPARSE_Z = 3.0         # wider cutoff, 12-24mo tier (noisier baseline, so demand more evidence)
FULL_Z = 2.5           # threshold, fixed across metrics (precision-leaning)


def min_periods_for(win):
    """75% of window, floor 3 -- tolerates a couple of missing months instead of demanding a
    100% complete window. This is what keeps iROAS scoreable despite its ~15% missingness."""
    return max(3, math.ceil(win * 0.75))


def nanmedian_raw(x):
    x = x[~np.isnan(x)]
    return np.nan if len(x) == 0 else np.median(x)


def mad_std_raw(x):
    """MAD scaled x1.4826 to estimate std under normality. Floored at 1e-4 rather than returning
    NaN on a perfectly flat window, which is well below any metric's observed noise floor."""
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan
    med = np.median(x)
    return max(np.median(np.abs(x - med)) * 1.4826, 1e-4)


def naive_z(pct_change, win):
    """Non-robust mean/std z-score. Look-ahead safe (shift(1) before rolling)."""
    mp = min_periods_for(win)
    m = pct_change.shift(1).rolling(win, min_periods=mp).mean()
    s = pct_change.shift(1).rolling(win, min_periods=mp).std()
    return (pct_change - m) / s


def robust_z(pct_change, win):
    """Median/MAD z-score. Look-ahead safe (shift(1) before rolling)."""
    mp = min_periods_for(win)
    med = pct_change.shift(1).rolling(win, min_periods=mp).apply(nanmedian_raw, raw=True)
    mad = pct_change.shift(1).rolling(win, min_periods=mp).apply(mad_std_raw, raw=True)
    return (pct_change - med) / mad


def business_floors(bm, metrics=ALL_PRIMARY):
    """75th percentile of each metric's own |month-over-month| change -- Task 1's per-metric floor.

    NOTE: Task 1's notebook rounds this to the nearest whole percent before thresholding; this returns
    the unrounded value. A handful of rows sit between the two, so flag counts here run 1-2 above
    Task 1's on ROAS, iROAS and spend (153/67/204 vs 151/66/203) and match exactly on RROI and MAS.
    Both print as `15% / 14% / 37% / 22% / 24%`, so the gap is invisible from the printout. Four extra
    candidates out of 466 changes nothing downstream; the delta is documented in the notebook rather
    than silently absorbed. Round here to make the two paths identical.
    """
    out = {}
    for m in metrics:
        mom = bm.groupby(KEY[:3])[m].pct_change(fill_method=None).dropna()
        mom = mom.replace([np.inf, -np.inf], np.nan).dropna()
        out[m] = float(mom.abs().quantile(0.75))
    return out


def score_metric(g, metric_col, floor):
    """Tier, z-scores and flags for one series on one metric column."""
    g = g.sort_values("month").reset_index(drop=True)
    n = len(g)
    tier = "excluded" if n < MIN_HIST_EXCLUDE else ("sparse" if n < MIN_HIST_FULL else "full")

    val = g[metric_col]
    mom = val.pct_change(fill_method=None)
    cum3 = val.pct_change(3, fill_method=None)
    g["tier"], g["mom_pct"], g["cum3_pct"] = tier, mom, cum3

    if tier == "sparse":
        g["z_stat"] = naive_z(mom, SPARSE_WIN)
        g["z_trend"] = np.nan
        g["flagged"] = (g["z_stat"].abs() > SPARSE_Z) & (g["mom_pct"].abs() >= floor)
        g["flag_type"] = np.where(g["flagged"], "level_shift", None)
    elif tier == "full":
        g["z_stat"] = robust_z(mom, FULL_WIN)
        g["z_trend"] = robust_z(cum3, FULL_WIN)
        level = (g["z_stat"].abs() > FULL_Z) & (g["mom_pct"].abs() >= floor)
        trend = (g["z_trend"].abs() > FULL_Z) & (g["cum3_pct"].abs() >= floor)
        g["flagged"] = level | trend
        g["flag_type"] = np.select(
            [level & trend, level, trend], ["both", "level_shift", "trend_break"], default=None
        )
    else:
        g["z_stat"] = np.nan
        g["z_trend"] = np.nan
        g["flagged"] = False
        g["flag_type"] = None
    return g


def score_all_series(bm, metric_col, floor):
    """Manual per-group loop, not groupby().apply(): recent pandas drops grouping columns from the
    sub-frame passed to apply()'d functions, which breaks the downstream joins on bu/brand/channel."""
    parts = [score_metric(g.copy(), metric_col, floor) for _, g in bm.groupby(KEY[:3])]
    return pd.concat(parts, ignore_index=True)


def run_task1(bm, metrics, floors):
    """Task 1 detector output for each requested metric."""
    return {m: score_all_series(bm, m, floors[m]) for m in metrics}


# --------------------------------------------------------------------------------------------
# Candidate table: outer union of the selected metrics' alerts, one row per investigation
# --------------------------------------------------------------------------------------------
def build_candidates(scored_by_metric, metrics):
    """Outer-union the alerts across `metrics` and pivot per-metric trigger evidence onto one row.

    A row flagged by two metrics is one investigation, not two training examples.
    """
    flagged_keys = set()
    for m in metrics:
        s = scored_by_metric[m]
        flagged_keys |= set(map(tuple, s.loc[s["flagged"], KEY].values))

    cand = pd.DataFrame(sorted(flagged_keys), columns=KEY)

    for m in metrics:
        s = scored_by_metric[m][KEY + ["tier", "flag_type", "z_stat", "z_trend"]]
        s = s.rename(
            columns={
                "flag_type": f"{m}_flag_type",
                "z_stat": f"{m}_z_level",
                "z_trend": f"{m}_z_trend",
            }
        )
        cand = cand.merge(s, on=KEY, how="left", suffixes=("", f"_{m}"))
        if "tier" in cand.columns and f"tier_{m}" in cand.columns:
            cand = cand.drop(columns=[f"tier_{m}"])
        trig_keys = set(
            map(tuple, scored_by_metric[m].loc[scored_by_metric[m]["flagged"], KEY].values)
        )
        cand[f"{m}_trig"] = [
            1 if tuple(r) in trig_keys else 0 for r in cand[KEY].values
        ]
        # flag_type is only meaningful where the metric actually fired
        cand.loc[cand[f"{m}_trig"] == 0, f"{m}_flag_type"] = None

    cand["n_metrics_triggered"] = cand[[f"{m}_trig" for m in metrics]].sum(axis=1)
    return cand


# --------------------------------------------------------------------------------------------
# Feature engineering -- everything below uses information available at month t only
# --------------------------------------------------------------------------------------------
def _robust_residual(s, win=FULL_WIN):
    """Trailing log-space robust residual: (log x_t - median(log x_{t-win:t-1})) / scaled MAD.

    shift(1) before rolling makes this look-ahead safe: month t never enters its own baseline.
    """
    lx = np.log(s.clip(lower=1e-9))
    mp = min_periods_for(win)
    prev = lx.shift(1)
    med = prev.rolling(win, min_periods=mp).apply(nanmedian_raw, raw=True)
    mad = prev.rolling(win, min_periods=mp).apply(mad_std_raw, raw=True)
    return (lx - med) / mad


def build_panel_features(bm):
    """Per-series, per-month feature panel. Computed on the whole panel (cheap) and joined onto
    candidates later; nothing here uses the flag status or any label."""
    cols = SELECTED_PRIMARY + LEADING
    parts = []
    for _, g in bm.groupby(KEY[:3]):
        g = g.sort_values("month").reset_index(drop=True)
        out = g[KEY].copy()
        for c in cols:
            s = g[c]
            lx = np.log(s.clip(lower=1e-9))
            out[f"{c}_resid"] = _robust_residual(s)
            out[f"{c}_d1"] = lx.diff(1)          # 1-month log change
            out[f"{c}_d3"] = lx.diff(3)          # 3-month log change
            out[f"{c}_miss"] = s.isna().astype(int)
        # temporal shape, only where a cause signature needs it (creative_refresh ramp/peak):
        # CTR and PICR rising for 1-2 months before t distinguishes a ramp from a one-month jump.
        for c in ["ctr", "picr"]:
            out[f"{c}_resid_lag1"] = out[f"{c}_resid"].shift(1)
            out[f"{c}_resid_lag2"] = out[f"{c}_resid"].shift(2)
        # months of usable history available at t (excludes t itself)
        out["months_history"] = np.arange(len(g))
        out["spend_raw"] = g["spend"]
        parts.append(out)
    panel = pd.concat(parts, ignore_index=True)

    # Mix context: this channel's share of its brand's total spend that month, and how that share
    # moved. Reallocation toward an already-efficient channel is the mix_shift signature -- brand
    # efficiency rises with no within-channel improvement.
    brand_spend = panel.groupby(["bu", "brand", "month"])["spend_raw"].transform("sum")
    panel["spend_share"] = panel["spend_raw"] / brand_spend.replace(0, np.nan)
    panel = panel.sort_values(KEY).reset_index(drop=True)
    ss = panel.groupby(KEY[:3])["spend_share"]
    panel["spend_share_d1"] = ss.diff(1)
    panel["spend_share_d3"] = ss.diff(3)
    panel = panel.drop(columns=["spend_raw"])

    # Peer context: leave-one-out median RROI residual across the same BU-month. Excluding the
    # target row is what makes this evidence about the market rather than about the target.
    panel = _add_peer_context(panel)
    return panel


def _add_peer_context(panel):
    """Leave-one-out BU-month peer aggregates on the RROI residual."""
    r = panel["rroi_resid"]
    grp = panel.groupby(["bu", "month"])["rroi_resid"]

    # LOO median: recomputing the median without each row is O(n) per group and the groups are
    # small (<=14 rows), so do it exactly rather than approximating with a mean-based shortcut.
    loo_med, loo_frac = np.full(len(panel), np.nan), np.full(len(panel), np.nan)
    for _, idx in panel.groupby(["bu", "month"]).indices.items():
        vals = r.values[idx]
        for j, i in enumerate(idx):
            others = np.delete(vals, j)
            others = others[~np.isnan(others)]
            if len(others):
                loo_med[i] = np.median(others)
                loo_frac[i] = np.mean(np.abs(others) > 2.0)  # |resid|>2 == "elevated", Task 1's scale
    panel["peer_median_resid"] = loo_med
    panel["peer_frac_elevated"] = loo_frac
    panel["target_minus_peer"] = panel["rroi_resid"] - panel["peer_median_resid"]
    return panel


def add_cross_metric(df):
    """Cross-metric consistency, using only the three selected primary metrics."""
    # ROAS-vs-RROI wedge: roas/rroi == attribution_inflation, so this isolates reporting movement
    # from real movement. It is the only signature that identifies measurement_artifact.
    df["wedge_roas_rroi"] = df["roas_resid"] - df["rroi_resid"]
    # iROAS-vs-RROI concordance: real incremental movement tracking attributable movement.
    df["conc_iroas_rroi"] = df["iroas_resid"] - df["rroi_resid"]
    # absolute outcome vs spend movement: a ROAS rise on falling spend is spend reduction, not gain
    df["mas_minus_spend"] = df["mas_d1"] - df["spend_d1"]
    df["n_primary_elevated"] = sum(
        (df[f"{m}_resid"].abs() > 2.0).astype(float) for m in SELECTED_PRIMARY
    )
    return df


def _primary_block(m):
    return [f"{m}_resid", f"{m}_d1", f"{m}_d3", f"{m}_trig", f"{m}_z_level", f"{m}_z_trend"]


FEATURE_BLOCKS = {f"primary_{m}": _primary_block(m) for m in SELECTED_PRIMARY}
FEATURE_BLOCKS.update({
    "leading_ctr": ["ctr_resid", "ctr_d1", "ctr_d3", "ctr_resid_lag1", "ctr_resid_lag2"],
    "leading_cpc": ["cpc_resid", "cpc_d1", "cpc_d3"],
    "leading_picr": ["picr_resid", "picr_d1", "picr_d3", "picr_resid_lag1", "picr_resid_lag2"],
    "leading_impression_share": ["impression_share_resid", "impression_share_d1", "impression_share_d3"],
    "cross_metric": ["wedge_roas_rroi", "conc_iroas_rroi", "mas_minus_spend",
                     "n_primary_elevated", "n_metrics_triggered"],
    "peer_context": ["peer_median_resid", "peer_frac_elevated", "target_minus_peer"],
    "mix_context": ["spend_share", "spend_share_d1", "spend_share_d3"],
    "reliability": ["iroas_miss", "n_missing", "months_history"],
})
FEATURES = [f for block in FEATURE_BLOCKS.values() for f in block]


def build_feature_matrix(cand, panel):
    """Join the panel features onto the candidate table and finish the derived blocks."""
    df = cand.merge(panel, on=KEY, how="left")
    df = add_cross_metric(df)
    df["n_missing"] = df[[f"{c}_miss" for c in SELECTED_PRIMARY + LEADING]].sum(axis=1)
    for m in SELECTED_PRIMARY:
        df[f"{m}_z_level"] = df[f"{m}_z_level"].fillna(0.0)
        df[f"{m}_z_trend"] = df[f"{m}_z_trend"].fillna(0.0)
    missing = [f for f in FEATURES if f not in df.columns]
    assert not missing, f"feature matrix is missing columns: {missing}"
    return df


# --------------------------------------------------------------------------------------------
# Labels and event grouping (ground truth -- targets and evaluation only, never features)
# --------------------------------------------------------------------------------------------
def attach_labels(df, gt):
    """Multi-label target, one binary column per cause. Candidates with no injected cause keep an
    all-zero vector: they teach the model to score low rather than to force an explanation."""
    out = df.merge(gt[KEY + ["cause_types", "event_ids"]], on=KEY, how="left")
    out["cause_types"] = out["cause_types"].fillna("")
    out["event_ids"] = out["event_ids"].fillna("")
    for c in CAUSES:
        out[f"y_{c}"] = out["cause_types"].apply(lambda s: int(c in s.split("|")))
    out["any_cause"] = out[[f"y_{c}" for c in CAUSES]].max(axis=1)
    return out


def event_groups(df):
    """Group id per candidate for event-grouped CV.

    Candidates sharing an event id go in the same fold; overlapping event ids are merged into one
    connected component via union-find, so a multi-causal event cannot straddle a split. Unlabelled
    candidates (no event) get their own singleton group -- they carry no event to leak.
    """
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for eids in df["event_ids"]:
        ids = [e for e in eids.split("|") if e]
        for e in ids[1:]:
            union(ids[0], e)

    groups = []
    for i, eids in enumerate(df["event_ids"]):
        ids = [e for e in eids.split("|") if e]
        groups.append(find(ids[0]) if ids else f"_noevent_{i}")
    return pd.Series(groups, index=df.index, name="event_group")


# --------------------------------------------------------------------------------------------
# Deterministic expert-signature baseline (also the scorer for the two rule-only causes)
# --------------------------------------------------------------------------------------------
def expert_signature_scores(df):
    """Hand-written signature score per cause, straight from data_dictionary.md's mechanism table.

    Squashed to (0,1) with a logistic so it is comparable to a model score. This is a *baseline*
    the fitted models must beat, and the only scorer for mix_shift_artifact / external_demand_spike.
    """

    def sig(x):
        return 1 / (1 + np.exp(-x))

    f = df.fillna(0.0)
    s = pd.DataFrame(index=df.index)
    # genuine efficiency: CTR and PICR up, efficiency up, impression share stable
    s["genuine_efficiency_gain"] = sig(
        f["ctr_resid"] + f["picr_resid"] + f["rroi_resid"] - f["impression_share_resid"].abs() - 2
    )
    # spend reduction: impression share and spend down, absolute outcome (MAS) flat or falling
    s["spend_reduction_artifact"] = sig(
        -f["impression_share_resid"] - f["spend_d1"] - f["mas_resid"].clip(lower=0) - 2
    )
    # survivorship: sharper impression-share drop, weak incremental corroboration
    s["survivorship_bias"] = sig(-1.5 * f["impression_share_resid"] - f["iroas_resid"] - 2)
    # creative refresh: CTR/PICR up now with a 1-2 month ramp behind it
    s["creative_refresh"] = sig(
        f["ctr_resid"] + f["picr_resid"] + 0.5 * (f["ctr_resid_lag1"] + f["picr_resid_lag1"]) - 2
    )
    # measurement artifact: ROAS moves, RROI does not
    s["measurement_artifact"] = sig(2 * f["wedge_roas_rroi"].abs() - 2)
    # mix shift: brand-level spend-share reallocation with no within-channel improvement
    s["mix_shift_artifact"] = sig(
        10 * f["spend_share_d1"].abs() - f["ctr_resid"].abs() - f["picr_resid"].abs() - 1.5
    )
    # external demand: peers in the same BU move at the same time
    s["external_demand_spike"] = sig(2 * f["peer_median_resid"] + 3 * f["peer_frac_elevated"] - 2)
    return s


# --------------------------------------------------------------------------------------------
# Analyst-facing narrative
# --------------------------------------------------------------------------------------------
def narrate(row, scores, contribs, abstain_below=0.30):
    """One-paragraph analyst explanation for a single investigation.

    Deliberately hedged language: these are diagnostic scores under a synthetic training
    distribution, not proof of cause.
    """
    top = scores.sort_values(ascending=False)
    lead, lead_p = top.index[0], top.iloc[0]
    where = f"{row['bu']} / {row['brand']} / {row['channel']} / {pd.Timestamp(row['month']):%Y-%m}"
    if lead_p < abstain_below:
        return (
            f"{where}: insufficient evidence for a known cause (top score "
            f"{lead.replace('_', ' ')} {lead_p:.2f}, below the {abstain_below:.2f} abstention "
            f"threshold). Route to manual review."
        )
    drivers = ", ".join(
        f"{n} ({v:+.2f})" for n, v in contribs.abs().sort_values(ascending=False).head(3).items()
    )
    second = f" Second candidate: {top.index[1].replace('_', ' ')} ({top.iloc[1]:.2f})." if len(top) > 1 else ""
    return (
        f"{where}: the observed pattern is most consistent with {lead.replace('_', ' ')} "
        f"({lead_p:.2f}) under the synthetic training distribution. Largest contributing signals: "
        f"{drivers}.{second} This is diagnostic evidence, not proof of cause."
    )
