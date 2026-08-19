"""
Purpose: generate a synthetic CPG paid-media panel (BU x brand x channel x month)
with mechanistically consistent metrics (ROAS/RROI/iROAS all derived from the
same primitives, not simulated independently) and injected, labeled spike events
covering the 7 cause types from the brief.

Inputs: none (config is inline below, this is a one-off generator, not a library).
Outputs are written beside this script:
  brief_metrics.csv       - brief-literal analysis columns.
  panel_data_full.csv     - analysis columns plus observable funnel/detail columns.
  ground_truth/ground_truth.csv        - hidden row-level labels; evaluation only.
  ground_truth/ground_truth_events.csv - hidden event-level log.

Design decisions this script depends on (locked in conversation before building):
  - bottom-up primitive simulation, all ratio metrics derived arithmetically
  - PICR := Conversions / Impressions
  - iROAS mostly observed (~85% of rows) + noisy (measurement error on top of true value)
  - missingness generated independently of spike ground truth (no leakage)
  - CTR/CPC applied uniformly across channels incl. TV/OOH (stated simplification)
"""

import numpy as np
import pandas as pd
from pathlib import Path

OUT = Path(__file__).resolve().parent
OUT.mkdir(exist_ok=True)
GROUND_TRUTH_OUT = OUT / "ground_truth"
GROUND_TRUTH_OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# 1. CONFIG
# ---------------------------------------------------------------------------
BUS = {
    "Beverages": ["AquaPure", "CitrusBoost", "BrewCo"],
    "Snacks": ["CrispBite", "NutHouse", "GrainSnax"],
    "HomeCare": ["CleanWave", "GlowSkin"],
}
CHANNEL_POOL = ["paid_search", "social", "digital_video", "display", "tv", "ooh", "audio"]
CHANNEL_BASE = {  # illustrative, not real industry benchmarks
    # picr = conversions / impressions (post-impression, not post-click -> much
    # smaller than ctr by construction). Calibrated so ROAS lands roughly 1.5-7x.
    "paid_search":   dict(ctr=0.035, cpc=1.2, picr=0.0050),
    "social":        dict(ctr=0.012, cpc=0.8, picr=0.0020),
    "digital_video": dict(ctr=0.006, cpc=0.6, picr=0.0013),
    "display":       dict(ctr=0.004, cpc=0.5, picr=0.0007),
    "tv":            dict(ctr=0.002, cpc=2.5, picr=0.0010),
    "ooh":           dict(ctr=0.0015, cpc=3.0, picr=0.0006),
    "audio":         dict(ctr=0.003, cpc=0.7, picr=0.0010),
}
MONTHS = pd.date_range("2022-01-01", periods=48, freq="MS")
# Preserve the original June/November 2023 launch dates after extending the history.
LATE_LAUNCH = {"GlowSkin": 22, "GrainSnax": 17}
IROAS_COVERAGE = 0.85

CAUSE_TYPES = [
    "genuine_efficiency_gain", "spend_reduction_artifact", "mix_shift_artifact",
    "survivorship_bias", "external_demand_spike", "creative_refresh",
    "measurement_artifact",
]

# ---------------------------------------------------------------------------
# 2. HELPERS
# ---------------------------------------------------------------------------
def ar1(n, sd=0.06, phi=0.65):
    """Mean-reverting multiplicative noise series, centered at 1.0."""
    x = np.zeros(n)
    x[0] = rng.normal(0, sd)
    innov_sd = sd * np.sqrt(1 - phi**2)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal(0, innov_sd)
    return 1.0 + x

def bu_seasonal(bu, n):
    """Asymmetric seasonality: sharp Nov/Dec peak, mild Jan/Feb trough, mild trend."""
    base = np.ones(n)
    for i, m in enumerate(MONTHS[:n]):
        if m.month == 11: base[i] = 1.30
        elif m.month == 12: base[i] = 1.55
        elif m.month in (1, 2): base[i] = 0.82
        if bu == "Beverages" and m.month in (6, 7, 8): base[i] *= 1.15
    trend = 1.0 + np.linspace(0, rng.uniform(-0.05, 0.15), n)
    return base * trend * ar1(n, sd=0.03)

# ---------------------------------------------------------------------------
# 3. BRAND / CHANNEL METADATA
# ---------------------------------------------------------------------------
brand_meta = {}
for bu, brands in BUS.items():
    for b in brands:
        scale = rng.uniform(0.4, 1.8)
        n_channels = 3 if scale < 0.7 else (4 if scale < 1.2 else 6)
        always_on = ["paid_search", "social"]
        rest = [c for c in CHANNEL_POOL if c not in always_on]
        extra = list(rng.choice(rest, size=max(0, n_channels - 2), replace=False))
        active_channels = always_on + extra
        brand_meta[b] = dict(
            bu=bu, scale=scale, channels=active_channels,
            launch_idx=LATE_LAUNCH.get(b, 0),
            organic_ratio=rng.uniform(0.12, 0.35),   # baseline_picr / picr
            ctr_mult=rng.uniform(0.8, 1.3),
            picr_mult=rng.uniform(0.8, 1.3),
            aov=rng.uniform(8, 45),
        )

# ---------------------------------------------------------------------------
# 4. BASE PRIMITIVES (pre-spike)
# ---------------------------------------------------------------------------
rows = []
for bu, brands in BUS.items():
    n = len(MONTHS)
    seasonal = bu_seasonal(bu, n)
    for b in brands:
        meta = brand_meta[b]
        launch = meta["launch_idx"]
        for c in meta["channels"]:
            base = CHANNEL_BASE[c]
            seasonal_only_ooh = c == "ooh"
            impression_share = np.clip(rng.uniform(0.5, 0.8) * ar1(n, sd=0.05), 0.05, 0.95)
            ctr = base["ctr"] * meta["ctr_mult"] * ar1(n, sd=0.08)
            picr = base["picr"] * meta["picr_mult"] * ar1(n, sd=0.08)
            cpc_base = base["cpc"] * ar1(n, sd=0.06)
            attribution_inflation = ar1(n, sd=0.03)
            eligible_impressions = meta["scale"] * 3_000_000 * seasonal * ar1(n, sd=0.05)
            baseline_picr = picr * meta["organic_ratio"] * ar1(n, sd=0.04)

            for i, m in enumerate(MONTHS):
                if i < launch:
                    continue
                if seasonal_only_ooh and m.month not in (11, 12):
                    continue
                rows.append(dict(
                    bu=bu, brand=b, channel=c, month=m, t=i,
                    eligible_impressions=eligible_impressions[i],
                    impression_share=impression_share[i],
                    ctr=ctr[i], cpc_base=cpc_base[i], picr=picr[i],
                    baseline_picr=min(baseline_picr[i], picr[i] * 0.9),
                    attribution_inflation=attribution_inflation[i],
                    aov=meta["aov"],
                ))

panel = pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# 5. SPIKE EVENTS: sample events, apply multipliers to primitives
# ---------------------------------------------------------------------------
events = []
event_id = 0
gt_tags = {}  # (bu,brand,channel,month) -> list of (event_id, cause_type)

def mark(bu, brand, channel, month, eid, cause):
    gt_tags.setdefault((bu, brand, channel, month), []).append((eid, cause))

def apply_mult(mask, col, mult_by_offset):
    """mask: boolean series aligned to panel; mult_by_offset: dict{t_rel: multiplier}"""
    idx = panel.index[mask]
    for i in idx:
        t_rel = panel.at[i, "_t_rel"]
        if t_rel in mult_by_offset:
            panel.at[i, col] *= mult_by_offset[t_rel]

# single-channel series list (for genuine gain, creative refresh, spend reduction, survivorship)
series_keys = panel[["bu", "brand", "channel"]].drop_duplicates().values.tolist()

for bu, brand, channel in series_keys:
    sub = panel[(panel.bu == bu) & (panel.brand == brand) & (panel.channel == channel)]
    active_months = sub["t"].tolist()
    if len(active_months) < 9:
        continue
    # Keep the v3 quiet/one/two-event distribution, with one exposure-scaled
    # opportunity for series that now contain more than the original 30 months.
    n_events = int(rng.choice([0, 1, 2], p=[0.40, 0.45, 0.15]))
    extra_event_probability = min(1.0, max(0, len(active_months) - 30) / 30)
    if rng.random() < extra_event_probability:
        n_events += 1
    used_starts = []
    for _ in range(n_events):
        start = rng.choice(active_months[3:-2]) if len(active_months) > 6 else active_months[0]
        if any(abs(start - u) < 4 for u in used_starts):
            continue
        used_starts.append(start)
        sudden = rng.random() < 0.6
        onset = 1 if sudden else 3
        n_causes = rng.choice([1, 2, 3], p=[0.5, 0.35, 0.15])
        causes = list(rng.choice(
            ["genuine_efficiency_gain", "spend_reduction_artifact", "survivorship_bias",
             "creative_refresh", "measurement_artifact"], size=n_causes, replace=False))
        eid = f"E{event_id:04d}"; event_id += 1
        events.append(dict(event_id=eid, scope="brand_channel", bu=bu, brand=brand,
                            channel=channel, start_month=str(MONTHS[start].date()),
                            duration_type="gradual" if not sudden else "sudden",
                            cause_types="|".join(causes)))
        for m_off in range(-2, onset + 4):
            t = start + m_off
            if t not in active_months:
                continue
            mrow_mask = (panel.bu == bu) & (panel.brand == brand) & (panel.channel == channel) & (panel.t == t)
            ramp = min(1.0, max(0.0, (m_off + 1) / onset)) if m_off >= 0 else 0.0
            for cause in causes:
                if cause == "genuine_efficiency_gain" and 0 <= m_off:
                    mag = rng.uniform(0.15, 0.30) * ramp
                    panel.loc[mrow_mask, "ctr"] *= (1 + mag)
                    panel.loc[mrow_mask, "picr"] *= (1 + mag)
                elif cause == "creative_refresh":
                    if 0 <= m_off <= onset:
                        mag = rng.uniform(0.20, 0.35) * ramp
                    else:
                        decay = max(0.0, 1 - (m_off - onset) / 3)
                        mag = rng.uniform(0.20, 0.35) * decay
                    if mag > 0:
                        panel.loc[mrow_mask, "ctr"] *= (1 + mag)
                        panel.loc[mrow_mask, "picr"] *= (1 + mag)
                elif cause == "spend_reduction_artifact" and 0 <= m_off:
                    mag = rng.uniform(0.20, 0.35) * ramp
                    panel.loc[mrow_mask, "impression_share"] *= (1 - mag)
                    panel.loc[mrow_mask, "picr"] *= (1 + 0.07 * ramp)
                elif cause == "survivorship_bias" and 0 <= m_off:
                    mag = rng.uniform(0.40, 0.60) * ramp
                    panel.loc[mrow_mask, "impression_share"] *= (1 - mag)
                    panel.loc[mrow_mask, "ctr"] *= (1 + 0.08 * ramp)
                    panel.loc[mrow_mask, "picr"] *= (1 + 0.08 * ramp)
                elif cause == "measurement_artifact" and 0 <= m_off:
                    mag = rng.uniform(0.10, 0.25) * ramp
                    panel.loc[mrow_mask, "attribution_inflation"] *= (1 + mag)
                if 0 <= m_off <= (onset + (3 if cause in ("creative_refresh",) else 0)):
                    for mm in panel.loc[mrow_mask, "month"]:
                        mark(bu, brand, channel, mm, eid, cause)

# mix_shift_artifact: brand-wide, needs >=2 channels
eligible_brands = [
    b for b, meta in brand_meta.items()
    if len(meta["channels"]) >= 2
]
k_mix = int(rng.choice([1, 2], p=[0.5, 0.5]))
for b in rng.choice(eligible_brands, size=k_mix, replace=False):
    meta = brand_meta[b]
    bu = meta["bu"]
    winner, loser = rng.choice(meta["channels"], size=2, replace=False)
    starts = [i for i in range(6, len(MONTHS) - 4)]
    start = int(rng.choice(starts))
    eid = f"E{event_id:04d}"; event_id += 1
    events.append(dict(event_id=eid, scope="brand", bu=bu, brand=b, channel=f"{winner}+{loser}",
                        start_month=str(MONTHS[start].date()), duration_type="sudden",
                        cause_types="mix_shift_artifact"))
    mag = rng.uniform(0.20, 0.35)
    for ch, sign in [(winner, +1), (loser, -1)]:
        mrow_mask = (panel.bu == bu) & (panel.brand == b) & (panel.channel == ch) & (panel.t >= start)
        panel.loc[mrow_mask, "impression_share"] *= (1 + sign * mag)
        for _, r in panel.loc[mrow_mask].iterrows():
            mark(bu, b, ch, r["month"], eid, "mix_shift_artifact")

# external_demand_spike: BU-wide, hits every active brand/channel that month
k_demand = 1
for bu in rng.choice(list(BUS.keys()), size=k_demand, replace=False):
    start = int(rng.integers(6, len(MONTHS) - 4))
    eid = f"E{event_id:04d}"; event_id += 1
    events.append(dict(event_id=eid, scope="bu", bu=bu, brand="ALL", channel="ALL",
                        start_month=str(MONTHS[start].date()), duration_type="gradual",
                        cause_types="external_demand_spike"))
    for m_off in range(0, 4):
        t = start + m_off
        decay = max(0.0, 1 - m_off / 4)
        mag = rng.uniform(0.15, 0.30) * decay
        mrow_mask = (panel.bu == bu) & (panel.t == t)
        panel.loc[mrow_mask, "eligible_impressions"] *= (1 + mag)
        panel.loc[mrow_mask, "baseline_picr"] *= (1 + mag * 0.8)
        for _, r in panel.loc[mrow_mask].iterrows():
            mark(bu, r["brand"], r["channel"], r["month"], eid, "external_demand_spike")

panel = panel.drop(columns=[c for c in panel.columns if c == "_t_rel"], errors="ignore")

# ---------------------------------------------------------------------------
# 6. DERIVE OBSERVABLE METRICS FROM (POST-SPIKE) PRIMITIVES
# ---------------------------------------------------------------------------
panel["impression_share"] = panel["impression_share"].clip(0.02, 0.98)
panel["ctr"] = panel["ctr"].clip(0.0001, 0.25)
panel["picr"] = panel["picr"].clip(0.0005, 0.35)
panel["baseline_picr"] = panel[["baseline_picr", "picr"]].min(axis=1) * 0.9

panel["impressions"] = np.round(panel["eligible_impressions"] * panel["impression_share"]).astype(int)
panel["clicks"] = rng.binomial(panel["impressions"].clip(upper=5_000_000), panel["ctr"])
panel["conversions"] = rng.binomial(panel["impressions"].clip(upper=5_000_000), panel["picr"])
panel["_baseline_conversions"] = rng.binomial(panel["impressions"].clip(upper=5_000_000), panel["baseline_picr"])

panel["cpc"] = panel["cpc_base"] * (0.7 + 0.6 * panel["impression_share"]) * np.exp(rng.normal(0, 0.03, len(panel)))
panel["spend"] = panel["clicks"] * panel["cpc"]
panel["mas"] = panel["conversions"] * panel["aov"]
panel["reported_revenue"] = panel["mas"] * panel["attribution_inflation"]
panel["_incremental_rev"] = np.maximum(panel["conversions"] - panel["_baseline_conversions"], 0) * panel["aov"]

panel["roas"] = panel["reported_revenue"] / panel["spend"]
panel["rroi"] = panel["mas"] / panel["spend"]
panel["_iroas_true"] = panel["_incremental_rev"] / panel["spend"]

# ---------------------------------------------------------------------------
# 7. iROAS: mostly observed + noisy
# ---------------------------------------------------------------------------
panel["iroas_test_flag"] = rng.random(len(panel)) < IROAS_COVERAGE
noise = np.exp(rng.normal(0, 0.20, len(panel)))
panel["iroas"] = np.where(panel["iroas_test_flag"], panel["_iroas_true"] * noise, np.nan)

# ---------------------------------------------------------------------------
# 8. GROUND TRUTH (row-level, hidden)
# ---------------------------------------------------------------------------
def lookup_gt(r):
    tags = gt_tags.get((r["bu"], r["brand"], r["channel"], r["month"]), [])
    if not tags:
        return pd.Series([False, "", ""])
    causes = sorted(set(t[1] for t in tags))
    eids = sorted(set(t[0] for t in tags))
    return pd.Series([True, "|".join(causes), "|".join(eids)])

panel[["is_anomaly", "cause_types", "event_ids"]] = panel.apply(lookup_gt, axis=1)
ground_truth = panel[["bu", "brand", "channel", "month", "is_anomaly", "cause_types", "event_ids"]].copy()
events_df = pd.DataFrame(events)

# ---------------------------------------------------------------------------
# 9. WEAK/NOISY analyst_tag (observable, separate from hidden ground truth)
# ---------------------------------------------------------------------------
def analyst_tag(r):
    if r["is_anomaly"]:
        causes = r["cause_types"].split("|")
        roll = rng.random()
        if roll < 0.75:
            return rng.choice(causes)
        elif roll < 0.90:
            wrong_pool = [c for c in CAUSE_TYPES if c not in causes]
            return rng.choice(wrong_pool)
        else:
            return np.nan
    else:
        return rng.choice(CAUSE_TYPES) if rng.random() < 0.03 else np.nan

panel["analyst_tag"] = panel.apply(analyst_tag, axis=1)

# ---------------------------------------------------------------------------
# 10. MISSINGNESS (independent of spike ground truth, by construction)
# ---------------------------------------------------------------------------
leading_cols = ["ctr", "cpc", "picr", "impression_share"]
rand_gap_mask = rng.random(len(panel)) < 0.03
for i in panel.index[rand_gap_mask]:
    col = rng.choice(leading_cols)
    panel.at[i, col] = np.nan

series_keys2 = panel[["bu", "brand", "channel"]].drop_duplicates().values.tolist()
burst_series = [series_keys2[i] for i in rng.choice(len(series_keys2),
                size=max(1, int(0.05 * len(series_keys2))), replace=False)]
metric_cols = ["impressions", "clicks", "conversions", "spend", "mas", "reported_revenue",
               "roas", "rroi", "ctr", "cpc", "picr", "impression_share"]
for bu, brand, channel in burst_series:
    sub_idx = panel.index[(panel.bu == bu) & (panel.brand == brand) & (panel.channel == channel)]
    if len(sub_idx) < 6:
        continue
    dur = int(rng.integers(1, 4))
    start_pos = int(rng.integers(2, len(sub_idx) - dur - 1))
    block = sub_idx[start_pos:start_pos + dur]
    panel.loc[block, metric_cols] = np.nan

# ---------------------------------------------------------------------------
# 11. FINALIZE OUTPUT COLUMNS
# ---------------------------------------------------------------------------
observable_cols = [
    "bu", "brand", "channel", "month",
    "impressions", "clicks", "conversions", "spend", "mas", "reported_revenue",
    "roas", "rroi", "iroas", "iroas_test_flag",
    "ctr", "cpc", "picr", "impression_share",
    "analyst_tag",
]
panel_out = panel[observable_cols].sort_values(["bu", "brand", "channel", "month"])
ground_truth_out = ground_truth.sort_values(["bu", "brand", "channel", "month"])

key_cols = ["bu", "brand", "channel", "month"]
brief_cols = key_cols + [
    "roas", "rroi", "iroas", "mas", "spend", "ctr", "cpc", "picr",
    "impression_share",
]
extra_observable_cols = [
    "impressions", "clicks", "conversions", "reported_revenue",
    "iroas_test_flag", "analyst_tag",
]
panel_data_full = panel_out[brief_cols + extra_observable_cols]

panel_out[brief_cols].to_csv(OUT / "brief_metrics.csv", index=False)
panel_data_full.to_csv(OUT / "panel_data_full.csv", index=False)
ground_truth_out.to_csv(GROUND_TRUTH_OUT / "ground_truth.csv", index=False)
events_df.to_csv(GROUND_TRUTH_OUT / "ground_truth_events.csv", index=False)

print("Rows written:", len(panel_out))
print("Events injected:", len(events_df))
print("iROAS coverage:", f"{panel_out['iroas'].notna().mean():.1%}")
print(panel_out.head(3).to_string())
