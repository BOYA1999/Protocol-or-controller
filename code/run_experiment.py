from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import core
import baselines

ROOT = core.ROOT
RESULTS = ROOT / "results"
MARKETS = core.MARKETS
CAPS = core.CAPACITIES
CAPS_PILOT = (0.0, 6.4)
CONTROLLERS = core.CONTROLLERS
PRIMARY_CAP = core.DEFAULT_CAP_KWH

_DATA = None


def _init_worker() -> None:
    global _DATA
    _DATA = core.load_main()


def _week_arrays(building: dict, seed: int):
    order = core.bootstrap_week_order(seed)
    load = core.bootstrap_series(building["load"], order)
    pv = core.bootstrap_series(building["pv_kw"], order)
    price = core.bootstrap_series(_DATA["price"], order)
    carbon = core.bootstrap_series(_DATA["carbon"], order)
    return load, pv, price, carbon


def _eval_condition(building: dict, seed: int, cap: float, ctrl: str) -> list[dict]:
    """Run one controller at one capacity and return metric records for both markets."""
    load, pv, price, carbon = _week_arrays(building, seed)
    power = core.DEFAULT_POWER_KW * (cap / core.DEFAULT_CAP_KWH) if cap > 0 else 0.0
    records = []
    if ctrl in ("NOB", "GREEDY", "PRICE", "CARBON"):
        res = core.simulate_rule(ctrl, load, pv, price, carbon, cap, power)
        for market in MARKETS:
            m = core.compute_metrics(res, load, pv, price, carbon, market, cap)
            records.append(_record(building, seed, market, cap, ctrl, m))
    elif ctrl == "ORACLE_COST":
        for market in MARKETS:
            res = core.solve_oracle(load, pv, price, carbon, cap, power, market, "cost")
            m = core.compute_metrics(res, load, pv, price, carbon, market, cap)
            records.append(_record(building, seed, market, cap, ctrl, m))
    elif ctrl == "ORACLE_CO2":
        res = core.solve_oracle(load, pv, price, carbon, cap, power, "M2_zero_export", "co2")
        for market in MARKETS:
            m = core.compute_metrics(res, load, pv, price, carbon, market, cap)
            records.append(_record(building, seed, market, cap, ctrl, m))
    else:
        raise ValueError(ctrl)
    return records


def _record(building, seed, market, cap, ctrl, m) -> dict:
    rec = dict(building_id=building["id"], seed=seed, market=market,
               capacity_kwh=cap, controller=ctrl)
    rec.update({k: float(v) for k, v in m.items()})
    return rec


def _task(payload):
    building, seed, caps = payload
    out = []
    for cap in caps:
        for ctrl in CONTROLLERS:
            out.extend(_eval_condition(building, seed, cap, ctrl))
    return out


def run_records(seeds, caps, workers=0) -> list[dict]:
    _init_worker()
    tasks = [(b, s, caps) for s in seeds for b in _DATA["buildings"]]
    if workers and workers > 1:
        with mp.Pool(processes=workers, initializer=_init_worker) as pool:
            chunks = pool.map(_task, tasks, chunksize=1)
        records = [r for chunk in chunks for r in chunk]
    else:
        records = [r for task in tasks for r in _task(task)]
    return records


# ---------------------------------------------------------------------------
# aggregation / statistics
# ---------------------------------------------------------------------------
def t_interval(values: list[float], level: float = 0.95) -> tuple[float, float, float]:
    """Student-t confidence interval of the mean (n small, no scipy.stats dependency)."""
    arr = np.asarray(values, dtype=float)
    n = arr.size
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1)) if n > 1 else 0.0
    if n < 2 or sd == 0.0:
        return mean, mean, mean
    # two-sided 95% t quantiles for df = n-1 (frozen table, df<=30)
    tcrit = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
             6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}[n - 1]
    half = tcrit * sd / math.sqrt(n)
    return mean, mean - half, mean + half


def _idx(records):
    return {(r["building_id"], r["seed"], r["market"], r["capacity_kwh"], r["controller"]): r
            for r in records}


def _metric_entry(name, values, description, effect_type="cohens_dz", effect_value=None):
    mean, lo, hi = t_interval(values)
    entry = dict(name=name, value=mean, ci_low=lo, ci_high=hi,
                 n_seeds=len(values), seed_values=[float(v) for v in values],
                 seed_std=float(np.std(values, ddof=0)) if len(values) > 1 else 0.0,
                 selection="all_seeds", dataset="citylearn2022_phase_all_plus_evs",
                 description=description)
    entry["effect_size"] = dict(type=effect_type, value=float(effect_value if effect_value is not None else abs(mean)))
    return entry


def derive_headline(records, seeds) -> dict:
    idx = _idx(records)
    b_ids = sorted({r["building_id"] for r in records})

    def get(b, s, market, ctrl, cap=PRIMARY_CAP):
        return idx[(b, s, market, cap, ctrl)]

    def value(b, s, market, ctrl, cap=PRIMARY_CAP):
        return get(b, s, market, "NOB", cap)["bill"] - get(b, s, market, ctrl, cap)["bill"]

    # ---------------- H1: market-design dominance ----------------
    h1_seeds, h1_effect = [], []
    for s in seeds:
        diffs = [value(b, s, "M2_zero_export", "GREEDY") - value(b, s, "M1_net_metering", "GREEDY")
                 for b in b_ids]
        h1_seeds.append(float(np.median(diffs)))
        sd = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
        h1_effect.append(float(np.mean(diffs) / sd) if sd > 0 else 0.0)

    # ---------------- H2: protocol effect vs intelligence gap ----------------
    h2_ratio_seeds, h2_a_seeds, h2_b_seeds = [], [], []
    for s in seeds:
        protocol = float(np.median([
            value(b, s, "M2_zero_export", "ORACLE_COST") - value(b, s, "M1_net_metering", "ORACLE_COST")
            for b in b_ids]))
        intelligence = float(np.median([
            value(b, s, "M2_zero_export", "ORACLE_COST") - value(b, s, "M2_zero_export", "PRICE")
            for b in b_ids]))
        h2_a_seeds.append(protocol)
        h2_b_seeds.append(intelligence)
        h2_ratio_seeds.append(protocol / intelligence if intelligence > 0 else float("nan"))

    # ---------------- H3: carbon-aware trade-off ----------------
    h3_emit_seeds, h3_premium_seeds = [], []
    for s in seeds:
        ratios, premia = [], []
        for b in b_ids:
            base = get(b, s, "M2_zero_export", "NOB", 0.0)["co2_gross"]
            co2_cost = get(b, s, "M2_zero_export", "ORACLE_COST")["co2_gross"]
            co2_min = get(b, s, "M2_zero_export", "ORACLE_CO2")["co2_gross"]
            ratios.append((co2_cost - co2_min) / base if base > 0 else 0.0)
            bill0 = get(b, s, "M2_zero_export", "NOB", 0.0)["bill"]
            d_bill = get(b, s, "M2_zero_export", "ORACLE_CO2")["bill"] - get(b, s, "M2_zero_export", "ORACLE_COST")["bill"]
            premia.append(d_bill / bill0 if bill0 > 0 else 0.0)
        h3_emit_seeds.append(float(np.median(ratios)))
        h3_premium_seeds.append(float(np.median(premia)))

    # ---------------- H4: causal capture share, M1 vs M2 ----------------
    h4_seeds = []
    h4_m1_seeds, h4_m2_seeds = [], []
    for s in seeds:
        s_m1 = float(np.median([value(b, s, "M1_net_metering", "PRICE") /
                                max(value(b, s, "M1_net_metering", "ORACLE_COST"), 1e-9) for b in b_ids]))
        s_m2 = float(np.median([value(b, s, "M2_zero_export", "PRICE") /
                                max(value(b, s, "M2_zero_export", "ORACLE_COST"), 1e-9) for b in b_ids]))
        h4_m1_seeds.append(s_m1)
        h4_m2_seeds.append(s_m2)
        h4_seeds.append(s_m2 - s_m1)

    # ---------------- descriptive: battery value heterogeneity at 6.4 kWh ----------------
    hetero_seeds = []
    for s in seeds:
        vals = [value(b, s, "M2_zero_export", "ORACLE_COST") for b in b_ids]
        hetero_seeds.append(float(np.percentile(vals, 75) - np.percentile(vals, 25)))

    # ---------------- descriptive: optimal capacity per market ----------------
    opt = {}
    for market in MARKETS:
        for ctrl in ("ORACLE_COST", "ORACLE_CO2"):
            caps_opt = []
            for s in seeds:
                # aggregate across buildings: mean value at each capacity
                means = {}
                for cap in [c for c in CAPS if c > 0]:
                    means[cap] = float(np.mean([value(b, s, market, ctrl, cap) for b in b_ids]))
                best = max(means, key=means.get)
                caps_opt.append(best)
            opt[f"{market}/{ctrl}"] = caps_opt

    metrics = []
    metrics.append(_metric_entry(
        "H1_market_design_value_gap_usd",
        h1_seeds,
        "Median across 17 buildings of [E6.4 GREEDY battery value under zero-export minus under retail net metering], USD/yr",
        effect_value=float(np.mean(h1_effect))))
    metrics.append(_metric_entry(
        "H2_protocol_to_intelligence_ratio",
        [v for v in h2_ratio_seeds if not math.isnan(v)],
        "Median protocol effect (M2-M1 oracle cost value) divided by median intelligence gap (M2 oracle minus M2 PRICE), ratio",
        effect_type="cohens_dz",
        effect_value=float(np.mean(h2_ratio_seeds) / max(np.std(h2_ratio_seeds, ddof=1), 1e-9))))
    metrics.append(_metric_entry(
        "H3_carbon_aware_emission_reduction_share",
        h3_emit_seeds,
        "Median across buildings of [co2(ORACLE_COST) - co2(ORACLE_CO2)] / co2(NOB), M2, 6.4 kWh",
        effect_type="relative_risk_reduction",
        effect_value=float(np.mean(h3_emit_seeds))))
    metrics.append(_metric_entry(
        "H3_carbon_aware_bill_premium_share",
        h3_premium_seeds,
        "Median across buildings of [bill(ORACLE_CO2) - bill(ORACLE_COST)] / bill(NOB), M2, 6.4 kWh",
        effect_type="relative_cost_increase",
        effect_value=float(np.mean(h3_premium_seeds))))
    metrics.append(_metric_entry(
        "H4_causal_capture_gain_M2_minus_M1",
        h4_seeds,
        "Median across buildings of PRICE/ORACLE_COST value share under M2 minus the same under M1",
        effect_value=float(np.mean(h4_seeds))))
    metrics.append(_metric_entry(
        "D_heterogeneity_value_IQR_usd",
        hetero_seeds,
        "Interquartile range across 17 buildings of ORACLE_COST battery value, M2, 6.4 kWh, USD/yr",
        effect_type="descriptive",
        effect_value=float(np.mean(hetero_seeds))))

    verdicts = {
        "H1": dict(
            supported=bool(np.mean(h1_seeds) >= 100.0 and min(h1_seeds) > 0.0),
            rule="median value gap >= 100 USD/yr and positive in every bootstrap seed"),
        "H2": dict(
            supported=bool(np.mean(h2_ratio_seeds) >= 1.0),
            rule="median protocol effect >= median intelligence gap (ratio >= 1)"),
        "H3": dict(
            supported=bool(np.mean(h3_emit_seeds) >= 0.05 and np.mean(h3_premium_seeds) <= 0.20),
            rule=">=5% emission reduction and <=20% bill premium"),
        "H4": dict(
            supported=bool(np.mean(h4_seeds) >= 0.10),
            rule="causal capture share improves by >=10 percentage points under M2"),
    }

    # ---------------- descriptive secondaries (reported, no hypothesis test) ----------------
    def _med(vals):
        return float(np.median(vals))

    sec_specs = []
    for s in seeds:
        def per_b(fn):
            return [fn(b) for b in b_ids]
        sec_specs.append({
            "no_battery_bill_M2_median_usd": _med(per_b(lambda b: get(b, s, "M2_zero_export", "NOB", 0.0)["bill"])),
            "greedy_value_M1_usd": _med(per_b(lambda b: value(b, s, "M1_net_metering", "GREEDY"))),
            "greedy_value_M2_usd": _med(per_b(lambda b: value(b, s, "M2_zero_export", "GREEDY"))),
            "buildings_positive_market_gap": float(np.sum([
                value(b, s, "M2_zero_export", "GREEDY") - value(b, s, "M1_net_metering", "GREEDY") > 0
                for b in b_ids])),
            "oracle_cost_value_M1_usd": _med(per_b(lambda b: value(b, s, "M1_net_metering", "ORACLE_COST"))),
            "oracle_cost_value_M2_usd": _med(per_b(lambda b: value(b, s, "M2_zero_export", "ORACLE_COST"))),
            "price_rule_value_M2_usd": _med(per_b(lambda b: value(b, s, "M2_zero_export", "PRICE"))),
            "carbon_rule_value_M2_usd": _med(per_b(lambda b: value(b, s, "M2_zero_export", "CARBON"))),
            "carbon_aware_emission_reduction_kg": _med([
                get(b, s, "M2_zero_export", "ORACLE_COST")["co2_gross"] -
                get(b, s, "M2_zero_export", "ORACLE_CO2")["co2_gross"] for b in b_ids]),
            "carbon_aware_bill_premium_usd": _med([
                get(b, s, "M2_zero_export", "ORACLE_CO2")["bill"] -
                get(b, s, "M2_zero_export", "ORACLE_COST")["bill"] for b in b_ids]),
            "peak_reduction_share_M2": _med([
                1.0 - get(b, s, "M2_zero_export", "ORACLE_COST")["peak"] /
                max(get(b, s, "M2_zero_export", "NOB", 0.0)["peak"], 1e-9) for b in b_ids]),
            "self_consumption_gain_M2": _med([
                get(b, s, "M2_zero_export", "GREEDY")["self_consumption"] -
                get(b, s, "M2_zero_export", "NOB", 0.0)["self_consumption"] for b in b_ids]),
            "marginal_value_per_kwh_M2": _med([
                (value(b, s, "M2_zero_export", "ORACLE_COST", 12.8) -
                 value(b, s, "M2_zero_export", "ORACLE_COST", 6.4)) / 6.4 for b in b_ids]),
            "greedy_cycles_M2": _med(per_b(lambda b: get(b, s, "M2_zero_export", "GREEDY")["cycles"])),
            "oracle_cost_cycles_M2": _med(per_b(lambda b: get(b, s, "M2_zero_export", "ORACLE_COST")["cycles"])),
            "nob_bill_M1_median_usd": _med(per_b(lambda b: get(b, s, "M1_net_metering", "NOB", 0.0)["bill"])),
            "buildings_greedy_positive_M1": float(np.sum([
                value(b, s, "M1_net_metering", "GREEDY") > 0 for b in b_ids])),
            "buildings_oracle_M2_gt_M1": float(np.sum([
                value(b, s, "M2_zero_export", "ORACLE_COST") > value(b, s, "M1_net_metering", "ORACLE_COST")
                for b in b_ids])),
            "greedy_share_of_oracle_M2": _med([
                value(b, s, "M2_zero_export", "GREEDY") / max(value(b, s, "M2_zero_export", "ORACLE_COST"), 1e-9)
                for b in b_ids]),
            "peak_ratio_oracle_over_nob_M2": _med([
                get(b, s, "M2_zero_export", "ORACLE_COST")["peak"] /
                max(get(b, s, "M2_zero_export", "NOB", 0.0)["peak"], 1e-9) for b in b_ids]),
            "buildings_peak_increase_M2": float(np.sum([
                get(b, s, "M2_zero_export", "ORACLE_COST")["peak"] >
                get(b, s, "M2_zero_export", "NOB", 0.0)["peak"] for b in b_ids])),
            "carbon_abatement_cost_usd_per_kg": _med([
                (get(b, s, "M2_zero_export", "ORACLE_CO2")["bill"] -
                 get(b, s, "M2_zero_export", "ORACLE_COST")["bill"]) /
                max(get(b, s, "M2_zero_export", "ORACLE_COST")["co2_gross"] -
                    get(b, s, "M2_zero_export", "ORACLE_CO2")["co2_gross"], 1e-9)
                for b in b_ids]),
            "h3_emission_reduction_pct": 100.0 * float(np.median([
                (get(b, s, "M2_zero_export", "ORACLE_COST")["co2_gross"] -
                 get(b, s, "M2_zero_export", "ORACLE_CO2")["co2_gross"]) /
                max(get(b, s, "M2_zero_export", "NOB", 0.0)["co2_gross"], 1e-9) for b in b_ids])),
            "h3_bill_premium_pct": 100.0 * float(np.median([
                (get(b, s, "M2_zero_export", "ORACLE_CO2")["bill"] -
                 get(b, s, "M2_zero_export", "ORACLE_COST")["bill"]) /
                max(get(b, s, "M2_zero_export", "NOB", 0.0)["bill"], 1e-9) for b in b_ids])),
            "h4_share_M1_pct": 100.0 * float(np.median([
                value(b, s, "M1_net_metering", "PRICE") /
                max(value(b, s, "M1_net_metering", "ORACLE_COST"), 1e-9) for b in b_ids])),
            "h4_share_M2_pct": 100.0 * float(np.median([
                value(b, s, "M2_zero_export", "PRICE") /
                max(value(b, s, "M2_zero_export", "ORACLE_COST"), 1e-9) for b in b_ids])),
            "greedy_share_of_oracle_pct": 100.0 * _med([
                value(b, s, "M2_zero_export", "GREEDY") /
                max(value(b, s, "M2_zero_export", "ORACLE_COST"), 1e-9) for b in b_ids]),
            "marginal_value_per_kwh_M1_usd": _med([
                (value(b, s, "M1_net_metering", "ORACLE_COST", 12.8) -
                 value(b, s, "M1_net_metering", "ORACLE_COST", 6.4)) / 6.4 for b in b_ids]),
            "greedy_marginal_value_per_kwh_M1_usd": _med([
                (value(b, s, "M1_net_metering", "GREEDY", 12.8) -
                 value(b, s, "M1_net_metering", "GREEDY", 6.4)) / 6.4 for b in b_ids]),
            "oracle_co2_cycles_M2": _med(per_b(lambda b: get(b, s, "M2_zero_export", "ORACLE_CO2")["cycles"])),
            "no_battery_gross_co2_kg": _med(per_b(lambda b: get(b, s, "M2_zero_export", "NOB", 0.0)["co2_gross"])),
            "no_battery_net_co2_kg": _med(per_b(lambda b: get(b, s, "M2_zero_export", "NOB", 0.0)["co2_net"])),
            "h2_intelligence_gap_greedy_usd": _med([
                value(b, s, "M2_zero_export", "ORACLE_COST") -
                value(b, s, "M2_zero_export", "GREEDY") for b in b_ids]),
            "h2_ratio_greedy": (_med([value(b, s, "M2_zero_export", "ORACLE_COST") -
                                      value(b, s, "M1_net_metering", "ORACLE_COST") for b in b_ids]) /
                                max(_med([value(b, s, "M2_zero_export", "ORACLE_COST") -
                                          value(b, s, "M2_zero_export", "GREEDY") for b in b_ids]), 1e-9)),
            "greedy_positive_cases_M1": float(np.sum([
                value(b, s, "M1_net_metering", "GREEDY") > 0 for b in b_ids])),
            "price_value_M1_usd": _med(per_b(lambda b: value(b, s, "M1_net_metering", "PRICE"))),
            "peak_ratio_p10": float(np.percentile([
                get(b, s, "M2_zero_export", "ORACLE_COST")["peak"] /
                max(get(b, s, "M2_zero_export", "NOB", 0.0)["peak"], 1e-9) for b in b_ids], 10)),
            "peak_ratio_p90": float(np.percentile([
                get(b, s, "M2_zero_export", "ORACLE_COST")["peak"] /
                max(get(b, s, "M2_zero_export", "NOB", 0.0)["peak"], 1e-9) for b in b_ids], 90)),
            "carbon_rule_emission_reduction_share": _med([
                (get(b, s, "M2_zero_export", "NOB", 0.0)["co2_gross"] -
                 get(b, s, "M2_zero_export", "CARBON")["co2_gross"]) /
                max(get(b, s, "M2_zero_export", "NOB", 0.0)["co2_gross"], 1e-9) for b in b_ids]),
            "oracle_cost_co2_M1": float(np.mean([
                get(b, s, "M1_net_metering", "ORACLE_COST")["co2_gross"] for b in b_ids])),
            "nob_co2_M1": float(np.mean([
                get(b, s, "M1_net_metering", "NOB", 0.0)["co2_gross"] for b in b_ids])),
            "oracle_cost_co2_M2": float(np.mean([
                get(b, s, "M2_zero_export", "ORACLE_COST")["co2_gross"] for b in b_ids])),
            "nob_co2_M2": float(np.mean([
                get(b, s, "M2_zero_export", "NOB", 0.0)["co2_gross"] for b in b_ids])),
            "h4_capture_gain_pp": 100.0 * (float(np.median([
                value(b, s, "M2_zero_export", "PRICE") /
                max(value(b, s, "M2_zero_export", "ORACLE_COST"), 1e-9) for b in b_ids])) -
                float(np.median([value(b, s, "M1_net_metering", "PRICE") /
                max(value(b, s, "M1_net_metering", "ORACLE_COST"), 1e-9) for b in b_ids]))),
            "oracle_value_min_M2_usd": float(np.min([
                value(b, s, "M2_zero_export", "ORACLE_COST") for b in b_ids])),
            "oracle_value_max_M2_usd": float(np.max([
                value(b, s, "M2_zero_export", "ORACLE_COST") for b in b_ids])),
        })
    secondary = []
    for key in sec_specs[0]:
        vals = [row[key] for row in sec_specs]
        secondary.append(dict(name=key, value=float(np.mean(vals)),
                              seed_values=[float(v) for v in vals],
                              n_seeds=len(vals),
                              description=key.replace("_", " ")))
    secondary.append(dict(name="h2_protocol_usd_mean", value=float(np.mean(h2_a_seeds)),
                          seed_values=[float(v) for v in h2_a_seeds], n_seeds=len(h2_a_seeds),
                          description="mean over seeds of the median protocol effect (USD/yr)"))
    secondary.append(dict(name="h2_intelligence_usd_mean", value=float(np.mean(h2_b_seeds)),
                          seed_values=[float(v) for v in h2_b_seeds], n_seeds=len(h2_b_seeds),
                          description="mean over seeds of the median intelligence gap (USD/yr)"))
    price_carbon_r = float(np.corrcoef(_DATA["price"], _DATA["carbon"])[0, 1])
    secondary.append(dict(name="price_carbon_correlation", value=price_carbon_r,
                          seed_values=[price_carbon_r], n_seeds=1,
                          description="Pearson correlation of hourly retail price and grid carbon intensity"))

    # artifact registry: JSON pointers into results.json
    artifacts = []
    for i, metric in enumerate(metrics):
        artifacts.append(dict(
            id=f"A-MET-{i + 1:02d}", path="results/results.json",
            pointer=f"/headline_metrics/{i}/value", metric=metric["name"],
            value=round(float(metric["value"]), 10), tolerance=1e-6))
    for i, item in enumerate(secondary):
        artifacts.append(dict(
            id=f"A-SEC-{i + 1:02d}", path="results/results.json",
            pointer=f"/secondary_metrics/{i}/value", metric=item["name"],
            value=round(float(item["value"]), 10), tolerance=1e-6))

    return dict(
        headline_metrics=metrics, verdicts=verdicts, artifacts=artifacts,
        secondary_metrics=secondary,
        optimal_capacity_by_seed=opt,
        building_ids=b_ids, seeds=list(seeds), primary_capacity_kwh=PRIMARY_CAP,
        secondary=dict(
            h2_protocol_usd=h2_a_seeds, h2_intelligence_usd=h2_b_seeds,
            h4_share_M1=h4_m1_seeds, h4_share_M2=h4_m2_seeds,
            h3_premium=h3_premium_seeds))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("pilot", "scale"))
    parser.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 4) - 2))
    args = parser.parse_args(argv)
    RESULTS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if args.stage == "pilot":
        seeds = [42]
        records = run_records(seeds, CAPS_PILOT, workers=args.workers)
        idx = _idx(records)
        summary = {
            "stage": "pilot", "seeds": seeds, "capacities": list(CAPS_PILOT),
            "n_records": len(records), "runtime_s": None,
            "sanity": {
                "oracle_cost_le_price_rule_M1": all(
                    idx[(b, 42, "M1_net_metering", 6.4, "ORACLE_COST")]["bill"]
                    <= idx[(b, 42, "M1_net_metering", 6.4, "PRICE")]["bill"] + 1e-6
                    for b in range(1, 18)),
                "oracle_cost_le_greedy_M2": all(
                    idx[(b, 42, "M2_zero_export", 6.4, "ORACLE_COST")]["bill"]
                    <= idx[(b, 42, "M2_zero_export", 6.4, "GREEDY")]["bill"] + 1e-6
                    for b in range(1, 18)),
                "oracle_co2_le_carbon_rule_M2": all(
                    idx[(b, 42, "M2_zero_export", 6.4, "ORACLE_CO2")]["co2_gross"]
                    <= idx[(b, 42, "M2_zero_export", 6.4, "CARBON")]["co2_gross"] + 1e-6
                    for b in range(1, 18)),
            },
        }
        summary["runtime_s"] = round(time.time() - t0, 1)
        (RESULTS / "pilot_results.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if all(summary["sanity"].values()) else 1

    # ---------------- scale ----------------
    records = run_records(list(core.SEEDS), CAPS, workers=args.workers)
    import csv
    with (RESULTS / "raw_metrics.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    derived = derive_headline(records, list(core.SEEDS))
    results = {
        "project": "battery-dispatch",
        "stage": "scale",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "seed_selection": "all_seeds",
        "n_seeds": len(core.SEEDS),
        "matrix": {
            "buildings": derived["building_ids"],
            "seeds": derived["seeds"],
            "markets": list(MARKETS),
            "capacities_kwh": list(CAPS),
            "controllers": list(CONTROLLERS),
            "primary_capacity_kwh": PRIMARY_CAP,
            "battery_power_rule": "P = 5 kW * capacity / 6.4 kWh",
            "efficiency": dict(round_trip=core.ROUND_TRIP, per_direction=round(core.ETA, 6)),
        },
        "datasets": [{"id": "citylearn2022_phase_all_plus_evs"}],
        **derived,
        "runtime_s": round(time.time() - t0, 1),
        "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                        "numpy": np.__version__},
    }
    (RESULTS / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                          encoding="utf-8")

    # ---------------- baselines (reference-schedule reproduction) ----------------
    base = baselines.evaluate(workers=args.workers)
    (RESULTS / "baselines.json").write_text(json.dumps(base, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    print(f"scale done in {time.time() - t0:.1f}s; records={len(records)}")
    for metric in results["headline_metrics"]:
        print(f"  {metric['name']}: {metric['value']:.4f} "
              f"[{metric['ci_low']:.4f}, {metric['ci_high']:.4f}]")
    print("  verdicts:", json.dumps(results["verdicts"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
