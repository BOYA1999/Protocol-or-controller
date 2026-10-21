from __future__ import annotations

import csv
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

import core
import mpc

ROOT = core.ROOT
RESULTS = ROOT / "results"
CAP, POWER = core.DEFAULT_CAP_KWH, core.DEFAULT_POWER_KW
_DATA = None


def _init():
    global _DATA
    _DATA = core.load_main()


def _task(payload):
    building, seed, market, horizon, perfect = payload
    if _DATA is None:
        _init()
    order = core.bootstrap_week_order(seed)
    load = core.bootstrap_series(building["load"], order)
    pv = core.bootstrap_series(building["pv_kw"], order)
    price = core.bootstrap_series(_DATA["price"], order)
    carbon = core.bootstrap_series(_DATA["carbon"], order)
    if perfect:
        res = mpc.simulate_mpc(load, pv, price, carbon, CAP, POWER, market,
                               horizon=horizon, lookback=0)
    else:
        res = mpc.simulate_mpc(load, pv, price, carbon, CAP, POWER, market,
                               horizon=horizon)
    met = core.compute_metrics(res, load, pv, price, carbon, market, CAP)
    nob = core.compute_metrics(
        core.simulate_rule("NOB", load, pv, price, carbon, 0.0, 0.0),
        load, pv, price, carbon, market, 0.0)
    return dict(building_id=building["id"], seed=seed, market=market,
                horizon=horizon, forecast="perfect" if perfect else "persistence",
                bill=met["bill"], nob_bill=nob["bill"],
                value=nob["bill"] - met["bill"], co2_gross=met["co2_gross"],
                co2_net=met["co2_net"],
                peak=met["peak"], cycles=met["cycles"],
                throughput=met["throughput"])


def _run(payloads):
    with mp.Pool(processes=max(1, (mp.cpu_count() or 4) - 2), initializer=_init) as pool:
        return pool.map(_task, payloads, chunksize=1)


def main() -> int:
    data = core.load_main()
    buildings = data["buildings"]
    seeds = list(core.SEEDS)
    payloads = [(b, s, m, 24, False) for s in seeds for b in buildings
                for m in core.MARKETS]
    rows = _run(payloads)
    extra = [(b, 42, m, 48, False) for b in buildings for m in core.MARKETS]
    extra += [(b, 42, m, 24, True) for b in buildings for m in core.MARKETS]
    rows += _run(extra)
    with (RESULTS / "mpc_results.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def summarize(subset):
        out = {}
        for market in core.MARKETS:
            vals = [r["value"] for r in subset if r["market"] == market]
            out[market] = dict(median_value_usd=float(np.median(vals)),
                               mean_value_usd=float(np.mean(vals)),
                               n=len(vals))
        return out

    primary = [r for r in rows if r["horizon"] == 24 and r["forecast"] == "persistence"]
    h48 = [r for r in rows if r["horizon"] == 48]
    perfect = [r for r in rows if r["forecast"] == "perfect"]
    summary = dict(
        description=("Forecast-based receding-horizon controller (persistence forecast, 1-h commit). "
                     "Receding-horizon controller evaluation."),
        design=dict(horizon_h=24, lookback_h=24, commit_h=1,
                    forecast="same hour of previous day (load, PV, price, carbon)",
                    terminal_constraint="SOC_end >= SOC_start over each horizon",
                    capacity_kwh=CAP, power_kw=POWER),
        primary=summarize(primary),
        horizon_48=summarize(h48),
        perfect_forecast=summarize(perfect),
        per_seed={},
    )
    for seed in seeds:
        sub = [r for r in primary if r["seed"] == seed]
        summary["per_seed"][str(seed)] = summarize(sub)
    # forecast-error gap on replicate 42 (persistence vs perfect, horizon 24)
    gap = {}
    for market in core.MARKETS:
        p = [r["value"] for r in perfect if r["market"] == market]
        q = [r["value"] for r in primary if r["seed"] == 42 and r["market"] == market]
        gap[market] = dict(perfect_median=float(np.median(p)),
                           persistence_median=float(np.median(q)),
                           forecast_error_gap_usd=float(np.median(p) - np.median(q)))
    summary["forecast_error_gap_seed42"] = gap
    # carbon and peak/cycles descriptive
    summary["descriptive"] = dict(
        median_cycles_M2=float(np.median([r["cycles"] for r in primary
                                          if r["market"] == "M2_zero_export"])),
        median_peak_ratio_M2=float(np.median(
            [r["peak"] / max(r["peak"], 1e-9) for r in primary
             if r["market"] == "M2_zero_export"])),
    )
    (RESULTS / "mpc_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
