from __future__ import annotations

import csv
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

import core

ROOT = core.ROOT
RESULTS = ROOT / "results"
SEEDS = list(core.SEEDS)
CAP, POWER = core.DEFAULT_CAP_KWH, core.DEFAULT_POWER_KW
_DATA = None


def _init():
    global _DATA
    _DATA = core.load_main()


def _rule(load, pv, price, carbon, signal, lo_q, hi_q, window):
    n = load.size
    charge = np.zeros(n)
    discharge = np.zeros(n)
    soc = 0.5 * CAP
    for t in range(n):
        surplus = pv[t] - load[t]
        deficit = -surplus
        lo, hi = signal[t], signal[t]
        if t >= window:
            past = signal[t - window:t]
            lo = float(np.quantile(past, lo_q))
            hi = float(np.quantile(past, hi_q))
        if surplus > 0:
            if soc < CAP:
                c = min(surplus, POWER, (CAP - soc) / core.ETA)
                charge[t] = c
                soc += core.ETA * c
        elif signal[t] >= hi and soc > 0:
            d = min(deficit, POWER, soc * core.ETA)
            discharge[t] = d
            soc -= d / core.ETA
        elif signal[t] <= lo and soc < CAP:
            c = min(POWER, (CAP - soc) / core.ETA)
            charge[t] = c
            soc += core.ETA * c
        soc = min(max(soc, 0.0), CAP)
    return load - pv + charge - discharge


def _task(payload):
    building, seed, lo_q, hi_q, window = payload
    order = core.bootstrap_week_order(seed)
    load = core.bootstrap_series(building["load"], order)
    pv = core.bootstrap_series(building["pv_kw"], order)
    price = core.bootstrap_series(_DATA["price"], order)
    carbon = core.bootstrap_series(_DATA["carbon"], order)
    net = _rule(load, pv, price, carbon, price, lo_q, hi_q, window)
    out = {}
    for market in core.MARKETS:
        bill = float(np.sum(price * net) if market == "M1_net_metering"
                     else np.sum(price * np.maximum(net, 0.0)))
        out[market] = bill
    return building["id"], seed, out


def main() -> int:
    data = core.load_main()
    # oracle values from the frozen raw metrics
    rows = list(csv.DictReader((RESULTS / "raw_metrics.csv").open(encoding="utf-8")))
    idx = {(int(r["building_id"]), int(r["seed"]), r["market"], float(r["capacity_kwh"]),
            r["controller"]): r for r in rows}

    def value(b, s, market, ctrl="ORACLE_COST"):
        return (float(idx[(b, s, market, 6.4, "NOB")]["bill"]) -
                float(idx[(b, s, market, 6.4, ctrl)]["bill"]))

    variants = [(0.30, 0.70, 168), (0.20, 0.80, 168), (0.40, 0.60, 168), (0.30, 0.70, 336)]
    out = {}
    for lo_q, hi_q, window in variants:
        payloads = [(b, s, lo_q, hi_q, window) for s in SEEDS for b in data["buildings"]]
        with mp.Pool(processes=max(1, (mp.cpu_count() or 4) - 2), initializer=_init) as pool:
            chunks = pool.map(_task, payloads, chunksize=1)
        gains, caps_m1, caps_m2 = [], [], []
        for market in core.MARKETS:
            pass
        by_seed = {}
        for b_id, seed, bills in chunks:
            by_seed.setdefault(seed, []).append((b_id, bills))
        for seed, items in by_seed.items():
            s1, s2 = [], []
            for b_id, bills in items:
                v1 = value(b_id, seed, "M1_net_metering")
                v2 = value(b_id, seed, "M2_zero_export")
                bill_m1 = float(idx[(b_id, seed, "M1_net_metering", 6.4, "NOB")]["bill"]) - bills["M1_net_metering"]
                bill_m2 = float(idx[(b_id, seed, "M2_zero_export", 6.4, "NOB")]["bill"]) - bills["M2_zero_export"]
                s1.append(bill_m1 / max(v1, 1e-9))
                s2.append(bill_m2 / max(v2, 1e-9))
            caps_m1.append(float(np.median(s1)))
            caps_m2.append(float(np.median(s2)))
            gains.append(float(np.median(s2) - np.median(s1)))
        key = f"q{int(lo_q*100)}_{int((1-hi_q)*100)}_w{window}"
        out[key] = dict(lo_quantile=lo_q, hi_quantile=hi_q, window_h=window,
                        capture_m1_mean=float(np.mean(caps_m1)),
                        capture_m2_mean=float(np.mean(caps_m2)),
                        gain_mean_pp=100.0 * float(np.mean(gains)),
                        gain_per_seed_pp=[100.0 * g for g in gains])
    (RESULTS / "threshold_sensitivity.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: dict(gain_mean_pp=round(v["gain_mean_pp"], 2),
                              capture_m1=round(v["capture_m1_mean"], 3),
                              capture_m2=round(v["capture_m2_mean"], 3))
                      for k, v in out.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
