from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

import core

ROOT = core.ROOT
RESULTS = ROOT / "results"
CAP = core.DEFAULT_CAP_KWH
POWER = core.DEFAULT_POWER_KW
_SEEDS = list(core.SEEDS)
_DATA = None


def _init():
    global _DATA
    _DATA = core.load_main()


def _year(building, seed):
    order = core.bootstrap_week_order(seed)
    return (core.bootstrap_series(building["load"], order),
            core.bootstrap_series(building["pv_kw"], order),
            core.bootstrap_series(_DATA["price"], order),
            core.bootstrap_series(_DATA["carbon"], order))


def _peak_task(payload):
    building, seed = payload
    load, pv, price, carbon = _year(building, seed)
    nob_rule = core.simulate_rule("NOB", load, pv, price, carbon, 0.0, 0.0)
    out = {}
    for market in core.MARKETS:
        nob = core.compute_metrics(nob_rule, load, pv, price, carbon, market, 0.0)
        unconstrained = core.solve_oracle(load, pv, price, carbon, CAP, POWER, market, "cost")
        m_un = core.compute_metrics(unconstrained, load, pv, price, carbon, market, CAP)
        constrained = core.solve_oracle(load, pv, price, carbon, CAP, POWER, market, "cost",
                                        peak_cap=nob["peak"])
        m_co = core.compute_metrics(constrained, load, pv, price, carbon, market, CAP)
        out[market] = dict(
            nob_bill=nob["bill"], nob_peak=nob["peak"],
            unconstrained_bill=m_un["bill"], unconstrained_peak=m_un["peak"],
            constrained_bill=m_co["bill"], constrained_peak=m_co["peak"],
            unconstrained_value=nob["bill"] - m_un["bill"],
            constrained_value=nob["bill"] - m_co["bill"])
    return building["id"], seed, out


def _soc_task(payload):
    building, seed = payload
    load, pv, price, carbon = _year(building, seed)
    out = {}
    for ctrl in ("GREEDY", "PRICE", "CARBON"):
        res = core.simulate_rule(ctrl, load, pv, price, carbon, CAP, POWER)
        out[ctrl] = float(res.soc[-1] / CAP)
    return building["id"], seed, out


def _export_credit_tasks(payload):
    """Gap(theta) for one building/seed: intermediate export-credit fractions."""
    building, seed = payload
    load, pv, price, carbon = _year(building, seed)
    nob_net = core.simulate_rule("NOB", load, pv, price, carbon, 0.0, 0.0)
    greedy = core.simulate_rule("GREEDY", load, pv, price, carbon, CAP, POWER)
    out = {}
    for theta in (0.0, 0.25, 0.5, 0.75, 1.0):
        def bill(net):
            return float(np.sum(price * np.maximum(net, 0.0)) +
                         theta * np.sum(price * np.minimum(net, 0.0)))
        out[str(theta)] = bill(nob_net.net) - bill(greedy.net)
    return building["id"], seed, out


def _run(fn, payloads):
    _init()
    with mp.Pool(processes=max(1, (mp.cpu_count() or 4) - 2), initializer=_init) as pool:
        return pool.map(fn, payloads, chunksize=1)


def main() -> int:
    data = core.load_main()
    payloads = [(b, s) for s in _SEEDS for b in data["buildings"]]
    peak_rows = _run(_peak_task, payloads)
    soc_rows = _run(_soc_task, payloads)
    theta_rows = _run(_export_credit_tasks, payloads)

    peak = {}
    for market in core.MARKETS:
        vals = [row[market] for _, _, row in peak_rows]
        un = np.array([v["unconstrained_value"] for v in vals])
        co = np.array([v["constrained_value"] for v in vals])
        peak[market] = dict(
            mean_unconstrained_value_usd=float(un.mean()),
            mean_peak_neutral_value_usd=float(co.mean()),
            value_retention_share=float(co.mean() / un.mean()) if un.mean() else None,
            median_peak_ratio_unconstrained=float(np.median(
                [v["unconstrained_peak"] / max(v["nob_peak"], 1e-9) for v in vals])),
            median_peak_ratio_constrained=float(np.median(
                [v["constrained_peak"] / max(v["nob_peak"], 1e-9) for v in vals])),
            n_rows=len(vals),
        )
    soc = {ctrl: dict(
        mean_final_soc_fraction=float(np.mean([row[ctrl] for _, _, row in soc_rows])),
        min_final_soc_fraction=float(np.min([row[ctrl] for _, _, row in soc_rows])),
        max_final_soc_fraction=float(np.max([row[ctrl] for _, _, row in soc_rows])),
    ) for ctrl in ("GREEDY", "PRICE", "CARBON")}
    theta = {}
    for th in ("0.0", "0.25", "0.5", "0.75", "1.0"):
        theta[th] = float(np.median([row[th] for _, _, row in theta_rows]))
    out = dict(
        description=("Post-hoc extensions (not pre-registered): peak-constrained cost "
                     "optimisation, terminal SOC of the rule controllers, and the "
                     "export-credit-fraction sweep."),
        peak_constrained=peak,
        terminal_soc=soc,
        export_credit_sweep_gap_usd=theta,
        rows_peak=[{"building": b, "seed": s, **row} for b, s, row in peak_rows],
    )
    (RESULTS / "extensions.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("peak_constrained", "terminal_soc",
                                          "export_credit_sweep_gap_usd")},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
