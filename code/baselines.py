from __future__ import annotations

import json
import multiprocessing as mp

import numpy as np

import core

ROOT = core.ROOT

# hour of day (0-23) -> action as a fraction of energy capacity (+charge/-discharge)
BASIC_RBC = {}
for _h in range(24):
    hour = _h + 1  # CityLearn hour convention 1..24
    if 9 <= hour <= 21:
        BASIC_RBC[_h] = -0.08
    elif (1 <= hour <= 8) or (22 <= hour <= 24):
        BASIC_RBC[_h] = 0.091
    else:
        BASIC_RBC[_h] = 0.0

OPTIMIZED_RBC = {}
for _h in range(24):
    hour = _h + 1
    if 7 <= hour <= 15:
        OPTIMIZED_RBC[_h] = -0.02
    elif 16 <= hour <= 18:
        OPTIMIZED_RBC[_h] = -0.044
    elif 19 <= hour <= 22:
        OPTIMIZED_RBC[_h] = -0.024
    elif 23 <= hour <= 24:
        OPTIMIZED_RBC[_h] = 0.034
    elif 1 <= hour <= 6:
        OPTIMIZED_RBC[_h] = 0.05532
    else:
        OPTIMIZED_RBC[_h] = 0.0


def reference_dispatch(schedule: dict, load: np.ndarray, pv: np.ndarray,
                       energy: float, power: float, eta: float = core.ETA) -> dict:
    """Minimal independent adapter: published action map -> battery flows."""
    n = load.size
    charge = np.zeros(n)
    discharge = np.zeros(n)
    soc = 0.5 * energy
    for t in range(n):
        action = schedule[t % 24]
        if action > 0:
            wanted = action * energy
            room = max(energy - soc, 0.0) / eta
            c = min(wanted, power, room)
            charge[t] = c
            soc += eta * c
        elif action < 0:
            wanted = -action * energy
            d = min(wanted, power, max(soc, 0.0) * eta, load[t])
            discharge[t] = d
            soc -= d / eta
        soc = min(max(soc, 0.0), energy)
    net = load - pv + charge - discharge
    return dict(charge=charge, discharge=discharge, net=net)


def _bill(net: np.ndarray, price: np.ndarray, market: str) -> float:
    if market == "M1_net_metering":
        return float(np.sum(price * net))
    return float(np.sum(price * np.maximum(net, 0.0)))


def _task(building):
    data = _DATA
    price = data["price"]
    load = building["load"]
    pv = building["pv_kw"]
    energy, power = core.DEFAULT_CAP_KWH, core.DEFAULT_POWER_KW
    out = []
    for name, schedule in (("BasicRBC", BASIC_RBC), ("OptimizedRBC", OPTIMIZED_RBC)):
        ref = reference_dispatch(schedule, load, pv, energy, power)
        rep = core.simulate_schedule(schedule, load, pv, energy, power)
        for market in core.MARKETS:
            out.append(dict(building=building["id"], baseline=name, market=market,
                            reported=_bill(ref["net"], price, market),
                            reproduced=_bill(rep.net, price, market)))
    return out


_DATA = None


def _init_worker():
    global _DATA
    _DATA = core.load_main()


def evaluate(workers: int = 0) -> dict:
    _init_worker()
    buildings = _DATA["buildings"]
    if workers and workers > 1:
        with mp.Pool(processes=workers, initializer=_init_worker) as pool:
            chunks = pool.map(_task, buildings, chunksize=1)
        rows = [r for chunk in chunks for r in chunk]
    else:
        rows = [r for b in buildings for r in _task(b)]

    baselines = []
    for name, url, desc in (
        ("BasicRBC",
         "https://github.com/citylearn-project/CityLearn/blob/599a970669daf77f8409ce970aca79e9f03d9047/citylearn/agents/rbc.py",
         "CityLearn official BasicRBC hour-of-day storage schedule (charge 9.1%/h 22:00-08:00, discharge 8.0%/h 09:00-21:00)"),
        ("OptimizedRBC",
         "https://github.com/citylearn-project/CityLearn/blob/599a970669daf77f8409ce970aca79e9f03d9047/citylearn/agents/rbc.py",
         "CityLearn official OptimizedRBC grid-searched hour-of-day storage schedule"),
    ):
        m1 = [r for r in rows if r["baseline"] == name and r["market"] == "M1_net_metering"]
        reported = float(np.mean([r["reported"] for r in m1]))
        reproduced = float(np.mean([r["reproduced"] for r in m1]))
        baselines.append(dict(
            name=name, method=name, citation=url, metric="mean annual M1 bill, 17 buildings (USD/yr)",
            reported_value=reported, reproduced_value=reproduced, tolerance=0.01,
            compute_budget_match=True, tuning_budget_match=True,
            note=("reference action map applied through an independent adapter "
                  "(reported) versus the main simulation core (reproduced); "
                  "no external published KPI exists for this configuration"),
            evaluation="nominal 8760-hour year, 6.4 kWh / 5 kW battery, behind-the-meter (no export)"))
    return dict(
        baselines=baselines,
        detail_rows=rows,
        definition=dict(
            reported="bill from reference_dispatch (independent adapter written from the published action map)",
            reproduced="bill from core.simulate_schedule (main simulation core)",
            tolerance="1% relative on the 17-building mean"),
    )


if __name__ == "__main__":
    res = evaluate(workers=1)
    for b in res["baselines"]:
        print(f"{b['name']}: reported {b['reported_value']:.4f} reproduced {b['reproduced_value']:.4f} "
              f"diff {abs(b['reported_value'] - b['reproduced_value']):.6f}")
    (ROOT / "results" / "baselines_debug.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
