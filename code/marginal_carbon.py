from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timedelta, timezone

import numpy as np

import core


def check_dispatch(res, load, pv):
    previous = np.r_[3.2, res.soc[:-1]]
    errors = {
        "soc_dynamics_kwh": float(np.max(np.abs(res.soc - previous - core.ETA * res.charge + res.discharge / core.ETA))),
        "terminal_soc_kwh": float(abs(res.soc[-1] - 3.2)),
        "soc_bounds_kwh": float(max(0, -res.soc.min(), res.soc.max() - 6.4)),
        "power_bounds_kw": float(max(0, res.charge.max() - 5, res.discharge.max() - 5, np.max(res.discharge - load))),
        "net_balance_kw": float(np.max(np.abs(res.net - load + pv - res.charge + res.discharge))),
        "simultaneous_kw": float(np.max(np.minimum(res.charge, res.discharge))),
    }
    if max(errors.values()) > 1e-6:
        raise AssertionError(errors)
    return errors


def main():
    source = core.ROOT / "data/marginal/sce_moer_2017_hourly.csv"
    with source.open(encoding="utf-8") as fh:
        hourly = {datetime.fromisoformat(r["timestamp_utc"].replace("Z", "+00:00")): r for r in csv.DictReader(fh)}
    start, stop = 3672, 8760
    dates = [datetime(2017, 1, 1) + timedelta(hours=i) for i in range(stop - start)]
    scenarios = {
        "v2_pst": ("v2", lambda dt: 8),
        "v1_pst": ("v1", lambda dt: 8),
        "v2_minus_1h": ("v2", lambda dt: 7),
        "v2_plus_1h": ("v2", lambda dt: 9),
        "v2_seasonal_dst_shift": ("v2", lambda dt: 7 if dt >= datetime(2017, 3, 12, 3) else 8),
    }
    factors = {}
    for name, (version, offset) in scenarios.items():
        values = np.array([float(hourly[(dt + timedelta(hours=offset(dt))).replace(tzinfo=timezone.utc)][f"moer_{version}_kg_co2_per_kwh"]) for dt in dates])
        assert np.isfinite(values).all() and values.min() >= 0
        factors[name] = values
    data = core.load_main()
    price, supplied = data["price"][start:stop], data["carbon"][start:stop]
    rows, checks, worst_objective_gap = [], {}, 0.0
    for building in data["buildings"]:
        load, pv = building["load"][start:stop], building["pv_kw"][start:stop]
        nob = core.simulate_rule("NOB", load, pv, price, supplied, 0, 0)
        schedules = {market: core.solve_oracle(load, pv, price, supplied, 6.4, 5, market, "cost") for market in core.MARKETS}
        supplied_oracle = core.solve_oracle(load, pv, price, supplied, 6.4, 5, core.MARKETS[0], "co2")
        for name, factor in factors.items():
            moer_oracle = core.solve_oracle(load, pv, price, factor, 6.4, 5, core.MARKETS[0], "co2_net")
            for controller, res in {"SUPPLIED_CO2": supplied_oracle, "NET_MOER": moer_oracle, **schedules}.items():
                errors = check_dispatch(res, load, pv)
                for key, value in errors.items():
                    checks[key] = max(checks.get(key, 0), value)
                if controller != "NET_MOER":
                    gap = float(np.dot(factor, moer_oracle.net - res.net))
                    worst_objective_gap = max(worst_objective_gap, gap)
                    assert gap <= 0.02, gap
            for market in core.MARKETS:
                for controller, res in {"NOB": nob, "ENERGY_COST": schedules[market], "SUPPLIED_CO2": supplied_oracle, "NET_MOER": moer_oracle}.items():
                    met = core.compute_metrics(res, load, pv, price, supplied, market, 0 if controller == "NOB" else 6.4)
                    rows.append(dict(building=building["id"], scenario=name, market=market, controller=controller,
                                     bill_usd=met["bill"], net_delta_co2_kg=float(np.dot(factor, res.net - nob.net)),
                                     gross_import_co2_kg=float(np.dot(factor, np.maximum(res.net, 0))),
                                     supplied_gross_co2_kg=met["co2_gross"], cycles=met["cycles"],
                                     discharge_during_pv_surplus_kwh=float(res.discharge[pv > load].sum())))
        print(f"Marginal-carbon analysis: building {building['id']} complete", flush=True)
    idx = {(r["building"], r["scenario"], r["market"], r["controller"]): r for r in rows}
    summary = {"design": dict(hours=stop-start, n_buildings=17, source_indices=[start, stop],
                              start_local="2017-01-01T00:00:00", stop_local_exclusive="2017-08-01T00:00:00",
                              primary_scenario="v2_pst", annualized=False, estimator="median across 17 buildings",
                              endpoint="sum MOER * (net_dispatch - net_NOB); exports included",
                              signal_role="retrospective marginal emissions estimate; not measured avoided emissions",
                              dispatch="perfect foresight, 6.4 kWh/5 kW, cyclic SOC; discharge <= gross building load",
                              tariff="frozen benchmark energy prices; not historical SCE tariff",
                              hourly_input_sha256=hashlib.sha256(source.read_bytes()).hexdigest()), "scenarios": {}}
    med = lambda values: float(np.median(values))
    for name, factor in factors.items():
        summary["scenarios"][name] = {}
        for market in core.MARKETS:
            cost = [idx[(b["id"], name, market, "ENERGY_COST")] for b in data["buildings"]]
            moer = [idx[(b["id"], name, market, "NET_MOER")] for b in data["buildings"]]
            old = [idx[(b["id"], name, market, "SUPPLIED_CO2")] for b in data["buildings"]]
            reduction = [c["net_delta_co2_kg"] - m["net_delta_co2_kg"] for c, m in zip(cost, moer)]
            premium = [m["bill_usd"] - c["bill_usd"] for c, m in zip(cost, moer)]
            summary["scenarios"][name][market] = dict(
                cost_net_delta_co2_kg=med([r["net_delta_co2_kg"] for r in cost]),
                supplied_oracle_net_delta_co2_kg=med([r["net_delta_co2_kg"] for r in old]),
                moer_oracle_net_delta_co2_kg=med([r["net_delta_co2_kg"] for r in moer]),
                moer_reduction_vs_cost_kg=med(reduction), moer_bill_premium_vs_cost_usd=med(premium),
                median_abatement_cost_usd_per_kg=med([p / r for p, r in zip(premium, reduction) if r > 1e-6]),
                cost_emission_increase_buildings=sum(r["net_delta_co2_kg"] > 1e-6 for r in cost),
                supplied_oracle_emission_increase_buildings=sum(r["net_delta_co2_kg"] > 1e-6 for r in old),
                moer_oracle_emission_decrease_buildings=sum(r["net_delta_co2_kg"] < -1e-6 for r in moer))
    summary["validation"] = dict(max_dispatch_errors=checks, maximum_moer_objective_gap_kg=worst_objective_gap,
                                  lp_solves=17 * (3 + len(factors)), rows=len(rows), tolerance=1e-6)
    with (core.ROOT / "results/marginal_carbon_results.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (core.ROOT / "results/marginal_carbon_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["# Retrospective marginal-carbon re-evaluation over 5,088 h", "",
             "Median across 17 buildings, 2017 January–July; negative net delta means lower estimated grid emissions than no battery. Values are not annualized. SCE SGIP v2 is the primary backcast; v1 and alignment variants are sensitivities.", "",
             "| Signal / alignment | Market | Cost dispatch net delta (kg CO2) | Supplied-factor dispatch net delta (kg CO2) | Net-MOER dispatch net delta (kg CO2) | Reduction vs cost (kg CO2) | Bill premium (USD) |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for scenario, markets in summary["scenarios"].items():
        for market, result in markets.items():
            values = [result[k] for k in ["cost_net_delta_co2_kg", "supplied_oracle_net_delta_co2_kg", "moer_oracle_net_delta_co2_kg", "moer_reduction_vs_cost_kg", "moer_bill_premium_vs_cost_usd"]]
            lines.append(f"| {scenario} | {market.split('_')[0]} | " + " | ".join(f"{v:.2f}" for v in values) + " |")
    lines += ["", "ΔCO2 = Σ MOER × (net dispatch − net no battery), including changes in exported PV. Primary alignment assumes fixed PST; ±1 h and a seasonal one-hour shift assess timestamp uncertainty. MOER minimization uses the frozen discharge ≤ gross-load constraint; it is not a strict prohibition on battery-associated exports. All dispatches have perfect foresight and cyclic state of charge; this is not a deployment test."]
    (core.ROOT / "outputs/tables/table8_marginal_carbon.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
