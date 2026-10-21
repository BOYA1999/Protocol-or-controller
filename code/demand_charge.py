from __future__ import annotations
import csv
import json
import time
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
import core
import tea

ROOT = core.ROOT
RATES = (0.0, 5.0, 10.0, 20.0)
E, P = core.DEFAULT_CAP_KWH, core.DEFAULT_POWER_KW


def billing(result, load, pv, price, carbon, month, market, rate):
    met = core.compute_metrics(result, load, pv, price, carbon, market, E)
    peak_sum = float(sum(np.maximum(result.net[month == m], 0).max() for m in np.unique(month)))
    return dict(energy_bill_usd=met["bill"], demand_bill_usd=rate * peak_sum,
                total_bill_usd=met["bill"] + rate * peak_sum,
                monthly_peak_sum_kw=peak_sum, annual_peak_kw=met["peak"], cycles=met["cycles"])


def physics(result, load, cap=None):
    residual = result.soc - np.r_[E / 2, result.soc[:-1]] - core.ETA * result.charge + result.discharge / core.ETA
    return dict(soc_dynamics_kwh=float(np.max(np.abs(residual))),
                terminal_soc_kwh=abs(float(result.soc[-1]) - E / 2),
                soc_bound_kwh=max(0., float(-result.soc.min()), float(result.soc.max()) - E),
                charge_power_kw=max(0., float(result.charge.max()) - P),
                discharge_power_kw=max(0., float(np.max(result.discharge - np.minimum(P, load)))),
                connection_cap_kw=max(0., float(result.net.max()) - cap) if cap is not None else 0.,
                simultaneous_charge_discharge_kw=float(np.minimum(result.charge, result.discharge).max()))


def independent_check():
    load = np.array([1., 2., .5, 3., 1., 4., .4, 2.])
    pv = np.array([0., 4., 0., 1., 3., 0., 2., 0.])
    price = np.array([.1, .3, .12, .35, .1, .04, .07, .28])
    month = np.repeat([1, 2], 4); n = len(load); carbon = np.ones(n)
    cumulative = np.tril(np.ones((n, n)))
    storage = np.column_stack([core.ETA * cumulative, -cumulative / core.ETA, np.zeros((n, 2))])
    net = np.column_stack([np.eye(n), -np.eye(n), np.zeros((n, 2))])
    monthly = net.copy(); monthly[np.arange(n), 2 * n + month - 1] = -1.
    out = []
    for rate in (0., 5.):
        for cap in (None, 4.):
            rows = [storage, -storage, monthly]; rhs = [np.ones(n), np.ones(n), pv - load]
            if cap is not None:
                rows.append(net); rhs.append(cap - load + pv)
            objective = np.r_[price + 1e-6, -price + 1e-6, [rate, rate]]
            bounds = [(0, 1.)] * n + [(0, min(1., float(v))) for v in load] + [(0, None)] * 2
            independent = linprog(objective, A_ub=np.vstack(rows), b_ub=np.concatenate(rhs),
                                  A_eq=storage[-1:].copy(), b_eq=[0.], bounds=bounds, method="highs")
            if not independent.success:
                raise RuntimeError(independent.message)
            x = independent.x; n_ind = load - pv + x[:n] - x[n:2*n]
            result = core.solve_oracle_demand(load, pv, price, carbon, month, 2., 1.,
                                               "M1_net_metering", rate, cap)
            def total(n):
                return float(price @ n + rate * sum(np.maximum(n[month == m], 0).max() for m in (1, 2)))
            out.append(dict(rate=rate, cap=cap, core_bill_usd=total(result.net),
                            independent_bill_usd=total(n_ind), error_usd=abs(total(result.net) - total(n_ind))))
    if max(r["error_usd"] for r in out) > 1e-7:
        raise AssertionError(out)
    return out


def objective_checks(rows):
    tariff, cap = 0., 0.
    for b in range(1, 18):
        for market in core.MARKETS:
            for limit in ("unconstrained", "nob_annual_peak"):
                sub = [r for r in rows if (r["building_id"], r["market"], r["connection_limit"]) == (b, market, limit)]
                for chosen in sub:
                    rate = chosen["demand_rate_usd_per_kw_month"]
                    tariff = max(tariff, chosen["total_bill_usd"] - min(r["energy_bill_usd"] + rate * r["monthly_peak_sum_kw"] for r in sub))
            for rate in RATES:
                sub = {r["connection_limit"]: r for r in rows if (r["building_id"], r["market"], r["demand_rate_usd_per_kw_month"]) == (b, market, rate)}
                cap = max(cap, sub["unconstrained"]["total_bill_usd"] - sub["nob_annual_peak"]["total_bill_usd"])
    return dict(cross_tariff_feasible_dispatch_dominance_violation_usd=tariff,
                unconstrained_vs_capped_dominance_violation_usd=cap)


def repair_peak(validation):
    start = time.time(); path = ROOT / "results/extensions.json"
    ext = json.loads(path.read_text(encoding="utf-8")); old = ext["peak_constrained"]
    previous_validation = ROOT / "results/demand_charge_validation.json"
    if previous_validation.exists():
        old = json.loads(previous_validation.read_text(encoding="utf-8")).get("bug_fix", {}).get("old_peak_summary", old)
    raw = list(csv.DictReader((ROOT / "results/raw_metrics.csv").open(encoding="utf-8")))
    idx = {(int(r["building_id"]), int(r["seed"]), r["market"], float(r["capacity_kwh"]), r["controller"]): r for r in raw}
    data = core.load_main(); rows = []; maxima = {}; dominance = 0.
    for seed in core.SEEDS:
        order = core.bootstrap_week_order(seed)
        price, carbon = [core.bootstrap_series(data[k], order) for k in ("price", "carbon")]
        for b in data["buildings"]:
            load, pv = [core.bootstrap_series(b[k], order) for k in ("load", "pv_kw")]
            row = dict(building=b["id"], seed=seed)
            for market in core.MARKETS:
                nob = idx[(b["id"], seed, market, 0., "NOB")]; un = idx[(b["id"], seed, market, E, "ORACLE_COST")]
                cap = float(nob["peak"])
                result = core.solve_oracle(load, pv, price, carbon, E, P, market, "cost", peak_cap=cap)
                met = core.compute_metrics(result, load, pv, price, carbon, market, E)
                for k, v in physics(result, load, cap).items(): maxima[k] = max(maxima.get(k, 0.), v)
                dominance = max(dominance, float(un["bill"]) - met["bill"])
                row[market] = dict(nob_bill=float(nob["bill"]), nob_peak=cap, unconstrained_bill=float(un["bill"]),
                                   unconstrained_peak=float(un["peak"]), constrained_bill=met["bill"], constrained_peak=met["peak"],
                                   unconstrained_value=float(nob["bill"]) - float(un["bill"]), constrained_value=float(nob["bill"]) - met["bill"])
            rows.append(row)
        print("peak-cap replicate", seed, "complete", flush=True)
    peak = {}
    for market in core.MARKETS:
        vals = [r[market] for r in rows]; un = np.mean([v["unconstrained_value"] for v in vals]); co = np.mean([v["constrained_value"] for v in vals])
        peak[market] = dict(mean_unconstrained_value_usd=float(un), mean_peak_neutral_value_usd=float(co),
                            value_retention_share=float(co / un), n_rows=len(vals),
                            median_peak_ratio_unconstrained=float(np.median([v["unconstrained_peak"] / v["nob_peak"] for v in vals])),
                            median_peak_ratio_constrained=float(np.median([v["constrained_peak"] / v["nob_peak"] for v in vals])))
    passed = max(maxima.values()) < 1e-6 and dominance < .02
    validation["bug_fix"] = dict(description="M1 peak-cap LP retains retail export credit; gross-import auxiliaries only enforce constraints.",
                                  n_capped_lps=170, n_hours_per_replicate=8736, physical_maxima=maxima,
                                  unconstrained_dominance_violation_usd=dominance, old_peak_summary=old,
                                  corrected_peak_summary=peak, passed=passed, runtime_s=time.time() - start)
    (ROOT / "results/demand_charge_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    if not passed: raise AssertionError(validation["bug_fix"])
    ext["peak_constrained"] = peak; ext["rows_peak"] = rows
    path.write_text(json.dumps(ext, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    start = time.time(); validation = {"independent_monthly_m1_checks": independent_check()}
    repair_peak(validation)
    data = core.load_main(); rows = []; maxima = {}; dominance = 0.; accounting_error = 0.
    for b in data["buildings"]:
        load, pv, month = [b[k] for k in ("load", "pv_kw", "month")]; price, carbon = data["price"], data["carbon"]
        nob = core.simulate_rule("NOB", load, pv, price, carbon, 0., 0.); annual_cap = float(np.maximum(nob.net, 0).max())
        for market in core.MARKETS:
            for limit in ("unconstrained", "nob_annual_peak"):
                cap = None if limit == "unconstrained" else annual_cap
                for rate in RATES:
                    result = core.solve_oracle_demand(load, pv, price, carbon, month, E, P, market, rate, cap)
                    base = billing(nob, load, pv, price, carbon, month, market, rate)
                    met = billing(result, load, pv, price, carbon, month, market, rate)
                    ev = base["energy_bill_usd"] - met["energy_bill_usd"]; dv = base["demand_bill_usd"] - met["demand_bill_usd"]
                    value = base["total_bill_usd"] - met["total_bill_usd"]
                    accounting_error = max(accounting_error, abs(value - ev - dv))
                    dominance = max(dominance, met["total_bill_usd"] - base["total_bill_usd"])
                    for k, v in physics(result, load, cap).items(): maxima[k] = max(maxima.get(k, 0.), v)
                    rows.append(dict(building_id=b["id"], market=market, connection_limit=limit, demand_rate_usd_per_kw_month=rate,
                                     capacity_kwh=E, power_kw=P, n_hours=len(load), n_billing_blocks=len(np.unique(month)),
                                     connection_cap_kw=cap, nob_energy_bill_usd=base["energy_bill_usd"], nob_demand_bill_usd=base["demand_bill_usd"],
                                     nob_total_bill_usd=base["total_bill_usd"], nob_annual_peak_kw=annual_cap, **met,
                                     energy_value_usd=ev, demand_value_usd=dv, total_value_usd=value,
                                     npv_usd=tea.npv(value, met["cycles"], E)["npv"]))
        print("demand scenarios building", b["id"], "complete", flush=True)
    target = ROOT / "results/demand_charge_results.csv"
    with target.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    groups = []
    for market in core.MARKETS:
        for limit in ("unconstrained", "nob_annual_peak"):
            for rate in RATES:
                sub = [r for r in rows if (r["market"], r["connection_limit"], r["demand_rate_usd_per_kw_month"]) == (market, limit, rate)]
                rec = dict(market=market, connection_limit=limit, demand_rate_usd_per_kw_month=rate, n_buildings=len(sub))
                for key in ("energy_value_usd", "demand_value_usd", "total_value_usd", "annual_peak_kw", "cycles", "npv_usd"):
                    rec["mean_" + key] = float(np.mean([r[key] for r in sub])); rec["median_" + key] = float(np.median([r[key] for r in sub]))
                rec["npv_positive_count"] = sum(r["npv_usd"] > 0 for r in sub); rec["npv_positive_share_pct"] = 100. * rec["npv_positive_count"] / len(sub)
                groups.append(rec)
    month = data["buildings"][0]["month"]; labels, counts = np.unique(month, return_counts=True)
    design = dict(post_hoc=True, n_hours=8760, time_order="original continuous row order; no bootstrap", n_buildings=17,
                  billing_blocks="12 original dataset month labels; August=745 h and July=743 h retained without calendar reconstruction",
                  hours_by_month=dict(zip(labels.tolist(), counts.tolist())), demand_rates_usd_per_kw_month=RATES,
                  tariff_scope="assumed parameter scenarios, not verified local retail tariffs; hourly maximum import is a proxy, not a 15-minute billing peak",
                  connection_scope="individual-building import limit only; no coincident network peak, feeder topology or power-flow model",
                  capacity_kwh=E, power_kw=P, npv_model="tea.npv; repeated benchmark year; fixed-capacity scenario, no capacity optimum inferred",
                  aggregation="each scenario uses 17 buildings; descriptive mean and median, no bootstrap interval")
    summary = dict(design=design, scenarios=groups)
    (ROOT / "results/demand_charge_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    objectives = objective_checks(rows)
    passed = max(maxima.values()) < 1e-6 and dominance < .02 and accounting_error < 1e-8 and len(rows) == 272 and max(objectives.values()) < .02
    validation["demand_charge"] = dict(n_lps=len(rows), physical_maxima=maxima, objective_checks=objectives, no_battery_dominance_violation_usd=dominance,
                                       bill_decomposition_max_error_usd=accounting_error, passed=passed, runtime_s=time.time() - start)
    (ROOT / "results/demand_charge_validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    if not passed: raise AssertionError(validation["demand_charge"])
    lines = ["# Individual-demand summary. Monthly-demand-charge scenarios at fixed 6.4 kWh/5 kW", "", "Original 8,760 rows in continuous order, grouped by the 12 supplied month labels. Rates are assumed USD/kW/month scenarios. Values are descriptive across 17 buildings. The import limit is the individual building's no-battery annual hourly peak. NPV reuses the existing 15-year cost model.", "", "| Market | Import limit | Rate | Median energy value (USD/y) | Median demand value (USD/y) | Median total value (USD/y) | Mean total value (USD/y) | Median NPV (USD) | Positive NPV |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in groups:
        lines.append(f"| {r['market']} | {r['connection_limit']} | {r['demand_rate_usd_per_kw_month']:g} | {r['median_energy_value_usd']:.2f} | {r['median_demand_value_usd']:.2f} | {r['median_total_value_usd']:.2f} | {r['mean_total_value_usd']:.2f} | {r['median_npv_usd']:.2f} | {r['npv_positive_count']}/17 ({r['npv_positive_share_pct']:.1f}%) |")
    lines += ["", "Energy and demand values add exactly within each building; their separate cross-building medians need not add to the median total. These hourly import maxima do not reproduce subhourly billing peaks or validate a distribution network. The fixed battery capacity does not identify an economic capacity optimum."]
    (ROOT / "outputs/tables/table7_demand_charge.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
