from __future__ import annotations

import csv
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np

import core

ROOT = core.ROOT
RESULTS = ROOT / "results"
TABLES = ROOT / "outputs" / "tables"
FIGS = ROOT / "outputs" / "figures"
CAP, POWER = core.DEFAULT_CAP_KWH, core.DEFAULT_POWER_KW
RATES = [0.0, 5.0, 10.0, 20.0]
N_PARTITIONS = 200
TEA = dict(c_e=350.0, c_p=80.0, om=0.015, rate=0.06, horizon=15.0, cycle_life=6000.0)
_DATA = None


def _init():
    global _DATA
    _DATA = core.load_main()


def annuity(rate: float, years: float) -> float:
    return (1.0 - (1.0 + rate) ** (-years)) / rate


def district_npv(value: float, efc: float, n: int) -> float:
    capex = (TEA["c_e"] * CAP + TEA["c_p"] * POWER) * n
    life = min(TEA["horizon"], TEA["cycle_life"] / max(efc, 1e-9))
    af = annuity(TEA["rate"], life)
    return (value - TEA["om"] * capex) * af - capex


def _task(building):
    if _DATA is None:
        _init()
    load = building["load"]
    pv = building["pv_kw"]
    price = _DATA["price"]
    carbon = _DATA["carbon"]
    month = building["month"].astype(int)
    out = {"month": month}
    out["NOB"] = core.simulate_rule("NOB", load, pv, price, carbon, 0.0, 0.0).net
    for ctrl in ("GREEDY", "PRICE"):
        res = core.simulate_rule(ctrl, load, pv, price, carbon, CAP, POWER)
        out[ctrl] = res.net
        out[f"{ctrl}_cycles"] = float(np.sum(res.discharge) / CAP)
    for market in core.MARKETS:
        for rate in RATES:
            res = core.solve_oracle_demand(load, pv, price, carbon, month, CAP, POWER,
                                           market, rate, conn_cap=None)
            out[f"ORACLE_DC|{market}|{rate}"] = res.net
            out[f"ORACLE_DC_cycles|{market}|{rate}"] = float(np.sum(res.discharge) / CAP)
    return building["id"], out


def monthly_peaks(series: np.ndarray, month: np.ndarray) -> dict:
    return {int(m): float(np.max(np.maximum(series[month == m], 0.0))) for m in np.unique(month)}


def sum_peaks(mat: np.ndarray, month: np.ndarray) -> float:
    return float(sum(sum(monthly_peaks(mat[i], month).values()) for i in range(mat.shape[0])))


def main() -> int:
    _init()
    data = core.load_main()
    with mp.Pool(processes=max(1, (mp.cpu_count() or 4) - 2), initializer=_init) as pool:
        chunks = pool.map(_task, data["buildings"], chunksize=1)
    store = {b_id: out for b_id, out in chunks}
    buildings = sorted(store)

    def mat_for(ctrl, market, rate):
        if ctrl == "ORACLE_DC":
            return np.vstack([store[b][f"ORACLE_DC|{market}|{rate}"] for b in buildings])
        return np.vstack([store[b][ctrl] for b in buildings])

    month = store[buildings[0]]["month"]
    price = _DATA["price"]
    rng = np.random.default_rng(20261020)
    parts = {k: [np.array_split(rng.permutation(buildings), k) for _ in range(N_PARTITIONS)]
             for k in (2, 4)}
    idx = {b: i for i, b in enumerate(buildings)}
    nob_mat = mat_for("NOB", "M1_net_metering", 0.0)
    nob_district_peaks = monthly_peaks(nob_mat.sum(axis=0), month)
    nob_ind_peak_sum = sum_peaks(nob_mat, month)
    nob_energy = {m: float(np.sum(price * (nob_mat if m == "M1_net_metering"
                                           else np.maximum(nob_mat, 0.0)))) for m in core.MARKETS}
    nob_annual_peak = float(np.max(np.maximum(nob_mat.sum(axis=0), 0.0)))
    nob_diversity = nob_annual_peak / max(float(np.sum([np.max(np.maximum(nob_mat[i], 0.0))
                                                        for i in range(len(buildings))])), 1e-9)

    rows = []
    for market in core.MARKETS:
        for rate in RATES:
            nob_ind_bill = nob_energy[market] + rate * nob_ind_peak_sum
            nob_dist_bill = nob_energy[market] + rate * float(sum(nob_district_peaks.values()))
            del nob_ind_bill, nob_dist_bill
            for ctrl in ("GREEDY", "PRICE", "ORACLE_DC"):
                mat = mat_for(ctrl, market, rate)
                efc = float(np.mean([store[b][f"{ctrl}_cycles"] if ctrl in ("GREEDY", "PRICE")
                                     else store[b][f"ORACLE_DC_cycles|{market}|{rate}"]
                                     for b in buildings]))
                energy = float(np.sum(price * (mat if market == "M1_net_metering"
                                               else np.maximum(mat, 0.0))))
                ind_peak_sum = sum_peaks(mat, month)
                dist_peaks = monthly_peaks(mat.sum(axis=0), month)
                energy_val = nob_energy[market] - energy
                dem_ind = rate * (nob_ind_peak_sum - ind_peak_sum)
                dem_dist = rate * (sum(nob_district_peaks.values()) - sum(dist_peaks.values()))
                part_vals = {}
                for k in (2, 4):
                    vals = []
                    for groups in parts[k]:
                        nob_p = sum(sum(monthly_peaks(nob_mat[[idx[x] for x in g]].sum(axis=0),
                                                      month).values()) for g in groups)
                        ctl_p = sum(sum(monthly_peaks(mat[[idx[x] for x in g]].sum(axis=0),
                                                      month).values()) for g in groups)
                        vals.append(rate * (nob_p - ctl_p))
                    part_vals[f"part{k}"] = [float(np.mean(vals)), float(np.percentile(vals, 25)),
                                              float(np.percentile(vals, 75))]
                div = float(np.max(np.maximum(mat.sum(axis=0), 0.0)) /
                            max(float(np.sum([np.max(np.maximum(mat[i], 0.0))
                                              for i in range(len(buildings))])), 1e-9))
                dist_peak_red = float(1.0 - sum(dist_peaks.values()) /
                                      max(sum(nob_district_peaks.values()), 1e-9))
                ind_peak_red = float(1.0 - ind_peak_sum / max(nob_ind_peak_sum, 1e-9))
                for group, dem in (("individual", dem_ind), ("single_district", dem_dist),
                                   ("part2_mean", part_vals["part2"][0]),
                                   ("part4_mean", part_vals["part4"][0]),
                                   ("part2_q25", part_vals["part2"][1]),
                                   ("part4_q25", part_vals["part4"][1]),
                                   ("part2_q75", part_vals["part2"][2]),
                                   ("part4_q75", part_vals["part4"][2])):
                    total = energy_val + dem
                    npv = district_npv(total, efc, len(buildings))
                    rows.append(dict(building_group=group, market=market, demand_rate=rate,
                                     controller=ctrl, energy_value=energy_val,
                                     demand_value=dem, total_value=total,
                                     district_peak_reduction_pct=100.0 * dist_peak_red,
                                     individual_peak_reduction_pct=100.0 * ind_peak_red,
                                     diversity_factor=div, npv=npv,
                                     npv_per_building=npv / len(buildings)))
    with (RESULTS / "coincident_peak_results.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def sel(**kw):
        out = rows
        for k, v in kw.items():
            out = [r for r in out if r[k] == v]
        return out

    summary = dict(
        design=dict(basis="original continuous 8,760 rows (no bootstrap), matching individual-demand scenarios",
                    n_buildings=len(buildings), demand_rates=RATES,
                    memberships=["individual", "single_district", "part2", "part4"],
                    n_random_partitions=N_PARTITIONS,
                    controllers=["GREEDY", "PRICE", "ORACLE_DC"],
                    no_battery_diversity_factor=nob_diversity,
                    no_battery_annual_district_peak_kw=nob_annual_peak,
                    no_battery_sum_individual_peak_kw=float(np.sum([np.max(np.maximum(nob_mat[i], 0.0))
                                                                    for i in range(len(buildings))])),
                    tea=TEA,
                    caveats=["no distribution topology: feeder membership is a scenario",
                             "hourly maxima are not 15-minute billing peaks (no 15-min data)",
                             "no feeder power flow/voltage/transformer constraints",
                             "no coordinated district controller is claimed",
                             "demand-aware dispatch is the per-building LP of individual-demand scenarios"]),
        diversity={}, headline={}, partitions={})
    for market in core.MARKETS:
        for ctrl in ("GREEDY", "PRICE", "ORACLE_DC"):
            for rate in RATES:
                dsub = sel(market=market, controller=ctrl, demand_rate=rate,
                           building_group="single_district")
                isub = sel(market=market, controller=ctrl, demand_rate=rate,
                           building_group="individual")
                summary["headline"][f"{market}/{ctrl}/d{rate:g}"] = dict(
                    district_demand_value=float(np.mean([r["demand_value"] for r in dsub])),
                    individual_demand_value=float(np.mean([r["demand_value"] for r in isub])),
                    district_total_value=float(np.mean([r["total_value"] for r in dsub])),
                    individual_total_value=float(np.mean([r["total_value"] for r in isub])),
                    district_npv=float(np.mean([r["npv"] for r in dsub])),
                    district_npv_per_building=float(np.mean([r["npv_per_building"] for r in dsub])),
                    district_peak_reduction_pct=float(np.mean([r["district_peak_reduction_pct"]
                                                               for r in dsub])),
                    individual_peak_reduction_pct=float(np.mean([r["individual_peak_reduction_pct"]
                                                                 for r in isub])),
                    diversity_factor=float(np.mean([r["diversity_factor"] for r in dsub])))
        for ctrl in ("GREEDY", "PRICE", "ORACLE_DC"):
            for grp in ("part2_mean", "part4_mean", "part2_q25", "part4_q25",
                        "part2_q75", "part4_q75"):
                vals = [r["total_value"] for r in sel(market=market, controller=ctrl,
                                                      demand_rate=10.0, building_group=grp)]
                summary["partitions"][f"{market}/{ctrl}/{grp}"] = float(np.mean(vals))
    (RESULTS / "coincident_peak_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["# Settlement summary. Coincident-peak (district) demand charge and feeder-membership scenarios",
             "",
             "Original continuous 8,760 rows. Monthly maximum of the aggregate district import "
             "('single district') vs the sum of individual building maxima ('individual', the individual-demand scenarios "
             "case). part2/part4 are random feeder partitions (mean over 200 draws). Values are "
             "district totals; NPV repeats the benchmark year under the battery-cost assumptions cost model.",
             "",
             "| Market | Controller | d (USD/kW/mo) | District demand value (USD/y) | Individual demand value (USD/y) | "
             "District total value (USD/y) | District NPV (USD) | District peak reduction (%) | Individual peak reduction (%) | Diversity |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for market in core.MARKETS:
        for ctrl in ("GREEDY", "PRICE", "ORACLE_DC"):
            for rate in RATES:
                h = summary["headline"][f"{market}/{ctrl}/d{rate:g}"]
                lines.append(f"| {market} | {ctrl} | {rate:g} | {h['district_demand_value']:.0f} | "
                             f"{h['individual_demand_value']:.0f} | {h['district_total_value']:.0f} | "
                             f"{h['district_npv']:.0f} | {h['district_peak_reduction_pct']:.1f} | "
                             f"{h['individual_peak_reduction_pct']:.1f} | {h['diversity_factor']:.3f} |")
    lines += ["", f"No-battery diversity factor: {nob_diversity:.3f} "
                  f"(district annual peak {nob_annual_peak:.1f} kW vs sum of individual peaks "
                  f"{summary['design']['no_battery_sum_individual_peak_kw']:.1f} kW).",
              "", "Feeder-membership sensitivity at d = 10 USD/kW/month "
                  "(district total annual value, USD):"]
    for key, val in summary["partitions"].items():
        lines.append(f"- {key}: {val:.0f}")
    (TABLES / "table9_coincident_peak.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "figure.dpi": 300, "savefig.dpi": 300})
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    colors = {"M1_net_metering": "#2b6cb0", "M2_zero_export": "#c05621"}
    for market in core.MARKETS:
        for ctrl, style in (("GREEDY", "o-"), ("PRICE", "s-"), ("ORACLE_DC", "^-")):
            axes[0].plot(RATES, [summary["headline"][f"{market}/{ctrl}/d{r:g}"]["district_total_value"]
                                 for r in RATES], style, ms=3, lw=1.0, color=colors[market],
                         label=f"{market.split('_')[0]} {ctrl}")
            axes[1].plot(RATES, [summary["headline"][f"{market}/{ctrl}/d{r:g}"]["individual_total_value"]
                                 for r in RATES], style, ms=3, lw=1.0, color=colors[market])
    axes[0].set_title("Single district (coincident peak)", fontsize=8)
    axes[1].set_title("Individual connections", fontsize=8)
    for ax in axes:
        ax.set_xlabel("Demand charge (USD kW$^{-1}$ month$^{-1}$)", fontsize=7)
    axes[0].set_ylabel("Total annual value, 17 buildings (USD)", fontsize=7)
    axes[0].legend(fontsize=5, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGS / "fig6_coincident_peak.png", bbox_inches="tight")
    plt.close(fig)

    print(json.dumps({"nob_diversity": nob_diversity,
                      "headline_d10": {k: {kk: round(vv, 2) for kk, vv in v.items()}
                                       for k, v in summary["headline"].items() if k.endswith("d10")},
                      "partitions": {k: round(v, 1) for k, v in summary["partitions"].items()}},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
