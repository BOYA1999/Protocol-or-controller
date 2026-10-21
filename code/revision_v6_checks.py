from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "revision_v6_checks.json"
NPZ = ROOT / "data" / "derived" / "coincident_d10_dispatch.npz"
C_E, C_P, OM, RATE, HORIZON, CYCLES = 350.0, 80.0, 0.015, 0.06, 15.0, 6000.0
CAP, POWER = 6.4, 5.0


def annuity(rate, y):
    return (1.0 - (1.0 + rate) ** (-y)) / rate


def npv(value, efc, n=1):
    capex = (C_E * CAP + C_P * POWER) * n
    life = min(HORIZON, CYCLES / max(efc, 1e-9))
    af = annuity(RATE, life)
    return (value - OM * capex) * af - capex, life


def peaks(mat, month):
    return {int(m): float(np.max(np.maximum(mat[:, month == m].sum(axis=0), 0.0)))
            for m in np.unique(month)}


def main() -> int:
    provenance = json.loads(NPZ.with_name(NPZ.stem + "_manifest.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(NPZ.read_bytes()).hexdigest()
    if digest != provenance["npz_sha256"]:
        raise ValueError("derived dispatch hash differs from its manifest")
    z = np.load(NPZ)
    nob = z["nob"]
    month = z["month"]
    price = z["price"]
    out = {"arrays": {k: z[k].shape for k in z.files}, "input_npz_sha256": digest}

    # ---- A. settlement boundaries, M2 d=10 ----
    nob_gross_energy = float(np.sum(price * np.maximum(nob, 0.0)))       # per building
    nob_pooled_energy = float(np.sum(price * np.maximum(nob.sum(axis=0), 0.0)))  # single meter
    mixed_hours = int(np.any((nob > 0) & (nob.sum(axis=0)[None, :] > 0), axis=0).sum())
    # hours where some building imports while another exports
    hours_mixed = int(np.sum(((nob > 0).sum(axis=0) > 0) & ((nob < 0).sum(axis=0) > 0)))
    no_battery = dict(
        energy_gross_usd=nob_gross_energy,
        energy_pooled_usd=nob_pooled_energy,
        difference_usd=nob_gross_energy - nob_pooled_energy,
        hours_with_simultaneous_import_and_export=hours_mixed,
    )
    # no-battery demand peaks
    nob_district = peaks(nob, month)
    nob_individual_sum = float(sum(
        np.max(np.maximum(nob[i], 0.0)) for i in range(nob.shape[0])))
    out["no_battery"] = dict(
        **no_battery,
        coincident_annual_peak_kw=float(np.max(np.maximum(nob.sum(axis=0), 0.0))),
        sum_individual_annual_peaks_kw=nob_individual_sum,
        coincidence_factor_annual=float(np.max(np.maximum(nob.sum(axis=0), 0.0)) /
                                        nob_individual_sum),
        coincidence_factor_monthly_peak_sum=float(
            sum(nob_district.values()) /
            max(sum(sum(np.max(np.maximum(nob[i][month == m], 0.0))
                        for i in range(nob.shape[0])) for m in np.unique(month)), 1e-9)),
        monthly_peak_sum_kw=float(sum(nob_district.values())),
    )

    controllers = {}
    for market in ("M1_net_metering", "M2_zero_export"):
        mat = z[market]
        gross_energy = float(np.sum(price * np.maximum(mat, 0.0)))
        pooled_energy = float(np.sum(price * np.maximum(mat.sum(axis=0), 0.0)))
        dist_peaks = peaks(mat, month)
        ind_peak_sum = float(sum(np.max(np.maximum(mat[i], 0.0)) for i in range(mat.shape[0])))
        annual_peak = float(np.max(np.maximum(mat.sum(axis=0), 0.0)))
        # per-building cycles are not stored in the npz; use demand_charge_results.csv below
        controllers[market] = dict(
            energy_gross_usd=gross_energy,
            energy_pooled_usd=pooled_energy,
            district_peak_sum_kw=float(sum(dist_peaks.values())),
            annual_district_peak_kw=annual_peak,
            individual_annual_peak_sum_kw=ind_peak_sum,
            coincidence_factor_annual=float(annual_peak / ind_peak_sum),
            annual_peak_reduction_pct=100.0 * (1.0 - annual_peak /
                                               (float(np.max(np.maximum(nob.sum(axis=0), 0.0))) + 1e-9)),
            monthly_peak_sum_reduction_pct=100.0 * (1.0 - sum(dist_peaks.values()) /
                                                    (sum(nob_district.values()) + 1e-9)),
        )
    out["controllers"] = controllers

    # ---- per-building demand-charge records: individual NPV and cycles ----
    rows = list(csv.DictReader((ROOT / "results" / "demand_charge_results.csv").open(encoding="utf-8")))
    # fixed capacity 6.4, no import cap, original rows; ORACLE_DC per-building demand-aware
    per_b = [r for r in rows if float(r["capacity_kwh"]) == 6.4
             and r["connection_limit"] == "unconstrained"]
    indiv = {}
    for market in ("M1_net_metering", "M2_zero_export"):
        sub = [r for r in per_b if r["market"] == market and float(r["demand_rate_usd_per_kw_month"]) == 10.0]
        values = [float(r["total_value_usd"]) for r in sub]
        efcs = [float(r["cycles"]) for r in sub]
        npv_sum, lives = 0.0, []
        for v, e in zip(values, efcs):
            n, life = npv(v, e)
            npv_sum += n
            lives.append(life)
        mean_efc = float(np.mean(efcs))
        common_npv = npv(np.sum(values), mean_efc, len(sub))[0]
        common_sum = sum(npv(v, mean_efc)[0] for v in values)
        np.testing.assert_allclose(common_npv, common_sum, rtol=0.0, atol=1e-8)
        indiv[market] = dict(
            n_buildings=len(sub),
            sum_annual_values_usd=float(np.sum(values)),
            sum_npv_per_asset_usd=float(npv_sum),
            npv_per_asset_min_usd=float(min(npv(v, e)[0] for v, e in zip(values, efcs))),
            npv_per_asset_max_usd=float(max(npv(v, e)[0] for v, e in zip(values, efcs))),
            common_life_npv_usd=float(common_npv),
            common_life_cost_identity_abs_error_usd=float(abs(common_npv - common_sum)),
            effective_life_min_y=float(min(lives)),
            effective_life_max_y=float(max(lives)),
            n_shorter_than_15=int(sum(1 for L in lives if L < HORIZON - 1e-9)),
        )
    out["individual_billing_d10"] = indiv

    # ---- hybrid vs pooled M2 d10 with cycles from the demand-charge records ----
    cycles_m2 = float(np.mean([float(r["cycles"]) for r in per_b
                               if r["market"] == "M2_zero_export"
                               and float(r["demand_rate_usd_per_kw_month"]) == 10.0]))
    m2 = controllers["M2_zero_export"]
    demand_val_m2 = 10.0 * (sum(nob_district.values()) - m2["district_peak_sum_kw"])
    nob_energy_hybrid = no_battery["energy_gross_usd"]
    nob_energy_pooled = no_battery["energy_pooled_usd"]
    hybrid = dict(
        energy_value_usd=nob_energy_hybrid - m2["energy_gross_usd"],
        demand_value_usd=demand_val_m2,
    )
    hybrid["total_usd"] = hybrid["energy_value_usd"] + hybrid["demand_value_usd"]
    hybrid["npv_common_life_usd"], hybrid["life_y"] = npv(hybrid["total_usd"], cycles_m2, 17)
    pooled = dict(
        energy_value_usd=nob_energy_pooled - m2["energy_pooled_usd"],
        demand_value_usd=demand_val_m2,
    )
    pooled["total_usd"] = pooled["energy_value_usd"] + pooled["demand_value_usd"]
    pooled["npv_common_life_usd"], pooled["life_y"] = npv(pooled["total_usd"], cycles_m2, 17)
    out["m2_d10_settlement"] = dict(cycles_mean=cycles_m2, hybrid=hybrid, pooled=pooled)

    # ---- break-even capex with effective life (M1/M2 ORACLE_COST, tea data) ----
    tea = json.loads((ROOT / "results" / "tea_summary.json").read_text(encoding="utf-8"))
    be = {}
    for market, key in (("M1_net_metering", "M1_net_metering/ORACLE_COST"),
                        ("M2_zero_export", "M2_zero_export/ORACLE_COST")):
        value = float(tea["base_case"][key]["median_npv_usd"])  # not used directly
        # use per-case mean value and mean cycles from raw metrics: recompute from tea_results
        tr = list(csv.DictReader((ROOT / "results" / "tea_results.csv").open(encoding="utf-8")))
        sub = [r for r in tr if r["market"] == market and r["controller"] == "ORACLE_COST"
               and float(r["capacity_kwh"]) == 6.4]
        vals = np.array([float(r["value_usd"]) for r in sub])
        efcs = np.array([float(r["cycles"]) for r in sub])
        # per-case break-even total capex and per-kWh figure
        bes = []
        for v, e in zip(vals, efcs):
            life = min(HORIZON, CYCLES / max(e, 1e-9))
            af = annuity(RATE, life)
            capex_be = v * af / (1.0 + OM * af)
            bes.append(capex_be / CAP)
        # aggregate as in the original estimator: mean over seeds of the per-seed median
        seeds_col = [int(r["seed"]) for r in sub]
        per_seed = {}
        for v, e, sd in zip(vals, efcs, seeds_col):
            life = min(HORIZON, CYCLES / max(e, 1e-9))
            af = annuity(RATE, life)
            per_seed.setdefault(sd, []).append((v * af / (1.0 + OM * af)) / CAP)
        seed_medians = [float(np.median(v)) for v in per_seed.values()]
        be[market] = dict(
            mean_value_usd=float(vals.mean()),
            mean_cycles=float(efcs.mean()),
            break_even_usd_per_kwh_mean_of_seed_medians=float(np.mean(seed_medians)),
            break_even_usd_per_kwh_median_all_cases=float(np.median(bes)),
            break_even_usd_per_kwh_mean_all_cases=float(np.mean(bes)),
            n_cases=len(bes),
            n_shorter_than_15=int(sum(1 for e in efcs if CYCLES / e < HORIZON)),
        )
    out["break_even_effective_life"] = be

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1)[:6000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
