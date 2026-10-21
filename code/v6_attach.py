from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import core

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def main() -> int:
    res = json.loads((RES / "results.json").read_text(encoding="utf-8"))
    v6 = json.loads((RES / "revision_v6_checks.json").read_text(encoding="utf-8"))
    cp = json.loads((RES / "coincident_peak_summary.json").read_text(encoding="utf-8"))
    m2 = v6["m2_d10_settlement"]
    ind = v6["individual_billing_d10"]
    m2h = cp["headline"]["M2_zero_export/ORACLE_DC/d10"]
    m1h = cp["headline"]["M1_net_metering/ORACLE_DC/d10"]

    # PRICE annual district peak increase (original 8760 rows)
    data = core.load_main()
    nob = np.vstack([b["load"] - b["pv_kw"] for b in data["buildings"]])
    price = data["price"]
    carbon = data["carbon"]
    price_net = np.vstack([
        core.simulate_rule("PRICE", b["load"], b["pv_kw"], price, carbon, 6.4, 5.0).net
        for b in data["buildings"]])
    nob_peak = float(np.max(np.maximum(nob.sum(axis=0), 0.0)))
    price_peak = float(np.max(np.maximum(price_net.sum(axis=0), 0.0)))
    price_annual_increase = 100.0 * (price_peak / nob_peak - 1.0)

    updates = {
        "tea_break_even_capex_M2_oracle_usd_per_kwh": v6["break_even_effective_life"]["M2_zero_export"]["break_even_usd_per_kwh_mean_of_seed_medians"],
    }
    new = {
        "v6_break_even_M1_oracle_usd_per_kwh": v6["break_even_effective_life"]["M1_net_metering"]["break_even_usd_per_kwh_mean_of_seed_medians"],
        "v6_break_even_M2_oracle_usd_per_kwh": v6["break_even_effective_life"]["M2_zero_export"]["break_even_usd_per_kwh_mean_of_seed_medians"],
        "v6_short_life_cases_M1": float(v6["break_even_effective_life"]["M1_net_metering"]["n_shorter_than_15"]),
        "v6_short_life_cases_M2": float(v6["break_even_effective_life"]["M2_zero_export"]["n_shorter_than_15"]),
        "v6_nob_energy_gross_usd": float(v6["no_battery"]["energy_gross_usd"]),
        "v6_nob_energy_pooled_usd": float(v6["no_battery"]["energy_pooled_usd"]),
        "v6_energy_boundary_gap_usd": float(v6["no_battery"]["difference_usd"]),
        "v6_mixed_hours": float(v6["no_battery"]["hours_with_simultaneous_import_and_export"]),
        "v6_coincidence_factor_nob": float(v6["no_battery"]["coincidence_factor_annual"]),
        "v6_coincidence_factor_m2_d10": float(m2h["diversity_factor"]),
        "v6_monthly_peak_sum_reduction_m2_pct": float(m2h["district_peak_reduction_pct"]),
        "v6_annual_peak_reduction_m2_pct": float(v6["controllers"]["M2_zero_export"]["annual_peak_reduction_pct"]),
        "v6_monthly_peak_sum_increase_price_pct": float(-m2h["district_peak_reduction_pct"]) if False else float(cp["headline"]["M2_zero_export/PRICE/d10"]["district_peak_reduction_pct"] * -1.0),
        "v6_annual_peak_increase_price_pct": float(price_annual_increase),
        "v6_hybrid_m2_d10_energy_usd": float(m2["hybrid"]["energy_value_usd"]),
        "v6_hybrid_m2_d10_demand_usd": float(m2["hybrid"]["demand_value_usd"]),
        "v6_hybrid_m2_d10_total_usd": float(m2["hybrid"]["total_usd"]),
        "v6_hybrid_m2_d10_npv_usd": float(m2["hybrid"]["npv_common_life_usd"]),
        "v6_pooled_m2_d10_energy_usd": float(m2["pooled"]["energy_value_usd"]),
        "v6_pooled_m2_d10_demand_usd": float(m2["pooled"]["demand_value_usd"]),
        "v6_pooled_m2_d10_total_usd": float(m2["pooled"]["total_usd"]),
        "v6_pooled_m2_d10_npv_usd": float(m2["pooled"]["npv_common_life_usd"]),
        "v6_hybrid_m1_d10_total_usd": float(m1h["district_total_value"]),
        "v6_hybrid_m1_d10_npv_usd": float(m1h["district_npv"]),
        "v6_individual_m2_d10_total_usd": float(m2h["individual_total_value"]),
        "v6_individual_m2_d10_npv_sum_usd": float(ind["M2_zero_export"]["sum_npv_per_asset_usd"]),
        "v6_individual_m1_d10_npv_sum_usd": float(ind["M1_net_metering"]["sum_npv_per_asset_usd"]),
        "v6_individual_m2_d10_npv_min_usd": float(ind["M2_zero_export"]["npv_per_asset_min_usd"]),
        "v6_individual_m2_d10_npv_max_usd": float(ind["M2_zero_export"]["npv_per_asset_max_usd"]),
    }
    changed = 0
    for item in res["secondary_metrics"]:
        if item["name"] in updates:
            item["value"] = float(updates[item["name"]])
            item["seed_values"] = [float(updates[item["name"]])]
            changed += 1
    existing = {s["name"] for s in res["secondary_metrics"]}
    for name, value in new.items():
        if name in existing:
            continue
        res["secondary_metrics"].append(dict(name=name, value=float(value),
                                             seed_values=[float(value)], n_seeds=1,
                                             description=name.replace("_", " ") +
                                             " (v6 revision; post-hoc)"))
    existing_ids = {a["id"] for a in res["artifacts"]}
    for i, item in enumerate(res["secondary_metrics"]):
        art_id = f"A-SEC-{i + 1:02d}"
        if art_id in existing_ids:
            continue
        res["artifacts"].append(dict(id=art_id, path="results/results.json",
                                     pointer=f"/secondary_metrics/{i}/value",
                                     metric=item["name"], value=round(float(item["value"]), 10),
                                     tolerance=1e-6))
    res["v6_revision"] = dict(
        note="Settlement-boundary and effective-life checks based on the fixed dispatch.",
        checks="results/revision_v6_checks.json")
    (RES / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print(json.dumps({"updated": changed, "new": len(new), "price_annual_increase_pct": price_annual_increase},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
