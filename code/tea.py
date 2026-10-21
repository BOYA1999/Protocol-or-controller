from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TABLES = ROOT / "outputs" / "tables"
C_E = 350.0
C_P = 80.0
OM_RATE = 0.015
RATE = 0.06
HORIZON = 15.0
CYCLE_LIFE = 6000.0
CAPS = [3.2, 6.4, 12.8, 25.6]
SEEDS = [42, 43, 44, 45, 46]
MARKETS = ("M1_net_metering", "M2_zero_export")


def annuity(rate: float, years: float) -> float:
    return (1.0 - (1.0 + rate) ** (-years)) / rate


def npv(value: float, efc: float, energy: float, capex_energy=C_E, capex_power=C_P) -> dict:
    power = 5.0 * energy / 6.4
    capex = capex_energy * energy + capex_power * power
    life = min(HORIZON, CYCLE_LIFE / max(efc, 1e-9))
    af = annuity(RATE, life)
    om = OM_RATE * capex
    net = (value - om) * af - capex
    return dict(capex=capex, om=om, life=life, annuity=af, npv=net,
                annual_value=value, efc=efc)


def main() -> int:
    raw = list(csv.DictReader((RESULTS / "raw_metrics.csv").open(encoding="utf-8")))
    idx = {(int(r["building_id"]), int(r["seed"]), r["market"], float(r["capacity_kwh"]),
            r["controller"]): r for r in raw}
    buildings = sorted({int(r["building_id"]) for r in raw})

    def value(b, s, market, ctrl, cap):
        return (float(idx[(b, s, market, 0.0, "NOB")]["bill"]) -
                float(idx[(b, s, market, cap, ctrl)]["bill"]))

    rows = []
    for b in buildings:
        for s in SEEDS:
            for market in MARKETS:
                for ctrl in ("GREEDY", "PRICE", "ORACLE_COST"):
                    for cap in CAPS:
                        efc = float(idx[(b, s, market, cap, ctrl)]["cycles"])
                        v = value(b, s, market, ctrl, cap)
                        rec = npv(v, efc, cap)
                        rows.append(dict(building=b, seed=s, market=market, controller=ctrl,
                                         capacity_kwh=cap, value_usd=v, cycles=efc,
                                         npv_usd=rec["npv"], capex_usd=rec["capex"],
                                         life_y=rec["life"]))
    with (RESULTS / "tea_results.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def med(vals):
        return float(np.median(vals))

    def summarize(capex_energy):
        out = {}
        # NPV at the primary capacity, per market and controller
        for market in MARKETS:
            for ctrl in ("GREEDY", "PRICE", "ORACLE_COST"):
                seed_npv, seed_share, be_capex, opt_caps, short15 = [], [], [], [], []
                for s in SEEDS:
                    npvs, bes, opts = [], [], []
                    for b in buildings:
                        v = value(b, s, market, ctrl, 6.4)
                        efc = float(idx[(b, s, market, 6.4, ctrl)]["cycles"])
                        rec = npv(v, efc, 6.4, capex_energy)
                        npvs.append(rec["npv"])
                        # break-even installed capex at 6.4 kWh using the case's effective life
                        life_be = min(HORIZON, CYCLE_LIFE / max(efc, 1e-9))
                        af = annuity(RATE, life_be)
                        capex_be = v * af / (1.0 + OM_RATE * af)
                        bes.append(capex_be / 6.4)
                        best_cap, best_npv = 0.0, 0.0
                        for cap in CAPS:
                            r2 = npv(value(b, s, market, ctrl, cap),
                                     float(idx[(b, s, market, cap, ctrl)]["cycles"]),
                                     cap, capex_energy)
                            if r2["npv"] > best_npv:
                                best_cap, best_npv = cap, r2["npv"]
                        opts.append(best_cap)
                    seed_npv.append(med(npvs))
                    seed_share.append(float(np.mean([x > 0 for x in npvs])))
                    be_capex.append(med(bes))
                    short15.append(int(sum(1 for L in [min(HORIZON, CYCLE_LIFE / max(float(idx[(b, s, market, 6.4, ctrl)]["cycles"]), 1e-9)) for b in buildings] if L < HORIZON - 1e-9)))
                    opt_caps.append(opts)
                out[f"{market}/{ctrl}"] = dict(
                    median_npv_usd=float(np.mean(seed_npv)),
                    median_npv_per_seed=[round(x, 3) for x in seed_npv],
                    share_positive_pct=100.0 * float(np.mean(seed_share)),
                    break_even_capex_usd_per_kwh=float(np.mean(be_capex)),
                    optimal_capacity_by_seed=opt_caps,
                    n_short_life_cases=int(np.sum(short15)),
                    optimal_capacity_median_kwh=float(np.median(
                        [c for caps in opt_caps for c in caps])),
                    interior_optimum_share_pct=100.0 * float(np.mean(
                        [0.0 < c < max(CAPS) for caps in opt_caps for c in caps])))
        return out

    base = summarize(C_E)
    low = summarize(250.0)
    high = summarize(500.0)

    # cost of the 6.4 kWh battery per kWh-year (levelised) at base capex
    power = 5.0 * 6.4 / 6.4
    capex = C_E * 6.4 + C_P * power
    cost_per_kwh_year = (capex * (1.0 / annuity(RATE, HORIZON) + OM_RATE)) / 6.4

    summary = dict(
        description=("Post-hoc techno-economic extension: NPV with battery capex, O&M and "
                     "degradation, and optimal capacity on the frozen grid."),
        assumptions=dict(capex_energy_usd_per_kwh=C_E, capex_power_usd_per_kw=C_P,
                         om_rate=OM_RATE, discount_rate=RATE, horizon_y=HORIZON,
                         cycle_life_efc=CYCLE_LIFE,
                         capex_sensitivity_usd_per_kwh=[250.0, 500.0],
                         cost_6p4_usd_per_kwh_year=cost_per_kwh_year),
        base_case=base,
        capex_250=low,
        capex_500=high,
    )
    (RESULTS / "tea_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                              encoding="utf-8")

    # battery-cost assumptions
    lines = ["# Battery-cost summary. Techno-economic extension: NPV and optimal capacity",
             "",
             f"Assumptions: capex {C_E:.0f} USD/kWh + {C_P:.0f} USD/kW, O&M {OM_RATE*100:.1f}% of capex "
             f"per year, discount rate {RATE*100:.0f}%, horizon {HORIZON:.0f} years, cycle life "
             f"{CYCLE_LIFE:.0f} equivalent full cycles, calendar life 15 years. Values are annual "
             f"operating values from the frozen 5-replicate simulation. Levelised cost of the 6.4 kWh "
             f"battery: {cost_per_kwh_year:.1f} USD kWh⁻¹ yr⁻¹.",
             "",
             "| Design / controller | Median NPV at 6.4 kWh (USD) | Buildings with NPV > 0 (%) | "
             "Break-even capex (USD kWh⁻¹) | Median optimal capacity (kWh) | Interior optimum (%) |",
             "|---|---:|---:|---:|---:|---:|"]
    for key, item in base.items():
        lines.append(f"| {key} | {item['median_npv_usd']:.0f} | {item['share_positive_pct']:.0f} | "
                     f"{item['break_even_capex_usd_per_kwh']:.0f} | "
                     f"{item['optimal_capacity_median_kwh']:.1f} | "
                     f"{item['interior_optimum_share_pct']:.0f} |")
    lines += ["", "Capex sensitivity (median NPV at 6.4 kWh, USD): "
              + "; ".join(f"{k}: {base[k]['median_npv_usd']:.0f} @350 / "
                          f"{low[k]['median_npv_usd']:.0f} @250 / "
                          f"{high[k]['median_npv_usd']:.0f} @500" for k in base)]
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "table6_tea.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"cost_6p4": round(cost_per_kwh_year, 2),
                      "base": {k: dict(npv=round(v["median_npv_usd"], 1),
                                       pos=round(v["share_positive_pct"], 1),
                                       be=round(v["break_even_capex_usd_per_kwh"], 1),
                                       opt=v["optimal_capacity_median_kwh"],
                                       interior=round(v["interior_optimum_share_pct"], 1))
                               for k, v in base.items()}}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
