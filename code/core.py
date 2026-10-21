from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import optimize, sparse

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "citylearn2022"
EXTERNAL = ROOT / "data" / "external"

N_BUILDINGS = 17
HOURS_PER_WEEK = 168
WEEKS = 52
DEFAULT_CAP_KWH = 6.4
DEFAULT_POWER_KW = 5.0
ROUND_TRIP = 0.9
ETA = math.sqrt(ROUND_TRIP)
SEEDS = (42, 43, 44, 45, 46)
CAPACITIES = (0.0, 3.2, 6.4, 12.8, 25.6)
MARKETS = ("M1_net_metering", "M2_zero_export")
CONTROLLERS = ("NOB", "GREEDY", "PRICE", "CARBON", "ORACLE_COST", "ORACLE_CO2")


# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------
def load_main() -> dict:
    """Load the 17-building hourly dataset into numpy arrays."""
    schema = json.loads((RAW / "schema.json").read_text(encoding="utf-8"))
    price = np.loadtxt(RAW / "pricing.csv", delimiter=",", skiprows=1,
                       usecols=(0,))
    carbon = np.loadtxt(RAW / "carbon_intensity.csv", delimiter=",", skiprows=1,
                        usecols=(0,))
    if price.size != 8760 or carbon.size != 8760:
        raise ValueError("unexpected pricing/carbon length")
    buildings = []
    for i in range(1, N_BUILDINGS + 1):
        rec = schema["buildings"][f"Building_{i}"]
        pv_kwp = float(rec["pv"]["attributes"]["nominal_power"])
        bat = rec["electrical_storage"]["attributes"]
        data = np.loadtxt(RAW / f"Building_{i}.csv", delimiter=",", skiprows=1,
                          usecols=(0, 7, 11))  # month, non_shiftable_load, solar_generation
        month = data[:, 0].astype(int)
        load = data[:, 1]
        pv_kw = data[:, 2] / 1000.0 * pv_kwp
        buildings.append(
            dict(id=i, load=load, pv_kw=pv_kw, pv_kwp=pv_kwp, month=month,
                 bat_kwh=float(bat["capacity"]), bat_kw=float(bat["nominal_power"]),
                 bat_eta=float(bat["efficiency"]))
        )
    return dict(price=price, carbon=carbon, buildings=buildings)


def load_external() -> dict:
    """Load the processed external residential dataset."""
    out = {}
    hh = EXTERNAL / "uci_household_hourly.csv"
    ns = EXTERNAL / "nsrdb_fujian_2020_hourly.csv"
    if hh.is_file():
        raw = np.genfromtxt(hh, delimiter=",", names=True, dtype=None,
                            encoding="utf-8")
        out["uci_load_kw"] = np.asarray(raw["load_kW"], dtype=float)
    if ns.is_file():
        raw = np.genfromtxt(ns, delimiter=",", names=True, dtype=None,
                            encoding="utf-8")
        out["nsrdb_ghi"] = np.asarray(raw["ghi_wm2"], dtype=float)
        out["nsrdb_dni"] = np.asarray(raw["dni_wm2"], dtype=float)
        out["nsrdb_dhi"] = np.asarray(raw["dhi_wm2"], dtype=float)
    return out


# ---------------------------------------------------------------------------
# bootstrap years
# ---------------------------------------------------------------------------
def bootstrap_week_order(seed: int) -> np.ndarray:
    """Return 52 week indices drawn with replacement (deterministic per seed)."""
    rng = np.random.default_rng(int(seed))
    return rng.integers(0, WEEKS, size=WEEKS)


def bootstrap_series(series: np.ndarray, week_order: np.ndarray) -> np.ndarray:
    """Concatenate whole weeks according to ``week_order`` (168 h each)."""
    pieces = []
    for w in week_order:
        start = int(w) * HOURS_PER_WEEK
        pieces.append(series[start:start + HOURS_PER_WEEK])
    return np.concatenate(pieces)


# ---------------------------------------------------------------------------
# battery physics
# ---------------------------------------------------------------------------
@dataclass
class BatteryResult:
    charge: np.ndarray
    discharge: np.ndarray
    soc: np.ndarray
    net: np.ndarray


def _apply_dispatch(charge: np.ndarray, discharge: np.ndarray, load: np.ndarray,
                    pv: np.ndarray, energy: float, eta: float,
                    soc0: float) -> BatteryResult:
    soc = np.empty_like(load)
    e = soc0
    for t in range(load.size):
        e += eta * charge[t] - discharge[t] / eta
        e = min(max(e, 0.0), energy)
        soc[t] = e
    net = load - pv + charge - discharge
    return BatteryResult(charge, discharge, soc, net)


def simulate_rule(kind: str, load: np.ndarray, pv: np.ndarray, price: np.ndarray,
                  carbon: np.ndarray, energy: float, power: float,
                  eta: float = ETA) -> BatteryResult:
    """Causal rule-based dispatch on one simulated year.

    kind in {NOB, GREEDY, PRICE, CARBON}.  All signals used at hour t are
    prices/carbon of hour t and the trailing window [t-168, t-1] only, so there
    is no look-ahead.
    """
    n = load.size
    charge = np.zeros(n)
    discharge = np.zeros(n)
    soc0 = 0.5 * energy
    if kind == "NOB" or energy <= 0:
        return BatteryResult(charge, discharge, np.full(n, soc0), load - pv)

    soc = soc0
    soc_hist = np.empty(n)
    window = 168
    for t in range(n):
        surplus = pv[t] - load[t]
        deficit = -surplus
        if kind == "GREEDY":
            if surplus > 0 and soc < energy:
                c = min(surplus, power, (energy - soc) / eta)
                charge[t] = c
                soc += eta * c
            elif deficit > 0 and soc > 0:
                d = min(deficit, power, soc * eta)
                discharge[t] = d
                soc -= d / eta
        else:
            signal = price if kind == "PRICE" else carbon
            lo, hi = signal[t], signal[t]
            if t >= window:
                past = signal[t - window:t]
                lo = float(np.quantile(past, 0.30))
                hi = float(np.quantile(past, 0.70))
            # self-consumption first: charge from PV surplus, never import while
            # the building is already exporting
            if surplus > 0:
                if soc < energy:
                    c = min(surplus, power, (energy - soc) / eta)
                    charge[t] = c
                    soc += eta * c
            elif signal[t] >= hi and soc > 0:
                d = min(deficit, power, soc * eta)
                discharge[t] = d
                soc -= d / eta
            elif signal[t] <= lo and soc < energy:
                c = min(power, (energy - soc) / eta)
                charge[t] = c
                soc += eta * c
        soc = min(max(soc, 0.0), energy)
        soc_hist[t] = soc
    return BatteryResult(charge, discharge, soc_hist, load - pv + charge - discharge)


def simulate_schedule(schedule: dict, load: np.ndarray, pv: np.ndarray,
                      energy: float, power: float,
                      eta: float = ETA) -> BatteryResult:
    """Apply an hour-of-day schedule (fraction of energy capacity per hour).

    Used only to reproduce published reference controllers (G4 baselines); the
    physical constraints (power limits, SOC window, behind-the-meter no export)
    are identical to every other controller in the study.
    """
    n = load.size
    charge = np.zeros(n)
    discharge = np.zeros(n)
    soc = 0.5 * energy
    for t in range(n):
        action = float(schedule[t % 24])
        if action > 0:
            room = max(energy - soc, 0.0) / eta
            c = min(action * energy, power, room)
            charge[t] = c
            soc += eta * c
        elif action < 0:
            d = min(-action * energy, power, max(soc, 0.0) * eta, load[t])
            discharge[t] = d
            soc -= d / eta
        soc = min(max(soc, 0.0), energy)
    return BatteryResult(charge, discharge, np.full(n, soc),
                         load - pv + charge - discharge)


def solve_oracle_demand(load: np.ndarray, pv: np.ndarray, price: np.ndarray,
                        carbon: np.ndarray, month: np.ndarray, energy: float,
                        power: float, market: str, demand_rate: float,
                        conn_cap: float | None = None, eta: float = ETA) -> BatteryResult:
    """Cost oracle with a charge on the maximum import in each supplied billing block."""
    return solve_oracle(load, pv, price, carbon, energy, power, market, "cost", eta,
                        peak_cap=conn_cap, month=month, demand_rate=demand_rate)


def solve_oracle(load: np.ndarray, pv: np.ndarray, price: np.ndarray,
                 carbon: np.ndarray, energy: float, power: float,
                 market: str, objective: str, eta: float = ETA,
                 peak_cap: float | None = None, month: np.ndarray | None = None,
                 demand_rate: float = 0.0) -> BatteryResult:
    """Perfect-foresight linear programme (HiGHS, sparse) for one simulated year.

    objective: 'cost', gross-import 'co2', or net-exchange 'co2_net'.
    market: 'M1_net_metering' or 'M2_zero_export'.

    The frozen discharge convention bounds discharge_t by load_t.
    A positive demand rate adds a charge on each supplied billing block's peak.
    """
    n = load.size
    if demand_rate < 0 or (demand_rate and objective != "cost"):
        raise ValueError("demand rate must be nonnegative and belongs to the cost objective")
    if demand_rate and (month is None or np.shape(month) != (n,)):
        raise ValueError("one billing-block label is required per hour")
    if energy <= 0:
        return BatteryResult(np.zeros(n), np.zeros(n), np.full(n, 0.0), load - pv)

    soc0 = 0.5 * energy
    # variables: charge[0:n], discharge[n:2n], soc[2n:3n] (+ import[3n:4n] when needed)
    nvar = 3 * n
    # Gross-import auxiliaries support M2 energy costs, CO2 and import limits;
    # their presence does not change M1's net-metered energy objective.
    gross = (objective == "co2") or (objective == "cost" and market == "M2_zero_export") or (peak_cap is not None) or demand_rate > 0
    n_extra = 1 if gross else 0
    month_index = np.unique(month, return_inverse=True)[1] if demand_rate else np.empty(0, dtype=int)
    n_months = int(month_index.max()) + 1 if demand_rate else 0
    peak_start = nvar + n * n_extra
    nvar_total = peak_start + n_months

    c_obj = np.zeros(nvar_total)
    if objective == "co2":
        c_obj[nvar:nvar + n] = carbon
    elif objective == "co2_net":
        c_obj[0:n] = carbon
        c_obj[n:2 * n] = -carbon
    elif market == "M2_zero_export":
        c_obj[nvar:nvar + n] = price
    else:  # M1 net metering cost: bill = sum(price * net)
        c_obj[0:n] = price
        c_obj[n:2 * n] = -price
    c_obj[peak_start:] = demand_rate
    # tiny penalty on simultaneous charge/discharge removes degenerate LP solutions
    c_obj[0:n] += 1e-6
    c_obj[n:2 * n] += 1e-6

    # SOC dynamics rows (n+1): soc_t - soc_{t-1} - eta*c_t + d_t/eta = 0, soc_{-1}=soc0
    rows, cols, vals = [], [], []
    for t in range(n):
        rows += [t, t, t]
        cols += [2 * n + t, t, n + t]
        vals += [1.0, -eta, 1.0 / eta]
        if t > 0:
            rows.append(t)
            cols.append(2 * n + t - 1)
            vals.append(-1.0)
    rows += [n]
    cols += [3 * n - 1]
    vals += [1.0]
    A_eq = sparse.csr_matrix((vals, (rows, cols)), shape=(n + 1, nvar_total))
    b_eq = np.zeros(n + 1)
    b_eq[0] = soc0
    b_eq[n] = soc0

    A_ub = None
    b_ub = None
    if gross:
        # import >= net = load - pv + c - d  <=>  -imp + c - d <= pv - load
        rows, cols, vals = [], [], []
        for t in range(n):
            rows += [t, t, t]
            cols += [nvar + t, t, n + t]
            vals += [-1.0, 1.0, -1.0]
        A_ub = sparse.csr_matrix((vals, (rows, cols)), shape=(n, nvar_total))
        b_ub = pv - load
        if peak_cap is not None:
            rows2, cols2, vals2 = [], [], []
            for t in range(n):
                rows2.append(t)
                cols2.append(nvar + t)
                vals2.append(1.0)
            A_cap = sparse.csr_matrix((vals2, (rows2, cols2)), shape=(n, nvar_total))
            A_ub = sparse.vstack([A_ub, A_cap])
            b_ub = np.concatenate([b_ub, np.full(n, float(peak_cap))])
        if demand_rate:
            rr = np.repeat(np.arange(n), 2)
            cc = np.column_stack([nvar + np.arange(n), peak_start + month_index]).ravel()
            vv = np.tile([1.0, -1.0], n)
            A_month = sparse.csr_matrix((vv, (rr, cc)), shape=(n, nvar_total))
            A_ub = sparse.vstack([A_ub, A_month])
            b_ub = np.concatenate([b_ub, np.zeros(n)])

    # Frozen discharge bound: discharge_t <= load_t.
    dis_max = np.minimum(power, load)
    bounds = [(0.0, power)] * n + [(0.0, float(dis_max[t])) for t in range(n)] + [(0.0, energy)] * n
    if gross:
        bounds += [(0.0, None)] * n
    bounds += [(0.0, None)] * n_months

    res = optimize.linprog(
        c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds,
        method="highs", options={"presolve": True},
    )
    if not res.success:
        raise RuntimeError(f"LP failed: {res.message}")
    x = res.x
    charge = np.clip(x[0:n], 0, None)
    discharge = np.clip(x[n:2 * n], 0, None)
    soc = np.clip(x[2 * n:3 * n], 0, energy)
    net = load - pv + charge - discharge
    return BatteryResult(charge, discharge, soc, net)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def compute_metrics(result: BatteryResult, load: np.ndarray, pv: np.ndarray,
                    price: np.ndarray, carbon: np.ndarray, market: str,
                    energy: float) -> dict:
    net = result.net
    gross_import = np.maximum(net, 0.0)
    if market == "M1_net_metering":
        bill = float(np.sum(price * net))
    else:
        bill = float(np.sum(price * gross_import))
    co2_gross = float(np.sum(carbon * gross_import))
    co2_net = float(np.sum(carbon * net))
    peak = float(gross_import.max()) if gross_import.size else 0.0
    sc_ratio = float(1.0 - np.maximum(net, 0.0).sum() / max(load.sum(), 1e-9))
    throughput = float(np.sum(result.charge) + np.sum(result.discharge)) / 2.0
    cycles = float(np.sum(result.discharge) / energy) if energy > 0 else 0.0
    return dict(bill=bill, co2_gross=co2_gross, co2_net=co2_net,
                peak=peak, self_consumption=sc_ratio, throughput=throughput,
                cycles=cycles, export=float(np.maximum(-net, 0.0).sum()))


def baseline_value(metrics_by_energy: dict, market: str, ctrl: str) -> float:
    """Battery value = NOB bill - controller bill (same market)."""
    return metrics_by_energy[(market, 0.0, "NOB")]["bill"] - \
        metrics_by_energy[(market, DEFAULT_CAP_KWH, ctrl)]["bill"]
