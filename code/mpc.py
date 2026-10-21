from __future__ import annotations

import math

import numpy as np
from scipy import optimize, sparse

import core

ETA = core.ETA
HORIZON = 24
LOOKBACK = 24  # persistence lag (one day)


def _lp(load_f, pv_f, price_f, soc0, energy, power, market):
    n = load_f.size
    gross = market == "M2_zero_export"
    nvar = 3 * n
    nvar_total = nvar + (n if gross else 0)
    c_obj = np.zeros(nvar_total)
    if gross:
        c_obj[nvar:] = price_f
    else:
        c_obj[0:n] = price_f
        c_obj[n:2 * n] = -price_f
    c_obj[0:n] += 1e-6
    c_obj[n:2 * n] += 1e-6

    rows, cols, vals = [], [], []
    for t in range(n):
        rows += [t, t, t]
        cols += [2 * n + t, t, n + t]
        vals += [1.0, -ETA, 1.0 / ETA]
        if t > 0:
            rows.append(t)
            cols.append(2 * n + t - 1)
            vals.append(-1.0)
    A_eq = sparse.csr_matrix((vals, (rows, cols)), shape=(n, nvar_total))
    b_eq = np.zeros(n)
    b_eq[0] = soc0

    # terminal SOC >= initial SOC (energy neutrality within the horizon)
    A_ub, b_ub = [], []
    rows, cols, vals = [], [], []
    rows += [0]
    cols += [3 * n - 1]
    vals += [-1.0]
    A_ub.append(sparse.csr_matrix((vals, (rows, cols)), shape=(1, nvar_total)))
    b_ub.append(np.array([-soc0]))

    if gross:
        rows, cols, vals = [], [], []
        for t in range(n):
            rows += [t, t, t]
            cols += [nvar + t, t, n + t]
            vals += [-1.0, 1.0, -1.0]
        A_ub.append(sparse.csr_matrix((vals, (rows, cols)), shape=(n, nvar_total)))
        b_ub.append(pv_f - load_f)

    dis_max = np.minimum(power, load_f)
    bounds = ([(0.0, power)] * n + [(0.0, float(dis_max[t])) for t in range(n)] +
              [(0.0, energy)] * n + ([(0.0, None)] * n if gross else []))
    A_ub_m = sparse.vstack(A_ub) if A_ub else None
    b_ub_v = np.concatenate(b_ub) if b_ub else None
    res = optimize.linprog(c_obj, A_ub=A_ub_m, b_ub=b_ub_v, A_eq=A_eq, b_eq=b_eq,
                           bounds=bounds, method="highs", options={"presolve": True})
    if not res.success:
        return 0.0, 0.0
    return float(max(res.x[0], 0.0)), float(max(res.x[n], 0.0))


def simulate_mpc(load, pv, price, carbon, energy, power, market,
                 horizon=HORIZON, lookback=LOOKBACK):
    """Run the receding-horizon controller on one simulated year."""
    T = load.size
    charge = np.zeros(T)
    discharge = np.zeros(T)
    soc = 0.5 * energy
    soc_hist = np.empty(T)
    for t in range(T):
        fut = np.arange(t, t + horizon)
        src = fut - lookback
        src[0] = t                      # the current hour is observed, not forecast
        src = np.clip(src, 0, T - 1)
        c, d = _lp(load[src], pv[src], price[src], soc, energy, power, market)
        # never let the applied action violate the physical limits of the realised hour
        c = min(c, max(0.0, energy - soc) / ETA)
        d = min(d, load[t], max(0.0, soc) * ETA)
        charge[t] = c
        discharge[t] = d
        soc += ETA * c - d / ETA
        if soc < 0.0:
            # the applied action would over-discharge: clip the discharge to what is stored
            discharge[t] = max(0.0, (soc + d / ETA) * ETA)
            soc = 0.0
        elif soc > energy:
            charge[t] = max(0.0, c - (soc - energy) / ETA)
            soc = energy
        soc_hist[t] = soc
    net = load - pv + charge - discharge
    return core.BatteryResult(charge, discharge, soc_hist, net)
