"""Risk-aware BESS dispatch: scenario generation + CVaR optimization.

Why this exists
---------------
`dispatch.optimize_dispatch` maximizes profit against a single price
curve, treating the forecast as certain. Our price forecast has ~23%
MAPE, so that is a strong assumption: the schedule it produces is optimal
for exactly one future that will not happen.

A DAM bid is a classic **here-and-now decision under uncertainty** — you
commit volumes before the price is known. The right formulation is a
two-stage stochastic program: one schedule, evaluated across many price
scenarios drawn from the forecast distribution.

We optimize a blend of expected profit and **CVaR** (Conditional
Value-at-Risk — the mean profit of the worst `1-alpha` share of
scenarios), linearized by the standard Rockafellar-Uryasev construction:

    maximize  (1 - lam) * E[profit]  +  lam * CVaR_alpha(profit)
    s.t.      u_s >= zeta - profit_s,  u_s >= 0
              CVaR_alpha = zeta - (1 / (1 - alpha)) * sum_s pi_s * u_s

`lam = 0` reproduces the risk-neutral LP; `lam = 1` maximizes the worst
tail alone. Everything stays linear, so CBC still solves it in about a
second.

Scenario generation
-------------------
Scenarios come from the conformal quantile forecast (P10/P50/P90) via a
Gaussian copula, so prices stay correlated across blocks — an expensive
day is expensive all day, which is how the market actually behaves.
`correlation=1.0` gives purely systematic (day-level) risk; lower values
add block-specific noise. Because the quantile band is wider in volatile
evening blocks than in quiet night blocks, even fully systematic
scenarios change the *shape* of the price curve, not just its level —
which is what matters for arbitrage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pulp
from scipy.stats import norm

from .dispatch import BLOCK_H, Bess


def make_scenarios(quantiles: pd.DataFrame, n_scenarios: int = 24,
                   correlation: float = 0.85,
                   seed: int = 42) -> np.ndarray:
    """Price scenarios (n_scenarios x n_blocks) from per-block quantiles.

    `quantiles` columns must be named qXX (e.g. q10/q50/q90) and sorted.
    Levels are read from the column names, so a 5-quantile model works
    unchanged.
    """
    levels = np.array([int(c[1:]) / 100 for c in quantiles.columns])
    order = np.argsort(levels)
    levels, vals = levels[order], quantiles.values[:, order]  # (blocks, q)
    n_blocks = len(quantiles)

    rng = np.random.default_rng(seed)
    # stratified day-level draws so the scenario set spans the distribution
    # evenly instead of clustering by luck
    strata = (np.arange(n_scenarios) + 0.5) / n_scenarios
    z_day = norm.ppf(rng.permutation(strata))
    z_block = rng.standard_normal((n_scenarios, n_blocks))
    rho = float(np.clip(correlation, 0.0, 1.0))
    z = rho * z_day[:, None] + np.sqrt(max(1 - rho**2, 0.0)) * z_block
    u = np.clip(norm.cdf(z), levels[0], levels[-1])

    scen = np.empty((n_scenarios, n_blocks))
    for t in range(n_blocks):
        scen[:, t] = np.interp(u[:, t], levels, vals[t])
    return np.clip(scen, 0, 10000)


def optimize_cvar(quantiles: pd.DataFrame, bess: Bess = Bess(),
                  lam: float = 0.5, alpha: float = 0.90,
                  n_scenarios: int = 24, correlation: float = 0.85,
                  scenarios: np.ndarray | None = None
                  ) -> tuple[pd.DataFrame, dict]:
    """One schedule, optimized across price scenarios.

    lam:   0 = risk-neutral, 1 = pure worst-tail. 0.5 balances them.
    alpha: tail definition — 0.90 means CVaR over the worst 10%.
    """
    scen = (make_scenarios(quantiles, n_scenarios, correlation)
            if scenarios is None else scenarios)
    S, n = scen.shape
    pi = 1.0 / S
    eta = np.sqrt(bess.round_trip_eff)
    soc0 = bess.soc0_frac * bess.energy_mwh
    soc_min = bess.soc_min_frac * bess.energy_mwh

    prob = pulp.LpProblem("bess_cvar", pulp.LpMaximize)
    ch = pulp.LpVariable.dicts("ch", range(n), 0, bess.power_mw)
    dis = pulp.LpVariable.dicts("dis", range(n), 0, bess.power_mw)
    soc = pulp.LpVariable.dicts("soc", range(n + 1), soc_min, bess.energy_mwh)
    zeta = pulp.LpVariable("zeta")                       # VaR level (free)
    u = pulp.LpVariable.dicts("u", range(S), 0)          # tail shortfalls

    # profit of the (single) schedule under each scenario
    profit = {
        s: pulp.lpSum(BLOCK_H * (scen[s, t] * dis[t] - scen[s, t] * ch[t])
                      - BLOCK_H * bess.degradation_rs_mwh * (ch[t] + dis[t])
                      for t in range(n))
        for s in range(S)
    }
    expected = pulp.lpSum(pi * profit[s] for s in range(S))
    cvar = zeta - (1.0 / (1.0 - alpha)) * pulp.lpSum(pi * u[s] for s in range(S))
    prob += (1 - lam) * expected + lam * cvar

    for s in range(S):
        prob += u[s] >= zeta - profit[s]

    prob += soc[0] == soc0
    for t in range(n):
        prob += soc[t + 1] == soc[t] + BLOCK_H * (ch[t] * eta - dis[t] / eta)
    prob += soc[n] >= soc0

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"CVaR LP status: {pulp.LpStatus[prob.status]}")

    out = pd.DataFrame(index=quantiles.index)
    out["price"] = quantiles.filter(like="q50").iloc[:, 0] \
        if any("q50" in c for c in quantiles.columns) else scen.mean(axis=0)
    out["charge_mw"] = [ch[t].value() for t in range(n)]
    out["discharge_mw"] = [dis[t].value() for t in range(n)]
    out["soc_mwh"] = [soc[t + 1].value() for t in range(n)]
    out["bess_mw"] = out["discharge_mw"] - out["charge_mw"]

    realized = np.array([pulp.value(profit[s]) for s in range(S)])
    stats = {
        "expected_pnl": float(realized.mean()),
        "cvar_pnl": float(realized[realized <= np.quantile(realized, 1 - alpha)].mean()),
        "worst_pnl": float(realized.min()),
        "best_pnl": float(realized.max()),
        "std_pnl": float(realized.std()),
        "lam": lam, "alpha": alpha, "n_scenarios": S,
        "scenario_pnl": realized,
    }
    return out, stats


def evaluate(schedule: pd.DataFrame, scen: np.ndarray,
             degradation_rs_mwh: float = 200.0) -> np.ndarray:
    """P&L of a fixed schedule under each scenario — lets us score a
    risk-neutral schedule on the same scenario set for a fair comparison."""
    e_dis = schedule["discharge_mw"].values * BLOCK_H
    e_ch = schedule["charge_mw"].values * BLOCK_H
    throughput = degradation_rs_mwh * (e_dis + e_ch).sum()
    return scen @ (e_dis - e_ch) - throughput
