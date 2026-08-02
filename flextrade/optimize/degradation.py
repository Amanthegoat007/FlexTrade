"""Physics-based battery degradation: rainflow cycle counting with
depth-of-discharge dependent cost, replacing the flat Rs/MWh proxy.

Why the flat proxy is wrong
---------------------------
Cycle life of Li-ion cells follows a Wohler-type power law: cycles to end
of life L(d) = L100 * d^-kp for cycle depth d (fraction of capacity). With
kp > 1, deep cycles consume proportionally MORE life per MWh than shallow
ones, so a flat Rs/MWh understates the cost of deep cycling and biases the
optimizer toward full-depth swings.

Parameters (documented assumptions, all configurable):
  L100  = 6000 full-depth cycles to 80% capacity — typical LFP datasheet
          value (grid-scale LFP is quoted 6000-8000 @ 0.5C, 25 C).
  KP    = 1.1 — empirical Wohler exponent for LFP; lab studies fit
          kp ~= 1.05-1.3 (e.g. Wang et al. 2011 J. Power Sources; DNV
          battery scorecards). Conservative low end used.
  CAPEX = Rs 1.5 crore per MWh installed — Indian grid BESS capex 2025-26
          (SECI/NTPC tender discovery trended Rs 1.3-1.8 Cr/MWh incl. PCS).
  CYCLE_SHARE = 0.7 — share of capex attributed to cycle aging (the rest
          is calendar aging that happens anyway and shouldn't steer
          dispatch).

Per-cycle cost of one cycle of depth d on a battery of E MWh:
    cost(d) = CYCLE_SHARE * CAPEX * E / L(d) = share * capex_total * d^kp / L100
Per discharged MWh at depth d (throughput per cycle = d*E discharged):
    rs_per_mwh(d) = share * CAPEX / L100 * d^(kp-1)

At the defaults that is ~Rs 1,750/MWh for full-depth cycles — an order of
magnitude above the old Rs 200 throughput proxy. This is the honest number:
LCOS studies put the storage adder for LFP at Rs 2-4/kWh, and it changes
the economics — arbitrage must clear a ~Rs 1,900/MWh round-trip hurdle
(degradation + efficiency), which Delhi's day/night spread still does.

The DoD-dependent cost is nonconvex, so the LP cannot carry it directly.
`optimize_physical()` keeps the LP linear and iterates a fixed point:
solve LP with flat rate r -> rainflow the resulting SoC path -> compute
the true physics cost -> set r = physics cost / throughput -> resolve.
Converges in 2-3 iterations because the schedule shape is stable in r.
"""
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from optimize.dispatch import BLOCK_H, Bess, optimize_dispatch  # noqa: E402


@dataclass
class DegParams:
    cycle_life_100: float = 6000.0   # full-DoD cycles to 80% capacity (LFP)
    kp: float = 1.1                  # Wohler exponent
    capex_rs_per_mwh: float = 1.5e7  # Rs 1.5 Cr per MWh installed
    cycle_share: float = 0.7         # capex share attributed to cycling

    def cost_of_cycle(self, depth: float, energy_mwh: float,
                      count: float = 1.0) -> float:
        """Rs cost of `count` cycles (0.5 = half cycle) at fractional depth."""
        if depth <= 0:
            return 0.0
        life = self.cycle_life_100 * depth ** (-self.kp)
        return count * self.cycle_share * self.capex_rs_per_mwh * energy_mwh / life

    def rs_per_mwh(self, depth: float) -> float:
        """Marginal Rs per discharged MWh at a given cycle depth."""
        if depth <= 0:
            return 0.0
        return (self.cycle_share * self.capex_rs_per_mwh / self.cycle_life_100
                * depth ** (self.kp - 1.0))


# ---------------------------------------------------------------- rainflow --

def _turning_points(x: np.ndarray) -> np.ndarray:
    """Strip monotone runs, keep local extrema (incl. endpoints)."""
    if len(x) < 3:
        return x
    d = np.diff(x)
    keep = [0]
    for i in range(1, len(x) - 1):
        if d[i - 1] == 0:
            continue
        if np.sign(d[i - 1]) != np.sign(d[i]) and d[i] != 0:
            keep.append(i)
        elif d[i] == 0 and (i + 1 == len(x) - 1 or np.sign(d[i - 1]) != np.sign(x[i + 1] - x[i])):
            keep.append(i)
    keep.append(len(x) - 1)
    return x[sorted(set(keep))]


def rainflow(soc_frac: np.ndarray) -> list[tuple[float, float]]:
    """ASTM E1049-85 rainflow (4-point method) on a SoC trajectory.

    Input: SoC as a fraction of capacity. Returns [(depth, count)] with
    count 1.0 for full cycles and 0.5 for residual half cycles.
    """
    pts = list(_turning_points(np.asarray(soc_frac, dtype=float)))
    cycles: list[tuple[float, float]] = []
    stack: list[float] = []
    for p in pts:
        stack.append(p)
        while len(stack) >= 4:
            x1, x2, x3, x4 = stack[-4:]
            r_inner = abs(x3 - x2)
            if r_inner <= abs(x2 - x1) and r_inner <= abs(x4 - x3):
                cycles.append((r_inner, 1.0))  # full cycle extracted
                del stack[-3:-1]
            else:
                break
    for a, b in zip(stack, stack[1:]):  # residuals are half cycles
        if abs(b - a) > 0:
            cycles.append((abs(b - a), 0.5))
    return [(d, c) for d, c in cycles if d > 1e-9]


def day_cost(soc_mwh: pd.Series, energy_mwh: float,
             params: DegParams = DegParams()) -> dict:
    """Physics degradation cost of one day's SoC trajectory."""
    soc_frac = np.asarray(soc_mwh, dtype=float) / energy_mwh
    cyc = rainflow(soc_frac)
    cost = sum(params.cost_of_cycle(d, energy_mwh, c) for d, c in cyc)
    fce = sum(d * c for d, c in cyc)  # full-cycle equivalents
    return {
        "cost_rs": round(cost, 0),
        "cycles": [(round(d, 3), c) for d, c in cyc],
        "full_cycle_equivalents": round(fce, 3),
        "throughput_mwh": round(2 * fce * energy_mwh, 2),
    }


# ----------------------------------------------------- physics-aware LP -----

def optimize_physical(prices: pd.Series, bess: Bess = Bess(),
                      params: DegParams = DegParams(),
                      iters: int = 4) -> dict:
    """Fixed-point iteration: LP with a flat rate calibrated to rainflow.

    Returns the converged schedule plus a comparison against the legacy
    flat Rs 200/MWh proxy, both settled at the same prices.
    """
    def solve(rate: float):
        b = Bess(**{**bess.__dict__, "degradation_rs_mwh": rate})
        sched, _ = optimize_dispatch(prices, b)
        gross = float(((sched["discharge_mw"] - sched["charge_mw"])
                       * BLOCK_H * prices.values).sum())
        deg = day_cost(sched["soc_mwh"], bess.energy_mwh, params)
        return sched, gross, deg

    history = []
    rate = bess.degradation_rs_mwh
    sched = gross = deg = None
    for _ in range(iters):
        sched, gross, deg = solve(rate)
        thr = deg["throughput_mwh"]
        new_rate = deg["cost_rs"] / thr if thr > 0 else rate
        history.append({"rate_rs_mwh": round(rate, 0), "gross_rs": round(gross, 0),
                        "physics_cost_rs": deg["cost_rs"],
                        "net_rs": round(gross - deg["cost_rs"], 0),
                        "fce": deg["full_cycle_equivalents"]})
        if abs(new_rate - rate) < 5:
            break
        rate = new_rate

    # the Rs 200 flat proxy this module was built to discredit — kept as a
    # comparison so the improvement stays visible after the default moved
    proxy_sched, proxy_gross, proxy_deg = solve(200.0)
    return {
        "schedule": sched,
        "converged_rate_rs_mwh": round(rate, 0),
        "gross_rs": round(gross, 0),
        "physics_cost_rs": deg["cost_rs"],
        "net_rs": round(gross - deg["cost_rs"], 0),
        "full_cycle_equivalents": deg["full_cycle_equivalents"],
        "iterations": history,
        "proxy200": {
            "gross_rs": round(proxy_gross, 0),
            "physics_cost_rs": proxy_deg["cost_rs"],
            "net_rs": round(proxy_gross - proxy_deg["cost_rs"], 0),
            "fce": proxy_deg["full_cycle_equivalents"],
        },
        "params": {**params.__dict__},
    }


# ---------------------------------------------------------------- selftest --

def selftest():
    p = DegParams()
    # 1) power law: one 100% cycle costs more than two 50% cycles (kp > 1)
    full = p.cost_of_cycle(1.0, 40.0)
    halves = 2 * p.cost_of_cycle(0.5, 40.0)
    assert full > halves, (full, halves)
    # 2) rainflow on a textbook sequence: known result
    seq = np.array([0.0, 1.0, 0.2, 0.8, 0.1, 0.9, 0.0])
    cyc = rainflow(seq)
    fce = sum(d * c for d, c in cyc)
    total_range = sum(d * (2 * c) for d, c in cyc)  # cycles*2 = excursions
    assert abs(total_range - np.abs(np.diff(seq)).sum()) < 1e-9, cyc
    # 3) simple triangle = one half-cycle pair of the full range
    tri = rainflow(np.array([0.2, 0.9, 0.2]))
    assert len(tri) == 2 and all(abs(d - 0.7) < 1e-9 and c == 0.5 for d, c in tri), tri
    # 4) flat line costs nothing
    assert day_cost(pd.Series([20.0] * 96), 40.0)["cost_rs"] == 0
    # 5) marginal rate at full depth matches the documented ~Rs 1,750
    assert 1600 < p.rs_per_mwh(1.0) < 1900, p.rs_per_mwh(1.0)
    print("degradation selftest: all 5 assertions pass")
    print(f"  full-DoD marginal cost  Rs {p.rs_per_mwh(1.0):,.0f}/MWh discharged")
    print(f"  50%-DoD marginal cost   Rs {p.rs_per_mwh(0.5):,.0f}/MWh discharged")
    print(f"  one 100% cycle Rs {full:,.0f} vs two 50% cycles Rs {halves:,.0f}")


if __name__ == "__main__":
    selftest()
    from ingest import store
    dam = store.read("dam_price")
    day = dam.index.date.max()
    prices = dam[dam.index.date == day]["mcp_rs_mwh"]
    print(f"\nphysics-aware dispatch on {day} actual DAM prices:")
    r = optimize_physical(prices)
    for it in r["iterations"]:
        print(f"  rate Rs {it['rate_rs_mwh']:>5,.0f}/MWh -> gross Rs {it['gross_rs']:>9,.0f}"
              f"  physics cost Rs {it['physics_cost_rs']:>9,.0f}  net Rs {it['net_rs']:>9,.0f}"
              f"  ({it['fce']:.2f} FCE)")
    print(f"  converged flat rate: Rs {r['converged_rate_rs_mwh']:,.0f}/MWh")
    pr = r["proxy200"]
    print(f"  legacy Rs200 proxy:  net Rs {pr['net_rs']:,.0f} ({pr['fce']:.2f} FCE)"
          f"  vs physics-aware net Rs {r['net_rs']:,.0f}"
          f"  -> uplift Rs {r['net_rs'] - pr['net_rs']:,.0f}")
