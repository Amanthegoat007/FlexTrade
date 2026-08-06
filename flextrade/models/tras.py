"""What reserving capacity for TRAS costs — and the price that makes it worth it.

WHY THIS IS NOT A PRICE FORECAST
--------------------------------
Ancillary services are reported as 15-20% of a merchant BESS revenue stack, and
CERC has opened the Tertiary Reserve Ancillary Services segment on all three
power exchanges. It is the largest revenue line this platform does not model.

It is also the one we cannot measure. Data discovery, run 6 Aug 2026:

  - IEX publishes exactly seven market-data segments (day-ahead, green
    day-ahead, green intra-day, high-price day-ahead, high-price intra-day,
    intra-day contingency, real-time). NONE is ancillary. Guessed slugs like
    /market-data/ancillary-services/market-snapshot return a 404 shell —
    83.9 KB with no MCP markers, against 257 KB for a real segment page.
  - Grid-India's ancillary portal (ancillary.grid-india.in) is an Angular app
    behind reCAPTCHA whose bundle carries 164 references to "token" and 26 to
    "login". It is a registered-participant portal, not an open feed. TRAS
    appears throughout it, so the data exists — we simply cannot read it.

Publicly available is VOLUME only (IEX reported TRAS at 603 MU in Q2 FY26
against 16.9 MU a year earlier), never cleared price.

So inventing a TRAS price and multiplying it out would produce a revenue number
with no evidence behind it, in the same place a lender looks hardest. Instead
this INVERTS the unknown, the same way bankability.capacity_payment_for_
bankability does: the cost side is fully measurable from our own dispatch, so
compute the price at which participation breaks even and let the operator
compare it against whatever they can actually contract.

WHAT IS MEASURED, AND WHAT IS ASSUMED
-------------------------------------
MEASURED   the arbitrage a battery gives up by holding reserve — our LP, our
           prices, the same objective the trading desk runs
ASSUMED    nothing about TRAS prices. The output is a required price, not a
           revenue.

THE RESERVE CONSTRAINT
----------------------
Upward tertiary reserve means the asset must be able to RAISE output by R MW
and sustain it. Two constraints, both physical:

    discharge[t] + R <= P                 headroom in the converter
    soc[t]           >= R * t_sustain/eta  energy actually behind the promise

The second is the one spreadsheets forget. Selling reserve you have no stored
energy to deliver is not a revenue line, it is a penalty waiting to happen.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pulp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from ingest import store  # noqa: E402
from optimize.dispatch import BLOCK_H, Bess, optimize_dispatch  # noqa: E402

OUT = HERE.parent / "output"
CACHE = OUT / "tras.json"

# How long the reserve must be deliverable for. CERC's tertiary product is a
# slower reserve than primary/secondary; one hour is used here as an explicit,
# adjustable parameter rather than a claim about the regulation.
SUSTAIN_H_DEFAULT = 1.0


def dispatch_with_reserve(prices: pd.Series, bess: Bess, reserve_mw: float,
                          sustain_h: float = SUSTAIN_H_DEFAULT
                          ) -> tuple[pd.DataFrame, float]:
    """Arbitrage LP that also holds `reserve_mw` of upward reserve all day.

    Same objective, efficiency, degradation cost and SoC bounds as the
    production optimizer — only the two reserve constraints are added, so the
    difference in P&L is attributable to the reserve and nothing else.
    """
    p = prices.to_numpy(dtype=float)
    n = len(p)
    eta = np.sqrt(bess.round_trip_eff)
    soc0 = bess.soc0_frac * bess.energy_mwh
    soc_min = bess.soc_min_frac * bess.energy_mwh
    # energy that must stay behind the reserve promise at every instant
    floor = max(soc_min, reserve_mw * sustain_h / eta)
    if floor > bess.energy_mwh or reserve_mw > bess.power_mw:
        return pd.DataFrame(), float("nan")     # promise the asset cannot keep

    prob = pulp.LpProblem("tras_reserve", pulp.LpMaximize)
    ch = pulp.LpVariable.dicts("ch", range(n), 0, bess.power_mw)
    dis = pulp.LpVariable.dicts("dis", range(n), 0, bess.power_mw - reserve_mw)
    soc = pulp.LpVariable.dicts("soc", range(n + 1), floor, bess.energy_mwh)

    prob += pulp.lpSum(
        BLOCK_H * p[t] * (dis[t] - ch[t])
        - BLOCK_H * bess.degradation_rs_mwh * (ch[t] + dis[t]) for t in range(n))
    prob += soc[0] == max(soc0, floor)
    prob += soc[n] == max(soc0, floor)
    for t in range(n):
        prob += soc[t + 1] == soc[t] + BLOCK_H * (eta * ch[t] - dis[t] / eta)
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[prob.status] != "Optimal":
        return pd.DataFrame(), float("nan")

    sched = pd.DataFrame({
        "charge_mw": [ch[t].value() or 0.0 for t in range(n)],
        "discharge_mw": [dis[t].value() or 0.0 for t in range(n)],
        "soc_mwh": [soc[t + 1].value() or 0.0 for t in range(n)],
    }, index=prices.index)
    return sched, float(pulp.value(prob.objective) or 0.0)


def run(days: int = 365, reserve_fracs=(0.10, 0.25, 0.50, 0.75),
        bess: Bess | None = None,
        sustain_h: float = SUSTAIN_H_DEFAULT) -> dict:
    """Breakeven TRAS price for each reserve level, measured on real prices."""
    bess = bess or Bess(power_mw=1.0, energy_mwh=2.0)
    dam = store.read("dam_price")["mcp_rs_mwh"]
    lo = dam.index.max().normalize() - pd.Timedelta(days=days)
    dam = dam[dam.index >= lo]
    day_list = [d for d, g in dam.groupby(dam.index.date) if len(g) == 96]
    if not day_list:
        raise RuntimeError("no complete price days in window")

    base = {d: optimize_dispatch(dam[dam.index.date == d], bess)[1]
            for d in day_list}
    base_annual = float(np.mean(list(base.values())) * 365)

    rows = []
    for frac in reserve_fracs:
        R = bess.power_mw * frac
        tot, n_ok, thr = 0.0, 0, 0.0
        for d in day_list:
            pr = dam[dam.index.date == d]
            sch, pnl = dispatch_with_reserve(pr, bess, R, sustain_h)
            if not np.isfinite(pnl):
                continue
            tot += pnl
            n_ok += 1
            thr += float((sch["charge_mw"] + sch["discharge_mw"]).sum() * BLOCK_H)
        if not n_ok:
            rows.append({"reserve_frac": frac, "reserve_mw": R,
                         "infeasible": True,
                         "why": "reserve exceeds power or stored-energy capability"})
            continue
        annual = tot / n_ok * 365
        forgone = base_annual - annual
        # what TRAS must pay, per MW of reserve per hour held, to cover it
        breakeven = forgone / (R * 8760) if R > 0 else float("nan")
        rows.append({
            "reserve_frac": frac,
            "reserve_mw": round(R, 3),
            "arbitrage_annual_rs_per_mw": round(annual, 0),
            "forgone_arbitrage_rs_per_mw_yr": round(forgone, 0),
            "forgone_pct": round(forgone / base_annual * 100, 1),
            "breakeven_tras_rs_per_mw_h": round(breakeven, 1),
            "throughput_mwh_per_mw_yr": round(thr / n_ok * 365, 1),
            "days": n_ok,
        })

    return {
        "window": {"from": str(day_list[0]), "to": str(day_list[-1]),
                   "n_days": len(day_list)},
        "asset": {"power_mw": bess.power_mw, "energy_mwh": bess.energy_mwh,
                  "duration_h": bess.energy_mwh / bess.power_mw},
        "sustain_h": sustain_h,
        "arbitrage_only_annual_rs_per_mw": round(base_annual, 0),
        "reserve_levels": rows,
        "measured": ["forgone arbitrage", "throughput", "feasibility"],
        "assumed": [],
        "not_available": {
            "tras_cleared_price": (
                "IEX publishes seven market-data segments and none is ancillary; "
                "Grid-India's ancillary portal is reCAPTCHA-gated and requires a "
                "registered participant login. Volume is public (603 MU in Q2 "
                "FY26 vs 16.9 MU a year earlier), price is not."),
        },
        "how_to_read": (
            "These are REQUIRED prices, not revenue. If TRAS clears above the "
            "breakeven for a given reserve level, holding that reserve beats "
            "pure arbitrage; below it, it does not. Nothing here assumes a TRAS "
            "price, so none of it inherits an error from one."),
        "caveats": [
            "upward reserve only — downward reserve uses charging headroom and "
            "is a separate, cheaper product",
            "assumes reserve is held every hour of the year; a real contract is "
            "for defined windows, which would lower the forgone arbitrage",
            "ignores activation payments and any energy actually delivered when "
            "called, both of which improve the case for participating",
            "DAM arbitrage is the only counterfactual; a desk also running RTM "
            "would give up more, so this understates the opportunity cost",
        ],
    }


if __name__ == "__main__":
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 365
    r = run(days=days)
    a = r["asset"]
    print(f"TRAS reserve cost — {r['window']['from']} -> {r['window']['to']} "
          f"({r['window']['n_days']} days)")
    print(f"asset {a['power_mw']:.0f} MW / {a['energy_mwh']:.0f} MWh "
          f"({a['duration_h']:.0f}h), reserve sustainable {r['sustain_h']:.0f}h")
    print(f"arbitrage only: Rs {r['arbitrage_only_annual_rs_per_mw']/1e5:,.1f} "
          f"lakh/MW/yr\n")
    print(f"{'reserve':>9}{'arb Rs L/MW/yr':>16}{'forgone':>10}"
          f"{'breakeven TRAS':>17}")
    print(f"{'':>9}{'':>16}{'':>10}{'Rs/MW/h':>17}")
    for x in r["reserve_levels"]:
        if x.get("infeasible"):
            print(f"{x['reserve_frac']:8.0%}   INFEASIBLE — {x['why']}")
            continue
        print(f"{x['reserve_frac']:8.0%}{x['arbitrage_annual_rs_per_mw']/1e5:16.1f}"
              f"{x['forgone_pct']:9.1f}%{x['breakeven_tras_rs_per_mw_h']:17,.0f}")
    CACHE.write_text(json.dumps(r, indent=2, default=float))
    print(f"\nwrote {CACHE.name}")
