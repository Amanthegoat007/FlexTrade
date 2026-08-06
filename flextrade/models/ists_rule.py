"""What the draft ISTS charging restriction does to a merchant BESS.

THE RULE, AND WHY IT MATTERS
---------------------------
Draft grid-security measures would require storage seeking ISTS connectivity to
(a) hold a minimum two-hour duration, (b) install commensurate renewable
capacity exclusively for charging, and (c) accept restrictions on withdrawing
power from the ISTS grid.

Read (b) and (c) together and they aim at the heart of the merchant model.
Merchant arbitrage IS buying cheap grid energy and selling it dear; a battery
that may only charge from its own co-located renewable is a different asset with
different economics. India has ~8.5 GWh of operational BESS and roughly 81% of
it runs merchant, so this is not a marginal rule.

THE QUESTION IS EMPIRICAL, NOT RHETORICAL
-----------------------------------------
It is tempting to assume the restriction is simply destructive. It may not be.
In a solar-heavy grid the price trough has moved to the middle of the day —
which is exactly when a co-located solar plant is generating. If the battery
wanted to charge at midday anyway, being forced to charge from midday solar
costs it very little. Whether that holds is a question about OUR price history
and OUR generation twin, and it is answerable rather than arguable.

So this module runs the same LP twice on the same days and the same prices,
changing one constraint:

    unrestricted   charge[t] <= P                  (today's merchant model)
    ISTS-bound     charge[t] <= min(P, re[t])      (charge only from own RE)

and reports the revenue difference as a function of how much RE is built per MW
of battery. What it deliberately does NOT do is price the RE plant itself: that
is a hybrid-project question with its own PPA and curtailment assumptions, and
mixing it in here would hide the one clean number this can produce — the
dispatch cost of the constraint.
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
from ingest import store, weather  # noqa: E402
from models import re_model as rm  # noqa: E402
from optimize.dispatch import BLOCK_H, Bess, optimize_dispatch  # noqa: E402

OUT = HERE.parent / "output"
CACHE = OUT / "ists_rule.json"

# Bhadla / western Rajasthan — India's solar belt and where the large merchant
# BESS projects actually sit (ACME's 2,031 MWh is in Rajasthan). Using Delhi's
# weather here would model a co-located plant nobody would build.
RE_LAT, RE_LON = 27.5, 71.9


def re_profile(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Per-MW-of-RE output at 15-min resolution from the physical twin.

    Normalised to 1 MW of solar and 1 MW of wind nameplate so the caller can
    scale to any RE:BESS ratio without refitting anything.
    """
    js = weather._get("https://archive-api.open-meteo.com/v1/archive", dict(
        latitude=RE_LAT, longitude=RE_LON,
        hourly="shortwave_radiation,wind_speed_100m,temperature_2m",
        start_date=str(start.date()), end_date=str(end.date()),
        timezone="Asia/Kolkata"))
    df = pd.DataFrame(js["hourly"])
    df["ts"] = pd.to_datetime(df.pop("time"))
    df = df.set_index("ts").sort_index()
    wx = pd.DataFrame({
        "ghi": pd.to_numeric(df["shortwave_radiation"], errors="coerce"),
        "wind100_kmh": pd.to_numeric(df["wind_speed_100m"], errors="coerce"),
        "temp_c": pd.to_numeric(df["temperature_2m"], errors="coerce"),
    }).dropna()
    twin = rm._twin(wx)
    # per 1 MW nameplate of each technology
    solar = twin["solar_mw"] / rm.SolarPlant().capacity_mw
    wind = twin["wind_mw"] / rm.WindFarm().capacity_mw
    return pd.DataFrame({"solar_pu": solar, "wind_pu": wind}).clip(lower=0)


def dispatch_re_bound(prices: pd.Series, re_mw: pd.Series,
                      bess: Bess = Bess()) -> tuple[pd.DataFrame, float]:
    """The same LP as optimize_dispatch, plus charge[t] <= re_mw[t].

    Deliberately a copy of the production objective rather than a new
    formulation: if the two differed in efficiency, degradation cost or SoC
    bounds, the comparison would measure the difference between two models
    instead of the effect of one constraint.
    """
    p = prices.to_numpy(dtype=float)
    re = np.asarray(re_mw, dtype=float)
    n = len(p)
    eta = np.sqrt(bess.round_trip_eff)
    soc0 = bess.soc0_frac * bess.energy_mwh
    soc_min = bess.soc_min_frac * bess.energy_mwh

    prob = pulp.LpProblem("ists", pulp.LpMaximize)
    ch = pulp.LpVariable.dicts("ch", range(n), 0, bess.power_mw)
    dis = pulp.LpVariable.dicts("dis", range(n), 0, bess.power_mw)
    soc = pulp.LpVariable.dicts("soc", range(n + 1), soc_min, bess.energy_mwh)

    prob += pulp.lpSum(
        BLOCK_H * p[t] * (dis[t] - ch[t])
        - BLOCK_H * bess.degradation_rs_mwh * (ch[t] + dis[t]) for t in range(n))

    prob += soc[0] == soc0
    prob += soc[n] == soc0
    for t in range(n):
        prob += soc[t + 1] == soc[t] + BLOCK_H * (eta * ch[t] - dis[t] / eta)
        prob += ch[t] <= max(float(re[t]), 0.0)      # THE RULE
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    sched = pd.DataFrame({
        "charge_mw": [ch[t].value() or 0.0 for t in range(n)],
        "discharge_mw": [dis[t].value() or 0.0 for t in range(n)],
        "soc_mwh": [soc[t + 1].value() or 0.0 for t in range(n)],
    }, index=prices.index)
    sched["bess_mw"] = sched["discharge_mw"] - sched["charge_mw"]
    pnl = float(pulp.value(prob.objective) or 0.0)
    return sched, pnl


def run(days: int = 365, re_ratios=(0.5, 1.0, 1.5, 2.0),
        mix=("solar", "hybrid"), bess: Bess | None = None) -> dict:
    """Revenue under the rule vs without it, across RE build ratios."""
    bess = bess or Bess(power_mw=1.0, energy_mwh=2.0)   # per MW, 2h (rule floor)
    dam = store.read("dam_price")["mcp_rs_mwh"]
    lo = dam.index.max().normalize() - pd.Timedelta(days=days)
    dam = dam[dam.index >= lo]
    day_list = [d for d, g in dam.groupby(dam.index.date) if len(g) == 96]
    if not day_list:
        raise RuntimeError("no complete price days in window")

    prof = re_profile(pd.Timestamp(day_list[0]), pd.Timestamp(day_list[-1]))
    prof = prof.resample("15min").interpolate(limit=8)

    # baseline: unrestricted merchant dispatch, same asset, same days
    base = {}
    for d in day_list:
        pr = dam[dam.index.date == d]
        _s, pnl = optimize_dispatch(pr, bess)
        base[d] = pnl
    base_annual = float(np.mean(list(base.values())) * 365)

    results = []
    for m in mix:
        for ratio in re_ratios:
            tot, n_days, curtail, charged = 0.0, 0, 0.0, 0.0
            for d in day_list:
                pr = dam[dam.index.date == d]
                sl = prof.reindex(pr.index)
                if sl["solar_pu"].isna().all():
                    continue
                sl = sl.fillna(0.0)
                re_mw = (sl["solar_pu"] * ratio if m == "solar"
                         else (sl["solar_pu"] * ratio * 0.7
                               + sl["wind_pu"] * ratio * 0.3))
                sch, pnl = dispatch_re_bound(pr, re_mw.to_numpy(), bess)
                tot += pnl
                n_days += 1
                charged += float(sch["charge_mw"].sum() * BLOCK_H)
                curtail += float((re_mw.to_numpy()
                                  - sch["charge_mw"].to_numpy()).clip(min=0).sum()
                                 * BLOCK_H)
            if not n_days:
                continue
            annual = tot / n_days * 365
            results.append({
                "mix": m, "re_mw_per_bess_mw": ratio,
                "annual_rs_per_mw": round(annual, 0),
                "vs_unrestricted_pct": round((annual / base_annual - 1) * 100, 1),
                "charged_mwh_per_mw_yr": round(charged / n_days * 365, 1),
                "re_unused_mwh_per_mw_yr": round(curtail / n_days * 365, 1),
                "days": n_days,
            })

    best = max(results, key=lambda r: r["annual_rs_per_mw"]) if results else None
    return {
        "window": {"from": str(day_list[0]), "to": str(day_list[-1]),
                   "n_days": len(day_list)},
        "site": {"lat": RE_LAT, "lon": RE_LON,
                 "note": "western Rajasthan solar belt — where merchant BESS is being built"},
        "asset": {"power_mw": bess.power_mw, "energy_mwh": bess.energy_mwh,
                  "duration_h": bess.energy_mwh / bess.power_mw},
        "unrestricted_annual_rs_per_mw": round(base_annual, 0),
        "scenarios": results,
        "best_restricted": best,
        "headline_pct": best["vs_unrestricted_pct"] if best else None,
        "caveats": [
            "prices are IEX DAM pan-India MCP; a co-located Rajasthan asset would "
            "settle on its own bidding zone if India ever moves to nodal pricing",
            "does NOT price the mandated RE plant itself — that is a hybrid-project "
            "question with its own PPA, curtailment and land assumptions. This "
            "isolates the DISPATCH cost of the charging constraint alone",
            "RE output is a physical twin driven by reanalysis weather, not metered "
            "generation; it carries the twin's error, not a plant's actual record",
            "assumes surplus RE beyond charging is not monetised here — reported "
            "separately as re_unused so it can be valued with a real PPA price",
        ],
    }


if __name__ == "__main__":
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 365
    r = run(days=days)
    print(f"ISTS charging restriction — {r['window']['from']} -> {r['window']['to']} "
          f"({r['window']['n_days']} days)")
    print(f"asset {r['asset']['power_mw']:.0f} MW / {r['asset']['energy_mwh']:.0f} MWh "
          f"({r['asset']['duration_h']:.0f}h)")
    print(f"\nunrestricted merchant: Rs {r['unrestricted_annual_rs_per_mw']/1e5:,.1f} "
          f"lakh/MW/yr\n")
    print(f"{'mix':8s}{'RE:BESS':>9s}{'Rs lakh/MW/yr':>15s}{'vs unrestricted':>17s}"
          f"{'RE unused MWh':>15s}")
    for s in r["scenarios"]:
        print(f"{s['mix']:8s}{s['re_mw_per_bess_mw']:9.1f}"
              f"{s['annual_rs_per_mw']/1e5:15.1f}{s['vs_unrestricted_pct']:16.1f}%"
              f"{s['re_unused_mwh_per_mw_yr']:15.1f}")
    CACHE.write_text(json.dumps(r, indent=2, default=float))
    print(f"\nwrote {CACHE.name}")
