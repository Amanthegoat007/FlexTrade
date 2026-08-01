"""Three-way settlement co-optimization: DAM + RTM + DSM as simultaneous
channels — the deviation itself becomes a priced decision.

The insight
-----------
Once the DAM position is firm, a physical deviation from it can be closed
two ways: trade it in the RTM, or let it settle through the DSM at the
Normal Rate (NR = 1/3 DAM + 1/3 RTM + 1/3 ancillary). These prices are
NOT equal, and they invert: on spike evenings RTM clears far above NR
(paying NR for a shortfall beats buying it back in RTM); in gluts RTM can
fall below the over-injection credit. Today operators treat DSM purely as
a penalty to avoid; treating it as the third settlement channel with a
price is, to our knowledge, not productized in India.

Deliberately conservative DSM leg (no free lunch)
-------------------------------------------------
CERC 2024 tolerance bands could be read as "deviation within the band is
free", which an optimizer would exploit as free energy — a modeling
artifact, not a trade. We refuse it:
  - under-delivery PAYS full NR from the FIRST MWh (no free band), plus a
    safety factor;
  - over-delivery is PAID at only 90% of NR within the band, NOTHING
    beyond it (capped in the LP).
So any uplift the LP finds survives a harsher settlement than the regs
themselves. The exact dsm.py engine then re-settles the chosen deviation
profile as verification, and both numbers are reported.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pulp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from optimize import rtm_reopt  # noqa: E402
from optimize.dispatch import BLOCK_H, Bess  # noqa: E402

OUT = HERE.parent / "output"

OVER_CREDIT_FRAC = 0.90   # over-injection paid at 90% of NR within band
UNDER_SAFETY = 1.05       # shortfall charged 105% of NR, from first MWh
BAND_FRAC = 0.10          # over-injection credited only within 10% band


def cooptimize(now: pd.Timestamp | None = None, bess: Bess = Bess()) -> dict:
    """Re-optimize the rest of today across RTM AND DSM channels.
    Returns schedule + uplift vs the RTM-only re-optimization."""
    now = pd.Timestamp.now() if now is None else pd.Timestamp(now)
    today = now.date()

    # reuse the RTM engine's inputs so the comparison is apples-to-apples
    base = rtm_reopt.reoptimize(now=now, bess=bess)
    if base.get("status") != "ok":
        return {"status": base.get("status", "unavailable"), **{
            k: base[k] for k in ("asof", "dam_basis") if k in base}}

    rtm_price, price_meta = rtm_reopt.expected_rtm_prices(today)
    from ingest import store
    dam = store.read("dam_price")
    dam_t = dam[dam.index.date == today]["mcp_rs_mwh"].reindex(rtm_price.index)
    # NR = 1/3 DAM + 1/3 RTM + 1/3 ancillary(proxied by RTM) — same proxy,
    # same flag, as models/dsm.py
    nr = (dam_t + 2.0 * rtm_price) / 3.0

    plan = rtm_reopt._dam_plan_today(today)
    idx = rtm_price.index
    if plan is not None:
        plan = plan.reindex(idx)
        dam_net = (plan["discharge_mw"] - plan["charge_mw"]).fillna(0.0)
    else:
        dam_net = pd.Series(0.0, index=idx)

    eta = np.sqrt(bess.round_trip_eff)
    soc0 = bess.soc0_frac * bess.energy_mwh
    soc_min = bess.soc_min_frac * bess.energy_mwh

    tradeable = idx >= (now + pd.Timedelta(minutes=rtm_reopt.LEAD_MIN)).floor("15min")
    T = list(np.where(tradeable)[0])
    if not T:
        return {"status": "day complete", "asof": str(now.floor("min")),
                "dam_basis": base["dam_basis"], **price_meta}

    # anchor on the DAM plan's own SoC path so the zero-deviation baseline is
    # feasible and uplift can't go spuriously negative (same fix as rtm_reopt)
    plan_soc = plan["soc_mwh"] if (plan is not None and "soc_mwh" in plan
                                   and plan["soc_mwh"].notna().any()) else None
    if plan_soc is not None:
        before = idx < idx[T[0]]
        soc_now = float(plan_soc[before].dropna().iloc[-1]) if before.any() \
            and plan_soc[before].notna().any() else soc0
        soc_target = float(plan_soc.dropna().iloc[-1])
    else:
        soc_now, soc_target = soc0, soc0
    soc_now = float(np.clip(soc_now, soc_min, bess.energy_mwh))
    p_rtm = rtm_price.values.astype(float)
    p_nr = nr.values.astype(float)
    dnet = dam_net.values.astype(float)
    band = np.maximum(BAND_FRAC * np.abs(dnet), BAND_FRAC * bess.power_mw)

    prob = pulp.LpProblem("threeway", pulp.LpMaximize)
    ch = pulp.LpVariable.dicts("ch", T, 0, bess.power_mw)
    dis = pulp.LpVariable.dicts("dis", T, 0, bess.power_mw)
    r = pulp.LpVariable.dicts("rtm", T, -bess.power_mw, bess.power_mw)
    dpos = pulp.LpVariable.dicts("dev_over", T, 0)
    dneg = pulp.LpVariable.dicts("dev_under", T, 0)
    soc = pulp.LpVariable.dicts("soc", range(len(T) + 1), soc_min, bess.energy_mwh)

    prob += pulp.lpSum(
        BLOCK_H * (p_rtm[t] * r[t]
                   + OVER_CREDIT_FRAC * p_nr[t] * dpos[t]
                   - UNDER_SAFETY * p_nr[t] * dneg[t])
        - BLOCK_H * bess.degradation_rs_mwh * (ch[t] + dis[t])
        for t in T)

    prob += soc[0] == soc_now
    for k, t in enumerate(T):
        # physical = DAM position + RTM trade + net deviation
        prob += dis[t] - ch[t] == dnet[t] + r[t] + dpos[t] - dneg[t]
        # BOTH deviation legs capped at the tolerance band. The first
        # version capped only the credit side and the LP immediately
        # discovered the gaming strategy (sell 20 MW in RTM, deliver
        # nothing, settle the whole shortfall through DSM) — profitable
        # under the rates, and exactly the deliberate-deviation conduct
        # CERC polices. Compliance is a hard constraint: deviations stay
        # inside the band, where they are defensible as forecast-error
        # management, never a trading channel of their own.
        prob += dpos[t] <= band[t]
        prob += dneg[t] <= band[t]
        prob += soc[k + 1] == soc[k] + BLOCK_H * (ch[t] * eta - dis[t] / eta)
    prob += soc[len(T)] >= soc_target

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"three-way LP: {pulp.LpStatus[prob.status]}")

    obj = float(pulp.value(prob.objective))
    base_deg = float(BLOCK_H * bess.degradation_rs_mwh * np.abs(dnet[T]).sum())
    # plan-anchored + strictly-more-channels than RTM-only, so guaranteed
    # >= rtm_only >= 0; clamp float noise (avoids a spurious "-0" on screen)
    rtm_only = base["expected_rtm_uplift_rs"]
    uplift = max(obj + base_deg, rtm_only)

    rows = []
    for k, t in enumerate(T):
        rows.append({
            "ts": idx[t], "rtm_price": round(p_rtm[t], 0),
            "normal_rate": round(p_nr[t], 0),
            "dam_net_mw": round(dnet[t], 2),
            "physical_net_mw": round(dis[t].value() - ch[t].value(), 2),
            "rtm_trade_mw": round(r[t].value(), 2),
            "dsm_over_mw": round(dpos[t].value(), 2),
            "dsm_under_mw": round(dneg[t].value(), 2),
            "soc_mwh": round(soc[k + 1].value(), 2),
        })
    sched = pd.DataFrame(rows).set_index("ts")
    sched.to_csv(OUT / "threeway_latest.csv", index_label="ts")

    dsm_blocks = sched[(sched["dsm_over_mw"] > 0.01) | (sched["dsm_under_mw"] > 0.01)]
    # verification: settle the chosen deviation profile with the exact engine
    verified_dsm_rs = None
    try:
        from models import dsm
        dev_actual = dam_net.copy()
        dev_actual.loc[sched.index] = sched["physical_net_mw"] - sched["rtm_trade_mw"]
        settled = dsm.settle_2024(
            actual_mw=dev_actual.loc[sched.index],
            scheduled_mw=dam_net.loc[sched.index],
            frequency_hz=50.0, dam_price=dam_t.loc[sched.index],
            rtm_price=rtm_price.loc[sched.index],
            available_capacity_mw=bess.power_mw, seller="gen")
        verified_dsm_rs = round(float(-settled["charge_rs"].sum()), 0)
    except Exception as e:
        verified_dsm_rs = f"verification failed: {str(e)[:80]}"

    return {
        "status": "ok", "asof": str(now.floor("min")),
        "delivery_day": str(today), "dam_basis": base["dam_basis"],
        "tradeable_blocks": len(T),
        "rtm_only_uplift_rs": rtm_only,
        "threeway_uplift_rs": round(uplift, 0),
        "dsm_leg_added_rs": round(uplift - rtm_only, 0),
        "dsm_blocks_used": int(len(dsm_blocks)),
        "lp_rates_note": (f"compliant envelope: BOTH deviation legs capped at the "
                          f"{BAND_FRAC:.0%} tolerance band (beyond-band deviation is "
                          f"gaming, which CERC polices); shortfall pays {UNDER_SAFETY:.0%} "
                          f"NR from first MWh, over-injection earns {OVER_CREDIT_FRAC:.0%} NR"),
        "exact_engine_dsm_rs": verified_dsm_rs,
        "ancillary_note": "NR ancillary leg proxied by RTM (no public feed), as everywhere",
        "schedule": sched, **price_meta,
    }


if __name__ == "__main__":
    import sys as _s
    now = pd.Timestamp(_s.argv[1]) if len(_s.argv) > 1 else None
    r = cooptimize(now=now)
    if r["status"] != "ok":
        print(f"three-way: {r['status']}")
    else:
        print(f"Three-way co-optimization  asof {r['asof']}  ({r['dam_basis']})")
        print(f"  RTM-only uplift  : Rs {r['rtm_only_uplift_rs']:>10,.0f}")
        print(f"  three-way uplift : Rs {r['threeway_uplift_rs']:>10,.0f}")
        print(f"  DSM leg added    : Rs {r['dsm_leg_added_rs']:>10,.0f} "
              f"across {r['dsm_blocks_used']} blocks")
        print(f"  ({r['lp_rates_note']})")
        print(f"  exact-engine check on chosen deviations: {r['exact_engine_dsm_rs']}")
        s = r["schedule"]
        used = s[(s["dsm_over_mw"] > 0.01) | (s["dsm_under_mw"] > 0.01)]
        if len(used):
            print(used.head(10).to_string())
