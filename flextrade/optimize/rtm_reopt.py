"""Intraday RTM re-optimization — the second revenue stream on the same asset.

After the DAM clears (13:00 on D-1), the 96-block DAM position is financially
firm. But the battery still has physical flexibility around that position on
delivery day, and IEX's Real-Time Market (48 half-hourly auctions, gate
closure ~1 hour before delivery) lets us monetise it:

    physical dispatch[t] = DAM position[t] + RTM trade[t]

Any deviation from the DAM position that we *trade in the RTM* is settled at
the RTM clearing price instead of leaking into DSM charges. So the LP here
re-optimises the remaining blocks of today:

  - decision vars: physical charge/discharge for every still-tradeable block
    (>= LEAD_MIN minutes away, respecting RTM gate closure)
  - the RTM trade per block is (physical net - DAM net); its revenue is
    valued at the expected RTM price
  - SoC starts from the position implied by executing the DAM plan up to now
    and must end the day where the DAM plan would have ended (so tomorrow's
    DAM plan stays feasible)
  - DAM revenue is sunk and excluded; the objective is pure incremental RTM
    revenue minus incremental degradation

Expected RTM price for remaining blocks — honest construction, labelled per
block in the output. Blocks already cleared use the ACTUAL RTM price; the
rest come from the trained cascade in models/rtm_model.py (intraday model
within reach of gate closure, sameday champion beyond it). Until that model
existed this file scaled today's DAM curve by an hour-of-day RTM/DAM ratio;
that path survives as a labelled fallback, because on a held-out 60 days the
model beat it on both the level (WAPE 26.6% vs 33.0%) and — the part that
decides whether we trade — the sign of the spread (76.6% vs 60.2%).
"""
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from ingest import store  # noqa: E402
from optimize.dispatch import BLOCK_H, Bess  # noqa: E402

import pulp  # noqa: E402

OUT = HERE.parent / "output"
LEAD_MIN = 60  # blocks starting closer than this are past RTM gate closure


def _dam_plan_today(today: date) -> pd.DataFrame | None:
    """The committed DAM schedule for today's delivery, if we planned it."""
    f = OUT / f"plan_{today}.csv"
    if not f.exists():
        return None
    plan = pd.read_csv(f, parse_dates=["ts"], index_col="ts")
    need = {"charge_mw", "discharge_mw", "soc_mwh"}
    return plan if need.issubset(plan.columns) else None


def _prices_today(today: date) -> tuple[pd.Series, pd.Series]:
    dam = store.read("dam_price")
    dam_t = dam[dam.index.date == today]["mcp_rs_mwh"] if len(dam) else pd.Series(dtype=float)
    if not len(dam_t):
        # today's DAM cleared yesterday 13:00 — fetch it if we haven't yet
        try:
            from ingest import iex
            iex.get_today()
            dam = store.read("dam_price")
            dam_t = dam[dam.index.date == today]["mcp_rs_mwh"]
        except Exception:
            pass
    rtm = store.read("rtm_price")
    rtm_t = rtm[rtm.index.date == today]["mcp_rs_mwh"] if len(rtm) else pd.Series(dtype=float)
    return dam_t, rtm_t


RATIO_LOOKBACK_DAYS = 120
RATIO_CLIP = (0.3, 3.0)


def _rtm_dam_ratio_profile(today: date) -> tuple[pd.Series, str, dict]:
    """RTM/DAM ratio BY HOUR OF DAY, from stored history.

    A single scalar ratio was the original implementation and it was a poor
    description of reality. With a year of RTM history now backfilled we can
    measure it: the ratio's median is 1.00 but its 5th-95th percentile spans
    0.36-1.79, and the *median by hour* runs from 0.91 at 18:00 to 1.20 at
    10:00 — a 1.31x swing that one number cannot represent. RTM also clears
    ABOVE DAM in 43% of blocks, so the two markets genuinely diverge rather
    than one tracking the other.

    Returns (ratio per hour 0-23, provenance label, dispersion stats). The
    dispersion is carried through to the UI so the projection is never
    mistaken for a forecast.
    """
    dam = store.read("dam_price")
    rtm = store.read("rtm_price")
    if len(dam) and len(rtm):
        lo = pd.Timestamp(today - timedelta(days=RATIO_LOOKBACK_DAYS))
        d = dam.loc[dam.index >= lo, "mcp_rs_mwh"]
        r = rtm.loc[rtm.index >= lo, "mcp_rs_mwh"]
        common = d.index.intersection(r.index)
        if len(common) >= 96 * 14:
            ratio = (r.loc[common] / d.loc[common].replace(0, pd.NA)).astype(float)
            ratio = ratio.clip(*RATIO_CLIP).dropna()
            byhour = ratio.groupby(ratio.index.hour).median()
            byhour = byhour.reindex(range(24)).interpolate(
                limit_direction="both").fillna(1.0)
            stats = {
                "n_blocks": int(len(ratio)),
                "median": round(float(ratio.median()), 3),
                "p05": round(float(ratio.quantile(0.05)), 3),
                "p95": round(float(ratio.quantile(0.95)), 3),
                "hour_min": round(float(byhour.min()), 3),
                "hour_max": round(float(byhour.max()), 3),
            }
            return byhour, (f"hour-of-day median over the last "
                            f"{RATIO_LOOKBACK_DAYS} days "
                            f"({len(ratio):,} paired blocks)"), stats
    return (pd.Series(1.0, index=range(24)),
            "no RTM history — ratio 1.0 assumed", {})


def expected_rtm_prices(today: date, now: pd.Timestamp | None = None
                        ) -> tuple[pd.Series, dict]:
    """Best-available RTM price per block for the whole day, labelled.

    Prefers the trained RTM model cascade (models/rtm_model.py): cleared
    blocks use the ACTUAL price, blocks within reach of gate closure use the
    intraday model, and the rest use the sameday champion.

    Falls back to the original hour-of-day RTM/DAM ratio if the models have
    not been trained yet. That ratio was the production method until the
    intraday model beat it on a held-out 60 days -- WAPE 26.6% vs 33.0%, and
    spread direction 76.6% vs 60.2% -- so the fallback is a real degradation
    and says so in its provenance label rather than pretending otherwise.
    """
    try:
        from models import rtm_model
        curve, meta = rtm_model.serve_curve(today, now)
        if curve.notna().all():
            return curve, {
                "ratio_basis": meta["basis"],
                "price_source": "rtm model cascade",
                "blocks_by_source": meta["blocks_by_source"],
                "blocks_actual_rtm": meta["blocks_by_source"].get("actual", 0),
                "blocks_projected": int(len(curve))
                - meta["blocks_by_source"].get("actual", 0),
                "model_wape_pct": meta.get("intraday_wape_pct"),
                "model_direction_pct": meta.get("intraday_direction_pct"),
                "sameday_champion": meta.get("sameday_champion"),
            }
    except Exception as e:
        print(f"  (RTM model unavailable: {e}; falling back to ratio profile)")

    dam_t, rtm_t = _prices_today(today)
    if not len(dam_t):
        raise RuntimeError(f"no DAM prices stored for {today}")
    byhour, basis, stats = _rtm_dam_ratio_profile(today)
    scale = pd.Series(dam_t.index.hour, index=dam_t.index).map(byhour).astype(float)
    exp = dam_t * scale
    common = exp.index.intersection(rtm_t.index)
    exp.loc[common] = rtm_t.loc[common]          # cleared = actual
    return exp.rename("rtm_price"), {
        "ratio": stats.get("median", 1.0),
        "ratio_basis": f"FALLBACK -- {basis}",
        "price_source": "hour-of-day ratio (model not trained)",
        "ratio_by_hour": {int(h): round(float(v), 3) for h, v in byhour.items()},
        "ratio_dispersion": stats,
        "blocks_actual_rtm": int(len(common)),
        "blocks_projected": int(len(exp) - len(common)),
    }


def reoptimize(now: pd.Timestamp | None = None,
               bess: Bess = Bess()) -> dict:
    """Re-optimise the rest of today against the RTM. Returns a full report."""
    now = pd.Timestamp.now() if now is None else pd.Timestamp(now)
    today = now.date()

    plan = _dam_plan_today(today)
    rtm_price, price_meta = expected_rtm_prices(today, now)

    idx = rtm_price.index
    if plan is not None:
        plan = plan.reindex(idx)
        dam_net = (plan["discharge_mw"] - plan["charge_mw"]).fillna(0.0)
        dam_basis = f"committed DAM plan (plan_{today}.csv)"
    else:
        dam_net = pd.Series(0.0, index=idx)
        dam_basis = "no DAM plan for today — pure-RTM mode (position = 0)"

    eta = np.sqrt(bess.round_trip_eff)
    soc0 = bess.soc0_frac * bess.energy_mwh
    soc_min = bess.soc_min_frac * bess.energy_mwh

    tradeable = idx >= (now + pd.Timedelta(minutes=LEAD_MIN)).floor("15min")
    T = list(np.where(tradeable)[0])
    if not T:
        return {"status": "day complete", "asof": str(now.floor("min")),
                "tradeable_blocks": 0, "dam_basis": dam_basis, **price_meta}

    # Anchor the tradeable window on the DAM plan's OWN SoC trajectory: the
    # window starts at the plan's SoC in the block just before it opens, and
    # must return to the plan's END-of-day SoC. This keeps the zero-deviation
    # baseline (= keep following the plan) feasible, so any uplift is genuine.
    # An earlier version re-integrated SoC from the day-start value and forced
    # a return to THAT — which, once the morning discharge had drained the
    # battery and the refill blocks were locked, forced the LP to buy energy
    # and produced a spurious NEGATIVE uplift. If no plan carries a SoC path
    # (pure-RTM mode), fall back to the day-start SoC both ends.
    plan_soc = plan["soc_mwh"] if (plan is not None and "soc_mwh" in plan
                                   and plan["soc_mwh"].notna().any()) else None
    if plan_soc is not None:
        before = idx < idx[T[0]]
        soc_start = float(plan_soc[before].dropna().iloc[-1]) if before.any() \
            and plan_soc[before].notna().any() else soc0
        soc_target = float(plan_soc.dropna().iloc[-1])
    else:
        soc_start, soc_target = soc0, soc0
    soc_start = float(np.clip(soc_start, soc_min, bess.energy_mwh))
    soc_now = soc_start

    p = rtm_price.values.astype(float)
    dnet = dam_net.values.astype(float)

    prob = pulp.LpProblem("rtm_reopt", pulp.LpMaximize)
    ch = pulp.LpVariable.dicts("ch", T, 0, bess.power_mw)
    dis = pulp.LpVariable.dicts("dis", T, 0, bess.power_mw)
    soc = pulp.LpVariable.dicts("soc", range(len(T) + 1), soc_min, bess.energy_mwh)

    # incremental objective: RTM revenue on the deviation from the DAM
    # position, minus degradation on physical throughput (DAM revenue sunk)
    prob += pulp.lpSum(
        BLOCK_H * p[t] * (dis[t] - ch[t] - dnet[t])
        - BLOCK_H * bess.degradation_rs_mwh * (ch[t] + dis[t])
        for t in T)

    prob += soc[0] == soc_start
    for k, t in enumerate(T):
        prob += soc[k + 1] == soc[k] + BLOCK_H * (ch[t] * eta - dis[t] / eta)
    prob += soc[len(T)] >= soc_target  # hand tomorrow the planned position

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"RTM LP status: {pulp.LpStatus[prob.status]}")

    # baseline = keep executing the DAM plan (RTM trade 0, physical = DAM)
    base_deg = float(BLOCK_H * bess.degradation_rs_mwh
                     * np.abs(dnet[T]).sum())
    obj = float(pulp.value(prob.objective))
    # the plan-anchored formulation guarantees uplift >= 0; clamp float noise
    uplift = max(obj + base_deg, 0.0)  # both sides carry degradation; net it out

    rows = []
    for k, t in enumerate(T):
        phys = dis[t].value() - ch[t].value()
        trade = phys - dnet[t]
        rows.append({
            "ts": idx[t], "rtm_price": round(p[t], 0),
            "dam_net_mw": round(dnet[t], 2), "physical_net_mw": round(phys, 2),
            "rtm_trade_mw": round(trade, 2),
            "side": "SELL" if trade > 0.01 else ("BUY" if trade < -0.01 else "-"),
            "soc_mwh": round(soc[k + 1].value(), 2),
        })
    sched = pd.DataFrame(rows).set_index("ts")
    sched.to_csv(OUT / "rtm_reopt_latest.csv", index_label="ts")

    n_trades = int((sched["side"] != "-").sum())
    return {
        "status": "ok", "asof": str(now.floor("min")),
        "delivery_day": str(today), "dam_basis": dam_basis,
        "soc_now_mwh": round(soc_now, 2),
        "tradeable_blocks": len(T), "n_trades": n_trades,
        "expected_rtm_uplift_rs": round(uplift, 0),
        "schedule": sched, **price_meta,
    }


if __name__ == "__main__":
    r = reoptimize()
    print(f"RTM re-optimization  asof {r['asof']}  ({r['dam_basis']})")
    if r["status"] != "ok":
        print(f"  {r['status']}")
    else:
        print(f"  price basis: {r['ratio_basis']}")
        print(f"    {r['blocks_actual_rtm']} actual RTM blocks + "
              f"{r['blocks_projected']} projected"
              + (f" | intraday model WAPE {r['model_wape_pct']}%, "
                 f"direction {r['model_direction_pct']}%"
                 if r.get('model_wape_pct') is not None else ""))
        print(f"  SoC now {r['soc_now_mwh']} MWh | {r['tradeable_blocks']} "
              f"tradeable blocks | {r['n_trades']} RTM trades")
        print(f"  expected incremental RTM uplift: Rs {r['expected_rtm_uplift_rs']:,.0f}")
        s = r["schedule"]
        print(s[s["side"] != "-"].head(12).to_string())
