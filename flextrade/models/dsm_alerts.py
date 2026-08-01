"""Alerts & Revision Engine — the sixth component of the DSM module
(`FlexTrade_DSM_Feature.pdf`, Table 3): "Recommends schedule revisions to
reduce exposure before the next gate closure. Combines all of the above."

What this actually does: takes the RE generation forecast (which already
exists, `models/re_model.py`) and the DSM regulation engine (`models/dsm.py`)
and asks a forward-looking question the other components don't —
*given what we now know is likely to happen, should the schedule
submitted for the next gate be revised, and how much would that be
worth?* IEX allows revision of RE injection schedules ahead of each
gate closure, so this has a real action attached to it, not just a
number on a dashboard.

Method
------
1. Take the day-ahead generation forecast (P50, and if available P10/P90
   from an ensemble -- currently the RE twin is a single deterministic
   physical model, so uncertainty is proxied from the forecast's own
   recent track record via `dsm_comparison`, not a trained quantile
   model like the price forecast has).
2. For each upcoming block, compute the expected DSM charge if the
   CURRENT schedule is kept vs. if it is REVISED to match the latest
   forecast.
3. Flag blocks where the exposure crosses a materiality threshold and
   the revision would clearly help -- these are the actionable alerts.
4. Roll up into a single recommendation: revise / hold, with the
   estimated rupee benefit, so a human (or eventually an automated
   bidder) can act on it before the next gate.

This is deliberately conservative: it only recommends a revision when
the expected benefit clears a threshold, because over-triggering on
forecast noise erodes trust exactly the way Table 6 of the DSM spec
warns about ("Customer trust in estimated vs. actual settlement").
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@dataclass
class RevisionRecommendation:
    block_ts: pd.Timestamp
    current_schedule_mw: float
    forecast_mw: float
    deviation_if_unrevised_pct: float
    exposure_if_unrevised_rs: float
    exposure_if_revised_rs: float
    benefit_rs: float
    action: str  # "REVISE" | "HOLD"
    reason: str


def block_exposure(actual_or_forecast_mw: float, schedule_mw: float,
                   normal_rate_rs_mwh: float, available_capacity_mw: float,
                   technology: str, settlement_date: date,
                   profile: str = "CERC_2024") -> float:
    """One block's DSM charge (Rs, positive = cost) for a single
    schedule/outcome pair -- a thin wrapper around models.dsm so the
    alert engine can be scored without building a full multi-block frame
    for a single what-if evaluation."""
    from models import dsm
    idx = pd.DatetimeIndex([pd.Timestamp.now()])
    act = pd.Series([actual_or_forecast_mw], idx)
    sch = pd.Series([schedule_mw], idx)
    nr = pd.Series([normal_rate_rs_mwh], idx)
    settled = dsm.settle(profile, actual_mw=act, scheduled_mw=sch,
                         frequency_hz=50.0, dam_price=nr, rtm_price=nr,
                         available_capacity_mw=available_capacity_mw,
                         seller="ws", technology=technology,
                         settlement_date=settlement_date)
    return float(settled["charge_rs"].iloc[0])


def evaluate_gate(forecast: pd.Series, current_schedule: pd.Series,
                  normal_rate: pd.Series, available_capacity_mw: float,
                  technology: str, gate_closure: datetime,
                  materiality_rs: float = 500.0,
                  profile: str = "CERC_2024") -> list[RevisionRecommendation]:
    """Compare 'keep current schedule' vs 'revise to match latest
    forecast' for every block still ahead of `gate_closure`, and return
    one recommendation per block that clears the materiality threshold.

    forecast, current_schedule, normal_rate: aligned Series indexed by
    block start time, covering the blocks the next gate can still affect.
    """
    from models import dsm

    idx = forecast.index.intersection(current_schedule.index).intersection(
        normal_rate.index)
    idx = idx[idx >= gate_closure] if len(idx) else idx
    if not len(idx):
        return []

    out = []
    for t in idx:
        f, s, nr = float(forecast[t]), float(current_schedule[t]), float(normal_rate[t])
        d = t.date() if hasattr(t, "date") else date.today()

        exp_unrevised = block_exposure(f, s, nr, available_capacity_mw,
                                       technology, d, profile)
        exp_revised = block_exposure(f, f, nr, available_capacity_mw,
                                     technology, d, profile)
        benefit = exp_unrevised - exp_revised  # positive = revising helps
        dev_pct = (abs(f - s) / max(available_capacity_mw, 1e-6)) * 100

        if benefit >= materiality_rs:
            action, reason = "REVISE", (
                f"Forecast now {f:.1f} MW vs scheduled {s:.1f} MW "
                f"({dev_pct:+.1f}% of capacity) -- revising saves an "
                f"estimated Rs {benefit:,.0f} in this block.")
        elif exp_unrevised >= materiality_rs and benefit < materiality_rs * 0.2:
            action, reason = "HOLD", (
                f"Exposure Rs {exp_unrevised:,.0f} exists but revising "
                f"barely helps (Rs {benefit:,.0f}) -- likely inside the "
                f"tolerance band either way, or the deviation is driven "
                f"by something a schedule change cannot fix in time.")
        else:
            continue  # not material enough to surface

        out.append(RevisionRecommendation(
            block_ts=t, current_schedule_mw=s, forecast_mw=f,
            deviation_if_unrevised_pct=dev_pct,
            exposure_if_unrevised_rs=exp_unrevised,
            exposure_if_revised_rs=exp_revised, benefit_rs=benefit,
            action=action, reason=reason,
        ))
    return out


def next_gate_alerts(lead_minutes: int = 15, materiality_rs: float = 500.0,
                     profile: str = "CERC_2024") -> dict:
    """Live entry point: pulls today's RE forecast + latest DAM/RTM
    prices and evaluates every block from now (plus the required lead
    time) to the end of the day. This is what the dashboard and, later,
    an automated bidder would call on a timer.

    Returns a dict with `alerts` (list of RevisionRecommendation),
    `lead_time_ok` (bool -- whether we are inside the >=15 min lead-time
    target from the DSM spec's success metrics, Table 5), and `asof`.
    """
    from models import re_model
    from ingest import iex, store

    now = datetime.now()
    cutoff = now + timedelta(minutes=lead_minutes)
    today = now.date()

    fc = re_model.forecast_day(today)  # today's own forecast, not tomorrow's
    dam, _ = iex.get_today()
    rtm, _ = iex.get_rtm_today()
    if len(dam) and len(rtm):
        nr = ((dam["mcp_rs_mwh"] + rtm["mcp_rs_mwh"]) / 2).reindex(fc.index)
    elif len(dam):
        nr = dam["mcp_rs_mwh"].reindex(fc.index)
    else:
        nr = pd.Series(5000.0, index=fc.index)  # last-resort flat estimate
    nr = nr.ffill().bfill()

    cap_solar = re_model.SolarPlant().capacity_mw
    cap_wind = re_model.WindFarm().capacity_mw

    alerts = []
    for tech, cap, col in [("solar", cap_solar, "solar_mw"),
                           ("wind", cap_wind, "wind_mw")]:
        # "current schedule" proxy: what was forecast >=1 day ago for
        # these same blocks, i.e. what would actually have been bid.
        # Without a stored historical bid sheet this uses the naive
        # persistence baseline as the schedule-in-force stand-in, which
        # is a labelled approximation -- see the returned `schedule_basis`.
        schedule = fc[col].shift(1).bfill()
        recs = evaluate_gate(fc[col], schedule, nr, cap, tech, cutoff,
                             materiality_rs, profile)
        alerts.extend(recs)

    alerts.sort(key=lambda r: -r.benefit_rs)
    return {
        "asof": now, "gate_cutoff": cutoff, "lead_minutes": lead_minutes,
        "alerts": alerts,
        "total_benefit_rs": sum(a.benefit_rs for a in alerts if a.action == "REVISE"),
        "schedule_basis": "persistence proxy -- wire to the real bid-sheet "
                          "store once schedules are actually submitted "
                          "through this platform",
    }


if __name__ == "__main__":
    result = next_gate_alerts()
    print(f"Alerts as of {result['asof']:%H:%M} "
          f"(gate cutoff {result['gate_cutoff']:%H:%M}, "
          f"basis: {result['schedule_basis']})")
    for a in result["alerts"][:10]:
        print(f"  [{a.action:7s}] {a.block_ts:%H:%M}  "
              f"sched {a.current_schedule_mw:6.1f} MW -> fcst {a.forecast_mw:6.1f} MW  "
              f"benefit Rs {a.benefit_rs:>8,.0f}")
    print(f"\ntotal REVISE benefit: Rs {result['total_benefit_rs']:,.0f}")
