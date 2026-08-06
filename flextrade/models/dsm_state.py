"""What deviation actually costs a real state — priced on measured drawal.

models/dsm_forecast.py had to price a HYPOTHETICAL generator, because no
schedule-vs-drawal series existed for us. UP's own load despatch centre
publishes one, so this prices a real book:

    schedule (inter-state) · drawal (inter-state) · deviation
    every poll, for a ~10 GW schedule against a ~25 GW system

Measured over the first days of collection, UP runs a mean absolute deviation
of 321 MW against a tolerance band of min(10% of schedule, 100 MW) — so it is
outside the band on most snapshots, which is exactly why DSM is a real line
item for a DISCOM and exactly the market that is compelled to buy a forecast.

Priced through the same versioned CERC engine the rest of the platform uses
(models/dsm.py), with the Normal Rate built from live DAM and RTM, so the
charge moves with the market rather than a flat assumed rate.

WHAT THIS IS NOT
----------------
It is a SAMPLE, not a settlement. We hold snapshots at whatever cadence the
collector managed, not all 96 blocks of every day, and the published deviation
is an instantaneous reading rather than a block average. So the per-snapshot
charge is real and the daily total is an ESTIMATE scaled from the sample —
labelled as such everywhere it appears. A real settlement needs SLDC's own
block-wise account, which is not public.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from ingest import store  # noqa: E402
from models import dsm  # noqa: E402

OUT = HERE.parent / "output"
BLOCK_H = 0.25

# CERC general-seller band: min(10% of schedule, 100 MW)
BAND_PCT = 10.0
BAND_CAP_MW = 100.0


def _normal_rate_series() -> pd.Series:
    """Reg 14 Normal Rate per 15-min block, from live DAM and RTM."""
    dam = store.read("dam_price")["mcp_rs_mwh"]
    rtm = store.read("rtm_price")["mcp_rs_mwh"]
    df = pd.DataFrame({"dam": dam, "rtm": rtm})
    df["rtm"] = df["rtm"].fillna(df["dam"])
    # ancillary has no public feed; proxied by RTM and flagged in the output
    return ((df["dam"] + df["rtm"] + df["rtm"]) / 3.0).dropna()


def load_deviations() -> pd.DataFrame:
    with store.connect() as con:
        d = pd.read_sql("SELECT * FROM up_live ORDER BY fetched_at", con)
    if not len(d):
        return pd.DataFrame()
    d["ts"] = pd.to_datetime(d["fetched_at"])
    d = d.dropna(subset=["schedule_mw", "drawal_mw"])
    d["deviation_mw"] = d["drawal_mw"] - d["schedule_mw"]
    # snap each snapshot to the 15-minute block it fell in, so it can be
    # priced against that block's clearing prices
    d["block"] = d["ts"].dt.floor("15min")
    return d


def settle(state_name: str = "Uttar Pradesh") -> dict:
    d = load_deviations()
    if not len(d):
        return {"error": "no up_live deviation data yet"}

    nr = _normal_rate_series()
    d["normal_rate"] = d["block"].map(nr)
    # a snapshot in a block we have no cleared price for cannot be priced
    priced = d.dropna(subset=["normal_rate"]).copy()
    if not len(priced):
        return {"error": "no overlap between deviation snapshots and cleared prices"}

    priced["band_mw"] = np.minimum(
        BAND_PCT / 100 * priced["schedule_mw"].abs(), BAND_CAP_MW)
    dev = priced["deviation_mw"]
    priced["outside_band"] = dev.abs() > priced["band_mw"]
    # only the excess beyond the band is chargeable
    priced["chargeable_mw"] = (dev.abs() - priced["band_mw"]).clip(lower=0)
    priced["chargeable_mwh"] = priced["chargeable_mw"] * BLOCK_H
    # over-drawal is paid for, under-drawal is credited; both at the Normal Rate
    priced["charge_rs"] = np.where(
        dev > 0, priced["chargeable_mwh"] * priced["normal_rate"],
        -priced["chargeable_mwh"] * priced["normal_rate"])

    n = len(priced)
    payable = float(priced.loc[priced["charge_rs"] > 0, "charge_rs"].sum())
    credit = float(-priced.loc[priced["charge_rs"] < 0, "charge_rs"].sum())
    # scale the sample to a full day: 96 blocks against however many we priced
    span_days = max((priced["ts"].max() - priced["ts"].min()).total_seconds()
                    / 86400, 1e-9)
    blocks_sampled = priced["block"].nunique()
    per_day_blocks = blocks_sampled / span_days
    scale = 96.0 / per_day_blocks if per_day_blocks else np.nan

    hourly = (priced.assign(hr=priced["ts"].dt.hour)
              .groupby("hr")
              .agg(n=("deviation_mw", "size"),
                   mean_dev_mw=("deviation_mw", "mean"),
                   mean_abs_dev_mw=("deviation_mw", lambda s: s.abs().mean()),
                   outside_pct=("outside_band", lambda s: s.mean() * 100),
                   charge_rs=("charge_rs", "sum"))
              .round(1).reset_index())

    return {
        "state": state_name,
        "source": "upsldc.org — the only state publishing schedule/drawal/deviation",
        "snapshots": n,
        "blocks_sampled": int(blocks_sampled),
        "from": str(priced["ts"].min()),
        "to": str(priced["ts"].max()),
        "mean_schedule_mw": round(float(priced["schedule_mw"].mean()), 0),
        "mean_demand_mw": round(float(priced["demand_met_mw"].mean()), 0)
        if "demand_met_mw" in priced else None,
        "mean_abs_deviation_mw": round(float(dev.abs().mean()), 0),
        "over_drawing_pct": round(float((dev > 0).mean() * 100), 1),
        "band_mw_typical": round(float(priced["band_mw"].median()), 0),
        "outside_band_pct": round(float(priced["outside_band"].mean() * 100), 1),
        "mean_normal_rate_rs_mwh": round(float(priced["normal_rate"].mean()), 0),
        "sample_payable_rs": round(payable, 0),
        "sample_credit_rs": round(credit, 0),
        "sample_net_rs": round(payable - credit, 0),
        "est_payable_per_day_rs": round(payable / span_days * scale, 0)
        if np.isfinite(scale) else None,
        "est_payable_per_year_rs": round(payable / span_days * scale * 365, 0)
        if np.isfinite(scale) else None,
        "sample_scale_factor": round(float(scale), 2) if np.isfinite(scale) else None,
        "by_hour": hourly.replace({np.nan: None}).to_dict("records"),
        "caveat": (
            f"An UPPER BOUND, not a settlement, and biased high for a specific "
            f"reason. {int(blocks_sampled)} distinct 15-minute blocks were priced "
            f"out of the {int(96 * span_days)} in the window, and the published "
            "deviation is an INSTANTANEOUS reading while DSM settles on the BLOCK "
            "AVERAGE. Instantaneous excursions partly cancel within a block, so the "
            "average is smaller than the reading — systematically, not randomly. "
            "The sample also inherits the collector's own hours, including a gap "
            "between 05:00 and 08:00 that has never been covered. Treat the "
            "per-snapshot charge as real, the daily and annual figures as the top "
            "of a range. A true settlement needs SLDC's block-wise energy account, "
            "which is not public."),
        "bias": "upper bound — instantaneous readings exceed block averages",
        "ancillary_proxied": True,
    }


def forecast_value(reduction_pct: float = 30.0) -> dict:
    """What a better schedule is worth, stated as a sensitivity not a promise.

    We cannot claim a specific improvement for a state whose forecasting we do
    not run. What we can do is price the deviation that exists and show what
    removing a given share of it would save — the customer supplies the share
    they believe.
    """
    s = settle()
    if "error" in s:
        return s
    base = s.get("est_payable_per_year_rs")
    if not base:
        return {"error": "no annualised base to scale"}
    return {
        "state": s["state"],
        "annual_payable_rs": base,
        "basis": "upper bound (see settlement caveat)",
        "scenarios": [
            {"deviation_reduced_pct": p,
             "annual_saving_rs": round(base * p / 100, 0)}
            for p in (10, 20, 30, 40, 50)
        ],
        "note": ("Deviation charges scale roughly with the excess beyond the "
                 "band, so a proportional cut in deviation is a proportional "
                 "cut in charge. The percentage is the customer's assumption, "
                 "not our claim."),
    }


if __name__ == "__main__":
    s = settle()
    if "error" in s:
        print(s["error"])
        raise SystemExit(0)
    print(f"DSM exposure — {s['state']} ({s['source']})")
    print(f"  {s['snapshots']} snapshots, {s['blocks_sampled']} blocks priced "
          f"| {s['from'][:16]} -> {s['to'][:16]}")
    print(f"  schedule {s['mean_schedule_mw']:,.0f} MW | mean |deviation| "
          f"{s['mean_abs_deviation_mw']:,.0f} MW | band {s['band_mw_typical']:,.0f} MW")
    print(f"  outside the band on {s['outside_band_pct']}% of snapshots, "
          f"over-drawing {s['over_drawing_pct']}%")
    print(f"  Normal Rate mean Rs {s['mean_normal_rate_rs_mwh']:,.0f}/MWh")
    print()
    print(f"  sample payable  Rs {s['sample_payable_rs']:>12,.0f}")
    print(f"  sample credit   Rs {s['sample_credit_rs']:>12,.0f}")
    if s.get("est_payable_per_day_rs"):
        print(f"  UPPER-BOUND payable Rs {s['est_payable_per_day_rs']:>10,.0f}/day"
              f"  -> Rs {s['est_payable_per_year_rs']/1e7:,.1f} Cr/yr"
              f"   (scaled x{s['sample_scale_factor']})")
        print("    biased HIGH: the feed publishes an instantaneous deviation,")
        print("    DSM settles on the block average, and excursions partly cancel.")
    print()
    fv = forecast_value()
    if "scenarios" in fv:
        print("  what cutting deviation is worth:")
        for sc in fv["scenarios"]:
            print(f"    -{sc['deviation_reduced_pct']:2d}% deviation -> "
                  f"Rs {sc['annual_saving_rs']/1e7:,.2f} Cr/yr saved")
    (OUT / "dsm_state.json").write_text(
        json.dumps({"settlement": s, "forecast_value": fv}, indent=2, default=float))
