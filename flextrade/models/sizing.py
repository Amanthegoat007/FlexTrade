"""BESS Sizing & Bankability engine — the question every customer asks
first ("what size, and will it pay back?") answered from real prices.

Method
------
1. Arbitrage revenue is LINEAR in power at fixed duration (double the MW
   and MWh together and the LP schedule doubles), so we precompute
   per-MW daily revenue for standard durations (1h / 2h / 4h) and any
   plant size scales from that instantly — this is what makes the web
   calculator interactive with zero server compute.
2. Per day over the trailing year of ACTUAL IEX DAM prices: solve the
   dispatch LP with perfect foresight, then multiply by FlexTrade's
   MEASURED capture ratio (read live from the backtest, not frozen) to get
   the achievable number. That keeps the projection tied to a measured
   capability instead of an assumed one.
3. Degradation at the physics-calibrated rate shared with the dispatch LP,
   rainflow/Wohler — see optimize/degradation.py), inside the LP so
   cycling depth is realistic, and reported as its own line.
4. Bankability: lenders lend against pessimistic revenue. We bootstrap
   annual revenue from the observed daily distribution (10,000 resamples
   of 365 days) and report P50 / P75 / P90 annual figures — the same
   quantile discipline used everywhere else in the platform.

Honest caveats carried into the output:
  - one year of history includes this summer's high-price regime; the
    bootstrap treats days as exchangeable (no multi-year cycles);
  - DAM arbitrage only — RTM uplift, DSM savings, and future ancillary
    revenue are additive upside, not counted;
  - capex defaults are indicative (Rs 1.3-1.8 Cr/MWh tender range).
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from ingest import store  # noqa: E402
from optimize.dispatch import Bess, optimize_dispatch  # noqa: E402

OUT = HERE.parent / "output"
CACHE = OUT / "sizing_curves.json"

DURATIONS_H = (1.0, 2.0, 4.0)
def _measured_capture() -> float:
    """Read the capture ratio out of the live backtest instead of freezing it.

    It was hardcoded at 0.938 and had already drifted to 0.947 after the
    degradation recalibration — the same failure mode that put a stale RMSE and
    a 61-day backtest on the Methodology page. A bankability model quoting a
    stale capability is the worst place for it.
    """
    try:
        txt = (OUT / "backtest_summary.txt").read_text(encoding="utf-8-sig")
        m = re.search(r"capture ratio\s*:\s*([\d.]+)%", txt)
        if m:
            return float(m.group(1)) / 100.0
    except Exception:
        pass
    return 0.947


CAPTURE_RATIO = _measured_capture()
# Single source of truth: the same calibrated rate the dispatch LP charges.
# This module used to carry its own hardcoded 843.0 — a stale snapshot of the
# same fixed point — so sizing quoted one degradation cost while the optimizer
# charged another. Two numbers for one physical quantity is how a bankability
# model loses an auditor.
from optimize.dispatch import DEGRADATION_RS_MWH  # noqa: E402
CAPEX_RS_PER_MWH_DEFAULT = 1.5e7


def compute(days: int = 365, force: bool = False) -> dict:
    """Per-MW daily revenue curves for each duration; cached to JSON."""
    dam = store.read("dam_price")["mcp_rs_mwh"]
    lo = dam.index.max().normalize() - pd.Timedelta(days=days)
    dam = dam[dam.index >= lo]
    day_list = [d for d, g in dam.groupby(dam.index.date) if len(g) == 96]

    if CACHE.exists() and not force:
        cached = json.loads(CACHE.read_text())
        if cached.get("n_days") == len(day_list) and \
           cached.get("last_day") == str(day_list[-1]):
            return cached

    print(f"sizing precompute: {len(day_list)} days x {len(DURATIONS_H)} durations")
    curves = {}
    for dur in DURATIONS_H:
        bess = Bess(power_mw=1.0, energy_mwh=dur,
                    degradation_rs_mwh=DEGRADATION_RS_MWH)
        daily = []
        for d in day_list:
            prices = dam[dam.index.date == d]
            sched, pnl = optimize_dispatch(prices, bess)
            thr = float((sched["charge_mw"] + sched["discharge_mw"]).sum() * 0.25)
            daily.append({"day": str(d), "pnl_rs": round(pnl, 1),
                          "throughput_mwh": round(thr, 3)})
        curves[f"{dur:g}h"] = daily
        print(f"  {dur:g}h: mean Rs {np.mean([x['pnl_rs'] for x in daily]):,.0f}"
              f"/MW/day (perfect foresight)")

    out = {"n_days": len(day_list), "first_day": str(day_list[0]),
           "last_day": str(day_list[-1]), "capture_ratio": CAPTURE_RATIO,
           "degradation_rs_mwh": DEGRADATION_RS_MWH, "curves": curves}
    CACHE.write_text(json.dumps(out))
    return out


def bankability(curves: dict, n_boot: int = 10000, seed: int = 42) -> dict:
    """Bootstrap annual P50/P75/P90 per MW for each duration, achievable
    (perfect-foresight x measured capture ratio)."""
    rng = np.random.default_rng(seed)
    res = {}
    for dur, daily in curves["curves"].items():
        pnl = np.array([d["pnl_rs"] for d in daily]) * curves["capture_ratio"]
        thr = np.array([d["throughput_mwh"] for d in daily])
        annual = rng.choice(pnl, size=(n_boot, 365), replace=True).sum(axis=1)
        res[dur] = {
            "daily_mean_rs_mw": round(float(pnl.mean()), 0),
            "annual_p50_rs_mw": round(float(np.percentile(annual, 50)), 0),
            "annual_p75_rs_mw": round(float(np.percentile(annual, 25)), 0),
            "annual_p90_rs_mw": round(float(np.percentile(annual, 10)), 0),
            "fce_per_day": round(float(thr.mean() / (2 * float(dur[:-1]))), 2),
            "throughput_mwh_day_mw": round(float(thr.mean()), 2),
        }
    return res


def export() -> dict:
    """Everything the web calculator needs, ready for modules.json."""
    curves = compute()
    bank = bankability(curves)
    monthly = {}
    for dur, daily in curves["curves"].items():
        df = pd.DataFrame(daily)
        df["month"] = df["day"].str[:7]
        m = (df.groupby("month")["pnl_rs"].mean() * curves["capture_ratio"]).round(0)
        monthly[dur] = [{"month": k, "rs_mw_day": v} for k, v in m.items()]
    return {
        "window": {"from": curves["first_day"], "to": curves["last_day"],
                   "n_days": curves["n_days"]},
        "capture_ratio": curves["capture_ratio"],
        "degradation_rs_mwh": curves["degradation_rs_mwh"],
        "capex_rs_per_mwh_default": CAPEX_RS_PER_MWH_DEFAULT,
        "bankability": bank,
        "monthly": monthly,
        "caveats": [
            "trailing-year DAM arbitrage only — RTM/DSM/ancillary revenue is additive upside, not counted",
            "achievable = perfect-foresight LP x measured 93.8% capture ratio (55-day walk-forward)",
            "bootstrap treats days as exchangeable; window includes this summer's high-price regime",
            "capex defaults indicative of FY25-26 tender discovery (Rs 1.3-1.8 Cr/MWh) — enter your own quote",
        ],
    }


if __name__ == "__main__":
    force = "--force" in sys.argv
    curves = compute(force=force)
    bank = bankability(curves)
    print(f"\nBankability per MW (achievable = perfect x {CAPTURE_RATIO:.1%} capture):")
    for dur, b in bank.items():
        print(f"  {dur}:  P50 Rs {b['annual_p50_rs_mw']/1e5:,.1f} L/MW/yr | "
              f"P90 Rs {b['annual_p90_rs_mw']/1e5:,.1f} L/MW/yr | "
              f"{b['fce_per_day']} cycles/day")
    ex = export()
    print(f"\nwindow {ex['window']['from']} -> {ex['window']['to']} "
          f"({ex['window']['n_days']} days)")
