"""Head-to-head: FlexTrade's LP schedule vs the real BRPL Kilokari BESS.

This is the platform's strongest claim, because it is not a simulation
against a hypothetical asset. The BRPL Kilokari battery (20 MW / 40 MWh,
India's first utility-scale standalone BESS) publishes live telemetry on
Delhi SLDC, and our optimizer's reference asset is the same spec. So for
any period where we have telemetry we can ask, block by block:

    what did the real battery do, and what would FlexTrade have done?

Both sides are valued at the SAME actual IEX DAM clearing prices, so the
comparison is like-for-like. FlexTrade's schedule is built from the
*price forecast* (bid-time information only) and then settled at actual
prices — exactly the discipline used in backtest/backtest.py.

Caveats printed with the result, because they matter:
  - The real battery is a regulated DISCOM asset. It is dispatched for
    grid support and DERC-approved duties, NOT purely for arbitrage
    profit. A revenue gap is therefore expected and is not a claim that
    the operator is doing a bad job — it quantifies the arbitrage value
    currently left on the table, which is precisely FlexTrade's pitch.
  - Telemetry is sampled by us (SLDC exposes no history), so coverage
    starts when polling started and gaps are reported honestly.
"""
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from ingest import bess, iex, store  # noqa: E402
from models import price_model  # noqa: E402
from optimize.dispatch import Bess, optimize_dispatch  # noqa: E402

OUT = HERE.parent / "output"
OUT.mkdir(exist_ok=True)
BLOCK_H = 0.25

# match the optimizer's asset to the real one
REAL_BESS = Bess(power_mw=bess.RATED_POWER_MW,
                 energy_mwh=bess.RATED_ENERGY_MWH,
                 round_trip_eff=0.90, degradation_rs_mwh=200.0)


def actual_revenue(profile: pd.DataFrame, prices: pd.Series) -> pd.DataFrame:
    """Value the real battery's measured dispatch at actual DAM prices."""
    df = pd.DataFrame({"discharge_mw": profile["discharge_mw"],
                       "soc_pct": profile["soc_pct"]}).join(
        prices.rename("price"), how="inner")
    # positive discharge earns, negative (charging) costs
    df["revenue_rs"] = df["discharge_mw"] * BLOCK_H * df["price"]
    df["throughput_mwh"] = df["discharge_mw"].abs() * BLOCK_H
    return df


def flextrade_schedule(day: date, prices_actual: pd.Series) -> pd.DataFrame:
    """LP schedule built on the price FORECAST, settled at actual prices."""
    try:
        fc = price_model.forecast_day(day)["forecast_mcp"]
    except Exception:
        fc = None
    if fc is None or not len(fc):
        return pd.DataFrame()
    sched, _ = optimize_dispatch(fc, REAL_BESS)
    df = sched[["charge_mw", "discharge_mw", "soc_mwh"]].join(
        prices_actual.rename("price"), how="inner")
    net_mw = df["discharge_mw"] - df["charge_mw"]
    df["revenue_rs"] = net_mw * BLOCK_H * df["price"]
    df["throughput_mwh"] = (df["discharge_mw"] + df["charge_mw"]) * BLOCK_H
    return df


def compare_day(day: date) -> dict | None:
    profile = bess.daily_profile(day)
    if profile.empty:
        return None
    try:
        prices = iex.fetch_dam(day)["mcp_rs_mwh"]
    except Exception:
        stored = store.read("dam_price")
        prices = stored[stored.index.date == day]["mcp_rs_mwh"]
    if not len(prices):
        return None

    real = actual_revenue(profile, prices)
    flex = flextrade_schedule(day, prices)
    if real.empty or flex.empty:
        return None

    # compare only the blocks we actually observed the real battery in
    common = real.index.intersection(flex.index)
    real_c, flex_c = real.loc[common], flex.loc[common]
    coverage = len(common) / 96.0

    # Partial-day coverage BIASES the revenue comparison: if sampling only
    # caught the evening discharge, the real battery's overnight charging
    # cost is missing and its revenue looks sell-side-only. Only days with
    # near-full coverage give a fair like-for-like number.
    fair = coverage >= 0.90

    return {
        "day": day,
        "blocks_observed": len(common),
        "coverage_pct": round(coverage * 100, 1),
        "fair_comparison": fair,
        "real_revenue_rs": float(real_c["revenue_rs"].sum()),
        "flex_revenue_rs": float(flex_c["revenue_rs"].sum()),
        "real_throughput_mwh": float(real_c["throughput_mwh"].sum()),
        "flex_throughput_mwh": float(flex_c["throughput_mwh"].sum()),
        "real_soc_range": (float(real_c["soc_pct"].min()),
                           float(real_c["soc_pct"].max())),
        "price_min": float(prices.loc[common].min()),
        "price_max": float(prices.loc[common].max()),
        "_real": real_c,
        "_flex": flex_c,
    }


def run(days: int = 7) -> pd.DataFrame:
    hist = bess.read_history()
    if not len(hist):
        print("No BESS telemetry yet — start `python poll_bess.py` and let it "
              "accumulate.\nThe validator needs at least a few hours of "
              "readings to be meaningful.")
        return pd.DataFrame()

    observed = sorted({d for d in hist.index.date})
    observed = [d for d in observed if d >= date.today() - timedelta(days=days)]
    rows, details = [], {}
    for d in observed:
        res = compare_day(d)
        if res:
            details[d] = (res.pop("_real"), res.pop("_flex"))
            rows.append(res)

    if not rows:
        print(f"Telemetry spans {observed[0]} → {observed[-1]} but no day had "
              "both prices and a usable schedule yet.")
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("day")
    df["uplift_rs"] = df["flex_revenue_rs"] - df["real_revenue_rs"]
    df.to_csv(OUT / "bess_validation.csv", index_label="day")

    fair = df[df["fair_comparison"]]
    hours = len(hist) and (hist.index.max() - hist.index.min()).total_seconds() / 3600

    print("=" * 68)
    print("FlexTrade vs BRPL Kilokari BESS (20 MW / 40 MWh, real asset)")
    print("=" * 68)
    print(f"telemetry: {len(hist):,} readings over {hours:.1f} h "
          f"({hist.index.min():%Y-%m-%d %H:%M} → {hist.index.max():%Y-%m-%d %H:%M})")
    print(f"days compared: {len(df)}   mean block coverage: "
          f"{df['coverage_pct'].mean():.0f}% of the 96-block day\n")
    for d, r in df.iterrows():
        tag = "FAIR   " if r["fair_comparison"] else "PARTIAL"
        print(f"  {d} {tag} {r['blocks_observed']:>3.0f} blocks "
              f"({r['coverage_pct']:>5.1f}%)  |  real Rs {r['real_revenue_rs']:>10,.0f}"
              f"  |  FlexTrade Rs {r['flex_revenue_rs']:>10,.0f}"
              f"  |  uplift Rs {r['uplift_rs']:>10,.0f}")
    if len(fair):
        print(f"\n  FAIR-DAYS TOTAL   real Rs {fair['real_revenue_rs'].sum():,.0f}   "
              f"FlexTrade Rs {fair['flex_revenue_rs'].sum():,.0f}   "
              f"uplift Rs {fair['uplift_rs'].sum():,.0f}")
    else:
        print("\n  No day yet has >=90% block coverage, so NO headline uplift is")
        print("  quoted: partial sampling that catches only the discharge window")
        print("  makes the real battery look sell-side-only (charging cost")
        print("  unobserved). The 5-min poller closes this within one full day.")
    print(f"  real battery throughput  {df['real_throughput_mwh'].sum():.1f} MWh")
    print(f"  FlexTrade throughput     {df['flex_throughput_mwh'].sum():.1f} MWh")
    print("\nCaveat: BRPL's battery is a regulated DISCOM asset dispatched for")
    print("grid support, not pure arbitrage. The gap measures arbitrage value")
    print("currently unmonetised — it is not a judgement of the operator.")
    print(f"\nsaved -> {OUT / 'bess_validation.csv'}")
    return df


if __name__ == "__main__":
    run(days=int(sys.argv[1]) if len(sys.argv) > 1 else 7)
