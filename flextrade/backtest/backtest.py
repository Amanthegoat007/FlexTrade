"""Backtest: replay history day by day.

For each day in the window:
  - forecast prices with the trained price model (bid-time-valid features)
  - build the LP schedule on the *forecast* curve
  - settle the schedule at *actual* cleared prices
Benchmarks: greedy on forecast, and LP on actual prices (perfect foresight
upper bound).

Outputs (./output): backtest_daily.csv, backtest_pnl.png, headline metrics.
"""
import sys
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ingest import store
from models import price_model
from optimize.dispatch import Bess, greedy_dispatch, optimize_dispatch, settle

OUT = Path(__file__).resolve().parent.parent / "output"
OUT.mkdir(exist_ok=True)


def run(test_days: int = 60, bess: Bess = Bess()) -> pd.DataFrame:
    f = price_model.build_features(price_model._table())
    f = f.dropna(subset=price_model.FEATURES + ["mcp_rs_mwh"])
    # same cap-hurdle predictor production uses (see price_model.predict_hurdle)
    f["mcp_pred"] = price_model.predict_hurdle(f)

    split = f.index.max().normalize() - pd.Timedelta(days=test_days)
    test = f[f.index >= split]

    rows = []
    for day, g in test.groupby(test.index.date):
        if len(g) != 96:
            continue
        fc, act = g["mcp_pred"], g["mcp_rs_mwh"]
        lp_sched, _ = optimize_dispatch(fc, bess)
        pnl_lp = settle(lp_sched, act, bess.degradation_rs_mwh)
        pnl_greedy = settle(greedy_dispatch(fc, bess), act, bess.degradation_rs_mwh)
        perfect_sched, pnl_perfect = optimize_dispatch(act, bess)
        rows.append({"date": day, "pnl_lp": pnl_lp, "pnl_greedy": pnl_greedy,
                     "pnl_perfect": pnl_perfect,
                     "price_mape": float(np.mean(np.abs(act - fc) /
                                                 np.maximum(act, 100)) * 100)})
    daily = pd.DataFrame(rows).set_index("date")
    daily.to_csv(OUT / "backtest_daily.csv")

    days = len(daily)
    ann = 365 / days
    head = [
        f"backtest window: {daily.index.min()} -> {daily.index.max()} ({days} days)",
        f"BESS: {bess.power_mw:.0f} MW / {bess.energy_mwh:.0f} MWh, "
        f"eff {bess.round_trip_eff:.0%}, degradation Rs {bess.degradation_rs_mwh}/MWh",
        f"LP on forecast : Rs {daily.pnl_lp.sum():12,.0f}  "
        f"(~Rs {daily.pnl_lp.sum() * ann / 1e7:.2f} Cr/yr)",
        f"greedy baseline: Rs {daily.pnl_greedy.sum():12,.0f}",
        f"perfect bound  : Rs {daily.pnl_perfect.sum():12,.0f}",
        f"capture ratio  : {daily.pnl_lp.sum() / daily.pnl_perfect.sum():.1%} "
        f"of perfect foresight",
        f"mean daily price MAPE: {daily.price_mape.mean():.1f}%",
    ]
    report = "\n".join(head)
    print(report)
    (OUT / "backtest_summary.txt").write_text(report)

    plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.3})
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = pd.to_datetime(daily.index)
    ax.plot(x, daily["pnl_perfect"].cumsum() / 1e5, "--", color="#999",
            label="Perfect foresight (upper bound)")
    ax.plot(x, daily["pnl_lp"].cumsum() / 1e5, color="#33577b", lw=2,
            label="FlexTrade (LP on forecast prices)")
    ax.plot(x, daily["pnl_greedy"].cumsum() / 1e5, color="#e07a30",
            label="Greedy baseline")
    ax.set_ylabel("Cumulative P&L (Rs lakh)")
    ax.set_title(f"BESS arbitrage backtest — {bess.power_mw:.0f} MW / "
                 f"{bess.energy_mwh:.0f} MWh")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "backtest_pnl.png")
    return daily


if __name__ == "__main__":
    run()
