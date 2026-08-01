"""Does risk-aware bidding actually pay? Backtest of CVaR vs point forecast.

Method, per day in the test window:
  1. Predict the price *distribution* (conformal P10/P50/P90) using only
     bid-time-valid features.
  2. Draw scenarios from it.
  3. Build three schedules — point-forecast LP (today's production
     behaviour), risk-neutral stochastic (lam=0), and risk-aware CVaR
     (lam=0.5).
  4. Settle all three at the **actual** cleared prices.

Nothing sees the delivery day. The comparison is like-for-like: same
asset, same day, same settlement prices — only the decision rule differs.

The claim being tested is *not* "risk-aware earns more on average". It is
"risk-aware gives up a little mean profit to substantially improve the
bad days", which is what an asset owner with financing covenants actually
wants. Judge it on the tail columns, not the total.
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from models import price_model  # noqa: E402
from optimize import stochastic as sto  # noqa: E402
from optimize.dispatch import Bess, optimize_dispatch, settle  # noqa: E402

OUT = HERE.parent / "output"
OUT.mkdir(exist_ok=True)


def _quantile_frame(part: pd.DataFrame, margin: float) -> pd.DataFrame:
    """Conformal P10/P50/P90 for one day's feature rows."""
    cols = {}
    for q in price_model.QUANTILES:
        b = lgb.Booster(model_file=str(price_model.QMODEL_PATH).format(q=q * 100))
        p = np.clip(np.exp(b.predict(part[price_model.FEATURES])), 0, 10000)
        cols[f"q{q * 100:02.0f}"] = p
    qf = pd.DataFrame(cols, index=part.index)
    lo, hi = qf.columns[0], qf.columns[-1]
    qf[lo] = (qf[lo] * np.exp(-margin)).clip(0, 10000)
    qf[hi] = (qf[hi] * np.exp(margin)).clip(0, 10000)
    qf[:] = np.sort(qf.values, axis=1)
    return qf


def run(test_days: int = 60, bess: Bess = Bess(), lam: float = 0.5,
        n_scenarios: int = 24) -> pd.DataFrame:
    import json
    margin = json.loads(price_model.CONFORMAL_PATH.read_text())["log_margin"] \
        if price_model.CONFORMAL_PATH.exists() else 0.0

    f = price_model.build_features(price_model._table())
    f = f.dropna(subset=price_model.FEATURES + ["mcp_rs_mwh"])
    point = lgb.Booster(model_file=str(price_model.MODEL_PATH))
    f["mcp_pred"] = np.clip(np.exp(point.predict(f[price_model.FEATURES])), 0, 10000)
    test = f[f.index >= f.index.max().normalize() - pd.Timedelta(days=test_days)]

    rows = []
    for day, g in test.groupby(test.index.date):
        if len(g) != 96:
            continue
        act = g["mcp_rs_mwh"]
        qf = _quantile_frame(g, margin)
        scen = sto.make_scenarios(qf, n_scenarios=n_scenarios)

        sched_pt, _ = optimize_dispatch(g["mcp_pred"], bess)
        sched_rn, _ = sto.optimize_cvar(qf, bess, lam=0.0, scenarios=scen)
        sched_cv, _ = sto.optimize_cvar(qf, bess, lam=lam, scenarios=scen)

        rows.append({
            "date": day,
            "pnl_point": settle(sched_pt, act, bess.degradation_rs_mwh),
            "pnl_stochastic": settle(sched_rn, act, bess.degradation_rs_mwh),
            "pnl_cvar": settle(sched_cv, act, bess.degradation_rs_mwh),
            "pnl_perfect": optimize_dispatch(act, bess)[1],
        })
        print(f"  {day}  point {rows[-1]['pnl_point']:>9,.0f}   "
              f"stoch {rows[-1]['pnl_stochastic']:>9,.0f}   "
              f"cvar {rows[-1]['pnl_cvar']:>9,.0f}", flush=True)

    daily = pd.DataFrame(rows).set_index("date")
    daily.to_csv(OUT / "risk_backtest_daily.csv")

    def stats(col):
        v = daily[col]
        return {
            "total": v.sum(),
            "mean_day": v.mean(),
            "worst_day": v.min(),
            "p10_day": v.quantile(0.10),
            "std_day": v.std(),
            "loss_days": int((v < 0).sum()),
            "capture": v.sum() / daily["pnl_perfect"].sum(),
        }

    names = {"pnl_point": "point forecast LP", "pnl_stochastic": "stochastic (lam=0)",
             "pnl_cvar": f"CVaR risk-aware (lam={lam})"}
    lines = [
        f"risk backtest: {daily.index.min()} -> {daily.index.max()} "
        f"({len(daily)} days), {n_scenarios} scenarios/day",
        f"BESS {bess.power_mw:.0f} MW / {bess.energy_mwh:.0f} MWh",
        "",
        f"{'strategy':<26} {'total Rs':>12} {'mean/day':>10} {'worst day':>11} "
        f"{'P10 day':>10} {'std':>9} {'capture':>8}",
    ]
    for col, label in names.items():
        s = stats(col)
        lines.append(f"{label:<26} {s['total']:>12,.0f} {s['mean_day']:>10,.0f} "
                     f"{s['worst_day']:>11,.0f} {s['p10_day']:>10,.0f} "
                     f"{s['std_day']:>9,.0f} {s['capture']:>7.1%}")
    sp, sc = stats("pnl_point"), stats("pnl_cvar")
    lines += [
        "",
        f"CVaR vs point forecast:",
        f"  mean/day    {(sc['mean_day'] / sp['mean_day'] - 1) * 100:+6.1f}%",
        f"  worst day   {(sc['worst_day'] / sp['worst_day'] - 1) * 100:+6.1f}%  "
        f"(Rs {sp['worst_day']:,.0f} -> Rs {sc['worst_day']:,.0f})",
        f"  P10 day     {(sc['p10_day'] / sp['p10_day'] - 1) * 100:+6.1f}%",
        f"  volatility  {(sc['std_day'] / sp['std_day'] - 1) * 100:+6.1f}%",
    ]
    report = "\n".join(lines)
    print("\n" + report)
    (OUT / "risk_backtest_summary.txt").write_text(report)
    return daily


if __name__ == "__main__":
    run(test_days=int(sys.argv[1]) if len(sys.argv) > 1 else 60)
