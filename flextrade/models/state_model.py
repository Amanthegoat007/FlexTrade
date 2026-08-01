"""Per-state demand forecasting — the Delhi recipe, replicated by data.

The Delhi load model works because we hold 4+ years of 5-min history.
No public source hands out that history for other states — but MERIT
gives us the *current* value for 23 states, and FlexTrade-StatesPoller
now samples it every 15 minutes into the `state_live` table. So per-state
forecasting is a **data-accrual problem with a known-good recipe**, and
this module is honest about where each state stands:

    readiness()   -> per-state: samples held, days of history, coverage %,
                     and a verdict — "training-ready" needs MIN_DAYS days
                     at >= MIN_COVERAGE completeness.
    train(code)   -> refuses below the readiness gate (no toy models that
                     would embarrass us in a demo); otherwise trains the
                     same LightGBM + chronological-split recipe as Delhi,
                     with lag features restricted to what is bid-time
                     valid (>= 48 h), and reports honest test metrics.

Why the gate matters: with days of history a model can only learn the
daily shape; weekly cycles need ~3 weeks, weather sensitivity needs the
weather join. Shipping a 2-day model as "AI forecasting" is exactly the
overclaim this codebase exists to avoid. The Delhi model (MAPE 4.98%)
is the proof of the recipe; this module is the proof of the pipeline.
"""
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from ingest import states, store  # noqa: E402

MIN_DAYS = 7          # least history we'll train a shape model on
MIN_COVERAGE = 0.60   # fraction of expected 15-min samples present
FULL_DAYS = 21        # history at which weekly features switch on
OUT = HERE.parent / "output"


def _history(code: str) -> pd.Series:
    with store.connect() as con:
        try:
            df = pd.read_sql(
                "SELECT fetched_at, demand_mw FROM state_live WHERE code=? "
                "AND demand_mw IS NOT NULL ORDER BY fetched_at",
                con, params=(code,), parse_dates=["fetched_at"])
        except Exception:
            return pd.Series(dtype=float)
    if not len(df):
        return pd.Series(dtype=float)
    s = df.set_index("fetched_at")["demand_mw"]
    return s.resample("15min").mean().dropna()


def readiness() -> pd.DataFrame:
    """Training-readiness of every state, from the data actually held."""
    rows = []
    for code in states.MERIT_CODES:
        s = _history(code)
        if len(s):
            span_days = max((s.index.max() - s.index.min()).total_seconds() / 86400,
                            1e-9)
            expected = span_days * 96
            coverage = min(len(s) / expected, 1.0)
            ready = span_days >= MIN_DAYS and coverage >= MIN_COVERAGE
            eta = (pd.Timestamp.now() + timedelta(days=MIN_DAYS - span_days)).date() \
                if not ready and coverage >= MIN_COVERAGE else None
        else:
            span_days, coverage, ready, eta = 0.0, 0.0, False, None
        rows.append({
            "code": code, "name": states.MERIT_CODES[code][1],
            "samples": len(s), "days": round(span_days, 2),
            "coverage_pct": round(coverage * 100, 1),
            "status": ("training-ready" if ready else
                       "accruing" if len(s) else "no data yet"),
            "ready_eta": str(eta) if eta else None,
        })
    return pd.DataFrame(rows)


def _features(s: pd.Series, weekly: bool) -> pd.DataFrame:
    """Bid-time-valid features only: lags >= 48 h, calendar shape."""
    df = pd.DataFrame({"y": s})
    idx = df.index
    df["block"] = idx.hour * 4 + idx.minute // 15
    df["dow"] = idx.dayofweek
    df["sin_d"] = np.sin(2 * np.pi * df["block"] / 96)
    df["cos_d"] = np.cos(2 * np.pi * df["block"] / 96)
    df["lag_48h"] = df["y"].shift(96 * 2)
    df["lag_72h"] = df["y"].shift(96 * 3)
    if weekly:
        df["lag_168h"] = df["y"].shift(96 * 7)
        df["is_weekend"] = (df["dow"] >= 5).astype(int)
    return df.dropna()


def train(code: str, force: bool = False) -> dict:
    """Train + evaluate one state's demand model. Chronological split
    (last 20% is the untouched test window). Refuses below the gate."""
    r = readiness()
    row = r[r["code"] == code].iloc[0]
    if row["status"] != "training-ready" and not force:
        return {"code": code, "name": row["name"], "trained": False,
                "reason": (f"{row['status']}: {row['days']} days held "
                           f"({row['coverage_pct']}% coverage); gate is "
                           f"{MIN_DAYS} days at ≥{MIN_COVERAGE:.0%}"),
                "ready_eta": row["ready_eta"]}

    import lightgbm as lgb
    s = _history(code)
    weekly = (s.index.max() - s.index.min()).days >= FULL_DAYS
    df = _features(s, weekly)
    cut = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:cut], df.iloc[cut:]
    feats = [c for c in df.columns if c != "y"]

    model = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05,
                              num_leaves=31, verbose=-1)
    model.fit(train_df[feats], train_df["y"])
    pred = model.predict(test_df[feats])
    mape = float(np.mean(np.abs(pred - test_df["y"]) / test_df["y"]) * 100)
    rmse = float(np.sqrt(np.mean((pred - test_df["y"]) ** 2)))

    mdir = OUT / "state_models"
    mdir.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(mdir / f"{code}.txt"))
    return {"code": code, "name": row["name"], "trained": True,
            "weekly_features": weekly, "n_train": len(train_df),
            "n_test": len(test_df), "test_mape_pct": round(mape, 2),
            "test_rmse_mw": round(rmse, 1),
            "note": ("shape-only model (needs ≥21 days for weekly cycle, "
                     "weather join pending)" if not weekly else
                     "weekly features on; weather join pending")}


if __name__ == "__main__":
    r = readiness()
    print("Per-state forecast readiness (gate: "
          f"{MIN_DAYS} days @ ≥{MIN_COVERAGE:.0%} coverage):")
    print(r.to_string(index=False))
    ready = r[r["status"] == "training-ready"]["code"].tolist()
    if ready:
        print()
        for c in ready:
            res = train(c)
            print(res)
    else:
        print("\nNo state past the gate yet — FlexTrade-StatesPoller is "
              "accruing 96 samples/state/day; Delhi's recipe (MAPE 4.98%) "
              "replicates as each state crosses it.")
