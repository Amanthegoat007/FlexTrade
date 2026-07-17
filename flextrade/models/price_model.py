"""Day-ahead DAM price (MCP Rs/MWh) forecast per 15-min block.

Trained on the IEX history scraped into the store by bootstrap_history.py.
Bid-time validity: DAM for delivery day D clears ~13:00 on D-1, so when
bidding for D+1 (gate closure ~12:00 on D) the latest known prices are
for delivery day D. All price lags here are >= 1 day, which is valid.

Load lags use >= 2 days (same rule as the load model).
"""
import sys
from datetime import date, timedelta
from pathlib import Path

import holidays
import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ingest import store

OUT = Path(__file__).resolve().parent.parent / "output"
OUT.mkdir(exist_ok=True)
MODEL_PATH = OUT / "price_model.txt"
B = 96  # blocks/day

FEATURES = [
    "block", "dow", "month", "is_weekend", "is_holiday",
    "hour_sin", "hour_cos", "doy_sin", "doy_cos",
    "p_lag_1d", "p_lag_2d", "p_lag_7d",
    "p_roll7d_mean", "p_roll7d_max", "p_roll7d_std",
    "p_prevday_mean", "p_prevday_max", "p_prevday_min",
    "pb_lag_1d", "sb_lag_1d", "bidgap_lag_1d",
    "load_lag_2d", "load_roll7d_mean",
    "temp_c", "cdh",
]


def _table() -> pd.DataFrame:
    """15-min table: mcp + bids + delhi load + temperature (gap-filled)."""
    p = store.read("dam_price")[["mcp_rs_mwh", "purchase_bid_mw", "sell_bid_mw"]]
    idx = pd.date_range(p.index.min(), p.index.max(), freq="15min")
    p = p.reindex(idx)

    load = store.read("load_5min", since=str(p.index.min() - pd.Timedelta(days=10)))
    load = load["delhi"].resample("15min").mean()
    # contiguous load history so lag/rolling features survive telemetry gaps
    load = load.reindex(idx).interpolate(limit_direction="both").ffill().bfill()

    w = store.read("weather", since=str(p.index.min() - pd.Timedelta(days=10)))
    w = w[~w.index.duplicated(keep="first")]
    actual = w[w["kind"] == "actual"]["temp_c"]
    fcst = w[w["kind"] == "forecast"]["temp_c"]
    temp = actual.combine_first(fcst).resample("15min").interpolate(limit=8)

    df = pd.concat([p, load.rename("load_mw"), temp.rename("temp_c")], axis=1)
    return df.loc[p.index.min(): p.index.max()]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    f = df.copy()
    idx = f.index
    f["block"] = idx.hour * 4 + idx.minute // 15
    hour = idx.hour + idx.minute / 60
    f["dow"] = idx.dayofweek
    f["month"] = idx.month
    f["is_weekend"] = (f["dow"] >= 5).astype(int)
    f["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    doy = idx.dayofyear
    f["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    ind = holidays.India(years=range(idx.year.min(), idx.year.max() + 1))
    f["is_holiday"] = pd.Series(idx.date, index=idx).isin(ind).astype(int)

    f["p_lag_1d"] = f["mcp_rs_mwh"].shift(1 * B)
    f["p_lag_2d"] = f["mcp_rs_mwh"].shift(2 * B)
    f["p_lag_7d"] = f["mcp_rs_mwh"].shift(7 * B)
    base = f["mcp_rs_mwh"].shift(1 * B)
    f["p_roll7d_mean"] = base.rolling(7 * B, min_periods=5 * B).mean()
    f["p_roll7d_max"] = base.rolling(7 * B, min_periods=5 * B).max()
    f["p_roll7d_std"] = base.rolling(7 * B, min_periods=5 * B).std()
    day_agg = f["mcp_rs_mwh"].groupby(idx.date).agg(["mean", "max", "min"]).shift(1)
    dates = pd.Series(idx.date, index=idx)
    f["p_prevday_mean"] = dates.map(day_agg["mean"])
    f["p_prevday_max"] = dates.map(day_agg["max"])
    f["p_prevday_min"] = dates.map(day_agg["min"])
    f["pb_lag_1d"] = f["purchase_bid_mw"].shift(1 * B)
    f["sb_lag_1d"] = f["sell_bid_mw"].shift(1 * B)
    f["bidgap_lag_1d"] = f["pb_lag_1d"] - f["sb_lag_1d"]

    f["load_lag_2d"] = f["load_mw"].shift(2 * B)
    f["load_roll7d_mean"] = f["load_mw"].shift(2 * B).rolling(7 * B).mean()

    f["cdh"] = np.maximum(f["temp_c"] - 24, 0)
    return f


def train(test_days: int = 60):
    """Model predicts log(MCP); exponentiate on the way out so relative
    error is weighted evenly across the 1,000-10,000 Rs price range."""
    f = build_features(_table()).dropna(subset=FEATURES + ["mcp_rs_mwh"])
    split = f.index.max().normalize() - pd.Timedelta(days=test_days)
    val_split = split - pd.Timedelta(days=30)
    train_ = f[f.index < val_split]
    val = f[(f.index >= val_split) & (f.index < split)]
    test = f[f.index >= split]
    print(f"price model: train {len(train_):,} | val {len(val):,} | "
          f"test {len(test):,} ({f.index.min():%Y-%m-%d} -> {f.index.max():%Y-%m-%d})")

    model = lgb.LGBMRegressor(
        n_estimators=2000, learning_rate=0.03, num_leaves=63,
        min_child_samples=40, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_lambda=1.0, random_state=42,
    )
    model.fit(train_[FEATURES], np.log(train_["mcp_rs_mwh"].clip(lower=50)),
              eval_set=[(val[FEATURES], np.log(val["mcp_rs_mwh"].clip(lower=50)))],
              callbacks=[lgb.early_stopping(100, verbose=False),
                         lgb.log_evaluation(0)])

    lines = []
    for name, part in [("train", train_), ("val", val), ("test", test)]:
        p = np.clip(np.exp(model.predict(part[FEATURES])), 0, 10000)
        y = part["mcp_rs_mwh"].values
        mape = np.mean(np.abs(y - p) / np.maximum(y, 100)) * 100
        rmse = float(np.sqrt(np.mean((y - p) ** 2)))
        corr = np.corrcoef(y, p)[0, 1]
        lines.append(f"{name:5s}  MAPE {mape:5.2f}%   RMSE {rmse:7.1f} Rs/MWh"
                     f"   corr {corr:.3f}")
    report = "\n".join(lines)
    print(report)
    (OUT / "metrics_price.txt").write_text(report)
    model.booster_.save_model(str(MODEL_PATH))
    return model


def forecast_day(target: date | None = None) -> pd.DataFrame:
    """Predict MCP for all 96 blocks of `target` (default: tomorrow)."""
    target = target or (date.today() + timedelta(days=1))
    df = _table()
    end = pd.Timestamp(target) + pd.Timedelta(hours=23, minutes=45)
    df = df.reindex(pd.date_range(df.index.min(), end, freq="15min"))
    # temperature for the target day comes from the live forecast
    w = store.read("weather")
    fc = w[w["kind"] == "forecast"]["temp_c"].resample("15min").interpolate()
    df["temp_c"] = df["temp_c"].combine_first(fc)

    feats = build_features(df)
    day = feats[feats.index.date == target]
    booster = lgb.Booster(model_file=str(MODEL_PATH))
    out = pd.DataFrame(index=day.index)
    out["forecast_mcp"] = np.clip(np.exp(booster.predict(day[FEATURES])), 0, 10000)
    return out


if __name__ == "__main__":
    train()
