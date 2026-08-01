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

import json
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


CAP = 10000.0
CAP_CLF_PATH = OUT / "price_cap_clf.txt"


def train(test_days: int = 60):
    """Cap-hurdle two-stage model (model-lab winner P3, adopted 24 Jul).

    Indian DAM prices pin at the Rs 10,000 cap on a large share of summer
    evening blocks — the target is right-censored, and a single regression
    smears the censoring. Two stages treat it explicitly:

      stage 1  P(cap): classifier for "this block clears at the cap"
      stage 2  E[MCP | below cap]: log-target regression on non-cap blocks

      forecast = P(cap) * 10000 + (1 - P(cap)) * E[MCP | below cap]

    On the held-out 60 days this cut evening MAPE from 15.3% -> 11.4% and
    lifted cap-block recall from 49% -> 78% vs the single-stage model.
    """
    f = build_features(_table()).dropna(subset=FEATURES + ["mcp_rs_mwh"])
    split = f.index.max().normalize() - pd.Timedelta(days=test_days)
    val_split = split - pd.Timedelta(days=30)
    train_ = f[f.index < val_split]
    val = f[(f.index >= val_split) & (f.index < split)]
    test = f[f.index >= split]
    print(f"price model: train {len(train_):,} | val {len(val):,} | "
          f"test {len(test):,} ({f.index.min():%Y-%m-%d} -> {f.index.max():%Y-%m-%d})")

    params = dict(n_estimators=2000, learning_rate=0.03, num_leaves=63,
                  min_child_samples=40, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, reg_lambda=1.0, random_state=42,
                  verbose=-1)

    # stage 1: cap classifier
    ycap_tr = (train_["mcp_rs_mwh"] >= CAP * 0.95).astype(int)
    ycap_va = (val["mcp_rs_mwh"] >= CAP * 0.95).astype(int)
    clf = lgb.LGBMClassifier(**{**params, "n_estimators": 1500})
    clf.fit(train_[FEATURES], ycap_tr, eval_set=[(val[FEATURES], ycap_va)],
            callbacks=[lgb.early_stopping(100, verbose=False),
                       lgb.log_evaluation(0)])
    clf.booster_.save_model(str(CAP_CLF_PATH))

    # stage 2: below-cap regression, log target
    below = train_["mcp_rs_mwh"] < CAP * 0.95
    vbelow = val["mcp_rs_mwh"] < CAP * 0.95
    model = lgb.LGBMRegressor(**params)
    model.fit(train_.loc[below, FEATURES],
              np.log(train_.loc[below, "mcp_rs_mwh"].clip(lower=50)),
              eval_set=[(val.loc[vbelow, FEATURES],
                         np.log(val.loc[vbelow, "mcp_rs_mwh"].clip(lower=50)))],
              callbacks=[lgb.early_stopping(100, verbose=False),
                         lgb.log_evaluation(0)])
    model.booster_.save_model(str(MODEL_PATH))

    def hurdle_predict(part):
        pcap = clf.predict_proba(part[FEATURES])[:, 1]
        pbelow = np.clip(np.exp(model.predict(part[FEATURES])), 0, CAP)
        return pcap * CAP + (1 - pcap) * pbelow

    lines = []
    for name, part in [("train", train_), ("val", val), ("test", test)]:
        p = hurdle_predict(part)
        y = part["mcp_rs_mwh"].values
        mape = np.mean(np.abs(y - p) / np.maximum(y, 100)) * 100
        rmse = float(np.sqrt(np.mean((y - p) ** 2)))
        corr = np.corrcoef(y, p)[0, 1]
        evening = (part.index.hour >= 17) & (part.index.hour <= 23)
        emape = np.mean(np.abs(y[evening] - p[evening])
                        / np.maximum(y[evening], 100)) * 100
        lines.append(f"{name:5s}  MAPE {mape:5.2f}%   RMSE {rmse:7.1f} Rs/MWh"
                     f"   corr {corr:.3f}   evening MAPE {emape:5.2f}%")
    report = "cap-hurdle two-stage (P(cap) x below-cap regression)\n" + "\n".join(lines)
    print(report)
    (OUT / "metrics_price.txt").write_text(report)
    return model


QUANTILES = (0.10, 0.50, 0.90)
QMODEL_PATH = OUT / "price_model_q{q:02.0f}.txt"
CONFORMAL_PATH = OUT / "price_conformal.json"


def train_quantiles(test_days: int = 60, quantiles=QUANTILES):
    """Quantile regression models — the distribution, not just the mean.

    The point forecast has ~23% MAPE because Indian DAM prices are
    genuinely volatile, and a single number hides that. Training with
    LightGBM's pinball (quantile) objective gives P10/P50/P90 per block,
    which the stochastic optimizer turns into scenarios: we stop
    pretending to know tomorrow's price and bid on its distribution
    instead.

    Trained on log(MCP) like the point model, so the intervals are
    multiplicative — wide where prices are high and volatile (evening
    peak), tight in the quiet night blocks.

    Reported metric is pinball loss (the proper scoring rule for a
    quantile) plus empirical coverage: the share of actual prices that
    fell at or below each predicted quantile, which should land near the
    nominal level if the model is calibrated.
    """
    f = build_features(_table()).dropna(subset=FEATURES + ["mcp_rs_mwh"])
    split = f.index.max().normalize() - pd.Timedelta(days=test_days)
    val_split = split - pd.Timedelta(days=30)
    train_ = f[f.index < val_split]
    val = f[(f.index >= val_split) & (f.index < split)]
    test = f[f.index >= split]

    y_tr = np.log(train_["mcp_rs_mwh"].clip(lower=50))
    y_va = np.log(val["mcp_rs_mwh"].clip(lower=50))
    lines, preds = [], {}
    for q in quantiles:
        m = lgb.LGBMRegressor(
            objective="quantile", alpha=q,
            n_estimators=2000, learning_rate=0.03, num_leaves=63,
            min_child_samples=40, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=1.0, random_state=42,
        )
        m.fit(train_[FEATURES], y_tr, eval_set=[(val[FEATURES], y_va)],
              callbacks=[lgb.early_stopping(100, verbose=False),
                         lgb.log_evaluation(0)])
        m.booster_.save_model(str(QMODEL_PATH).format(q=q * 100))
        p = np.clip(np.exp(m.predict(test[FEATURES])), 0, 10000)
        preds[q] = p
        y = test["mcp_rs_mwh"].values
        err = y - p
        pinball = np.mean(np.maximum(q * err, (q - 1) * err))
        coverage = np.mean(y <= p) * 100
        lines.append(f"P{q * 100:02.0f}  pinball {pinball:7.1f} Rs/MWh   "
                     f"coverage {coverage:5.1f}% (nominal {q * 100:.0f}%)")

    qlo, qhi = min(quantiles), max(quantiles)
    lo, hi = preds[qlo], preds[qhi]
    y = test["mcp_rs_mwh"].values
    raw_inside = np.mean((y >= lo) & (y <= hi)) * 100
    nominal = (qhi - qlo) * 100
    lines.append(f"\nraw P{qlo*100:.0f}-P{qhi*100:.0f}: {raw_inside:.1f}% inside "
                 f"(nominal {nominal:.0f}%), mean width Rs {np.mean(hi - lo):,.0f}/MWh")

    # --- conformal calibration (CQR, Romano et al. 2019) -----------------
    # The raw intervals under-cover badly here, and the cause is a genuine
    # regime shift rather than a coding error: the test window's mean MCP
    # is ~35% above the training history and cap-pinned blocks triple.
    # Models fitted on cheaper history therefore predict low.
    #
    # CQR fixes this without retraining: score each calibration point by
    # how far outside the predicted band it fell,
    #     E_i = max(q_lo(x_i) - y_i,  y_i - q_hi(x_i)),
    # then widen the band by the (1-alpha) empirical quantile of E. We do
    # it in log space, so the correction is multiplicative — it widens the
    # expensive, volatile blocks more than the quiet ones.
    #
    # Calibrating on the most recent 30 days (val) is what lets the band
    # track the current regime. Honest caveat: the coverage guarantee
    # assumes exchangeability, which time series violate; treat the result
    # as well-calibrated-in-practice, not as a proof.
    cal_lo = np.log(np.clip(np.exp(lgb.Booster(
        model_file=str(QMODEL_PATH).format(q=qlo * 100)).predict(val[FEATURES])),
        50, 10000))
    cal_hi = np.log(np.clip(np.exp(lgb.Booster(
        model_file=str(QMODEL_PATH).format(q=qhi * 100)).predict(val[FEATURES])),
        50, 10000))
    y_cal = np.log(val["mcp_rs_mwh"].clip(lower=50).values)
    scores = np.maximum(cal_lo - y_cal, y_cal - cal_hi)
    n = len(scores)
    level = min((qhi - qlo) * (1 + 1 / n), 1.0)
    margin = float(np.quantile(scores, level, method="higher"))

    lo_c = np.clip(lo * np.exp(-margin), 0, 10000)
    hi_c = np.clip(hi * np.exp(margin), 0, 10000)
    cal_inside = np.mean((y >= lo_c) & (y <= hi_c)) * 100
    lines.append(f"conformal P{qlo*100:.0f}-P{qhi*100:.0f}: {cal_inside:.1f}% inside "
                 f"(target {nominal:.0f}%), mean width Rs {np.mean(hi_c - lo_c):,.0f}/MWh"
                 f"   [log-margin {margin:.3f} = x{np.exp(margin):.2f}]")

    CONFORMAL_PATH.write_text(json.dumps({
        "log_margin": margin, "q_lo": qlo, "q_hi": qhi,
        "calibration_n": n, "raw_coverage_pct": round(raw_inside, 1),
        "conformal_coverage_pct": round(cal_inside, 1),
    }, indent=2))

    report = "\n".join(lines)
    print(report)
    (OUT / "metrics_price_quantile.txt").write_text(report)
    return report


def forecast_day_quantiles(target: date | None = None, quantiles=QUANTILES,
                           conformal: bool = True) -> pd.DataFrame:
    """Per-block price quantiles for `target` — columns q10/q50/q90.

    With conformal=True the outer quantiles are widened by the stored CQR
    margin so the band actually achieves its nominal coverage.
    """
    target = target or (date.today() + timedelta(days=1))
    df = _table()
    end = pd.Timestamp(target) + pd.Timedelta(hours=23, minutes=45)
    df = df.reindex(pd.date_range(df.index.min(), end, freq="15min"))
    w = store.read("weather")
    fc = w[w["kind"] == "forecast"]["temp_c"].resample("15min").interpolate()
    df["temp_c"] = df["temp_c"].combine_first(fc)
    day = build_features(df)
    day = day[day.index.date == target]

    out = pd.DataFrame(index=day.index)
    for q in quantiles:
        path = str(QMODEL_PATH).format(q=q * 100)
        if not Path(path).exists():
            raise FileNotFoundError(
                f"{path} missing — run `python models/price_model.py quantiles`")
        b = lgb.Booster(model_file=path)
        out[f"q{q * 100:02.0f}"] = np.clip(np.exp(b.predict(day[FEATURES])), 0, 10000)
    if conformal and CONFORMAL_PATH.exists():
        cfg = json.loads(CONFORMAL_PATH.read_text())
        m = cfg["log_margin"]
        lo_col = f"q{cfg['q_lo'] * 100:02.0f}"
        hi_col = f"q{cfg['q_hi'] * 100:02.0f}"
        if lo_col in out:
            out[lo_col] = (out[lo_col] * np.exp(-m)).clip(0, 10000)
        if hi_col in out:
            out[hi_col] = (out[hi_col] * np.exp(m)).clip(0, 10000)
        out.attrs["conformal"] = True
    else:
        out.attrs["conformal"] = False
    # quantile models are fitted independently and can cross on rare blocks;
    # enforce monotonicity so downstream scenario interpolation stays sane
    out[:] = np.sort(out.values, axis=1)
    return out


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
    out = pd.DataFrame(index=day.index)
    out["forecast_mcp"] = predict_hurdle(day)
    return out


def predict_hurdle(feats: pd.DataFrame) -> np.ndarray:
    """The production point forecast: cap-hurdle expectation combine.
    Single implementation used by forecast_day AND the backtest, so the
    two can never quietly diverge."""
    booster = lgb.Booster(model_file=str(MODEL_PATH))
    pbelow = np.clip(np.exp(booster.predict(feats[FEATURES])), 0, CAP)
    if CAP_CLF_PATH.exists():
        clf = lgb.Booster(model_file=str(CAP_CLF_PATH))
        pcap = clf.predict(feats[FEATURES])  # raw score = P(class 1) for binary
        return pcap * CAP + (1 - pcap) * pbelow
    return pbelow


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].startswith("q"):
        train_quantiles()
    else:
        train()
        print()
        train_quantiles()
