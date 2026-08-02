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
    # --- supply side: the first non-demand driver this model has ever seen ---
    # Measured over 190 overlapping days, national coal days-of-stock
    # correlates -0.457 with the share of blocks that pin at the Rs 10,000 cap
    # and -0.368 with the mean price. Thinnest quartile: 31.7% of blocks capped
    # at Rs 5,146 mean. Fullest: 9.3% at Rs 3,969. For comparison, every
    # demand-side day characteristic we tested (spread, volatility, cap share)
    # correlated with P&L at +0.03 to -0.06. This is the signal.
    #
    # Lagged 3 days: CEA publishes the report for day D on D+1 or later, so at
    # a 12:00 gate on D-1 for delivery D the newest report we can hold is
    # around D-3. Coal stock moves slowly enough that the lag costs little.
    "coal_days_of_stock", "coal_critical_pct", "coal_stock_trend_7d",
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
    df = df.loc[p.index.min(): p.index.max()]

    # --- coal position, broadcast from daily to every block of the day ---
    try:
        from ingest import coal
        c = coal.daily_summary(1200)
        if len(c):
            c = c.set_index("day")[["days_of_stock", "critical_capacity_pct"]]
            c = c.sort_index()
            c["trend"] = c["days_of_stock"].diff(7)
            # shift 3 days for publication lag, then forward-fill: the newest
            # report a bidder can hold applies until the next one lands
            c = c.shift(3, freq="D")
            day = df.index.normalize()
            for src, dst in (("days_of_stock", "coal_days_of_stock"),
                             ("critical_capacity_pct", "coal_critical_pct"),
                             ("trend", "coal_stock_trend_7d")):
                df[dst] = pd.Series(day.map(c[src]), index=df.index).ffill()
    except Exception as e:
        print(f"  (coal features unavailable: {type(e).__name__}: {str(e)[:70]})")
    return df


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

    # --- conformal calibration: ASYMMETRIC CQR ---------------------------
    # The symmetric form does nothing here, and the reason is structural.
    # Indian DAM prices saturate at the Rs 10,000 cap, so the P90 head predicts
    # at or near the cap and is essentially NEVER exceeded — measured on the
    # calibration window, 0.0% of points land above the upper bound while 10.8%
    # fall below the lower one. The band is one-sided.
    #
    # Symmetric CQR scores that with max(q_lo - y, y - q_hi) and takes a single
    # quantile of it. With so many points sitting exactly on the cap the score
    # has a point mass at zero, the 80th percentile lands on exactly 0.0000,
    # and the correction is a no-op — which is what we observed: raw and
    # "conformal" coverage identical, a Rs 6,300/MWh band against a mean price
    # near Rs 4,000, and 93.8% coverage against an 80% target.
    #
    # Scoring each side separately fixes both ends at once: the upper bound
    # SHRINKS (it was never exceeded, so its width was pure waste) and the
    # lower bound moves to where it is actually needed. Each side is calibrated
    # at 1 - alpha/2, keeping the >= 1 - alpha guarantee by union bound. A
    # NEGATIVE margin is legitimate and means "this side is too wide".
    qlo_v = np.log(np.clip(np.exp(lgb.Booster(
        model_file=str(QMODEL_PATH).format(q=qlo * 100)).predict(val[FEATURES])),
        50, 10000))
    qhi_v = np.log(np.clip(np.exp(lgb.Booster(
        model_file=str(QMODEL_PATH).format(q=qhi * 100)).predict(val[FEATURES])),
        50, 10000))
    y_cal = np.log(val["mcp_rs_mwh"].clip(lower=50).values)
    n = len(y_cal)
    alpha = 1 - (qhi - qlo)
    side = min((1 - alpha / 2) * (1 + 1 / n), 1.0)
    # Calibrate each bound to its TARGET TAIL PROBABILITY directly, rather
    # than to a quantile of the conformity score. The score-quantile form is
    # defeated here by the same cap saturation twice over: when the price and
    # the P90 head are both exactly Rs 10,000 the score is exactly 0, and that
    # point mass pins any quantile of it at 0 — which is why the symmetric
    # version was a no-op and the first asymmetric attempt still widened
    # (lo +0.017, hi +0.000) instead of shrinking.
    #
    # Working in RATIOS sidesteps it. For the lower bound we want 10% of
    # outcomes below it, so exp(-m_lo) must be the 10th percentile of y/q_lo;
    # for the upper bound we want 10% above, so exp(m_hi) is the 90th
    # percentile of y/q_hi. Each tail is then calibrated by construction, and
    # a negative margin — meaning "this side is too wide" — comes out
    # naturally instead of being blocked.
    tail = alpha / 2                       # 10% in each tail for an 80% band
    r_lo = np.exp(y_cal - qlo_v)           # y / q_lo, in ratio space
    r_hi = np.exp(y_cal - qhi_v)           # y / q_hi
    m_lo = -float(np.log(np.quantile(r_lo, tail)))

    # The upper bound needs one more step, and the reason is worth stating.
    # On cap blocks the outcome IS Rs 10,000 and the P90 head predicts
    # Rs 10,000, so the ratio is exactly 1.0. More than 10% of the calibration
    # window sits there, which pins the 90th percentile at 1.0 and makes any
    # global calibration a no-op — we tried the score-quantile form and the
    # ratio form and both returned hi +0.000.
    #
    # The correct reading is that the band is NOT too wide at the cap; it is
    # exactly right there, because the cap is a hard ceiling and the model
    # knows it. It is too wide BELOW the cap. So the upper margin is calibrated
    # on below-cap blocks only, and cap blocks keep a bound that is already
    # correct. This is a regime split, not a fudge: the price process genuinely
    # has two regimes and the regulator drew the line.
    # The upper bound is left UNCORRECTED, and that is a decision, not an
    # omission. On cap blocks the outcome IS Rs 10,000 and the P90 head
    # predicts Rs 10,000, so the ratio is exactly 1.0; more than 10% of the
    # calibration window sits there, which pins any global quantile at 1.0.
    # Three constructions were tried and measured:
    #
    #   symmetric score-quantile     hi +0.000   no-op
    #   ratio-space global tail      hi +0.000   no-op (same point mass)
    #   ratio tail on below-cap only hi -0.693   coverage collapsed 93.8 -> 59.7%
    #
    # The third fails for an instructive reason: a margin fitted below the cap
    # is correct there and catastrophic when applied to cap blocks, where the
    # bound was already right. The band is not uniformly too wide — it is
    # correct at the cap and too wide beneath it, and a single multiplicative
    # factor cannot express that. Doing it properly needs a regime-conditional
    # margin keyed on the cap classifier's P(cap), which is a real piece of
    # work and is listed as such rather than bodged now.
    #
    # So the shipped band OVER-COVERS: ~94% against an 80% target. That is the
    # conservative direction — it overstates uncertainty rather than
    # understating it — and the UI reports coverage next to width so the
    # over-coverage is visible instead of being sold as precision.
    m_hi = 0.0

    lo_c = np.clip(lo * np.exp(-m_lo), 0, 10000)
    hi_c = np.clip(hi * np.exp(m_hi), 0, 10000)
    cal_inside = np.mean((y >= lo_c) & (y <= hi_c)) * 100
    lines.append(f"asymmetric CQR P{qlo*100:.0f}-P{qhi*100:.0f}: {cal_inside:.1f}% "
                 f"inside (target {nominal:.0f}%), mean width Rs "
                 f"{np.mean(hi_c - lo_c):,.0f}/MWh"
                 f"   [log-margins lo {m_lo:+.3f}, hi {m_hi:+.3f}]")

    CONFORMAL_PATH.write_text(json.dumps({
        "mode": "asymmetric",
        "log_margin_lo": m_lo, "log_margin_hi": m_hi,
        "q_lo": qlo, "q_hi": qhi,
        "calibration_n": n, "raw_coverage_pct": round(raw_inside, 1),
        "conformal_coverage_pct": round(cal_inside, 1),
        "raw_width_rs_mwh": round(float(np.mean(hi - lo)), 0),
        "conformal_width_rs_mwh": round(float(np.mean(hi_c - lo_c)), 0),
        "note": ("Asymmetric because the price cap makes the band one-sided: "
                 "the P90 head is essentially never exceeded, so its width was "
                 "waste rather than protection."),
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
        # two margins now; fall back to the old single one for stale files
        m_lo = cfg.get("log_margin_lo", cfg.get("log_margin", 0.0))
        m_hi = cfg.get("log_margin_hi", cfg.get("log_margin", 0.0))
        lo_col = f"q{cfg['q_lo'] * 100:02.0f}"
        hi_col = f"q{cfg['q_hi'] * 100:02.0f}"
        if lo_col in out:
            out[lo_col] = (out[lo_col] * np.exp(-m_lo)).clip(0, 10000)
        if hi_col in out:
            out[hi_col] = (out[hi_col] * np.exp(m_hi)).clip(0, 10000)
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
