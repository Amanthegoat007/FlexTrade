"""Day-ahead peak forecast for Delhi -- when tomorrow's peak lands, and how big.

"When is the peak and how big is it" is the question a DISCOM control room
asks every single day, and it is not answered by a good average. It sets
merit-order commitment, short-term purchase, and -- for us -- the block a
battery must be full before.

Delhi makes this genuinely hard: the peak hour is multi-modal, not seasonal
drift around one time. Over 1,806 days it lands at 10:00 on 27.6% of days,
15:00 on 20.0%, 23:00 on 14.0% and 00:00 on 8.7% -- an afternoon
air-conditioning regime and a winter evening regime that swap places. A
model that learns "the peak is in the evening" is wrong a third of the year.

Two targets, each scored against the incumbent that already exists:

  magnitude   tomorrow's maximum load, MW
  timing      which HOUR the peak falls in, as a probability over all 24 --
              because a control room hedges across a window, and a single
              predicted hour throws away exactly the information needed to
              decide how wide that window should be

The baseline that matters is `blockfc`: we ALREADY publish a 96-block load
forecast, so its own argmax is a free peak prediction. A dedicated peak
model has to beat that to justify existing at all. It is scored on the same
held-out window where the block model is genuinely out-of-sample.
"""
import importlib.util
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import holidays
import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

LF_DIR = HERE.parent.parent / "load_forecast"
OUT = HERE.parent / "output"
OUT.mkdir(exist_ok=True)
MAG_PATH = OUT / "peak_magnitude.txt"
HOUR_PATH = OUT / "peak_hour.txt"
METRICS_PATH = OUT / "metrics_peak.txt"
MAG_BIAS_PATH = OUT / "peak_bias.json"
SUMMARY_PATH = OUT / "peak_summary.json"

_spec = importlib.util.spec_from_file_location("lf_train", LF_DIR / "02_train_model.py")
lf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lf)

# Bid-time rule, same as everywhere else in this repo: planning for delivery
# day D happens around 12:00 on D-1, so D-1's own peak has not happened yet.
# Every load-derived lag is therefore >= 2 days. Weather is the exception --
# tomorrow's FORECAST is legitimately available, and it is the feature that
# carries most of the signal.
MAG_FEATURES = [
    "dow", "month", "is_weekend", "is_holiday", "doy_sin", "doy_cos",
    "peak_lag2", "peak_lag3", "peak_lag7", "peak_roll7", "peak_roll28",
    "peak_max7", "trough_lag2", "mean_lag2", "range_lag2",
    "tmax", "tmin", "tmean", "cdd", "hdd", "rh_mean", "cloud_mean",
    "tmax_lag1", "tmax_delta", "heat_streak",
]
HOUR_FEATURES = MAG_FEATURES + ["peakhour_lag2", "peakhour_lag3", "peakhour_lag7",
                                "peakhour_mode7"]


def _live_tail() -> pd.DataFrame:
    """Days after the training parquet ends, rebuilt from the live store.

    The parquet is frozen at the last training refresh, so on its own the
    panel stops months before today and nothing can be served. SLDC load and
    Open-Meteo weather in the store carry it forward -- and because weather
    includes the FORECAST rows, the tail extends past today to cover the
    delivery day we are actually predicting.
    """
    from ingest import store
    load = store.read("load_5min")["delhi"].resample("15min").mean()
    w = store.read("weather")
    w = w[~w.index.duplicated(keep="first")]
    wx = (w[w["kind"] == "actual"][["temp_c", "rh_pct", "cloud_pct"]]
          .combine_first(w[w["kind"] == "forecast"][["temp_c", "rh_pct", "cloud_pct"]]))
    if not len(wx):
        return pd.DataFrame()

    g = load.groupby(load.index.date)
    d = pd.DataFrame({
        "peak_mw": g.max(), "trough_mw": g.min(), "mean_mw": g.mean(),
        "peak_hour": pd.Series({k: v.hour for k, v in g.idxmax().items()}),
    }) if len(load) else pd.DataFrame(
        columns=["peak_mw", "trough_mw", "mean_mw", "peak_hour"])
    d.index = pd.to_datetime(d.index)

    wd = wx.groupby(wx.index.date)
    wdf = pd.DataFrame({"tmax": wd["temp_c"].max(), "tmin": wd["temp_c"].min(),
                        "tmean": wd["temp_c"].mean(), "rh_mean": wd["rh_pct"].mean(),
                        "cloud_mean": wd["cloud_pct"].mean()})
    wdf.index = pd.to_datetime(wdf.index)
    # weather runs ahead of load; keep those rows -- that is the whole point
    return wdf.join(d, how="left")


def _daily_panel(live: bool = True) -> pd.DataFrame:
    """One row per day: peak, peak hour, and the weather that drives them."""
    df = pd.read_parquet(LF_DIR / "data" / "model_table.parquet")
    load = df["load_mw"]
    g = load.groupby(load.index.date)
    d = pd.DataFrame({
        "peak_mw": g.max(),
        "trough_mw": g.min(),
        "mean_mw": g.mean(),
        "peak_hour": pd.Series({k: v.hour for k, v in g.idxmax().items()}),
    })
    d.index = pd.to_datetime(d.index)
    d = d.reindex(pd.date_range(d.index.min(), d.index.max(), freq="D"))

    wd = df[["temp_c", "rh_pct", "cloud_pct"]].groupby(df.index.date)
    w = pd.DataFrame({"tmax": wd["temp_c"].max(), "tmin": wd["temp_c"].min(),
                      "tmean": wd["temp_c"].mean(), "rh_mean": wd["rh_pct"].mean(),
                      "cloud_mean": wd["cloud_pct"].mean()})
    w.index = pd.to_datetime(w.index)
    d = d.join(w)

    if live:
        tail = _live_tail()
        tail = tail[tail.index > d.index.max()] if len(tail) else tail
        if len(tail):
            d = pd.concat([d, tail.reindex(columns=d.columns)])
            d = d.reindex(pd.date_range(d.index.min(), d.index.max(), freq="D"))
    return d


def build_features(d: pd.DataFrame) -> pd.DataFrame:
    f = d.copy()
    idx = f.index
    f["dow"] = idx.dayofweek
    f["month"] = idx.month
    f["is_weekend"] = (f["dow"] >= 5).astype(int)
    doy = idx.dayofyear
    f["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    ind = holidays.India(years=range(idx.year.min(), idx.year.max() + 1))
    f["is_holiday"] = pd.Series(idx.date, index=idx).isin(ind).astype(int)

    for lag in (2, 3, 7):
        f[f"peak_lag{lag}"] = f["peak_mw"].shift(lag)
        f[f"peakhour_lag{lag}"] = f["peak_hour"].shift(lag)
    base = f["peak_mw"].shift(2)
    f["peak_roll7"] = base.rolling(7, min_periods=3).mean()
    f["peak_roll28"] = base.rolling(28, min_periods=10).mean()
    f["peak_max7"] = base.rolling(7, min_periods=3).max()
    f["trough_lag2"] = f["trough_mw"].shift(2)
    f["mean_lag2"] = f["mean_mw"].shift(2)
    f["range_lag2"] = f["peak_lag2"] - f["trough_lag2"]
    f["peakhour_mode7"] = (f["peak_hour"].shift(2)
                           .rolling(7, min_periods=3)
                           .apply(lambda s: s.mode().iloc[0] if len(s.mode()) else np.nan))

    f["cdd"] = np.maximum(f["tmean"] - 24, 0)
    f["hdd"] = np.maximum(18 - f["tmean"], 0)
    f["tmax_lag1"] = f["tmax"].shift(1)
    f["tmax_delta"] = f["tmax"] - f["tmax_lag1"]
    f["heat_streak"] = (f["tmax"] > 38).astype(int).rolling(3, min_periods=1).sum()
    return f


def _splits(f: pd.DataFrame):
    test_start = f.index.max() - pd.DateOffset(months=6)
    val_start = test_start - pd.DateOffset(months=6)
    return (f[f.index < val_start],
            f[(f.index >= val_start) & (f.index < test_start)],
            f[f.index >= test_start])


def _block_forecast_peaks(days: pd.DatetimeIndex) -> pd.DataFrame:
    """Peak MW and peak hour implied by our own 96-block load forecast.

    This is the incumbent: the block model already exists, so its argmax is
    a peak forecast we get for nothing. Predicted here over the same days
    the peak models are tested on, where the block model is out-of-sample.
    """
    from models import load_model
    df = pd.read_parquet(LF_DIR / "data" / "model_table.parquet")
    feats = lf.build_features(df).dropna(subset=lf.FEATURES)
    keep = feats.index.normalize().isin(days)
    part = feats[keep]
    if not len(part):
        return pd.DataFrame(columns=["bf_peak_mw", "bf_peak_hour"])
    pred = np.mean([b.predict(part[lf.FEATURES])
                    for b in load_model._boosters()], axis=0)
    s = pd.Series(pred, index=part.index)
    g = s.groupby(s.index.normalize())
    return pd.DataFrame({
        "bf_peak_mw": g.max(),
        "bf_peak_hour": pd.Series({k: v.hour for k, v in g.idxmax().items()}),
    })


def train() -> str:
    # Trained on the curated parquet only. The live tail is for SERVING: it
    # is raw SLDC telemetry with gaps, and letting it drift into training
    # would quietly move the reported test window every time the poller runs.
    f = build_features(_daily_panel(live=False))
    f = f.dropna(subset=HOUR_FEATURES + ["peak_mw", "peak_hour"])
    train_, val, test = _splits(f)
    lines = ["day-ahead peak forecast -- Delhi",
             f"train {len(train_):,} days | val {len(val):,} | test {len(test):,}",
             f"test window {test.index.min():%Y-%m-%d} -> {test.index.max():%Y-%m-%d}"]

    bf = _block_forecast_peaks(test.index)
    bf = bf.reindex(test.index)

    # ---------------- magnitude ----------------------------------------
    mag = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.02, num_leaves=31,
                            min_child_samples=15, subsample=0.8, subsample_freq=1,
                            colsample_bytree=0.8, reg_lambda=2.0,
                            random_state=42, verbose=-1)
    mag.fit(train_[MAG_FEATURES], train_["peak_mw"],
            eval_set=[(val[MAG_FEATURES], val["peak_mw"])],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    mag.booster_.save_model(str(MAG_PATH))

    # ---- bias correction, fitted on VALIDATION only --------------------
    # The peak model under-forecasts, for the same reason the block model
    # does: Delhi's load is growing and a learner fitted on older history
    # sits below the recent level. The shortfall is stable across windows
    # (99 MW on validation, 168 MW on test), which is what makes it a bias
    # rather than noise -- so it is worth removing, and removing it with a
    # constant measured on validation keeps test untouched. Under-forecasting
    # the peak is also the dangerous direction for a control room: it
    # under-commits generation.
    raw_test = mag.predict(test[MAG_FEATURES])
    val_bias = float(np.mean(val["peak_mw"].values - mag.predict(val[MAG_FEATURES])))
    MAG_BIAS_PATH.write_text(json.dumps({
        "bias_mw": round(val_bias, 1),
        "fitted_on": "validation",
        "val_from": str(val.index.min().date()),
        "val_to": str(val.index.max().date()),
    }, indent=2))

    y = test["peak_mw"].values
    cands = {
        "model": raw_test + val_bias,
        "model_uncorrected": raw_test,
        "blockfc": bf["bf_peak_mw"].values,
        "persist_d2": test["peak_lag2"].values,
        "sameweekday": test["peak_lag7"].values,
    }
    lines += ["", "PEAK MAGNITUDE (MW):",
              f"  (model = raw + {val_bias:+.0f} MW bias correction fitted on validation)",
              f"  {'':<18} {'MAE':>8} {'MAPE':>8} {'bias':>8} {'p90 err':>9}"]
    mag_scores = {}
    for name, p in cands.items():
        ok = ~np.isnan(p)
        e = y[ok] - p[ok]
        mag_scores[name] = {
            "mae": float(np.mean(np.abs(e))),
            "mape": float(np.mean(np.abs(e) / y[ok]) * 100),
            "bias": float(np.mean(e)),
            "p90_abs_err": float(np.quantile(np.abs(e), 0.9)),
        }
        s = mag_scores[name]
        lines.append(f"  {name:<18} {s['mae']:7.0f} {s['mape']:7.2f}% "
                     f"{s['bias']:+7.0f} {s['p90_abs_err']:8.0f}")

    # ---------------- timing -------------------------------------------
    # Multiclass over all 24 hours, so the output is a DISTRIBUTION. The
    # top-1 hour is reported, but the number a control room can act on is
    # "the smallest window that holds 80% of the probability".
    hrs = sorted(train_["peak_hour"].unique())
    remap = {h: i for i, h in enumerate(hrs)}
    ytr = train_["peak_hour"].map(remap)
    yva = val["peak_hour"].map(remap)
    ok_va = yva.notna()
    clf = lgb.LGBMClassifier(objective="multiclass", num_class=len(hrs),
                             n_estimators=1500, learning_rate=0.03, num_leaves=31,
                             min_child_samples=15, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, reg_lambda=2.0,
                             random_state=42, verbose=-1)
    clf.fit(train_[HOUR_FEATURES], ytr,
            eval_set=[(val.loc[ok_va, HOUR_FEATURES], yva[ok_va])],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    clf.booster_.save_model(str(HOUR_PATH))
    json.dump({"hours": [int(h) for h in hrs]},
              open(OUT / "peak_hour_classes.json", "w"))

    proba = clf.predict_proba(test[HOUR_FEATURES])
    hours = np.array(hrs)
    pred_hour = hours[proba.argmax(axis=1)]
    yh = test["peak_hour"].values

    def circ_err(a, b):
        """Hours are circular: 23:00 and 00:00 are one hour apart, not 23."""
        d = np.abs(a - b) % 24
        return np.minimum(d, 24 - d)

    thours = {
        "model": pred_hour,
        "blockfc": bf["bf_peak_hour"].values,
        "persist_d2": test["peakhour_lag2"].values,
        "sameweekday": test["peakhour_lag7"].values,
    }
    lines += ["", "PEAK TIMING (hour of day):",
              f"  {'':<13} {'exact':>8} {'+/-1h':>8} {'+/-2h':>8} {'MAE h':>8}"]
    hour_scores = {}
    for name, p in thours.items():
        ok = ~pd.isna(p)
        e = circ_err(yh[ok], np.asarray(p, dtype=float)[ok])
        hour_scores[name] = {
            "exact_pct": float(np.mean(e == 0) * 100),
            "within1_pct": float(np.mean(e <= 1) * 100),
            "within2_pct": float(np.mean(e <= 2) * 100),
            "mae_hours": float(np.mean(e)),
        }
        s = hour_scores[name]
        lines.append(f"  {name:<13} {s['exact_pct']:7.1f}% {s['within1_pct']:7.1f}% "
                     f"{s['within2_pct']:7.1f}% {s['mae_hours']:7.2f}")

    # the distributional output: smallest set of hours holding 80% mass
    order = np.argsort(-proba, axis=1)
    cum = np.take_along_axis(proba, order, axis=1).cumsum(axis=1)
    k80 = (cum < 0.80).sum(axis=1) + 1
    hit80 = np.array([yh[i] in hours[order[i, :k80[i]]] for i in range(len(yh))])
    lines += ["",
              f"  probability output: an 80% window needs {k80.mean():.1f} hours "
              f"on average and actually contains the peak "
              f"{hit80.mean() * 100:.1f}% of the time",
              f"  (a flat guess over 24 hours would need 19.2 hours for the same 80%)"]

    best_m = min(mag_scores, key=lambda k: mag_scores[k]["mae"])
    best_t = max(hour_scores, key=lambda k: hour_scores[k]["within1_pct"])
    lines += ["",
              f"best magnitude: {best_m} "
              f"(MAE {mag_scores[best_m]['mae']:.0f} MW)",
              f"best timing:    {best_t} "
              f"(+/-1h {hour_scores[best_t]['within1_pct']:.1f}%)"]
    if best_m != "model" or best_t != "model":
        lines.append("  a baseline wins here and we say so -- the block forecast "
                     "we already ship is hard to beat,")
        lines.append("  which is itself the finding: peak intelligence comes free "
                     "with the 96-block model.")

    report = "\n".join(lines)
    print(report)
    METRICS_PATH.write_text(report)
    SUMMARY_PATH.write_text(json.dumps({
        "test_from": str(test.index.min().date()),
        "test_to": str(test.index.max().date()),
        "n_test_days": int(len(test)),
        "magnitude": mag_scores, "timing": hour_scores,
        "window80_hours": round(float(k80.mean()), 2),
        "window80_hit_pct": round(float(hit80.mean() * 100), 1),
        "best_magnitude": best_m, "best_timing": best_t,
        "peak_hour_distribution": {
            int(h): round(float(v) * 100, 1) for h, v in
            test["peak_hour"].value_counts(normalize=True).sort_index().items()},
    }, indent=2))
    return report


def forecast(target: date | None = None) -> dict:
    """Tomorrow's peak: magnitude, most likely hour, and the 80% window."""
    target = target or (date.today() + timedelta(days=1))
    f = build_features(_daily_panel())
    ts = pd.Timestamp(target)
    if ts not in f.index:
        raise ValueError(f"no daily panel row for {target}")
    row = f.loc[[ts]]

    out = {"day": str(target)}
    if MAG_PATH.exists():
        raw = float(lgb.Booster(model_file=str(MAG_PATH)).predict(row[MAG_FEATURES])[0])
        bias = (json.loads(MAG_BIAS_PATH.read_text())["bias_mw"]
                if MAG_BIAS_PATH.exists() else 0.0)
        out["peak_mw"] = raw + bias
        out["peak_mw_uncorrected"] = raw
        out["bias_correction_mw"] = bias
    if HOUR_PATH.exists():
        hours = np.array(json.load(open(OUT / "peak_hour_classes.json"))["hours"])
        p = lgb.Booster(model_file=str(HOUR_PATH)).predict(row[HOUR_FEATURES])[0]
        order = np.argsort(-p)
        k = int((np.cumsum(p[order]) < 0.80).sum() + 1)
        out["peak_hour"] = int(hours[order[0]])
        out["peak_hour_confidence_pct"] = round(float(p[order[0]]) * 100, 1)
        out["window80_hours"] = sorted(int(h) for h in hours[order[:k]])
        out["hour_probabilities"] = {int(h): round(float(v) * 100, 1)
                                     for h, v in zip(hours, p) if v >= 0.01}
    return out


if __name__ == "__main__":
    train()
