"""RTM price and DAM->RTM spread forecast.

The Real-Time Market is where a battery actually earns in India: 48
half-hourly auctions a day, gate closure ~1 hour before delivery, and a
spread against DAM whose standard deviation is Rs 1,630/MWh -- rising to
Rs 2,382/MWh in the 19:00 hour. The DAM position is financially firm once
it clears; every rupee after that comes from trading the deviation.

Two horizons, because they have genuinely different information sets and
mixing them would leak:

  dayahead  bidding the DAM at ~12:00 on D-1 for delivery day D. Tomorrow's
            DAM MCP has NOT cleared yet, so it cannot be a feature. Only
            lags of >= 1 day are legal. Feeds the three-way co-optimizer,
            which needs an RTM view at DAM bid time.

  intraday  re-optimising during delivery day D. Today's DAM MCP is known
            for every block (it cleared 13:00 yesterday), and RTM prices
            are known up to gate closure. This is the model that replaces
            the hour-of-day ratio currently used by optimize/rtm_reopt.py.

Both are scored against the incumbent they must beat, not against nothing:

  dam_flat   RTM = DAM                      (the naive "markets converge" view)
  dam_ratio  RTM = DAM x hour-of-day ratio  (what rtm_reopt.py does today)
  persist    RTM = same block yesterday / last known block (intraday)

Level accuracy is reported, but the number that decides money is the
SPREAD: getting sign(RTM - DAM) right is what tells the optimizer whether
to sell into the RTM or hold the DAM position.
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
MODEL_PATH = OUT / "rtm_model_{horizon}.txt"
METRICS_PATH = OUT / "metrics_rtm.txt"
CHAMPIONS_PATH = OUT / "rtm_champions.json"
B = 96          # 15-min blocks per day
CAP = 10000.0
FLOOR = 50.0    # MAPE denominator floor, same convention as price_model
LEAD_BLOCKS = 6  # 90 min: RTM gate closure (~1h) plus publication lag

CAL = ["block", "dow", "month", "is_weekend", "is_holiday",
       "hour_sin", "hour_cos", "doy_sin", "doy_cos"]

# lags of >= 1 day only -- legal at DAM bid time on D-1
DAYAHEAD = CAL + [
    "r_lag_1d", "r_lag_2d", "r_lag_7d",
    "r_roll7d_mean", "r_roll7d_std", "r_block7d_mean",
    "r_prevday_mean", "r_prevday_max", "r_prevday_min",
    "d_lag_1d", "d_lag_7d", "d_prevday_mean", "d_prevday_max",
    "spread_lag_1d", "spread_block7d_mean", "ratio_block7d_mean",
    "rbid_lag_1d", "sbid_lag_1d", "rbidgap_lag_1d",
    "load_lag_2d", "temp_c", "cdh",
]

# delivery day has begun: today's DAM curve is known, but for a block many
# hours out no RTM session near it has cleared yet
DAM_KNOWN = ["dam_now", "dam_day_mean", "dam_day_max", "dam_rank_in_day"]
SAMEDAY = DAYAHEAD + DAM_KNOWN

# within reach of gate closure: recent RTM clearings are known too
INTRADAY = SAMEDAY + [
    "r_intra_lag", "r_intra_lag2", "r_intra_lag4",
    "r_intra_roll8_mean", "r_intra_roll8_std",
    "spread_intra_lag", "spread_intra_roll8_mean",
    "blocks_elapsed",
]

FEATURES = {"dayahead": DAYAHEAD, "sameday": SAMEDAY, "intraday": INTRADAY}
DAM_IS_KNOWN = {"dayahead": False, "sameday": True, "intraday": True}

# The features that DEFINE each horizon's information advantage. At serving
# time these must be present or the model is not entitled to be used; every
# other feature may be NaN, because LightGBM routes missing values natively
# and a stale 7-day lag is no reason to fall back to persistence.
CRITICAL = {
    "dayahead": ["r_lag_1d", "r_block7d_mean"],
    "sameday": ["dam_now", "r_block7d_mean"],
    "intraday": ["dam_now", "r_intra_lag"],
}


def _table() -> pd.DataFrame:
    """15-min frame: RTM + DAM + Delhi load + temperature on one grid."""
    r = store.read("rtm_price")[["mcp_rs_mwh", "purchase_bid_mw", "sell_bid_mw"]]
    r = r.rename(columns={"mcp_rs_mwh": "rtm", "purchase_bid_mw": "rbid",
                          "sell_bid_mw": "sbid"})
    d = store.read("dam_price")[["mcp_rs_mwh"]].rename(columns={"mcp_rs_mwh": "dam"})

    # Span BOTH feeds, not just RTM. They end at different times by design:
    # tomorrow's DAM clears at 13:00 today, while RTM only exists up to the
    # session that just cleared. Indexing on RTM alone silently truncated
    # today's DAM tail, which then read as "DAM unknown" for exactly the
    # evening blocks the intraday model is meant to price.
    idx = pd.date_range(min(r.index.min(), d.index.min()),
                        max(r.index.max(), d.index.max()), freq="15min")
    r, d = r.reindex(idx), d.reindex(idx)

    load = store.read("load_5min", since=str(idx.min() - pd.Timedelta(days=10)))
    load = load["delhi"].resample("15min").mean()
    load = load.reindex(idx).interpolate(limit_direction="both").ffill().bfill()

    w = store.read("weather", since=str(idx.min() - pd.Timedelta(days=10)))
    w = w[~w.index.duplicated(keep="first")]
    temp = (w[w["kind"] == "actual"]["temp_c"]
            .combine_first(w[w["kind"] == "forecast"]["temp_c"])
            .resample("15min").interpolate(limit=8))

    return pd.concat([r, d, load.rename("load_mw"), temp.rename("temp_c")],
                     axis=1).loc[idx]


def _same_block_prev_days(s: pd.Series, days: int = 7) -> pd.Series:
    """Mean of this same 15-min block over the previous `days` days.

    Built from explicit day-shifts rather than a rolling window: on a
    regular 15-min grid s.shift(k*B) IS "same block, k days ago", so this
    is hour-specific by construction and can never see the present.
    """
    return pd.concat([s.shift(k * B) for k in range(1, days + 1)], axis=1).mean(axis=1)


def build_features(df: pd.DataFrame, horizon: str = "intraday") -> pd.DataFrame:
    f = df.copy()
    idx = f.index
    hour = idx.hour + idx.minute / 60
    f["block"] = idx.hour * 4 + idx.minute // 15
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

    # ---- day-lagged RTM ------------------------------------------------
    f["r_lag_1d"] = f["rtm"].shift(B)
    f["r_lag_2d"] = f["rtm"].shift(2 * B)
    f["r_lag_7d"] = f["rtm"].shift(7 * B)
    base = f["rtm"].shift(B)
    f["r_roll7d_mean"] = base.rolling(7 * B, min_periods=3 * B).mean()
    f["r_roll7d_std"] = base.rolling(7 * B, min_periods=3 * B).std()
    f["r_block7d_mean"] = _same_block_prev_days(f["rtm"])

    dates = pd.Series(idx.date, index=idx)
    ragg = f["rtm"].groupby(idx.date).agg(["mean", "max", "min"]).shift(1)
    f["r_prevday_mean"] = dates.map(ragg["mean"])
    f["r_prevday_max"] = dates.map(ragg["max"])
    f["r_prevday_min"] = dates.map(ragg["min"])

    # ---- day-lagged DAM and spread -------------------------------------
    f["d_lag_1d"] = f["dam"].shift(B)
    f["d_lag_7d"] = f["dam"].shift(7 * B)
    dagg = f["dam"].groupby(idx.date).agg(["mean", "max"]).shift(1)
    f["d_prevday_mean"] = dates.map(dagg["mean"])
    f["d_prevday_max"] = dates.map(dagg["max"])

    spread = f["rtm"] - f["dam"]
    ratio = (f["rtm"] / f["dam"].replace(0, np.nan)).clip(0.2, 5.0)
    f["spread_lag_1d"] = spread.shift(B)
    f["spread_block7d_mean"] = _same_block_prev_days(spread)
    f["ratio_block7d_mean"] = _same_block_prev_days(ratio)

    f["rbid_lag_1d"] = f["rbid"].shift(B)
    f["sbid_lag_1d"] = f["sbid"].shift(B)
    f["rbidgap_lag_1d"] = f["rbid_lag_1d"] - f["sbid_lag_1d"]

    f["load_lag_2d"] = f["load_mw"].shift(2 * B)
    f["cdh"] = np.maximum(f["temp_c"] - 24, 0)

    if DAM_IS_KNOWN.get(horizon):
        # today's DAM curve cleared at 13:00 yesterday -- fully known
        f["dam_now"] = f["dam"]
        today = f["dam"].groupby(idx.date)
        f["dam_day_mean"] = dates.map(today.mean())
        f["dam_day_max"] = dates.map(today.max())
        f["dam_rank_in_day"] = today.rank(pct=True)

    if horizon == "intraday":
        # RTM history up to gate closure: never closer than LEAD_BLOCKS
        f["r_intra_lag"] = f["rtm"].shift(LEAD_BLOCKS)
        f["r_intra_lag2"] = f["rtm"].shift(LEAD_BLOCKS + 2)
        f["r_intra_lag4"] = f["rtm"].shift(LEAD_BLOCKS + 4)
        rl = f["rtm"].shift(LEAD_BLOCKS)
        f["r_intra_roll8_mean"] = rl.rolling(8, min_periods=3).mean()
        f["r_intra_roll8_std"] = rl.rolling(8, min_periods=3).std()
        sl = spread.shift(LEAD_BLOCKS)
        f["spread_intra_lag"] = sl
        f["spread_intra_roll8_mean"] = sl.rolling(8, min_periods=3).mean()
        f["blocks_elapsed"] = f["block"]
    return f


# --------------------------------------------------------------------------
# baselines the model has to beat
# --------------------------------------------------------------------------
def _hour_ratio_from(train_: pd.DataFrame) -> pd.Series:
    """The incumbent in optimize/rtm_reopt.py: median RTM/DAM by hour.

    Fitted on the TRAIN slice only so the baseline gets exactly the same
    information the model does -- otherwise we would be flattering ourselves.
    """
    ratio = (train_["rtm"] / train_["dam"].replace(0, np.nan)).clip(0.3, 3.0).dropna()
    byhour = ratio.groupby(ratio.index.hour).median()
    return byhour.reindex(range(24)).interpolate(limit_direction="both").fillna(1.0)


def _dam_forecast(rows: pd.DataFrame) -> np.ndarray:
    """Our own day-ahead DAM forecast on `rows` -- the honest day-ahead anchor.

    Falls back to the previous day's DAM if the price model is unavailable,
    which is still implementable at bid time (unlike the actual price).
    """
    try:
        from models import price_model as pm
        pf = pm.build_features(pm._table())
        pf = pf.reindex(rows.index)
        ok = pf[pm.FEATURES].notna().all(axis=1)
        out = rows["d_lag_1d"].to_numpy(dtype=float).copy()
        if ok.any():
            out[ok.to_numpy()] = pm.predict_hurdle(pf.loc[ok])
        return out
    except Exception as e:                       # model not trained yet
        print(f"  (DAM-forecast anchor unavailable: {e}; using D-1 DAM)")
        return rows["d_lag_1d"].to_numpy(dtype=float)


def _mape(y, p):
    return float(np.mean(np.abs(y - p) / np.maximum(y, 100)) * 100)


def _wape(y, p):
    """Weighted absolute percentage error = sum|err| / sum|actual|.

    The headline metric here, and MAPE is NOT, for a measured reason: RTM
    clears at Rs 0 on some blocks and 3.2% of the test window sits below
    Rs 100 (1st percentile: Rs 23). Plain MAPE divides by those, so a
    Rs 500 miss on a Rs 23 block scores 2,000% and a handful of cheap
    night blocks swamp the whole report. WAPE divides once, at the end,
    by the total energy value -- so it says what a trading desk means by
    "how far off were we", in rupees per rupee traded.
    """
    return float(np.sum(np.abs(y - p)) / np.sum(np.abs(y)) * 100)


def _smape(y, p):
    """Symmetric MAPE, bounded at 200% -- the near-zero-safe percentage."""
    denom = (np.abs(y) + np.abs(p)) / 2
    return float(np.mean(np.abs(y - p) / np.maximum(denom, 1e-9)) * 100)


def _score(y, p, index) -> dict:
    evening = (index.hour >= 17) & (index.hour <= 23)
    return {
        "wape": _wape(y, p),
        "smape": _smape(y, p),
        "mape": _mape(y, p),
        "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
        "mae": float(np.mean(np.abs(y - p))),
        "corr": float(np.corrcoef(y, p)[0, 1]),
        "evening_wape": _wape(y[evening], p[evening]),
    }


def train(horizon: str = "intraday", test_days: int = 60, val_days: int = 30):
    """Fit one horizon and score it against all three baselines."""
    if horizon not in FEATURES:
        raise ValueError(f"horizon must be one of {list(FEATURES)}")
    feats = FEATURES[horizon]
    f = build_features(_table(), horizon).dropna(subset=feats + ["rtm", "dam"])

    split = f.index.max().normalize() - pd.Timedelta(days=test_days)
    val_split = split - pd.Timedelta(days=val_days)
    train_ = f[f.index < val_split]
    val = f[(f.index >= val_split) & (f.index < split)]
    test = f[f.index >= split]
    if not len(test):
        raise RuntimeError("empty test window -- not enough RTM history")

    model = lgb.LGBMRegressor(
        n_estimators=3000, learning_rate=0.03, num_leaves=63,
        min_child_samples=40, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, reg_lambda=1.0, random_state=42, verbose=-1)
    model.fit(train_[feats], np.log(train_["rtm"].clip(lower=FLOOR)),
              eval_set=[(val[feats], np.log(val["rtm"].clip(lower=FLOOR)))],
              callbacks=[lgb.early_stopping(150, verbose=False),
                         lgb.log_evaluation(0)])
    model.booster_.save_model(str(MODEL_PATH).format(horizon=horizon))

    # The DAM anchor a baseline is ALLOWED to use depends on the horizon.
    # Intraday/sameday, today's DAM has cleared, so "RTM = DAM" is
    # implementable. Day-ahead it has NOT: scoring against actual DAM would
    # hand the baseline tomorrow's clearing price and rig the comparison
    # against our own model. So day-ahead baselines are anchored on our DAM
    # *forecast*, which is what a desk would really have at 12:00 on D-1.
    anchor_name = "dam" if DAM_IS_KNOWN[horizon] else "damhat"
    incumbent = f"{anchor_name}_ratio"
    byhour = _hour_ratio_from(train_)

    def candidates(part: pd.DataFrame) -> tuple[dict, np.ndarray]:
        anchor = (part["dam"].to_numpy(dtype=float) if DAM_IS_KNOWN[horizon]
                  else _dam_forecast(part))
        scale = part.index.hour.map(byhour).values
        cands = {
            "model": np.clip(np.exp(model.predict(part[feats])), 0, CAP),
            f"{anchor_name}_flat": np.clip(anchor, 0, CAP),
            incumbent: np.clip(anchor * scale, 0, CAP),
            "persist": (part["r_intra_lag"] if horizon == "intraday"
                        else part["r_lag_1d"]).to_numpy(dtype=float),
        }
        if not DAM_IS_KNOWN[horizon]:
            # reference line only -- it needs a price we cannot have at bid time
            cands["dam_oracle"] = part["dam"].to_numpy(dtype=float)
        return cands, anchor

    # ---- champion picked on VALIDATION, never on test -------------------
    # The sameday model leans hard on today's cleared DAM curve, and across
    # the price regime shift in this window that hurt it. Rather than hand
    # the optimizer whichever candidate happens to win on test (that is
    # leakage, and it is how you ship a model that only worked in a
    # backtest), each horizon selects on the validation slice and the
    # selection is then measured on test, whatever it says.
    val_cands, _ = candidates(val)
    yv = val["rtm"].values
    val_wape = {k: _wape(yv, p) for k, p in val_cands.items() if k != "dam_oracle"}
    champion = min(val_wape, key=val_wape.get)

    cands, anchor = candidates(test)
    y = test["rtm"].values
    dam = test["dam"].values
    info = {"intraday": f"lead {LEAD_BLOCKS} blocks = {LEAD_BLOCKS * 15} min; "
                        "today's DAM + recent RTM known",
            "sameday": "today's DAM known, no RTM session near the block yet",
            "dayahead": "lags >= 1 day only; tomorrow's DAM has not cleared"}
    lines = [f"RTM model -- horizon={horizon}  ({info[horizon]})",
             f"train {len(train_):,} | val {len(val):,} | test {len(test):,}"
             f"   test window {test.index.min():%Y-%m-%d} -> {test.index.max():%Y-%m-%d}",
             "",
             f"{'':<10} {'WAPE':>8} {'sMAPE':>8} {'MAE':>8} {'RMSE':>8} "
             f"{'corr':>7} {'even.WAPE':>10}"]
    scores = {}
    for name, p in cands.items():
        s = _score(y, p, test.index)
        scores[name] = s
        lines.append(f"{name:<10} {s['wape']:7.2f}% {s['smape']:7.1f}% "
                     f"{s['mae']:7.0f} {s['rmse']:7.0f} {s['corr']:7.3f} "
                     f"{s['evening_wape']:9.2f}%")

    # ---- the metric that actually drives money -------------------------
    # sign(RTM - DAM) tells the optimizer whether to sell into the RTM or
    # hold the DAM position. A level forecast that is close on average but
    # wrong about the sign is worth nothing.
    act_spread = y - dam
    lines += ["", f"spread (RTM - DAM) vs the {anchor_name} anchor, "
                  f"the trading signal:"]
    live = np.abs(act_spread) > 100          # ignore blocks where it makes no odds
    for name, p in cands.items():
        ps = p - anchor
        hit = float(np.mean(np.sign(ps[live]) == np.sign(act_spread[live])) * 100)
        mae = float(np.mean(np.abs(ps - act_spread)))
        scores[name]["spread_mae"] = mae
        scores[name]["direction_pct"] = hit
        if name in ("model", incumbent, "persist"):
            lines.append(f"  {name:<12} direction {hit:5.1f}%   "
                         f"spread MAE Rs {mae:,.0f}/MWh")
    lines.append(f"  {'coin flip':<12} direction  50.0%   "
                 f"(blocks scored: {int(live.sum()):,} of {len(y):,} with |spread| > Rs 100)")

    scorable = {k: v for k, v in cands.items() if k != "dam_oracle"}
    best = min(scorable, key=lambda k: scores[k]["wape"])
    gain = (scores[incumbent]["wape"] - scores["model"]["wape"]) \
        / scores[incumbent]["wape"] * 100
    served = scores[champion]
    lines += ["",
              f"CHAMPION (picked on validation): {champion}"
              f"   val WAPE {val_wape[champion]:.2f}%"
              f"  ->  served test WAPE {served['wape']:.2f}%, "
              f"direction {served.get('direction_pct', float('nan')):.1f}%",
              f"best on test would have been: {best} "
              f"({scores[best]['wape']:.2f}%)"
              + ("  <- selection agreed" if best == champion
                 else "  <- we do NOT switch to it; that would be test leakage"),
              f"model vs incumbent ({incumbent}): {gain:+.1f}% relative",
              "WAPE is the headline, not MAPE: RTM clears at Rs 0 on real blocks "
              "(1st pct Rs 23),",
              "so a per-block percentage divides by near-zero and reports nonsense. "
              "MAE in Rs/MWh is the",
              f"other honest read -- compare it to the Rs {np.std(act_spread):,.0f}/MWh "
              "standard deviation of the spread itself."]
    report = "\n".join(lines)
    print(report)
    scores["_meta"] = {
        "horizon": horizon, "champion": champion, "incumbent": incumbent,
        "val_wape": {k: round(v, 2) for k, v in val_wape.items()},
        "n_train": len(train_), "n_val": len(val), "n_test": len(test),
        "test_from": str(test.index.min().date()),
        "test_to": str(test.index.max().date()),
        "hour_ratio": {int(h): round(float(v), 3) for h, v in byhour.items()},
    }
    return model, scores, report


def train_all(test_days: int = 60) -> dict:
    import json
    reports, all_scores = [], {}
    for h in ("intraday", "sameday", "dayahead"):
        _, scores, rep = train(h, test_days=test_days)
        reports.append(rep)
        all_scores[h] = scores
        print()
    METRICS_PATH.write_text("\n\n".join(reports))
    CHAMPIONS_PATH.write_text(json.dumps({
        h: {**s["_meta"],
            "served": {k: round(v, 3) for k, v in s[s["_meta"]["champion"]].items()},
            "model": {k: round(v, 3) for k, v in s["model"].items()},
            "incumbent_scores": {k: round(v, 3)
                                 for k, v in s[s["_meta"]["incumbent"]].items()}}
        for h, s in all_scores.items()}, indent=2))
    return all_scores


# --------------------------------------------------------------------------
# serving
# --------------------------------------------------------------------------
def _frame_through(target: date) -> pd.DataFrame:
    """Store frame extended to cover `target`, with forecast temperature."""
    df = _table()
    end = pd.Timestamp(target) + pd.Timedelta(hours=23, minutes=45)
    if end > df.index.max():
        df = df.reindex(pd.date_range(df.index.min(), end, freq="15min"))
    w = store.read("weather")
    fc = w[w["kind"] == "forecast"]["temp_c"].resample("15min").interpolate()
    df["temp_c"] = df["temp_c"].combine_first(fc)
    return df


def forecast_day(target: date | None = None, horizon: str = "dayahead",
                 dam_curve: pd.Series | None = None) -> pd.DataFrame:
    """Per-block RTM forecast for `target`.

    horizon="dayahead" needs nothing about the target day beyond weather.
    horizon="intraday" needs the target day's cleared DAM curve; pass it as
    `dam_curve` (or leave it to be read from the store).
    """
    path = Path(str(MODEL_PATH).format(horizon=horizon))
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} missing -- run `python models/rtm_model.py`")
    target = target or (date.today() + timedelta(days=1))
    df = _frame_through(target)
    if dam_curve is not None:
        df.loc[dam_curve.index, "dam"] = dam_curve.values

    day = build_features(df, horizon)
    day = day[day.index.date == target]
    feats = FEATURES[horizon]
    missing = [c for c in feats if day[c].isna().all()]
    if missing:
        raise ValueError(f"no data for features {missing} on {target}")

    booster = lgb.Booster(model_file=str(path))
    out = pd.DataFrame(index=day.index)
    out["forecast_rtm"] = np.clip(np.exp(booster.predict(day[feats])), 0, CAP)
    out["dam"] = day["dam"].values
    out["forecast_spread"] = out["forecast_rtm"] - out["dam"]
    return out


def _champions() -> dict:
    import json
    if CHAMPIONS_PATH.exists():
        return json.loads(CHAMPIONS_PATH.read_text())
    return {}


def _predict(part: pd.DataFrame, horizon: str) -> np.ndarray:
    path = Path(str(MODEL_PATH).format(horizon=horizon))
    booster = lgb.Booster(model_file=str(path))
    return np.clip(np.exp(booster.predict(part[FEATURES[horizon]])), 0, CAP)


def serve_curve(today: date, now: pd.Timestamp | None = None
                ) -> tuple[pd.Series, dict]:
    """Best-available RTM price for every block of `today`, block by block.

    This is what optimize/rtm_reopt.py trades against, and it is a cascade
    rather than one model, because the information available differs across
    the day:

      cleared blocks            ACTUAL RTM clearing price
      up to 90 min past those   intraday model  (its lags exist -- champion,
                                WAPE 26.6% vs 33.0% for the ratio it replaces)
      the rest of the day       sameday champion (no RTM near those blocks
                                has cleared, so the intraday model would be
                                fed lags it will never have at bid time)

    Each block is labelled with the source that produced it, so nothing in
    the UI can present a projection as a measurement.
    """
    now = pd.Timestamp.now() if now is None else pd.Timestamp(now)
    champs = _champions()
    df = _frame_through(today)
    day_mask = df.index.date == today
    if not day_mask.any():
        raise RuntimeError(f"no price grid for {today}")

    rtm_actual = df.loc[day_mask, "rtm"]
    cleared = rtm_actual.notna()
    last_rtm = rtm_actual[cleared].index.max() if cleared.any() else None

    out = rtm_actual.copy()
    source = pd.Series("", index=out.index, dtype=object)
    source[cleared] = "actual"

    # intraday reach: exactly the blocks whose LEAD-lagged RTM exists
    reach = (out.index <= last_rtm + pd.Timedelta(minutes=15 * LEAD_BLOCKS)) \
        if last_rtm is not None else np.zeros(len(out), dtype=bool)
    counts = {"actual": int(cleared.sum())}

    for horizon, mask in (("intraday", ~cleared & reach),
                          ("sameday", ~cleared & ~reach)):
        if not mask.any():
            continue
        champ = champs.get(horizon, {}).get("champion", "model")
        part = build_features(df, horizon)[day_mask][mask]

        # Every candidate the champion selection considered, computed for
        # these blocks. The champion is tried first, then the others in
        # order of their validation score. Coalescing like this matters:
        # persistence is NaN wherever yesterday's block is missing (the RTM
        # feed has real gaps), and an earlier version filled those holes
        # with the day's median. That painted a flat Rs 3,279 plateau next
        # to cap-priced blocks and handed the LP a large arbitrage that
        # existed only in the fallback. A missing price must degrade to the
        # next real estimate, never to a constant.
        cand = {}
        mpath = Path(str(MODEL_PATH).format(horizon=horizon))
        usable = part[CRITICAL[horizon]].notna().all(axis=1)
        if mpath.exists() and usable.any():
            v = pd.Series(np.nan, index=part.index)
            v[usable.values] = _predict(part[usable], horizon)
            cand["model"] = v
        cand["persist"] = (part["r_intra_lag"] if horizon == "intraday"
                           else part["r_lag_1d"])
        byhour = pd.Series(champs.get(horizon, {}).get("hour_ratio", {}),
                           dtype=float)
        if len(byhour):
            byhour.index = byhour.index.astype(int)
            cand["dam_ratio"] = part["dam"] * part.index.hour.map(byhour).astype(float)
        cand["dam_flat"] = part["dam"]
        cand["r_block7d_mean"] = part["r_block7d_mean"]   # last resort, still real

        order = [champ] + [k for k in ("model", "persist", "dam_ratio",
                                       "dam_flat", "r_block7d_mean")
                           if k != champ]
        vals = pd.Series(np.nan, index=part.index)
        src = pd.Series("", index=part.index, dtype=object)
        for name in order:
            if name not in cand:
                continue
            fill = vals.isna() & cand[name].notna()
            if not fill.any():
                continue
            vals[fill] = cand[name][fill]
            tag = f"{horizon} {name}" + (" (champion)" if name == champ else " (fill)")
            src[fill] = tag
            counts[tag] = counts.get(tag, 0) + int(fill.sum())
        out.loc[part.index] = vals.values
        source.loc[part.index] = src.values

    gap = out.isna()
    if gap.any():
        # nothing real left for these blocks -- keep them out of the LP's
        # reach by pricing them at the day's median, and say so loudly
        out[gap] = float(out.dropna().median()) if out.notna().any() else 0.0
        source[gap] = "median fallback (NO real estimate available)"
        counts["median fallback"] = int(gap.sum())

    meta = {
        "basis": "cascade: actual RTM -> intraday model -> sameday champion",
        "blocks_by_source": counts,
        "last_cleared_rtm": str(last_rtm) if last_rtm is not None else None,
        "intraday_champion": champs.get("intraday", {}).get("champion"),
        "intraday_wape_pct": champs.get("intraday", {}).get("served", {}).get("wape"),
        "intraday_direction_pct": champs.get("intraday", {})
                                        .get("served", {}).get("direction_pct"),
        "sameday_champion": champs.get("sameday", {}).get("champion"),
        "asof": str(now.floor("min")),
    }
    return out.rename("rtm_price"), meta


if __name__ == "__main__":
    horizon = sys.argv[1] if len(sys.argv) > 1 else None
    if horizon:
        train(horizon)
    else:
        train_all()
