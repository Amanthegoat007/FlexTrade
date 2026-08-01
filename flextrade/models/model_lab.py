"""Model improvement lab — controlled experiments, honest leaderboards.

Rules of the lab:
  * One change per experiment, measured against the SAME chronological
    split the production script uses. No cherry-picked windows.
  * The test window is touched only for the final number — model choice
    (early stopping) happens on the validation window.
  * A change is adopted only if it beats baseline on the test metric AND
    the mechanism is explainable (we present these models to judges).

Diagnosed error sources driving the experiment set:

LOAD (baseline 4.98% test MAPE):
  D1  Level drift — Delhi load grows YoY and the test window is the most
      recent 6 months, so a model fitted mostly on older, lower levels
      under-predicts. Two attacks: RECENCY sample weights (half-life),
      and a RELATIVE TARGET y / sameblock_4w_mean (forecast the shape
      deviation, let the recent baseline carry the level — the classic
      stationarization trick in STLF).
  D2  Missing thermal-inertia / streak signals — AC load depends on how
      hot it HAS BEEN, not just how hot it is: prev-day temperature,
      3-day heat-streak, evening cooling demand interaction.
  D3  Capacity/growth trend — lag ratios (lag_2d / lag_7d etc.) encode
      short-run growth explicitly.

PRICE (baseline 22.6% test MAPE, corr 0.92):
  D4  Regime shift — test-window mean MCP ~35% above training history
      (documented when CQR was added). RECENCY weights attack this
      directly.
  D5  Cap censoring — MCP pins at the Rs 10,000 cap on ~10-30% of summer
      evening blocks; a regression through the cap smears it. A TWO-STAGE
      HURDLE (P(cap) classifier x below-cap regressor, expectation
      combined) treats the censoring explicitly.
  D6  Cross-market information — yesterday's RTM level/peak and the
      GDAM-DAM spread carry supply-tightness signal DAM lags alone miss;
      cap_share_prev7d is an explicit regime dial.
"""
import importlib.util
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from ingest import store  # noqa: E402
from models import price_model  # noqa: E402

LF_DIR = HERE.parent.parent / "load_forecast"
_spec = importlib.util.spec_from_file_location("lf_train", LF_DIR / "02_train_model.py")
lf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lf)

OUT = HERE.parent / "output"
B = 96


# ---------------------------------------------------------------- helpers --

def mape(y, p, floor=None):
    den = np.maximum(y, floor) if floor else y
    return float(np.mean(np.abs(y - p) / den) * 100)


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def recency_weights(idx: pd.DatetimeIndex, half_life_days: float) -> np.ndarray:
    age = np.asarray((idx.max() - idx).total_seconds()) / 86400
    return 0.5 ** (age / half_life_days)


LGB_BASE = dict(n_estimators=3000, learning_rate=0.03, num_leaves=127,
                min_child_samples=50, subsample=0.8, subsample_freq=1,
                colsample_bytree=0.8, reg_lambda=1.0, random_state=42,
                verbose=-1)
LGB_TUNED = dict(n_estimators=6000, learning_rate=0.015, num_leaves=255,
                 min_child_samples=20, subsample=0.8, subsample_freq=1,
                 colsample_bytree=0.7, reg_lambda=2.0, random_state=42,
                 verbose=-1)


# ============================================================ LOAD MODEL ==

def load_table() -> pd.DataFrame:
    df = pd.read_parquet(LF_DIR / "data" / "model_table.parquet")
    return lf.build_features(df)


EXTRA_LOAD_FEATURES = [
    "temp_lag_1d", "heat_streak_3d", "cdh_evening",
    "r_2_7", "r_2_14", "ewma_sameblock",
]


def add_load_features(f: pd.DataFrame) -> pd.DataFrame:
    f = f.copy()
    # D2: what the building stock has already absorbed
    f["temp_lag_1d"] = f["temp_c"].shift(B)
    daily_cdh = f["cdh"].groupby(f.index.date).transform("mean")
    f["heat_streak_3d"] = daily_cdh.shift(B).rolling(3 * B).mean()
    hour = f.index.hour
    f["cdh_evening"] = f["cdh"] * (((hour >= 17) | (hour <= 1)).astype(int))
    # D3: short-run growth encoded explicitly
    f["r_2_7"] = f["lag_2d"] / f["lag_7d"]
    f["r_2_14"] = f["lag_2d"] / f["lag_14d"]
    # D1: adaptive same-block baseline (EWM over same block, info to D-2)
    f["ewma_sameblock"] = (f["load_mw"].shift(2 * B)
                           .groupby(f.index.hour * 4 + f.index.minute // 15)
                           .transform(lambda s: s.ewm(halflife=7).mean()))
    return f


# frozen copy of the ORIGINAL production feature set (pre-24-Jul), so the
# lab's baseline stays the historical baseline even after production adopts
# lab winners (lf.FEATURES drifts; this list must not)
BASE_FEATURES_V0 = [
    "block", "hour_sin", "hour_cos", "dow", "month", "doy_sin", "doy_cos",
    "is_weekend", "is_holiday",
    "lag_2d", "lag_3d", "lag_7d", "lag_14d",
    "roll7d_mean", "roll7d_max", "roll7d_min", "sameblock_4w_mean",
    "temp_c", "temp_sq", "cdh", "hdh", "rh_pct", "temp_rh", "rain_mm",
    "cloud_pct", "apparent_temp_c", "temp_24h_mean",
]


def run_load_experiments() -> pd.DataFrame:
    f = add_load_features(load_table())
    feats_base = BASE_FEATURES_V0
    feats_plus = feats_base + [c for c in EXTRA_LOAD_FEATURES
                               if c not in feats_base]

    test_start = f.index.max() - pd.DateOffset(months=6)
    val_start = test_start - pd.DateOffset(months=6)

    def split(cols, extra=()):
        d = f.dropna(subset=list(cols) + ["load_mw"] + list(extra))
        return (d[d.index < val_start],
                d[(d.index >= val_start) & (d.index < test_start)],
                d[d.index >= test_start])

    results = []

    def run(name, cols, params, target="abs", weights_hl=None, seeds=(42,)):
        tr, va, te = split(cols, ("sameblock_4w_mean",) if target == "rel" else ())
        if target == "rel":
            ytr = tr["load_mw"] / tr["sameblock_4w_mean"]
            yva = va["load_mw"] / va["sameblock_4w_mean"]
        else:
            ytr, yva = tr["load_mw"], va["load_mw"]
        w = recency_weights(tr.index, weights_hl) if weights_hl else None
        preds = np.zeros(len(te))
        for seed in seeds:
            m = lgb.LGBMRegressor(**{**params, "random_state": seed})
            m.fit(tr[cols], ytr, sample_weight=w,
                  eval_set=[(va[cols], yva)], eval_metric="mape",
                  callbacks=[lgb.early_stopping(150, verbose=False),
                             lgb.log_evaluation(0)])
            p = m.predict(te[cols])
            preds += (p * te["sameblock_4w_mean"].values if target == "rel" else p)
        p = preds / len(seeds)
        y = te["load_mw"].values
        row = {"experiment": name, "test_mape_pct": round(mape(y, p), 3),
               "test_rmse_mw": round(rmse(y, p), 1), "n_test": len(te)}
        results.append(row)
        print(f"  {name:34s} MAPE {row['test_mape_pct']:.3f}%  RMSE {row['test_rmse_mw']:.0f} MW")
        return row

    print("LOAD experiments (test = last 6 months, identical split):")
    run("L0 baseline (production)", feats_base, LGB_BASE)
    run("L1 +thermal/growth features", feats_plus, LGB_BASE)
    run("L2 relative target", feats_base, LGB_BASE, target="rel")
    run("L3 recency weights (hl=180d)", feats_base, LGB_BASE, weights_hl=180)
    run("L4 tuned hyperparams", feats_base, LGB_TUNED)
    run("L5 L1+L2 (features + rel target)", feats_plus, LGB_BASE, target="rel")
    run("L6 L1+L3 (features + recency)", feats_plus, LGB_BASE, weights_hl=180)
    run("L7 L1+L3+L4", feats_plus, LGB_TUNED, weights_hl=180)
    run("L8 L7 + 3-seed ensemble", feats_plus, LGB_TUNED, weights_hl=180,
        seeds=(42, 7, 2026))
    return pd.DataFrame(results)


# =========================================================== PRICE MODEL ==

EXTRA_PRICE_FEATURES = ["rtm_prevday_mean", "rtm_prevday_max",
                        "gdam_spread_prevday", "cap_share_prev7d",
                        "p_momentum", "cdh_evening"]
CAP = 10000.0


def price_table() -> pd.DataFrame:
    f = price_model.build_features(price_model._table())
    idx = f.index
    dates = pd.Series(idx.date, index=idx)
    # D6: cross-market signals, all lagged >= 1 day (bid-time valid)
    try:
        rtm = store.read("rtm_price")["mcp_rs_mwh"]
        rday = rtm.groupby(rtm.index.date).agg(["mean", "max"]).shift(1)
        f["rtm_prevday_mean"] = dates.map(rday["mean"])
        f["rtm_prevday_max"] = dates.map(rday["max"])
    except Exception:
        f["rtm_prevday_mean"] = np.nan
        f["rtm_prevday_max"] = np.nan
    try:
        gdam = store.read("gdam_price")["mcp_rs_mwh"]
        gday = gdam.groupby(gdam.index.date).mean().shift(1)
        f["gdam_spread_prevday"] = dates.map(gday) - f["p_prevday_mean"]
    except Exception:
        f["gdam_spread_prevday"] = np.nan
    cap_flag = (f["mcp_rs_mwh"] >= CAP * 0.95).astype(float)
    f["cap_share_prev7d"] = cap_flag.shift(B).rolling(7 * B).mean()
    f["p_momentum"] = f["p_prevday_mean"] / f["p_roll7d_mean"]
    hour = idx.hour
    f["cdh_evening"] = f["cdh"] * (((hour >= 17) | (hour <= 1)).astype(int))
    return f


def run_price_experiments() -> pd.DataFrame:
    f = price_table()
    feats_base = price_model.FEATURES
    # RTM/GDAM history may not span the whole price history; only demand
    # the features that actually have data, and report the loss of rows
    extra_ok = [c for c in EXTRA_PRICE_FEATURES
                if f[c].notna().mean() > 0.5]
    feats_plus = feats_base + extra_ok
    dropped = [c for c in EXTRA_PRICE_FEATURES if c not in extra_ok]
    if dropped:
        print(f"  (skipping sparse features: {dropped})")

    split_t = f.index.max().normalize() - pd.Timedelta(days=60)
    val_t = split_t - pd.Timedelta(days=30)

    def split(cols):
        d = f.dropna(subset=list(cols) + ["mcp_rs_mwh"])
        return (d[d.index < val_t],
                d[(d.index >= val_t) & (d.index < split_t)],
                d[d.index >= split_t])

    results = []

    def evaluate(name, y, p, te_idx, n_test):
        evening = (te_idx.hour >= 17) & (te_idx.hour <= 23)
        capped = y >= CAP * 0.95
        row = {"experiment": name,
               "test_mape_pct": round(mape(y, p, floor=100), 2),
               "test_rmse": round(rmse(y, p), 0),
               "corr": round(float(np.corrcoef(y, p)[0, 1]), 4),
               "evening_mape_pct": round(mape(y[evening], p[evening], floor=100), 2),
               "cap_recall_pct": round(float(np.mean(p[capped] >= 9000) * 100), 1)
               if capped.any() else None,
               "n_test": n_test}
        results.append(row)
        print(f"  {name:34s} MAPE {row['test_mape_pct']:5.2f}%  corr {row['corr']:.3f}"
              f"  evening {row['evening_mape_pct']:5.2f}%  capRecall {row['cap_recall_pct']}")
        return row

    def run(name, cols, params=None, weights_hl=None, hurdle=False):
        params = params or dict(n_estimators=2000, learning_rate=0.03,
                                num_leaves=63, min_child_samples=40,
                                subsample=0.8, subsample_freq=1,
                                colsample_bytree=0.8, reg_lambda=1.0,
                                random_state=42, verbose=-1)
        tr, va, te = split(cols)
        w = recency_weights(tr.index, weights_hl) if weights_hl else None
        ytr = np.log(tr["mcp_rs_mwh"].clip(lower=50))
        yva = np.log(va["mcp_rs_mwh"].clip(lower=50))
        reg = lgb.LGBMRegressor(**params)
        reg.fit(tr[cols], ytr, sample_weight=w, eval_set=[(va[cols], yva)],
                callbacks=[lgb.early_stopping(100, verbose=False),
                           lgb.log_evaluation(0)])
        p = np.clip(np.exp(reg.predict(te[cols])), 0, CAP)
        if hurdle:
            # D5: explicit cap-censoring stage. Expectation combine:
            # E[y] = P(cap) * CAP + (1 - P(cap)) * E[y | below cap]
            ycap_tr = (tr["mcp_rs_mwh"] >= CAP * 0.95).astype(int)
            ycap_va = (va["mcp_rs_mwh"] >= CAP * 0.95).astype(int)
            clf = lgb.LGBMClassifier(**{**params, "n_estimators": 1500})
            clf.fit(tr[cols], ycap_tr, sample_weight=w,
                    eval_set=[(va[cols], ycap_va)],
                    callbacks=[lgb.early_stopping(100, verbose=False),
                               lgb.log_evaluation(0)])
            pcap = clf.predict_proba(te[cols])[:, 1]
            below = tr["mcp_rs_mwh"] < CAP * 0.95
            wb = w[below.values] if w is not None else None
            reg2 = lgb.LGBMRegressor(**params)
            vb = va["mcp_rs_mwh"] < CAP * 0.95
            reg2.fit(tr.loc[below, cols],
                     np.log(tr.loc[below, "mcp_rs_mwh"].clip(lower=50)),
                     sample_weight=wb,
                     eval_set=[(va.loc[vb, cols],
                                np.log(va.loc[vb, "mcp_rs_mwh"].clip(lower=50)))],
                     callbacks=[lgb.early_stopping(100, verbose=False),
                                lgb.log_evaluation(0)])
            pbelow = np.clip(np.exp(reg2.predict(te[cols])), 0, CAP)
            p = pcap * CAP + (1 - pcap) * pbelow
        return evaluate(name, te["mcp_rs_mwh"].values, p, te.index, len(te))

    print("\nPRICE experiments (test = last 60 days, identical split):")
    run("P0 baseline (production)", feats_base)
    run("P1 recency weights (hl=60d)", feats_base, weights_hl=60)
    run("P2 +cross-market features", feats_plus)
    run("P3 cap-hurdle two-stage", feats_base, hurdle=True)
    run("P4 P1+P2", feats_plus, weights_hl=60)
    run("P5 P1+P2+P3 (full)", feats_plus, weights_hl=60, hurdle=True)
    run("P6 P5 + tuned params", feats_plus, weights_hl=60, hurdle=True,
        params=dict(n_estimators=4000, learning_rate=0.02, num_leaves=127,
                    min_child_samples=25, subsample=0.8, subsample_freq=1,
                    colsample_bytree=0.7, reg_lambda=2.0, random_state=42,
                    verbose=-1))
    return pd.DataFrame(results)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    # merge with previous results so a partial run never erases the other side
    prev = OUT / "model_lab.json"
    out = json.loads(prev.read_text()) if prev.exists() else {}
    if which in ("load", "both"):
        out["load"] = run_load_experiments().to_dict("records")
    if which in ("price", "both"):
        out["price"] = run_price_experiments().to_dict("records")
    (OUT / "model_lab.json").write_text(json.dumps(out, indent=1))
    print(f"\nresults -> {OUT / 'model_lab.json'}")
