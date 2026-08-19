"""Re-measure every published accuracy claim under rolling-origin evaluation.

Each adapter reproduces its model's PRODUCTION recipe exactly — same features,
same hyperparameters, same calibration construction, including the parts that
are arguably wrong. The purpose is to measure what ships, not an improved
variant; where a recipe has a known flaw it is reproduced and noted rather than
quietly fixed here, because fixing it here would mean the audit no longer
describes the thing on the website.

Run one model:      python backtest/audit.py load
Run everything:     python backtest/audit.py all
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import lightgbm as lgb
import os

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from walkforward import Origin, Spec, report, run, save  # noqa: E402

OUT = HERE.parent / "output"


# ------------------------------------------------------------- load band --
# Production calibrates on a 6-month window (models/load_quantile._splits), so
# the audit uses the same. Shortening it would measure a different recipe.
LOAD_CAL_DAYS = 182
_load_panel = None


def _panel():
    global _load_panel
    if _load_panel is None:
        from models import load_quantile as lq
        _load_panel = lq._panel()
    return _load_panel


def load_band_task(o: Origin) -> pd.DataFrame:
    from models import load_quantile as lq
    # the recency half-life lives in load_forecast/02_train_model.py, which
    # load_quantile loads dynamically as `lf`; reuse its handle rather than
    # re-importing, so the audit can never drift onto a different constant
    lf = lq.lf

    f = _panel()
    cal_start = o.train_end - pd.Timedelta(days=LOAD_CAL_DAYS)
    tr = f[f.index < cal_start]
    cal = f[(f.index >= cal_start) & (f.index < o.train_end)]
    te = f[(f.index >= o.test_start) & (f.index < o.test_end)]
    if len(tr) < 5000 or len(cal) < 2000 or len(te) < 200:
        raise RuntimeError(f"thin split: train {len(tr)}, cal {len(cal)}, test {len(te)}")

    # recency weighting, exactly as production
    age = np.asarray((tr.index.max() - tr.index).total_seconds()) / 86400
    weights = 0.5 ** (age / lf.RECENCY_HALF_LIFE_DAYS)

    pc, pt = {}, {}
    for q in (0.10, 0.50, 0.90):
        m = lgb.LGBMRegressor(
            objective="quantile", alpha=q,
            n_estimators=3000, learning_rate=0.03, num_leaves=127,
            min_child_samples=20, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.7, reg_lambda=2.0, random_state=42, verbose=-1)
        # NOTE: production uses this same window for early stopping AND for the
        # conformal margins. That double use leaks model selection into the
        # calibration set and biases coverage optimistically. Reproduced here on
        # purpose — the audit's job is to score the shipped recipe.
        m.fit(tr[lq.FEATURES], tr["load_mw"], sample_weight=weights,
              eval_set=[(cal[lq.FEATURES], cal["load_mw"])],
              callbacks=[lgb.early_stopping(150, verbose=False),
                         lgb.log_evaluation(0)])
        pc[q] = m.predict(cal[lq.FEATURES])
        pt[q] = m.predict(te[lq.FEATURES])

    # Trailing-window CQR margins — the shipped construction as of 18 Aug 2026.
    #
    # Production used a single asymmetric margin fitted on the whole
    # calibration block until today. That is now lo_static/hi_static and is NOT
    # what gets served: the margin is re-read each day off the previous 45 days
    # of realized residuals, because a block fitted months earlier is drawn
    # from the wrong season. Reproduced here so the audit keeps scoring the
    # shipped recipe rather than a retired one.
    yv = cal["load_mw"].values
    yt_ = te["load_mw"].values
    alpha = 0.2
    cal_y = np.concatenate([yv, yt_])
    cal_lo = np.concatenate([pc[0.10], pt[0.10]])
    cal_hi = np.concatenate([pc[0.90], pt[0.90]])
    trail_days = int(os.environ.get("FT_TRAIL_DAYS", str(lq.TRAIL_DAYS)))
    cal_day = cal.index.append(te.index).normalize()
    te_day = te.index.normalize()

    # the retired static margin, kept only as the fallback for a thin trail
    n = len(yv)
    side = min((1 - alpha / 2) * (1 + 1 / n), 1.0)
    m_lo = max(float(np.quantile(pc[0.10] - yv, side, method="higher")), 0.0)
    m_hi = max(float(np.quantile(yv - pc[0.90], side, method="higher")), 0.0)

    lo_out = pt[0.10].copy()
    hi_out = pt[0.90].copy()
    for d in sorted(set(te_day)):
        today = te_day == d
        trail = (cal_day < d) & (cal_day >= d - pd.Timedelta(days=trail_days))
        if trail.sum() < 1000 or not today.any():
            lo_out[today] -= m_lo
            hi_out[today] += m_hi
            continue
        nt = int(trail.sum())
        side_t = min((1 - alpha / 2) * (1 + 1 / nt), 1.0)
        lo_out[today] -= max(float(np.quantile(
            cal_lo[trail] - cal_y[trail], side_t, method="higher")), 0.0)
        hi_out[today] += max(float(np.quantile(
            cal_y[trail] - cal_hi[trail], side_t, method="higher")), 0.0)

    return pd.DataFrame({"actual": yt_, "lo": lo_out, "hi": hi_out,
                         "mid": pt[0.50], "q10": pt[0.10], "q50": pt[0.50],
                         "q90": pt[0.90]}, index=te.index)


# ------------------------------------------------------------- peak model --
PEAK_CAL_DAYS = 182          # production's validation window, mirrored
_peak_panel = None


def _peak_frame():
    global _peak_panel
    if _peak_panel is None:
        from models import peak_model as pk
        f = pk.build_features(pk._daily_panel(live=False))
        _peak_panel = f.dropna(subset=pk.MAG_FEATURES + ["peak_mw"])
    return _peak_panel


def peak_task(o: Origin) -> pd.DataFrame:
    """Day-ahead peak MW — the shipped magnitude model plus its bias correction.

    Reproduces models.peak_model.train(): the same MAG_FEATURES, the same
    LightGBM settings, and the same constant bias correction fitted on the
    validation window only (never on the scored window).
    """
    from models import peak_model as pk

    f = _peak_frame()
    cal_start = o.train_end - pd.Timedelta(days=PEAK_CAL_DAYS)
    tr = f[f.index < cal_start]
    cal = f[(f.index >= cal_start) & (f.index < o.train_end)]
    te = f[(f.index >= o.test_start) & (f.index < o.test_end)]
    if len(tr) < 300 or len(cal) < 90 or len(te) < 10:
        raise RuntimeError(f"thin split: train {len(tr)}, cal {len(cal)}, test {len(te)}")

    mag = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.02, num_leaves=31,
                            min_child_samples=15, subsample=0.8, subsample_freq=1,
                            colsample_bytree=0.8, reg_lambda=2.0,
                            random_state=42, verbose=-1)
    mag.fit(tr[pk.MAG_FEATURES], tr["peak_mw"],
            eval_set=[(cal[pk.MAG_FEATURES], cal["peak_mw"])],
            callbacks=[lgb.early_stopping(100, verbose=False),
                       lgb.log_evaluation(0)])
    # bias measured on the calibration window only, exactly as production
    bias = float(np.mean(cal["peak_mw"].values - mag.predict(cal[pk.MAG_FEATURES])))
    pred = mag.predict(te[pk.MAG_FEATURES]) + bias
    return pd.DataFrame({"actual": te["peak_mw"].values, "point": pred},
                        index=te.index)


def peak_naive_task(o: Origin) -> pd.DataFrame:
    """Seasonal-naive benchmark: last same-weekday peak. The bar any peak model
    has to clear before it has earned its complexity."""
    from models import peak_model as pk

    f = _peak_frame()
    te = f[(f.index >= o.test_start) & (f.index < o.test_end)]
    hist = f[f.index < o.test_start]["peak_mw"]
    full = pd.concat([hist, te["peak_mw"]])
    pred = full.shift(7).reindex(te.index)      # same weekday, one week back
    return pd.DataFrame({"actual": te["peak_mw"].values,
                         "point": pred.values}, index=te.index).dropna()


# -------------------------------------------------------------- registry --
# ------------------------------------------------------------ RTM intraday --
RTM_CAL_DAYS = 30            # production's val_days, mirrored
_rtm_frame = None


def _rtm():
    global _rtm_frame
    if _rtm_frame is None:
        from models import rtm_model as rt
        f = rt.build_features(rt._table(), "intraday")
        _rtm_frame = f.dropna(subset=rt.FEATURES["intraday"] + ["rtm", "dam"])
    return _rtm_frame


def rtm_task(o: Origin) -> pd.DataFrame:
    """Intraday RTM price — the shipped model, refitted at each origin."""
    from models import rtm_model as rt

    feats = rt.FEATURES["intraday"]
    f = _rtm()
    cal_start = o.train_end - pd.Timedelta(days=RTM_CAL_DAYS)
    tr = f[f.index < cal_start]
    cal = f[(f.index >= cal_start) & (f.index < o.train_end)]
    te = f[(f.index >= o.test_start) & (f.index < o.test_end)]
    if len(tr) < 3000 or len(cal) < 500 or len(te) < 200:
        raise RuntimeError(f"thin split: train {len(tr)}, cal {len(cal)}, test {len(te)}")

    m = lgb.LGBMRegressor(n_estimators=3000, learning_rate=0.03, num_leaves=63,
                          min_child_samples=40, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.8, reg_lambda=1.0,
                          random_state=42, verbose=-1)
    m.fit(tr[feats], np.log(tr["rtm"].clip(lower=rt.FLOOR)),
          eval_set=[(cal[feats], np.log(cal["rtm"].clip(lower=rt.FLOOR)))],
          callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)])
    pred = np.clip(np.exp(m.predict(te[feats])), 0, rt.CAP)
    return pd.DataFrame({"actual": te["rtm"].values, "point": pred}, index=te.index)


def rtm_incumbent_task(o: Origin) -> pd.DataFrame:
    """The hour-ratio incumbent this model replaced: DAM x an hour-of-day scale
    fitted on the training window. Intraday, today's DAM has cleared, so this
    baseline is genuinely implementable — no oracle price is handed to it."""
    from models import rtm_model as rt

    f = _rtm()
    cal_start = o.train_end - pd.Timedelta(days=RTM_CAL_DAYS)
    tr = f[f.index < cal_start]
    te = f[(f.index >= o.test_start) & (f.index < o.test_end)]
    byhour = rt._hour_ratio_from(tr)
    anchor = te["dam"].to_numpy(dtype=float)
    scale = te.index.hour.map(byhour).values
    return pd.DataFrame({"actual": te["rtm"].values,
                         "point": np.clip(anchor * scale, 0, rt.CAP)}, index=te.index)


# ------------------------------------------------------- DAM price model --
PRICE_CAL_DAYS = 30          # production's validation window, mirrored
_price_panel = None


def _price_frame():
    """Feature frame for the DAM hurdle model, built once and reused.

    Dropped on REQUIRED_FEATURES only, exactly as production does since
    18 Aug — listing the coal block here would silently discard 72% of the
    history and the audit would then be scoring a model nobody ships.
    """
    global _price_panel
    if _price_panel is None:
        from models import price_model as pm
        _price_panel = pm.build_features(pm._table()).dropna(
            subset=pm.REQUIRED_FEATURES + ["mcp_rs_mwh"])
    return _price_panel


def price_task(o: Origin) -> pd.DataFrame:
    """Day-ahead DAM price — the shipped cap-hurdle, refitted at each origin.

    BOTH stages are refitted: the classifier that predicts P(block pins at the
    Rs 10,000 cap) and the regressor for the below-cap level. Refitting only
    one would leak the other's view of the test window, which is the mistake
    that contaminated an earlier version of this harness.

    This is the model the bid sheet is built on and it had never been audited
    under rolling origins — it was added on 19 Aug after the training set went
    from 31,683 rows to 136,839 and no walk-forward number existed to say
    whether that helped out of sample.
    """
    from models import price_model as pm

    f = _price_frame()
    cal_start = o.train_end - pd.Timedelta(days=PRICE_CAL_DAYS)
    tr = f[f.index < cal_start]
    cal = f[(f.index >= cal_start) & (f.index < o.train_end)]
    te = f[(f.index >= o.test_start) & (f.index < o.test_end)]
    if len(tr) < 5000 or len(cal) < 500 or len(te) < 200:
        raise RuntimeError(f"thin split: train {len(tr)}, cal {len(cal)}, test {len(te)}")

    params = dict(n_estimators=2000, learning_rate=0.03, num_leaves=63,
                  min_child_samples=40, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.8, reg_lambda=1.0, random_state=42, verbose=-1)
    ytr = (tr["mcp_rs_mwh"] >= pm.CAP * 0.95).astype(int)
    ycal = (cal["mcp_rs_mwh"] >= pm.CAP * 0.95).astype(int)
    clf = lgb.LGBMClassifier(**params)
    clf.fit(tr[pm.FEATURES], ytr, eval_set=[(cal[pm.FEATURES], ycal)],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])

    btr, bcal = tr[ytr == 0], cal[ycal == 0]
    reg = lgb.LGBMRegressor(**params)
    reg.fit(btr[pm.FEATURES], np.log(btr["mcp_rs_mwh"].clip(lower=50)),
            eval_set=[(bcal[pm.FEATURES], np.log(bcal["mcp_rs_mwh"].clip(lower=50)))],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])

    pcap = clf.predict_proba(te[pm.FEATURES])[:, 1]
    pbelow = np.clip(np.exp(reg.predict(te[pm.FEATURES])), 0, pm.CAP)
    pred = pcap * pm.CAP + (1 - pcap) * pbelow
    return pd.DataFrame({"actual": te["mcp_rs_mwh"].values, "point": pred},
                        index=te.index)


def price_naive_task(o: Origin) -> pd.DataFrame:
    """Seasonal naive: the same 15-minute block one day earlier.

    The honest baseline for a day-ahead price, and a hard one — DAM is strongly
    autocorrelated block to block across days. p_lag_1d is already a feature of
    the real model, so this asks whether everything else earns its keep.
    """
    f = _price_frame()
    te = f[(f.index >= o.test_start) & (f.index < o.test_end)]
    prev = f["mcp_rs_mwh"].reindex(te.index - pd.Timedelta(days=1))
    return pd.DataFrame({"actual": te["mcp_rs_mwh"].values,
                         "point": prev.values}, index=te.index).dropna()


SPECS = {
    "load_band": lambda: Spec(
        name="Delhi load band (P10-P90, trailing-window CQR)",
        task=load_band_task, alpha=0.2, unit="MW",
        quantiles=(0.10, 0.50, 0.90)),
    "peak": lambda: Spec(
        name="Delhi day-ahead peak (LightGBM + validation bias correction)",
        task=peak_task, unit="MW", benchmark=peak_naive_task),
    # RTM has only 367 days of scraped history, so it gets a shorter training
    # floor and fewer origins than the 5-year load panel. Fewer origins means a
    # noisier worst-window figure, which is reported rather than smoothed over.
    "price": lambda: Spec(
        name="DAM day-ahead price (cap-hurdle, vs seasonal naive)",
        task=price_task, unit="Rs/MWh", benchmark=price_naive_task,
        min_train_days=400, n_origins=6),
    "rtm": lambda: Spec(
        name="RTM intraday price (vs the hour-ratio incumbent)",
        task=rtm_task, unit="Rs/MWh", benchmark=rtm_incumbent_task,
        min_train_days=150, n_origins=5),
}

INDEX_FOR = {"load_band": _panel, "peak": _peak_frame, "rtm": _rtm}


def main(which: str = "all", n_origins: int = 8, test_days: int = 30):
    names = list(SPECS) if which == "all" else [which]
    results, texts = [], []
    for nm in names:
        if nm not in SPECS:
            print(f"unknown model '{nm}'. known: {', '.join(SPECS)}")
            continue
        spec = SPECS[nm]()
        print(f"\n=== {spec.name} ===")
        idx = INDEX_FOR[nm]().index
        res = run(spec, n_origins=n_origins, test_days=test_days,
                  min_train_days=730, index=idx)
        res["key"] = nm
        results.append(res)
        txt = report(res)
        texts.append(txt)
        print("\n" + txt)

    if results:
        # merge with any previously audited models so one run does not erase
        # the others' results from the published file
        path = OUT / "walkforward.json"
        prev = []
        if path.exists():
            try:
                import json
                prev = [m for m in json.loads(path.read_text()).get("models", [])
                        if m.get("key") not in {r["key"] for r in results}]
            except Exception:
                prev = []
        save(prev + results, path)
        (OUT / "metrics_walkforward.txt").write_text("\n\n".join(texts),
                                                     encoding="utf-8")
        print(f"\nwrote {path.name} and metrics_walkforward.txt")
    return results


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    main(args[0] if args else "all",
         n_origins=int(args[1]) if len(args) > 1 else 8)
