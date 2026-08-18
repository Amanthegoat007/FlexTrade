"""Probabilistic day-ahead load forecast for Delhi -- P05/P10/P50/P90/P95.

A point forecast is the wrong product shape for a utility. Under the CERC
Deviation Settlement Mechanism a DISCOM is not paid for being right on
average; it is charged when actual drawal leaves the band around its
schedule. So the operational question is never "what will load be" but
"how wide does my band have to be to stay inside the DSM tolerance", and
that is a quantile question.

Same feature pipeline, same data and the same chronological split as the
production point model in load_forecast/02_train_model.py -- this is that
model's distribution, not a different model with a different opinion. Five
LightGBM learners fitted with the pinball objective give the band directly.

Calibration is not assumed, it is measured and then corrected. Raw quantile
regression under-covers on held-out data (each quantile is fitted
independently and none of them knows about the others), so we apply
conformalised quantile regression -- CQR, Romano et al. 2019 -- on the
validation window: score every calibration point by how far outside the
band it fell, then widen the band by the empirical quantile of that score.
The margin is additive in MW here rather than multiplicative, because load
is a bounded, strictly positive series around 2-8 GW where an absolute
MW band is also what a control-room engineer actually wants to read.

Honest caveat kept in the report: the CQR coverage guarantee assumes
exchangeability, which a time series violates. Treat the result as
well-calibrated in practice, not as a proof.
"""
import importlib.util
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

LF_DIR = HERE.parent.parent / "load_forecast"
OUT = HERE.parent / "output"
OUT.mkdir(exist_ok=True)
QMODEL_PATH = OUT / "load_model_q{q:02.0f}.txt"
CONFORMAL_PATH = OUT / "load_conformal.json"
METRICS_PATH = OUT / "metrics_load_quantile.txt"

_spec = importlib.util.spec_from_file_location("lf_train", LF_DIR / "02_train_model.py")
lf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lf)
FEATURES = lf.FEATURES
build_features = lf.build_features

QUANTILES = (0.05, 0.10, 0.50, 0.90, 0.95)
BANDS = ((0.10, 0.90), (0.05, 0.95))     # nominal 80% and 90%

# CERC DSM: drawal outside this band around schedule attracts deviation
# charges. Used to express the forecast band in the unit the regulation
# speaks, not to claim a settlement calculation.
DSM_BAND_PCT = 12.0


def _panel() -> pd.DataFrame:
    df = pd.read_parquet(LF_DIR / "data" / "model_table.parquet")
    return build_features(df).dropna(subset=FEATURES + ["load_mw"])


# --- band calibration: regime-conditional and adaptive -------------------
#
# The static asymmetric CQR below produced a band that LOOKS calibrated on
# average and is not calibrated in any individual month. Measured across 8
# rolling origins (backtest/audit.py load_band):
#
#     coverage 81.4% mean against 80% nominal  -- fine
#     but the windows run 70.9% to 96.9%
#     Kupiec REJECTS the correct rate in 7 of 8 windows
#     Christoffersen REJECTS independence in 8 of 8 -- failures CLUSTER
#
# Two distinct faults, and the mean hides both. The rate is wrong window to
# window, and the misses arrive in bursts rather than scattered. For a DISCOM
# scheduling against DSM penalties those are different risks: a band that fails
# six days running through a heatwave is a drawdown, not noise.
#
# Both point at weather. Delhi's load band fails when cooling demand runs hot
# for several days together, which is exactly what a margin fitted on a fixed
# past window cannot track. So the same two mechanisms that fixed the price
# band apply here:
#
#   regime-conditional (Mondrian, Vovk 2003)  margins per cooling-degree-hour
#                                             band, so a heatwave is calibrated
#                                             as a heatwave
#   adaptive (ACI, Gibbs & Candes 2021)       tail levels move against realised
#                                             miscoverage, so drift is tracked
#                                             instead of assumed away
CDH_EDGES = (-0.01, 2.0, 8.0, 1e9)      # mild / warm / hot, in cooling deg-hours
ACI_GAMMA = 0.02
REGIME_MIN_N = 400                       # below this a regime uses the global margin


def regime_of(cdh) -> np.ndarray:
    """Map cooling-degree-hours to a load regime index.

    Shared by calibration and serving so the two cannot disagree about which
    bin a block is in — the same guarantee price_model.regime_of gives.
    """
    return np.clip(np.digitize(np.asarray(cdh, dtype=float), CDH_EDGES) - 1,
                   0, len(CDH_EDGES) - 2)


def _adaptive_regime_margins(val: pd.DataFrame, val_pred: dict,
                             lo_q: float, hi_q: float) -> dict:
    """Per-regime margins, walked forward over the calibration window.

    Replays the validation window day by day. Each day the margins are read off
    a trailing window at the CURRENT adaptive tail level, the day is scored, and
    the level is pushed against whatever miscoverage actually occurred:

        t <- t + gamma * (target - observed_exceedance)

    which holds long-run coverage under arbitrary drift, with no exchangeability
    assumption to violate — and Delhi's load plainly violates it, growing
    through the window while the weather swings.

    Replayed from scratch on every retrain rather than carried in a state file:
    reproducible, auditable, and a stale state file cannot silently mis-size a
    band the way a stale margin already did once.
    """
    alpha = 1 - (hi_q - lo_q)
    target = alpha / 2
    nb = len(CDH_EDGES) - 1
    y = val["load_mw"].values
    lo_p, hi_p = val_pred[lo_q], val_pred[hi_q]
    reg = regime_of(val["cdh"].values)
    days = val.index.normalize()
    uniq = sorted(set(days))

    tails = [[target, target] for _ in range(nb)]
    g_tail = [target, target]
    margins = {b: (0.0, 0.0) for b in range(nb)}
    g_margin = (0.0, 0.0)

    def q_at(arr, level, default):
        if len(arr) < REGIME_MIN_N:
            return default
        return float(np.quantile(arr, float(np.clip(level, 0.001, 0.999)),
                                 method="higher"))

    for d in uniq:
        today = days == d
        trail = (days < d) & (days >= d - pd.Timedelta(days=45))
        if trail.sum() < 1000 or not today.any():
            continue
        # residual beyond each bound, in MW — the additive form this model has
        # always used, unlike the price band's multiplicative ratios
        s_lo, s_hi = lo_p[trail] - y[trail], y[trail] - hi_p[trail]
        tb = reg[trail]
        g_margin = (max(q_at(s_lo, 1 - g_tail[0], 0.0), 0.0),
                    max(q_at(s_hi, 1 - g_tail[1], 0.0), 0.0))
        for b in range(nb):
            mb = tb == b
            margins[b] = (max(q_at(s_lo[mb], 1 - tails[b][0], g_margin[0]), 0.0),
                          max(q_at(s_hi[mb], 1 - tails[b][1], g_margin[1]), 0.0))

        yt, bt = y[today], reg[today]
        lo_t = lo_p[today].copy()
        hi_t = hi_p[today].copy()
        for b in range(nb):
            m = bt == b
            if m.any():
                lo_t[m] -= margins[b][0]
                hi_t[m] += margins[b][1]
        for b in range(nb):
            m = bt == b
            if not m.any():
                continue
            tails[b][0] += ACI_GAMMA * (target - np.mean(yt[m] < lo_t[m]))
            tails[b][1] += ACI_GAMMA * (target - np.mean(yt[m] > hi_t[m]))
            tails[b] = [float(np.clip(x, 0.001, 0.45)) for x in tails[b]]

    return {
        "mode": "adaptive-regime",
        "cdh_edges": list(CDH_EDGES),
        "global": {"lo": round(g_margin[0], 1), "hi": round(g_margin[1], 1)},
        "regimes": [
            {"lo_edge": CDH_EDGES[b], "hi_edge": CDH_EDGES[b + 1],
             "lo": round(margins[b][0], 1), "hi": round(margins[b][1], 1),
             "tail_lo": round(tails[b][0], 4), "tail_hi": round(tails[b][1], 4),
             "blocks": int((reg == b).sum())}
            for b in range(nb)],
    }


def _splits(f: pd.DataFrame):
    """The exact split the point model uses, so the two are comparable."""
    test_start = f.index.max() - pd.DateOffset(months=6)
    val_start = test_start - pd.DateOffset(months=6)
    return (f[f.index < val_start],
            f[(f.index >= val_start) & (f.index < test_start)],
            f[f.index >= test_start])


def train(quantiles=QUANTILES) -> str:
    f = _panel()
    train_, val, test = _splits(f)
    print(f"load quantiles: train {len(train_):,} | val {len(val):,} | "
          f"test {len(test):,}   ({f.index.min():%Y-%m-%d} -> {f.index.max():%Y-%m-%d})")

    age = np.asarray((train_.index.max() - train_.index).total_seconds()) / 86400
    weights = 0.5 ** (age / lf.RECENCY_HALF_LIFE_DAYS)

    val_pred, test_pred = {}, {}
    for q in quantiles:
        m = lgb.LGBMRegressor(
            objective="quantile", alpha=q,
            n_estimators=3000, learning_rate=0.03, num_leaves=127,
            min_child_samples=20, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.7, reg_lambda=2.0, random_state=42, verbose=-1)
        m.fit(train_[FEATURES], train_["load_mw"], sample_weight=weights,
              eval_set=[(val[FEATURES], val["load_mw"])],
              callbacks=[lgb.early_stopping(150, verbose=False),
                         lgb.log_evaluation(0)])
        m.booster_.save_model(str(QMODEL_PATH).format(q=q * 100))
        val_pred[q] = m.predict(val[FEATURES])
        test_pred[q] = m.predict(test[FEATURES])
        print(f"  P{q * 100:02.0f} fitted (best iter {m.best_iteration_})")

    yv, yt = val["load_mw"].values, test["load_mw"].values
    lines = [f"probabilistic load forecast -- Delhi, 15-min blocks",
             f"train {len(train_):,} | val {len(val):,} | test {len(test):,}",
             f"test window {test.index.min():%Y-%m-%d} -> {test.index.max():%Y-%m-%d}",
             "", "per-quantile (test):",
             f"  {'':<5} {'pinball':>9} {'coverage':>10} {'nominal':>9}"]
    for q in quantiles:
        p = test_pred[q]
        err = yt - p
        pinball = float(np.mean(np.maximum(q * err, (q - 1) * err)))
        cov = float(np.mean(yt <= p) * 100)
        lines.append(f"  P{q * 100:02.0f} {pinball:8.1f} MW {cov:9.1f}% "
                     f"{q * 100:8.0f}%")

    # ---- conformal calibration on the validation window -----------------
    # ASYMMETRIC CQR: a separate margin for each side. The symmetric form
    # takes max(lo - y, y - hi) and widens both bounds by it, which is the
    # wrong instrument here -- the median under-forecasts (actual exceeds
    # P50 on ~76% of test blocks as Delhi's load grew), so the miss is
    # almost entirely through the UPPER bound. Widening the lower bound by
    # the same amount buys no coverage and costs band width, and band width
    # is what the customer pays for in procured reserve. Scoring each side
    # at level 1 - alpha/2 keeps the >= 1 - alpha guarantee by union bound.
    margins = {}
    lines += ["", "interval coverage (raw -> symmetric CQR -> asymmetric CQR):"]
    for lo_q, hi_q in BANDS:
        if lo_q not in quantiles or hi_q not in quantiles:
            continue
        alpha = 1 - (hi_q - lo_q)
        nominal = (hi_q - lo_q) * 100
        n = len(yv)

        sym_s = np.maximum(val_pred[lo_q] - yv, yv - val_pred[hi_q])
        sym = float(np.quantile(sym_s, min((hi_q - lo_q) * (1 + 1 / n), 1.0),
                                method="higher"))
        side = min((1 - alpha / 2) * (1 + 1 / n), 1.0)
        m_lo = float(np.quantile(val_pred[lo_q] - yv, side, method="higher"))
        m_hi = float(np.quantile(yv - val_pred[hi_q], side, method="higher"))
        m_lo, m_hi = max(m_lo, 0.0), max(m_hi, 0.0)
        margins[f"{lo_q}-{hi_q}"] = {"lo": m_lo, "hi": m_hi, "symmetric": sym}

        # --- Mondrian CQR (Vovk 2003): the same asymmetric margin, taken
        # WITHIN each regime instead of once across all of them.
        #
        # One global margin assumes the residual scale is the same on a mild
        # day and a 40C one, and it is not: measured on this test window the
        # global asymmetric band covers 88.2% of mild blocks and 72.5% of hot
        # ones against a nominal 80%. The average looked fine at 84.2% and hid
        # a regime that is 7.5 points short — on the days a DISCOM actually
        # needs the band.
        #
        # This is NOT the adaptive-regime candidate that was rejected below.
        # That one pushed each regime's tail LEVEL around with an ACI update
        # and then froze whatever level the last calibration day happened to
        # land on; the levels drifted, the frozen ones did not transfer, and
        # its hot regime came out worse (63.7%) than the global margin it was
        # meant to repair. Here the level is fixed at nominal and only the
        # QUANTILE is taken per regime — no state, no drift, nothing to go
        # stale. The finite-sample (1 + 1/n) correction uses each regime's own
        # n, and a regime thinner than REGIME_MIN_N falls back to the global
        # margin rather than trusting a quantile of a handful of points.
        vreg = regime_of(val["cdh"].values)
        s_lo_all, s_hi_all = val_pred[lo_q] - yv, yv - val_pred[hi_q]
        mond = []
        for b in range(len(CDH_EDGES) - 1):
            mb = vreg == b
            nb_ = int(mb.sum())
            if nb_ < REGIME_MIN_N:
                mond.append({"lo": m_lo, "hi": m_hi, "n": nb_, "fallback": True})
                continue
            side_b = min((1 - alpha / 2) * (1 + 1 / nb_), 1.0)
            mond.append({
                "lo": max(float(np.quantile(s_lo_all[mb], side_b, method="higher")), 0.0),
                "hi": max(float(np.quantile(s_hi_all[mb], side_b, method="higher")), 0.0),
                "n": nb_, "fallback": False})
        margins[f"{lo_q}-{hi_q}"]["mondrian"] = {
            "cdh_edges": list(CDH_EDGES),
            "regimes": [{k: (round(v, 1) if isinstance(v, float) else v)
                         for k, v in r.items()} for r in mond]}

        lo, hi = test_pred[lo_q], test_pred[hi_q]
        raw = float(np.mean((yt >= lo) & (yt <= hi)) * 100)
        s_cov = float(np.mean((yt >= lo - sym) & (yt <= hi + sym)) * 100)
        a_lo, a_hi = lo - m_lo, hi + m_hi
        a_cov = float(np.mean((yt >= a_lo) & (yt <= a_hi)) * 100)

        # Mondrian scored on the SAME test window as everything else
        treg0 = regime_of(test["cdh"].values)
        d_lo, d_hi = lo.copy(), hi.copy()
        for b, spec in enumerate(mond):
            mb = treg0 == b
            if mb.any():
                d_lo[mb] -= spec["lo"]
                d_hi[mb] += spec["hi"]
        d_cov = float(np.mean((yt >= d_lo) & (yt <= d_hi)) * 100)
        d_per = [round(float(np.mean((yt[treg0 == b] >= d_lo[treg0 == b])
                                     & (yt[treg0 == b] <= d_hi[treg0 == b])) * 100), 1)
                 if (treg0 == b).sum() >= 100 else None
                 for b in range(len(CDH_EDGES) - 1)]

        # --- trailing recalibration, scored walk-forward -------------------
        #
        # Both regime-conditional attempts failed for one structural reason:
        # the calibration block and the test window do not contain the same
        # seasons. Measured on this split, validation is 28/53/19 percent
        # mild/warm/hot and test is 63/19/18 — and validation's "hot" blocks
        # are monsoon-adjacent Jun-Sep 2025 while test's are peak-summer
        # Apr-Jun 2026. Those are different physical regimes at the same CDH,
        # so a margin fitted on one cannot size a band on the other, however
        # the bins are drawn.
        #
        # The price band already solved this by recalibrating on a trailing
        # window instead of a fixed block. Same idea here: at each day of the
        # test window the margin is read off the previous 45 days of realized
        # residuals, which are always recent and always the current season.
        # Scored walk-forward, so no margin ever sees the day it sizes.
        cal_y = np.concatenate([yv, yt])
        cal_lo = np.concatenate([val_pred[lo_q], test_pred[lo_q]])
        cal_hi = np.concatenate([val_pred[hi_q], test_pred[hi_q]])
        cal_idx = val.index.append(test.index)
        cal_day = cal_idx.normalize()
        t_lo, t_hi = lo.copy(), hi.copy()
        test_day = test.index.normalize()
        for d in sorted(set(test_day)):
            trail = (cal_day < d) & (cal_day >= d - pd.Timedelta(days=45))
            today = test_day == d
            if trail.sum() < 1000 or not today.any():
                t_lo[today] -= m_lo
                t_hi[today] += m_hi
                continue
            nt = int(trail.sum())
            side_t = min((1 - alpha / 2) * (1 + 1 / nt), 1.0)
            tl = max(float(np.quantile(cal_lo[trail] - cal_y[trail], side_t,
                                       method="higher")), 0.0)
            th = max(float(np.quantile(cal_y[trail] - cal_hi[trail], side_t,
                                       method="higher")), 0.0)
            t_lo[today] -= tl
            t_hi[today] += th
        t_cov = float(np.mean((yt >= t_lo) & (yt <= t_hi)) * 100)
        if (lo_q, hi_q) == BANDS[0]:
            served_lo, served_hi = t_lo.copy(), t_hi.copy()

        # ADOPTED. forecast_day() reads margins_mw[band]["lo"/"hi"], so the
        # served band switches by writing the trailing margin into those keys
        # — computed here from the most recent 45 days available, which is
        # what "trailing" means on the day the model is retrained. The daily
        # pipeline retrains, so serving reproduces the walk-forward behaviour
        # scored above rather than approximating it.
        #
        # The static asymmetric margins are kept beside them under
        # lo_static/hi_static: they are what every report before 18 Aug 2026
        # was measured against, and dropping them would make the change
        # untraceable.
        last = cal_day.max()
        fin = (cal_day > last - pd.Timedelta(days=45))
        if fin.sum() >= 1000:
            nf = int(fin.sum())
            side_f = min((1 - alpha / 2) * (1 + 1 / nf), 1.0)
            f_lo = max(float(np.quantile(cal_lo[fin] - cal_y[fin], side_f,
                                         method="higher")), 0.0)
            f_hi = max(float(np.quantile(cal_y[fin] - cal_hi[fin], side_f,
                                         method="higher")), 0.0)
            margins[f"{lo_q}-{hi_q}"].update({
                "lo": f_lo, "hi": f_hi,
                "lo_static": m_lo, "hi_static": m_hi,
                "mode": "trailing45",
                "trailing_days": 45, "trailing_n": nf,
                "trailing_from": str((last - pd.Timedelta(days=45)).date()),
                "trailing_to": str(last.date())})
        t_per = [round(float(np.mean((yt[treg0 == b] >= t_lo[treg0 == b])
                                     & (yt[treg0 == b] <= t_hi[treg0 == b])) * 100), 1)
                 if (treg0 == b).sum() >= 100 else None
                 for b in range(len(CDH_EDGES) - 1)]

        # --- adaptive regime-conditional, scored on the SAME test window ---
        ar = _adaptive_regime_margins(val, val_pred, lo_q, hi_q)
        margins[f"{lo_q}-{hi_q}"]["adaptive_regime"] = ar
        treg = regime_of(test["cdh"].values)
        r_lo, r_hi = lo.copy(), hi.copy()
        for b, spec in enumerate(ar["regimes"]):
            m = treg == b
            if m.any():
                r_lo[m] -= spec["lo"]
                r_hi[m] += spec["hi"]
        r_cov = float(np.mean((yt >= r_lo) & (yt <= r_hi)) * 100)
        # coverage WITHIN each regime — the property a mean cannot show
        per = []
        for b in range(len(ar["regimes"])):
            m = treg == b
            per.append(round(float(np.mean((yt[m] >= r_lo[m]) & (yt[m] <= r_hi[m]))
                                   * 100), 1) if m.sum() >= 100 else None)
        ar["test_coverage_pct"] = round(r_cov, 1)
        ar["test_coverage_by_regime_pct"] = per
        ar["test_width_mw"] = round(float(np.mean(r_hi - r_lo)), 0)
        # Conditional coverage for the band we ACTUALLY SERVE. A marginal
        # number can be perfect while a regime the customer cares about is
        # badly short — the candidate below was rejected for exactly that, and
        # until now the served band was never held to the same test.
        a_per, a_n = [], []
        for b in range(len(CDH_EDGES) - 1):
            m = treg == b
            a_n.append(int(m.sum()))
            a_per.append(round(float(np.mean((yt[m] >= a_lo[m])
                                             & (yt[m] <= a_hi[m])) * 100), 1)
                         if m.sum() >= 100 else None)

        # Interval score (Gneiting & Raftery 2007, eq. 43): width plus a
        # 2/alpha penalty per unit of miss. Lower is better. This is the
        # proper rule for adjudicating band vs band, because coverage alone
        # always rewards the wider one and width alone always rewards the
        # narrower.
        def interval_score(l, h, y, alpha):
            return float(np.mean((h - l)
                                 + (2 / alpha) * (l - y) * (y < l)
                                 + (2 / alpha) * (y - h) * (y > h)))
        alpha_ = 1 - (hi_q - lo_q)
        is_a = interval_score(a_lo, a_hi, yt, alpha_)
        is_r = interval_score(r_lo, r_hi, yt, alpha_)
        is_s = interval_score(lo - sym, hi + sym, yt, alpha_)
        is_d = interval_score(d_lo, d_hi, yt, alpha_)
        is_t = interval_score(t_lo, t_hi, yt, alpha_)

        lines += [
            f"  P{lo_q * 100:02.0f}-P{hi_q * 100:02.0f} (nominal {nominal:.0f}%)",
            f"      raw         {raw:5.1f}%  width {np.mean(hi - lo):6.0f} MW",
            f"      symmetric   {s_cov:5.1f}%  width "
            f"{np.mean(hi + sym - lo + sym):6.0f} MW  [+/-{sym:.0f} MW]",
            f"      asymmetric  {a_cov:5.1f}%  width {np.mean(a_hi - a_lo):6.0f} MW"
            f"  [-{m_lo:.0f} / +{m_hi:.0f} MW]  <- previous default",
            f"        by CDH regime {a_per}   n {a_n}",
            f"      mondrian    {d_cov:5.1f}%  width {np.mean(d_hi - d_lo):6.0f} MW"
            f"   by CDH regime {d_per}",
            f"        margins per regime (MW): "
            + "  ".join(f"[{CDH_EDGES[b]:.0f}-{CDH_EDGES[b+1]:.0f}h] "
                        f"-{g['lo']:.0f}/+{g['hi']:.0f}"
                        + ("*" if g["fallback"] else "")
                        for b, g in enumerate(mond)),
            f"      trailing45  {t_cov:5.1f}%  width {np.mean(t_hi - t_lo):6.0f} MW"
            f"   by CDH regime {t_per}   <- SERVED (recalibrated daily, walk-forward)",
            f"        interval score (lower better): symmetric {is_s:6.1f}"
            f"   asymmetric {is_a:6.1f}   mondrian {is_d:6.1f}"
            f"   trailing45 {is_t:6.1f}   adaptive+regime {is_r:6.1f}",
            f"      adaptive+regime {r_cov:5.1f}%  width {ar['test_width_mw']:6.0f} MW"
            f"   by CDH regime {ar['test_coverage_by_regime_pct']}"
            f"   <- CANDIDATE, NOT SERVED (hot regime under-covers)",
            f"        margins per regime (MW): "
            + "  ".join(f"[{g['lo_edge']:.0f}-{g['hi_edge']:.0f}h] "
                        f"-{g['lo']:.0f}/+{g['hi']:.0f}"
                        for g in ar["regimes"])]

    # ---- what the band means operationally ------------------------------
    lo_q, hi_q = BANDS[0]
    # Use the WALK-FORWARD band, not today's frozen margin applied backwards.
    # The margin now tracks the season: the one standing at the end of the
    # window is sized for peak summer, and painting that across a year of
    # winter blocks reports a band 22% wider than the one a customer would
    # ever have been served. The walk-forward series is what was scored above
    # and what serving reproduces, so it is what gets described here.
    if "served_lo" in dir():
        lo_c, hi_c = served_lo, served_hi
    else:
        m = margins.get(f"{lo_q}-{hi_q}", {"lo": 0.0, "hi": 0.0})
        lo_c = test_pred[lo_q] - m["lo"]
        hi_c = test_pred[hi_q] + m["hi"]
    width_pct = float(np.mean((hi_c - lo_c) / yt) * 100)
    p50 = test_pred[0.50]
    inside_dsm = float(np.mean(np.abs(yt - p50) / yt * 100 <= DSM_BAND_PCT) * 100)
    band_vs_dsm = float(np.mean((hi_c - lo_c) / 2 / yt * 100 <= DSM_BAND_PCT) * 100)
    bias = float(np.mean(yt - p50))
    under = float(np.mean(yt > p50) * 100)
    lines += [
        "", "operational reading:",
        f"  P50 point error (MAPE)            {np.mean(np.abs(yt - p50) / yt) * 100:5.2f}%",
        f"  P50 signed bias                   {bias:+5.0f} MW "
        f"(actual exceeds P50 on {under:.1f}% of blocks)",]
    if under > 60:
        lines += [
            f"  NOTE: the median under-forecasts. Delhi load grew through the",
            f"  test window and a quantile fitted on older history sits below it.",
            f"  This matters for DSM in one direction only -- under-scheduling is",
            f"  what draws deviation charges -- so it is corrected by the conformal",
            f"  margin above rather than left for the operator to discover.",]
    lines += [
        f"  80% band width, mean              {np.mean(hi_c - lo_c):5.0f} MW "
        f"({width_pct:.1f}% of load)",
        f"  blocks where P50 lands inside the CERC DSM +/-{DSM_BAND_PCT:.0f}% "
        f"band  {inside_dsm:5.1f}%",
        f"  blocks where our whole 80% band fits inside it            "
        f"{band_vs_dsm:5.1f}%",
        "",
        "The second number is the one a DISCOM buys: it is the share of the day",
        "we can promise a schedule with quantified confidence rather than a",
        "single number and a hope. CQR coverage assumes exchangeability, which",
        "a time series violates -- this is calibrated in practice, not proven.",
    ]

    CONFORMAL_PATH.write_text(json.dumps({
        # round the scalar margins; adaptive_regime is a nested block and is
        # already rounded where it was built
        "margins_mw": {k: {n: (round(x, 1) if isinstance(x, (int, float)) else x)
                           for n, x in v.items()}
                       for k, v in margins.items()},
        "quantiles": list(quantiles),
        "calibration_n": len(yv),
        "calibration_from": str(val.index.min().date()),
        "calibration_to": str(val.index.max().date()),
    }, indent=2))
    report = "\n".join(lines)
    print()
    print(report)
    METRICS_PATH.write_text(report)
    return report


def forecast_day(target: date | None = None, quantiles=QUANTILES,
                 conformal: bool = True) -> pd.DataFrame:
    """Per-block load quantiles for `target` -- columns p05/p10/p50/p90/p95."""
    from models import load_model
    target = target or (date.today() + timedelta(days=1))
    frame = load_model._history_frame(target)
    feats = build_features(frame)
    day = feats[feats.index.date == target]
    missing = day[FEATURES].isna().sum()
    missing = missing[missing > 0]
    if len(missing):
        raise ValueError(f"missing features for {target}: {missing.to_dict()}")

    out = pd.DataFrame(index=day.index)
    for q in quantiles:
        path = Path(str(QMODEL_PATH).format(q=q * 100))
        if not path.exists():
            raise FileNotFoundError(
                f"{path.name} missing -- run `python models/load_quantile.py`")
        out[f"p{q * 100:02.0f}"] = lgb.Booster(
            model_file=str(path)).predict(day[FEATURES])

    if conformal and CONFORMAL_PATH.exists():
        margins = json.loads(CONFORMAL_PATH.read_text())["margins_mw"]
        for key, m in margins.items():
            lo_q, hi_q = (float(x) for x in key.split("-"))
            lo_col, hi_col = f"p{lo_q * 100:02.0f}", f"p{hi_q * 100:02.0f}"
            if lo_col in out:
                out[lo_col] = out[lo_col] - m["lo"]
            if hi_col in out:
                out[hi_col] = out[hi_col] + m["hi"]
        out.attrs["conformal"] = True
    else:
        out.attrs["conformal"] = False

    # independently fitted quantiles can cross on rare blocks; enforce order
    out[:] = np.sort(out.values, axis=1)
    return out.clip(lower=0)


if __name__ == "__main__":
    train()
