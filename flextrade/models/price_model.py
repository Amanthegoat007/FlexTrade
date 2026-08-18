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
    # --- outages: MEASURED AND NOT ADOPTED, 18 Aug 2026 ---
    # National outage MW correlates -0.466 with the share of blocks pinning at
    # the cap, and -0.321 after deseasonalising against two Fourier harmonics
    # of day-of-year, so the association is real and not the calendar in
    # disguise. The sign is the opposite of the naive reading because outages
    # are endogenous: operators schedule maintenance into slack, so a large
    # parked fleet signals expected thin margins rather than scarcity. April
    # averages 13,903 MW out at a 0.308 cap share; November 32,874 MW at 0.063.
    #
    # None of which earned a place. Retrained at 6 rolling origins, 30-day test
    # windows, both stages refit per origin so no test window sat inside
    # training data:
    #
    #     MAE   base 766.9 -> 796.0   +3.80% WORSE   outage wins 3/6
    #     WAPE  base 16.21 ->  16.84  +3.89% WORSE   paired t -1.21
    #
    # A true association that adds nothing once the calendar, coal position and
    # price lags are already in the model. The columns are still built in
    # _table() because the State Stress index may want them; they stay out of
    # FEATURES until something measures better. Do not re-add on the strength
    # of the correlation alone — that is what was tested here.
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

    # --- outage position, same daily-to-block broadcast as coal ---
    try:
        from ingest import outages
        o = outages.daily_summary(1200)
        if len(o):
            o = o.set_index("day")[["total_out_mw", "forced_mw"]].sort_index()
            o["trend"] = o["total_out_mw"].diff(7)
            o = o.shift(3, freq="D")
            day = df.index.normalize()
            for src, dst in (("total_out_mw", "outage_total_mw"),
                             ("forced_mw", "outage_forced_mw"),
                             ("trend", "outage_trend_7d")):
                df[dst] = pd.Series(day.map(o[src]), index=df.index).ffill()
    except Exception as e:
        print(f"  (outage features unavailable: {type(e).__name__}: {str(e)[:70]})")
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

    lines_ = []
    for name, part in [("train", train_), ("val", val), ("test", test)]:
        p = hurdle_predict(part)
        y = part["mcp_rs_mwh"].values
        mae = float(np.mean(np.abs(y - p)))
        wape = 100 * float(np.sum(np.abs(y - p)) / np.sum(np.abs(y)))
        mape = np.mean(np.abs(y - p) / np.maximum(y, 100)) * 100
        rmse = float(np.sqrt(np.mean((y - p) ** 2)))
        corr = np.corrcoef(y, p)[0, 1]
        evening = (part.index.hour >= 17) & (part.index.hour <= 23)
        emae = float(np.mean(np.abs(y[evening] - p[evening])))
        lines_.append(f"{name:5s}  MAE {mae:7.1f} Rs/MWh   WAPE {wape:5.2f}%"
                      f"   RMSE {rmse:7.1f}   corr {corr:.3f}"
                      f"   evening MAE {emae:7.1f}   [MAPE {mape:5.2f}%]")
    # MAE is the headline and MAPE is bracketed, because MAPE is close to
    # meaningless on this target and is kept only for continuity with older
    # reports. Over the last 365 days 20.1% of blocks cleared below Rs 2,000
    # and 16.1% pinned exactly at the Rs 10,000 cap, so one fixed Rs 800/MWh
    # error reads as 80% APE at the 5th percentile and 8% at the 85th. It
    # scores hardest where the rupees are smallest, and its level says more
    # about the price distribution in the window than about the model.
    header = ("cap-hurdle two-stage (P(cap) x below-cap regression)"
              "   [MAPE shown in brackets: unreliable on a capped, near-zero"
              "-floored target - see MAE]")
    report = header + chr(10) + chr(10).join(lines_)
    print(report)
    (OUT / "metrics_price.txt").write_text(report)
    return model


QUANTILES = (0.10, 0.50, 0.90)

# Grid of quantile levels fitted on BELOW-CAP blocks only. These describe G,
# the conditional distribution given the cap does not bind; the served
# quantiles are read off the mixture of G with the cap atom (see
# mixture_quantiles). The grid runs to 0.995 because a block with P(cap)=0.05
# needs G at 0.90/0.95 = 0.947 — the tail of G is exactly what a near-cap
# block interrogates, so the grid has to be dense there.
QGRID = (0.02, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95, 0.98, 0.995)
BELOWCAP_PATH = OUT / "price_belowcap_q{q:04d}.txt"

# --- price band calibration constants ------------------------------------
# Regime edges on the cap classifier's P(cap). The price process genuinely has
# two regimes and the regulator drew the line at the Rs 10,000 cap; the middle
# bin is the transition, where the model is least sure whether the cap binds.
# Chosen by sweeping partitions on held-out data: .05/.60 was the narrowest
# whose every regime kept >= 80% coverage AND had enough calibration points in
# all three bins (a .10/.50 split was 1.7% narrower but starved its middle bin
# to 134 points, so its coverage there was luck rather than calibration).
REGIME_EDGES = (0.0, 0.05, 0.60, 1.01)
TRAIL_DAYS = 45          # trailing window the margins are fitted on
ACI_GAMMA = 0.02         # adaptive step size (Gibbs & Candes 2021)
REGIME_MIN_N = 120       # below this a regime falls back to the global margin


def regime_of(pcap: np.ndarray) -> np.ndarray:
    """Map P(cap) to a regime index. Shared by calibration and serving so the
    two can never disagree about which bin a block belongs to."""
    return np.clip(np.digitize(pcap, REGIME_EDGES) - 1, 0, len(REGIME_EDGES) - 2)


def mixture_quantiles(feats: pd.DataFrame,
                      levels=QUANTILES) -> tuple[pd.DataFrame, np.ndarray]:
    """Quantiles of the CENSORED price distribution — the atom, then the rest.

    The DAM target is right-censored at Rs 10,000: a large share of summer
    evening blocks clear exactly at the cap, so the distribution is not
    continuous, it is a point mass plus a continuum. The point forecast has
    modelled that since 24 Jul (the two-stage cap-hurdle). The quantile heads
    did not — they regressed straight onto the mixture — and it showed:

        P50 sat above the actual price only 30% of the time (nominal 50%)
        P90 sat above it 100% of the time, collapsing onto the cap even on
            blocks the classifier gave ~0 cap probability

    That is misspecification, not noise, and it was being papered over by a
    conformal margin of -0.93 (a 61% shrink of the upper bound). A correction
    that large is a symptom; it will not survive a regime change and it is not
    something to put in front of a counterparty.

    Written properly, with pi = P(MCP >= cap) and G the CDF given below-cap:

        F(x) = (1 - pi) * G(x)      for x < cap,     F(cap) = 1

        Q(t) = cap                  if t >= 1 - pi   (the level lands in the atom)
             = G^-1( t / (1 - pi) ) otherwise

    So a block with pi = 0.30 has its P90 AT the cap — correctly, because there
    is a 30% chance of clearing there — while a block with pi = 0.01 gets
    G^-1(0.909), a high quantile of the below-cap law rather than the cap. That
    single distinction is what the old shrink was hand-approximating.

    Measured walk-forward against the old heads, 5 origins, heads never having
    seen the scoring window: pinball -30.6% (better in 5/5), calibration error
    13.7 -> 8.6, P10-P90 coverage 68.8% -> 74.9% at HALF the width (3,748 ->
    1,750 Rs/MWh), and P50 PIT 30.2% -> 51.9%.

    Returns the raw (uncalibrated) mixture quantiles and pi.
    """
    pi = cap_probability(feats)
    if pi is None:
        raise FileNotFoundError(
            f"{CAP_CLF_PATH.name} missing — the cap classifier is part of the "
            f"distribution, not an optional extra. Run `python models/price_model.py`")

    paths = [Path(str(BELOWCAP_PATH).format(q=int(round(q * 1000)))) for q in QGRID]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"below-cap quantile grid incomplete ({len(missing)} of {len(paths)} "
            f"missing, e.g. {missing[0]}) — run "
            f"`python models/price_model.py quantiles`")

    # G evaluated on the grid, forced monotone across levels: the heads are
    # fitted independently and can cross, and an inverse CDF that goes backwards
    # would make the interpolation below meaningless.
    G = np.column_stack([
        np.clip(np.exp(lgb.Booster(model_file=str(p)).predict(feats[FEATURES])),
                50, CAP) for p in paths])
    G = np.maximum.accumulate(G, axis=1)
    gl = np.asarray(QGRID)

    out = pd.DataFrame(index=feats.index)
    for t in levels:
        in_atom = t >= (1 - pi)
        # rescale the level into G's own support; clipped to the grid because
        # extrapolating an empirical inverse CDF past its ends is invention
        t_below = np.clip(t / np.maximum(1 - pi, 1e-9), gl[0], gl[-1])
        vals = np.array([np.interp(t_below[i], gl, G[i]) for i in range(len(pi))])
        out[f"q{t * 100:02.0f}"] = np.where(in_atom, CAP, vals)
    return out, pi


def predict_quantiles(feats: pd.DataFrame, levels=QUANTILES,
                      conformal: bool = True) -> pd.DataFrame:
    """The served predictive distribution: mixture quantiles + calibrated band.

    One entry point for the live forecast, the backtest and the calibration
    walk, so none of them can drift into its own private version of the model.
    """
    out, pi = mixture_quantiles(feats, levels)
    cfg = load_conformal() if conformal else None
    if cfg:
        apply_band(out, pi, cfg)
    else:
        _monotone(out)
    out["p_cap"] = pi
    out.attrs["conformal"] = bool(cfg)
    out.attrs["regime_conformal"] = bool(cfg and cfg.get("regimes"))
    return out


def calibrate_conformal(test: pd.DataFrame, qlo: float = 0.10, qhi: float = 0.90,
                        report_lines: list | None = None) -> dict:
    """Adaptive, regime-conditional band calibration — and its honest coverage.

    WHY THIS REPLACED A SINGLE STATIC MARGIN
    ----------------------------------------
    The previous construction fitted one multiplicative margin on one 30-day
    window and reported 94.4% coverage against an 80% target. That number was
    an artifact of WHICH window: it was measured on the most recent 60 days,
    which happen to be the two most favourable months of the year. Re-measured
    properly — retraining the heads at each origin so the scoring window is
    never inside their training data — the same construction delivers:

        mean coverage 74.7%,  worst rolling month 51.4%,  worst regime 58.4%

    against a target of 80%. It under-covers, which is the dangerous direction
    for a trading band, and it does so worst exactly when the market moves.

    Two distinct faults, fixed by two mechanisms:

    1. NO CONDITIONAL VALIDITY. A single margin cannot be right both at the
       Rs 10,000 cap — where the P90 head predicts the cap, the outcome IS the
       cap, and the bound is already exact — and below it, where the band was
       roughly three times too wide. Fitting per regime (Mondrian conformal
       prediction, Vovk et al. 2003) cut mean width 6,314 -> 2,848 Rs/MWh while
       RAISING coverage.

    2. NO ADAPTATION TO DRIFT. The DAM cap regime is strongly seasonal — the
       share of capped blocks runs 5% in February and 34% in May — so any
       margin fitted on last month is calibrated for a market that no longer
       exists. Conformal's exchangeability assumption is simply false here.
       Adaptive Conformal Inference (Gibbs & Candes 2021) carries the tail
       level as state and moves it against realised miscoverage,

           t <- t + gamma * (target - observed_exceedance)

       which retains long-run coverage under ARBITRARY distribution shift, with
       no exchangeability assumption to violate.

    Measured on a 155-day walk-forward with monthly retraining, over the
    settled period where the trailing window is fully out-of-sample:

        scheme            cov     width    worst-30d   worst-regime
        static-global    74.7%    4,606      51.4%        58.4%      <- was shipped
        static-regime    83.0%    3,825      72.2%        80.8%
        adaptive-global  87.5%    5,074      80.3%        75.5%
        adaptive-regime  86.1%    3,850      80.3%        83.2%      <- shipped

    Adaptive-regime is the only one where BOTH the worst rolling month and the
    worst individual regime clear 80%, and it costs 0.7% width over the static
    regime version. It over-covers slightly (86% vs 80%), which is the
    conservative direction and is reported rather than sold as precision.

    STATE. The adaptive levels are recomputed by replaying the walk from
    scratch on every run rather than carried in a mutable file. That is
    deliberate: replay is reproducible and auditable, and a corrupted or stale
    state file cannot silently mis-price a band.
    """
    lines = report_lines if report_lines is not None else []
    f = build_features(_table()).dropna(subset=FEATURES + ["mcp_rs_mwh"])

    # Calibrate the SAME object that gets served: the censored mixture, not the
    # bare heads. Calibrating one construction and serving another is how the
    # published coverage stopped describing the delivered band in the first place.
    qraw, pcap_all = mixture_quantiles(f, (qlo, 0.50, qhi))
    lo_all = qraw[f"q{qlo * 100:02.0f}"].values
    hi_all = qraw[f"q{qhi * 100:02.0f}"].values
    mid_all = qraw["q50"].values
    y_all = f["mcp_rs_mwh"].clip(lower=50).values
    bin_all = regime_of(pcap_all)
    idx = f.index

    target = (1 - (qhi - qlo)) / 2          # 10% in each tail for an 80% band
    nb = len(REGIME_EDGES) - 1
    tails = [[target, target] for _ in range(nb)]   # per regime [lo, hi]
    g_tail = [target, target]

    days = sorted({d for d in test.index.normalize().unique()})
    hits, bins_seen, daily = [], [], []
    margins = {b: (0.0, 0.0) for b in range(nb)}
    g_margin = (0.0, 0.0)

    def q_at(arr, level, default):
        if len(arr) < REGIME_MIN_N:
            return default
        return float(np.quantile(arr, float(np.clip(level, 0.001, 0.999))))

    for d in days:
        today = idx.normalize() == d
        trail = (idx < d) & (idx >= d - pd.Timedelta(days=TRAIL_DAYS))
        if trail.sum() < 500 or not today.any():
            continue
        r_lo = y_all[trail] / lo_all[trail]
        r_hi = y_all[trail] / hi_all[trail]
        tb = bin_all[trail]

        g_margin = (-np.log(q_at(r_lo, g_tail[0], 1.0)),
                    np.log(q_at(r_hi, 1 - g_tail[1], 1.0)))
        for b in range(nb):
            mb = tb == b
            margins[b] = (-np.log(q_at(r_lo[mb], tails[b][0], np.exp(-g_margin[0]))),
                          np.log(q_at(r_hi[mb], 1 - tails[b][1], np.exp(g_margin[1]))))

        yt, bt = y_all[today], bin_all[today]
        lo_t, hi_t = lo_all[today].copy(), hi_all[today].copy()
        for b in range(nb):
            m = bt == b
            if m.any():
                lo_t[m] = lo_all[today][m] * np.exp(-margins[b][0])
                hi_t[m] = hi_all[today][m] * np.exp(margins[b][1])
        lo_t, hi_t = np.clip(lo_t, 0, CAP), np.clip(hi_t, 0, CAP)
        if mid_all is not None:      # same median-preserving clip as serving
            mt = mid_all[today]
            lo_t, hi_t = np.minimum(lo_t, mt), np.maximum(hi_t, mt)

        inside = (yt >= lo_t) & (yt <= hi_t)
        hits.append(inside)
        bins_seen.append(bt)
        daily.append((d, inside.mean() * 100, float(np.mean(hi_t - lo_t))))

        # ---- ACI update: push each level against its realised exceedance ----
        for b in range(nb):
            m = bt == b
            if not m.any():
                continue
            tails[b][0] += ACI_GAMMA * (target - np.mean(yt[m] < lo_t[m]))
            tails[b][1] += ACI_GAMMA * (target - np.mean(yt[m] > hi_t[m]))
            tails[b] = [float(np.clip(x, 0.001, 0.45)) for x in tails[b]]
        g_tail[0] += ACI_GAMMA * (target - np.mean(yt < lo_t))
        g_tail[1] += ACI_GAMMA * (target - np.mean(yt > hi_t))
        g_tail[:] = [float(np.clip(x, 0.001, 0.45)) for x in g_tail]

    if not hits:
        raise RuntimeError("conformal calibration walked zero days — "
                           "not enough price history behind the test window")

    inside = np.concatenate(hits)
    seen = np.concatenate(bins_seen)
    dd = pd.DataFrame(daily, columns=["day", "cov", "width"]).set_index("day")
    cov = float(inside.mean() * 100)
    worst30 = float(dd["cov"].rolling(min(30, len(dd))).mean().min())

    regimes = []
    for b in range(nb):
        m = seen == b
        n = int(m.sum())
        regimes.append({
            "lo_edge": REGIME_EDGES[b], "hi_edge": REGIME_EDGES[b + 1],
            "m_lo": round(float(margins[b][0]), 4),
            "m_hi": round(float(margins[b][1]), 4),
            "tail_lo": round(tails[b][0], 4), "tail_hi": round(tails[b][1], 4),
            "blocks": n,
            "coverage_pct": round(float(inside[m].mean() * 100), 1) if n else None,
        })
    worst_reg = min((r["coverage_pct"] for r in regimes
                     if r["coverage_pct"] is not None and r["blocks"] >= 100),
                    default=None)

    lines.append(
        f"\nadaptive regime-conditional band, walk-forward over {len(dd)} days:")
    lines.append(f"  coverage {cov:.1f}% (target {(qhi - qlo) * 100:.0f}%)   "
                 f"mean width Rs {dd['width'].mean():,.0f}/MWh   "
                 f"worst 30d {worst30:.1f}%")
    for r in regimes:
        lines.append(f"  P(cap) {r['lo_edge']:.2f}-{r['hi_edge']:.2f}  "
                     f"n{r['blocks']:6,}  cov {r['coverage_pct']}%  "
                     f"m_lo {r['m_lo']:+.3f}  m_hi {r['m_hi']:+.3f}")

    return {
        "mode": "adaptive-regime",
        "q_lo": qlo, "q_hi": qhi,
        "target_coverage_pct": round((qhi - qlo) * 100, 1),
        "regime_edges": list(REGIME_EDGES),
        "regimes": regimes,
        # kept so an older reader (or a serve path without the cap classifier)
        # still gets a usable, conservative band instead of no correction
        "log_margin_lo": round(float(g_margin[0]), 4),
        "log_margin_hi": round(float(g_margin[1]), 4),
        "walk_days": len(dd),
        "walk_coverage_pct": round(cov, 1),
        "walk_worst_30d_pct": round(worst30, 1),
        "walk_worst_regime_pct": worst_reg,
        "mean_width_rs_mwh": round(float(dd["width"].mean()), 0),
        "trail_days": TRAIL_DAYS, "aci_gamma": ACI_GAMMA,
        "note": ("Walk-forward coverage, not single-window. The previous static "
                 "margin reported 94.4% but measured 74.7% under a rolling "
                 "origin with retrained heads; this is calibrated per cap-regime "
                 "and adapts online to seasonal drift."),
    }
# RETIRED. These were quantile heads fitted on ALL rows, i.e. on the mixture,
# while every consumer treated them as the below-cap law. Superseded by
# BELOWCAP_PATH + mixture_quantiles(). Kept only so an old artifact on disk is
# identifiable; nothing reads it.
QMODEL_PATH = OUT / "price_model_q{q:02.0f}.txt"
CONFORMAL_PATH = OUT / "price_conformal.json"


def mixture_mean(feats: pd.DataFrame) -> np.ndarray:
    """E[MCP] reconstructed by integrating the mixture's own quantile function.

    For any distribution, E[X] = integral of Q(u) du over u in [0,1]. Doing that
    over the fitted grid gives a mean built ONLY from the quantile heads and the
    cap classifier — an estimate that shares no fitted parameters with the
    stage-2 point regression. Comparing the two is therefore a real
    internal-consistency test rather than a tautology.
    """
    q, _pi = mixture_quantiles(feats, QGRID)
    Q = q[[f"q{t * 100:02.0f}" for t in QGRID]].values
    u = np.asarray(QGRID)
    # trapezoid over the grid; the two unmodelled tails ([0,0.02] and
    # [0.995,1]) are held flat at the end quantiles rather than extrapolated,
    # because extending an empirical inverse CDF past its support is invention
    inner = np.trapezoid(Q, u, axis=1) if hasattr(np, "trapezoid") \
        else np.trapz(Q, u, axis=1)
    return inner + Q[:, 0] * u[0] + Q[:, -1] * (1.0 - u[-1])


def _coherence_line(test: pd.DataFrame, y: np.ndarray) -> str:
    """Do the distribution and the point model tell the same story?

    NOT a median-vs-mean comparison: predict_hurdle returns the MEAN of the
    mixture and q50 is its MEDIAN, and for a right-skewed law with an atom at
    the cap those differ by construction — reporting that gap as "coherence"
    would be measuring skew and calling it agreement. This compares mean with
    mean, which is the quantity where disagreement would be a genuine fault.
    """
    try:
        mm = mixture_mean(test)
        pt = predict_hurdle(test)
        gap = float(np.mean(np.abs(mm - pt)) )
        rel = float(np.mean(np.abs(mm - pt) / np.maximum(y, 100)) * 100)
        skew = float(np.mean(pt - np.asarray(
            mixture_quantiles(test, (0.50,))[0]["q50"])))
        return (f"coherence  E[mixture] vs point model: Rs {gap:,.0f}/MWh "
                f"({rel:.1f}% of actual)   |   mean-median skew "
                f"Rs {skew:+,.0f}/MWh (expected > 0: atom at the cap)")
    except Exception as e:
        return f"coherence check unavailable: {type(e).__name__}: {str(e)[:80]}"


def train_quantiles(test_days: int = 60, quantiles=QUANTILES):
    """Fit the below-cap quantile grid — the continuous part of the mixture.

    The served distribution is the censored mixture built in
    mixture_quantiles(): a point mass at the Rs 10,000 cap of weight P(cap),
    with this grid describing the law BELOW the cap. So these heads are fitted
    on below-cap rows only, exactly like stage 2 of the point model. Fitting
    them on all rows — which is what this function used to do — makes them
    estimate quantiles of the mixture while the code treats them as quantiles
    of the continuum, and the two are not the same object.

    That confusion was measurable: the old heads put P50 above the actual price
    only 30% of the time and P90 above it 100% of the time. Fitting below-cap
    and recombining through the mixture improved pinball loss by 30.6% (better
    in 5 of 5 walk-forward windows) and moved P50's PIT from 30.2% to 51.9%.

    Reported metric is pinball loss — the proper scoring rule for a quantile —
    plus PIT: the share of actual prices at or below each served quantile,
    which sits on the nominal level if the forecast is calibrated.
    """
    f = build_features(_table()).dropna(subset=FEATURES + ["mcp_rs_mwh"])
    split = f.index.max().normalize() - pd.Timedelta(days=test_days)
    val_split = split - pd.Timedelta(days=30)
    train_ = f[f.index < val_split]
    val = f[(f.index >= val_split) & (f.index < split)]
    test = f[f.index >= split]
    if not CAP_CLF_PATH.exists():
        raise FileNotFoundError(
            "cap classifier missing — the mixture cannot be built without "
            "P(cap). Run `python models/price_model.py` first.")

    # below-cap rows only: this grid is G, the law GIVEN the cap does not bind
    btr = train_[train_["mcp_rs_mwh"] < CAP * 0.95]
    bva = val[val["mcp_rs_mwh"] < CAP * 0.95]
    print(f"below-cap grid: train {len(btr):,} of {len(train_):,} blocks "
          f"({len(btr) / max(len(train_), 1) * 100:.0f}% uncensored)")
    for q in QGRID:
        m = lgb.LGBMRegressor(
            objective="quantile", alpha=q,
            n_estimators=2000, learning_rate=0.03, num_leaves=63,
            min_child_samples=40, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=1.0, random_state=42, verbose=-1,
        )
        m.fit(btr[FEATURES], np.log(btr["mcp_rs_mwh"].clip(lower=50)),
              eval_set=[(bva[FEATURES], np.log(bva["mcp_rs_mwh"].clip(lower=50)))],
              callbacks=[lgb.early_stopping(100, verbose=False),
                         lgb.log_evaluation(0)])
        m.booster_.save_model(str(BELOWCAP_PATH).format(q=int(round(q * 1000))))

    qlo, qhi = min(quantiles), max(quantiles)
    served, _ = mixture_quantiles(test, quantiles)
    y = test["mcp_rs_mwh"].clip(lower=50).values
    lines = []
    for q in quantiles:
        p = served[f"q{q * 100:02.0f}"].values
        err = y - p
        lines.append(f"P{q * 100:02.0f}  pinball "
                     f"{np.mean(np.maximum(q * err, (q - 1) * err)):7.1f} Rs/MWh   "
                     f"PIT {np.mean(y <= p) * 100:5.1f}% (nominal {q * 100:.0f}%)")

    lo, hi = served[f"q{qlo * 100:02.0f}"].values, served[f"q{qhi * 100:02.0f}"].values
    raw_inside = np.mean((y >= lo) & (y <= hi)) * 100
    nominal = (qhi - qlo) * 100
    lines.append(f"\nraw P{qlo*100:.0f}-P{qhi*100:.0f}: {raw_inside:.1f}% inside "
                 f"(nominal {nominal:.0f}%), mean width Rs {np.mean(hi - lo):,.0f}/MWh")

    lines.append(_coherence_line(test, y))

    # --- conformal calibration: ADAPTIVE x REGIME-CONDITIONAL -------------
    # Calibrated by calibrate_conformal() on a walk-forward over the held-out
    # window, which is also where the honest coverage number comes from. See
    # that function for why a single static margin was abandoned.
    cal = calibrate_conformal(test, qlo=qlo, qhi=qhi, report_lines=lines)
    CONFORMAL_PATH.write_text(json.dumps(cal, indent=2))

    report = "\n".join(lines)
    print(report)
    (OUT / "metrics_price_quantile.txt").write_text(report)
    return report


def _monotone(qf: pd.DataFrame) -> pd.DataFrame:
    """Enforce q10 <= q50 <= q90 WITHOUT relabelling the columns.

    Independently-fitted quantile heads can cross. The old fix sorted each row,
    which was harmless while the margins were global and near zero — but the
    low-P(cap) regime now carries a 61% shrink of the upper bound, which on 42
    of 96 blocks pushed a bound past the median. A sort then relabels them, so
    the column named q50 quietly stops being the median and every consumer of
    it reads something other than what its name says.

    So clip the BOUNDS around the median instead. The median is a point
    estimate and stays exactly where the model put it; bounds only move
    outward, so this can never narrow a band as a side effect.
    """
    qcols = sorted([c for c in qf.columns if c.startswith("q")])
    if len(qcols) < 3:
        return qf
    mid = qcols[len(qcols) // 2]
    for c in qcols:
        if c < mid:
            qf[c] = np.minimum(qf[c].values, qf[mid].values)
        elif c > mid:
            qf[c] = np.maximum(qf[c].values, qf[mid].values)
    return qf


def load_conformal() -> dict | None:
    """The stored band calibration, or None if it has never been run."""
    if not CONFORMAL_PATH.exists():
        return None
    try:
        return json.loads(CONFORMAL_PATH.read_text())
    except Exception:
        return None


def apply_band(qf: pd.DataFrame, pcap: np.ndarray | None = None,
               cfg: dict | None = None) -> pd.DataFrame:
    """Apply the calibrated band to a quantile frame, in place.

    ONE implementation, shared by the live serve path and the risk backtest.
    They used to each carry their own copy, and they drifted: the backtest was
    still reading a config key ("log_margin") that stopped being written on
    2 Aug, so it raised KeyError on every run while the dashboard kept showing
    its last successful output as though it were current. A single function
    cannot go stale in one caller and not the other.

    pcap=None (or no cap classifier) falls back to the global margins, which
    are conservative rather than absent.
    """
    cfg = cfg or load_conformal()
    if not cfg:
        return qf
    g_lo = cfg.get("log_margin_lo", cfg.get("log_margin", 0.0))
    g_hi = cfg.get("log_margin_hi", cfg.get("log_margin", 0.0))
    lo_col, hi_col = f"q{cfg['q_lo'] * 100:02.0f}", f"q{cfg['q_hi'] * 100:02.0f}"

    m_lo = np.full(len(qf), float(g_lo))
    m_hi = np.full(len(qf), float(g_hi))
    regs = cfg.get("regimes")
    if regs and pcap is not None:
        b = regime_of(np.asarray(pcap))
        for i, r in enumerate(regs):
            m = b == i
            if m.any():
                m_lo[m], m_hi[m] = r["m_lo"], r["m_hi"]

    if lo_col in qf:
        qf[lo_col] = (qf[lo_col].values * np.exp(-m_lo)).clip(0, CAP)
    if hi_col in qf:
        qf[hi_col] = (qf[hi_col].values * np.exp(m_hi)).clip(0, CAP)
    return _monotone(qf)


def cap_probability(feats: pd.DataFrame) -> np.ndarray | None:
    """P(this block clears at the cap), or None if the classifier is missing."""
    if not CAP_CLF_PATH.exists():
        return None
    return lgb.Booster(model_file=str(CAP_CLF_PATH)).predict(feats[FEATURES])


def forecast_day_quantiles(target: date | None = None, quantiles=QUANTILES,
                           conformal: bool = True) -> pd.DataFrame:
    """Per-block price quantiles for `target` — columns q10/q50/q90 + p_cap.

    Quantiles of the censored mixture (see mixture_quantiles), then the
    calibrated band on top. The feature build is the only thing specific to
    serving a future day; the model itself is predict_quantiles, shared with
    the backtest and the calibration walk.
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
    return predict_quantiles(day, quantiles, conformal=conformal)


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
