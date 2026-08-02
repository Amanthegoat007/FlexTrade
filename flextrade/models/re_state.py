"""Solar and wind generation forecast on REAL measured output, per state.

This closes the biggest honesty gap in the stack. Everything RE and DSM rested
on models/re_model.py, a physics twin of a hypothetical 50 MW solar + 50 MW
wind plant in Delhi NCR: the schedule and the "actual" both came out of the
same deterministic model, so the deviation being settled was pure weather
error with no plant in it. We labelled that as a lower bound, which was
honest, but it was still a simulation priced in rupees.

It turns out we were sitting on the real thing. MERIT's plant-level generation
endpoint has been backfilled since July and holds **measured daily solar and
wind output for 9 states**, up to 361 days each:

    RJ  43,444 MWh/day solar   16,839 wind
    MH  34,143                 19,045
    AP  33,890                 21,765
    GJ  24,777                 18,248
    TN  23,471                 12,767

So the forecast below is trained on generation that actually happened, scored
against generation that actually happened, and its error is a real RE
forecaster's error rather than a weather model's error propagated through a
power curve.

Design follows models/state_forecast.py, for the same reason: no single state
has enough history to train alone, so one pooled learner sees every state at
once with state identity and scale as features.

Bid-time validity: generation lags are >= 2 days (MERIT publishes with a lag),
while the target day's WEATHER is legitimately known — a day-ahead irradiance
and wind forecast is exactly what an RE scheduler has at gate closure. That
asymmetry is the whole point: weather drives RE, and weather is forecastable.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from ingest import merit_history, states  # noqa: E402
from models import state_forecast as sf  # noqa: E402

OUT = HERE.parent / "output"
MODEL_DIR = OUT / "state_models"
TARGETS = ("solar_mwh", "wind_mwh")

# Below this a state's RE fleet is rounding noise, not a book worth forecasting
MIN_MEAN_MWH_DAY = 500.0

BASE = ["state_id", "state_scale", "dow", "month", "doy_sin", "doy_cos",
        "lag2", "lag3", "lag7", "roll7", "roll28", "rel_lag2", "ratio_7_28"]
# target-day weather: a real day-ahead forecast exists for all of these
WX = ["shortwave_radiation_sum", "temperature_2m_mean", "temperature_2m_max",
      "precipitation_sum", "cloud_proxy",
      # wind: hub height and its CUBE. A turbine's power is proportional to
      # v^3 at ~100 m, so feeding a 10 m screen reading linearly — which is
      # what the first version did — asks the model to learn both the height
      # extrapolation and the cubic in one go. It could not: wind lost to a
      # seasonal-naive baseline on all 5 states.
      "wind_site_mean", "wind_site_max", "wind_cube",
      "wind_speed_100m_mean", "wind_10m_max"]


# Where the WIND FLEET actually is, which is not where the state capital is.
# This was the whole reason the wind model lost to a seasonal-naive baseline on
# every state: we were feeding it the capital's wind speed. Tamil Nadu is the
# extreme case — Chennai (13.1N) is ~550 km from the Muppandal cluster (8.3N),
# and Muppandal sits in the Palghat Gap, a monsoon wind tunnel whose regime has
# nothing to do with the coast at Chennai. Solar is far more spatially uniform,
# so it keeps the capital point.
WIND_SITES = {
    "GJ": (23.7, 69.7),   # Kutch
    "TN": (8.3, 77.6),    # Muppandal / Aralvaimozhi, Palghat Gap
    "RJ": (26.9, 70.9),   # Jaisalmer
    "MH": (17.3, 74.2),   # Sangli-Satara belt
    "AP": (14.7, 77.6),   # Anantapur
    "KL": (10.4, 76.9),   # Palakkad
    "MP": (22.8, 75.9),   # Dewas-Ratlam
}

WIND_TABLE = "wind_site_weather_daily"


def cache_wind_weather(codes, start: str, end: str) -> pd.DataFrame:
    """Daily hub-height wind AT THE FLEET, not at the capital."""
    import requests
    from ingest import store
    with store.connect() as con:
        con.execute(f"""CREATE TABLE IF NOT EXISTS {WIND_TABLE} (
            code TEXT, day TEXT, w100_max REAL, w100_mean REAL,
            w10_max REAL, PRIMARY KEY (code, day))""")
        try:
            have = set(pd.read_sql(
                f"SELECT DISTINCT code FROM {WIND_TABLE}", con)["code"])
        except Exception:
            have = set()

    for c in [c for c in codes if c in WIND_SITES and c not in have]:
        lat, lon = WIND_SITES[c]
        try:
            r = requests.get(
                "https://archive-api.open-meteo.com/v1/archive",
                params=dict(latitude=lat, longitude=lon, start_date=start,
                            end_date=end,
                            daily="wind_speed_100m_max,wind_speed_100m_mean,"
                                  "wind_speed_10m_max",
                            timezone="Asia/Kolkata"), timeout=60)
            r.raise_for_status()
            d = pd.DataFrame(r.json()["daily"])
            d["day"] = pd.to_datetime(d.pop("time")).dt.strftime("%Y-%m-%d")
            d = d.rename(columns={"wind_speed_100m_max": "w100_max",
                                  "wind_speed_100m_mean": "w100_mean",
                                  "wind_speed_10m_max": "w10_max"})
            d["code"] = c
            with store.connect() as con:
                d.to_sql("_tmp_ws", con, if_exists="replace", index=False)
                cols = ",".join(d.columns)
                con.execute(f"INSERT OR REPLACE INTO {WIND_TABLE} ({cols}) "
                            f"SELECT {cols} FROM _tmp_ws")
                con.execute("DROP TABLE _tmp_ws")
            print(f"  wind site {c} @ {lat},{lon}: {len(d)} days")
        except Exception as e:
            print(f"  wind site {c}: FAILED {str(e)[:70]}")

    with store.connect() as con:
        return pd.read_sql(f"SELECT * FROM {WIND_TABLE}", con, parse_dates=["day"])


def build_panel(target: str) -> pd.DataFrame:
    """One row per (state, day): measured RE output + bid-time-valid features."""
    g = merit_history.read_generation()
    if not len(g):
        raise RuntimeError("no MERIT generation history — run ingest/merit_history.py")
    g = g[["code", "day", target]].dropna(subset=[target])

    keep = [c for c, m in g.groupby("code")[target].mean().items()
            if m >= MIN_MEAN_MWH_DAY]
    if not keep:
        raise RuntimeError(f"no state averages >= {MIN_MEAN_MWH_DAY} MWh/day of {target}")
    g = g[g["code"].isin(keep)]

    lo, hi = g["day"].min().strftime("%Y-%m-%d"), g["day"].max().strftime("%Y-%m-%d")
    w = sf.cache_weather(sorted(g["code"].unique()), lo, hi)
    df = g.merge(w, on=["code", "day"], how="left") if len(w) else g.copy()
    if target == "wind_mwh":
        ws = cache_wind_weather(sorted(g["code"].unique()), lo, hi)
        if len(ws):
            df = df.merge(ws, on=["code", "day"], how="left")

    # continuous calendar per state BEFORE any shift, or lag2 silently means
    # "two stored rows back" and can span a fortnight (the bug that once made
    # the demand model lose to a naive baseline by 2x)
    df = df.sort_values(["code", "day"])
    full = []
    for code, grp in df.groupby("code"):
        idx = pd.date_range(grp["day"].min(), grp["day"].max(), freq="D")
        grp = grp.set_index("day").reindex(idx)
        grp["code"] = code
        grp.index.name = "day"
        full.append(grp.reset_index())
    df = pd.concat(full, ignore_index=True).sort_values(["code", "day"])
    gb = df.groupby("code", group_keys=False)

    for lag in (2, 3, 7):
        df[f"lag{lag}"] = gb[target].shift(lag)
    df["roll7"] = gb[target].transform(
        lambda s: s.shift(2).rolling(7, min_periods=3).mean())
    df["roll28"] = gb[target].transform(
        lambda s: s.shift(2).rolling(28, min_periods=7).mean())
    df["ratio_7_28"] = df["roll7"] / df["roll28"].replace(0, np.nan)
    df["state_id"] = df["code"].astype("category").cat.codes
    df["state_scale"] = gb[target].transform(
        lambda s: s.shift(2).expanding(min_periods=5).mean())
    df["rel_lag2"] = df["lag2"] / df["state_scale"].replace(0, np.nan)

    idx = df["day"].dt
    df["dow"] = idx.dayofweek
    df["month"] = idx.month
    doy = idx.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    # prefer the fleet-site reading; fall back to the capital where absent
    site_mean = df["w100_mean"] if "w100_mean" in df else None
    if site_mean is not None and "wind_speed_100m_mean" in df:
        site_mean = site_mean.fillna(df["wind_speed_100m_mean"])
    elif site_mean is None and "wind_speed_100m_mean" in df:
        site_mean = df["wind_speed_100m_mean"]
    if site_mean is not None:
        df["wind_site_mean"] = site_mean
        df["wind_cube"] = site_mean ** 3
    if "w100_max" in df:
        df["wind_site_max"] = df["w100_max"]
    if "wind_speed_10m_max" in df:
        df["wind_10m_max"] = df["wind_speed_10m_max"]
    if "shortwave_radiation_sum" in df:
        # clear-sky proxy: radiation relative to this state's seasonal normal
        df["cloud_proxy"] = df["shortwave_radiation_sum"] / gb[
            "shortwave_radiation_sum"].transform(
            lambda s: s.rolling(28, min_periods=7, center=True).mean()).replace(0, np.nan)
    return df


def _features(df: pd.DataFrame) -> list[str]:
    return ([c for c in BASE if c in df.columns]
            + [c for c in WX if c in df.columns and df[c].notna().mean() > 0.5])


def train(target: str = "solar_mwh", test_days: int = 45) -> dict:
    import lightgbm as lgb
    df = build_panel(target)
    feats = _features(df)
    required = [c for c in feats if c not in WX]
    d = df.dropna(subset=required + [target])
    d = d[d[target] > 0]

    cut = d["day"].max() - pd.Timedelta(days=test_days)
    vcut = cut - pd.Timedelta(days=max(45, test_days))
    tr, va, te = d[d.day < vcut], d[(d.day >= vcut) & (d.day < cut)], d[d.day >= cut]
    if not len(tr) or not len(te):
        raise RuntimeError(f"not enough history for {target}")

    model = lgb.LGBMRegressor(
        n_estimators=1500, learning_rate=0.03, num_leaves=31,
        min_child_samples=15, subsample=0.85, subsample_freq=1,
        colsample_bytree=0.85, reg_lambda=2.0, random_state=42, verbose=-1)
    model.fit(tr[feats], np.log1p(tr[target]),
              eval_set=[(va[feats], np.log1p(va[target]))],
              callbacks=[lgb.early_stopping(100, verbose=False),
                         lgb.log_evaluation(0)])
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(MODEL_DIR / f"re_{target}.txt"))

    y = te[target].values
    pred = np.expm1(model.predict(te[feats])).clip(min=0)
    naive = te["lag7"].values          # same weekday last week
    naive = np.where(np.isnan(naive), te["roll7"].values, naive)

    def wape(a, p):
        p = np.asarray(p, float)
        ok = ~np.isnan(p)
        return float(np.sum(np.abs(a[ok] - p[ok])) / np.sum(np.abs(a[ok])) * 100)

    # Same serving rule as the demand model, for the same measured reason:
    # per-state selection on ~1 year of daily data is too noisy to trust, so
    # weights are fixed and equal, with one coarse guard for states where the
    # two candidates are not in the same league.
    BLEND_W = 0.5
    naive_ok = ~np.isnan(naive)
    served = np.where(naive_ok, BLEND_W * pred + (1 - BLEND_W) * naive, pred)
    per_state_served = {}
    for code in te["code"].unique():
        m = (te["code"] == code).values
        wm, wn = wape(y[m], pred[m]), wape(y[m], naive[m])
        if wm > 1.6 * wn:          # model is not close — serve the baseline
            served[m] = naive[m]
            per_state_served[code] = "naive"
        elif wn > 1.6 * wm:        # baseline is not close — serve the model
            served[m] = pred[m]
            per_state_served[code] = "model"
        else:
            per_state_served[code] = f"blend {BLEND_W:.0%}"

    per = []
    for code, grp in te.groupby("code"):
        m = (te["code"] == code).values
        per.append({
            "code": code,
            "name": states.MERIT_CODES.get(code, (None, code))[1],
            "n_days": int(len(grp)),
            "mean_mwh": round(float(grp[target].mean()), 0),
            "model_wape_pct": round(wape(y[m], pred[m]), 2),
            "naive_wape_pct": round(wape(y[m], naive[m]), 2),
            "served_wape_pct": round(wape(y[m], served[m]), 2),
            "served": per_state_served.get(code, "blend"),
        })
    per.sort(key=lambda r: r["served_wape_pct"])
    beat = sum(1 for r in per if r["served_wape_pct"] < r["naive_wape_pct"])

    return {
        "target": target,
        "n_states": int(d["code"].nunique()),
        "n_train_rows": int(len(tr)),
        "n_test_rows": int(len(te)),
        "test_from": str(te["day"].min().date()),
        "test_days": test_days,
        "history_from": str(d["day"].min().date()),
        "history_to": str(d["day"].max().date()),
        "model_wape_pct": round(wape(y, pred), 2),
        "naive_wape_pct": round(wape(y, naive), 2),
        "served_wape_pct": round(wape(y, served), 2),
        "states_beating_naive": f"{beat}/{len(per)}",
        "features": feats,
        "per_state": per,
        "basis": ("Trained and scored on MEASURED daily generation from MERIT's "
                  "plant-level endpoint — not a physics twin. Generation lags are "
                  ">= 2 days (MERIT publishes with a lag); the target day's weather "
                  "is used directly, because a day-ahead irradiance and wind "
                  "forecast is exactly what an RE scheduler holds at gate closure."),
    }


def run_all(test_days: int = 45) -> dict:
    out = {"generated_at": pd.Timestamp.now().isoformat(sep=" ", timespec="seconds")}
    for t in TARGETS:
        try:
            out[t] = train(t, test_days)
        except Exception as e:
            out[t] = {"error": f"{type(e).__name__}: {e}"}
    (OUT / "re_state_forecast.json").write_text(json.dumps(out, indent=2, default=float))
    return out


if __name__ == "__main__":
    r = run_all()
    for t in TARGETS:
        v = r.get(t, {})
        if "error" in v:
            print(f"=== {t}: {v['error']}")
            continue
        print(f"\n=== {t} — MEASURED generation, {v['n_states']} states")
        print(f"  {v['n_train_rows']:,} train rows | test {v['test_from']} + "
              f"{v['test_days']}d | history {v['history_from']} -> {v['history_to']}")
        print(f"  SERVED WAPE {v['served_wape_pct']}%  (model {v['model_wape_pct']}%, "
              f"naive {v['naive_wape_pct']}%)  | beat naive {v['states_beating_naive']}")
        print(f"  {'state':16s} {'served%':>8} {'model%':>8} {'naive%':>8} "
              f"{'what':>11} {'mean MWh/d':>12}")
        for s in v["per_state"]:
            print(f"  {s['name'][:16]:16s} {s['served_wape_pct']:8} "
                  f"{s['model_wape_pct']:8} {s['naive_wape_pct']:8} "
                  f"{s['served']:>11} {s['mean_mwh']:12,.0f}")
    print(f"\nsaved -> {OUT / 're_state_forecast.json'}")
