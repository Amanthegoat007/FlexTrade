"""
Step 2: Train the day-ahead load forecasting model.

Forecast target: Delhi load (MW) for every 15-min block of day D+1,
predicted at DAM bid time (~noon on day D). All features are restricted
to information available at that moment:
  - load lags >= 48h (same block on D-1 and earlier)
  - rolling statistics ending on D-1
  - calendar features and public holidays for the target day
  - weather for the target day (proxy for a day-ahead weather forecast)

Model: LightGBM gradient boosting, one global model for all 96 blocks.
Split: chronological (train -> validation -> test), no shuffling.

Outputs (in ./output):
  - model.txt                trained LightGBM model
  - test_predictions.csv     block-level actual vs predicted on the test set
  - metrics.txt              MAPE / RMSE / R2 per split
  - plots: forecast_week.png, error_by_hour.png, feature_importance.png,
           scatter.png
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import holidays
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "output"
OUT.mkdir(exist_ok=True)

BLOCKS_PER_DAY = 96  # 15-min blocks


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    f = df.copy()
    idx = f.index

    # --- calendar ---
    f["block"] = idx.hour * 4 + idx.minute // 15
    f["hour"] = idx.hour + idx.minute / 60
    f["dow"] = idx.dayofweek
    f["month"] = idx.month
    f["doy"] = idx.dayofyear
    f["is_weekend"] = (f["dow"] >= 5).astype(int)
    # cyclic encodings so 23:45 sits next to 00:00 and Dec next to Jan
    f["hour_sin"] = np.sin(2 * np.pi * f["hour"] / 24)
    f["hour_cos"] = np.cos(2 * np.pi * f["hour"] / 24)
    f["doy_sin"] = np.sin(2 * np.pi * f["doy"] / 365.25)
    f["doy_cos"] = np.cos(2 * np.pi * f["doy"] / 365.25)

    ind_holidays = holidays.India(years=range(idx.year.min(), idx.year.max() + 1))
    dates = pd.Series(idx.date, index=idx)
    f["is_holiday"] = dates.isin(ind_holidays).astype(int)

    # --- load lags (only >= 48h: known at day-ahead bid time) ---
    lag_blocks = {"lag_2d": 2, "lag_3d": 3, "lag_7d": 7, "lag_14d": 14}
    for name, days in lag_blocks.items():
        f[name] = f["load_mw"].shift(days * BLOCKS_PER_DAY)

    # rolling stats over the 7 days ending 2 days before the target
    base = f["load_mw"].shift(2 * BLOCKS_PER_DAY)
    f["roll7d_mean"] = base.rolling(7 * BLOCKS_PER_DAY).mean()
    f["roll7d_max"] = base.rolling(7 * BLOCKS_PER_DAY).max()
    f["roll7d_min"] = base.rolling(7 * BLOCKS_PER_DAY).min()
    # same-block average over the last 4 same weekdays (captures weekly shape)
    f["sameblock_4w_mean"] = (
        f["load_mw"].shift(7 * BLOCKS_PER_DAY)
        .add(f["load_mw"].shift(14 * BLOCKS_PER_DAY))
        .add(f["load_mw"].shift(21 * BLOCKS_PER_DAY))
        .add(f["load_mw"].shift(28 * BLOCKS_PER_DAY)) / 4
    )

    # --- weather (target-day forecast proxy) ---
    f["temp_sq"] = f["temp_c"] ** 2                      # AC load is nonlinear in temp
    f["cdh"] = np.maximum(f["temp_c"] - 24, 0)           # cooling degree
    f["hdh"] = np.maximum(14 - f["temp_c"], 0)           # heating degree
    f["temp_rh"] = f["temp_c"] * f["rh_pct"] / 100       # humidity discomfort
    f["temp_24h_mean"] = f["temp_c"].rolling(96).mean()  # thermal inertia of the day

    # --- thermal inertia & growth (model-lab winners, adopted 24 Jul) ---
    # AC load depends on how hot it HAS BEEN, not just how hot it is
    f["temp_lag_1d"] = f["temp_c"].shift(BLOCKS_PER_DAY)
    daily_cdh = f["cdh"].groupby(idx.date).transform("mean")
    f["heat_streak_3d"] = daily_cdh.shift(BLOCKS_PER_DAY).rolling(3 * BLOCKS_PER_DAY).mean()
    hour_i = idx.hour
    f["cdh_evening"] = f["cdh"] * (((hour_i >= 17) | (hour_i <= 1)).astype(int))
    # short-run demand growth, encoded explicitly so level drift is learnable
    f["r_2_7"] = f["lag_2d"] / f["lag_7d"]
    f["r_2_14"] = f["lag_2d"] / f["lag_14d"]
    # adaptive same-block baseline (EWM, information through D-2 only)
    f["ewma_sameblock"] = (f["load_mw"].shift(2 * BLOCKS_PER_DAY)
                           .groupby(f["block"] if "block" in f else
                                    (idx.hour * 4 + idx.minute // 15))
                           .transform(lambda s: s.ewm(halflife=7).mean()))

    return f


FEATURES = [
    "block", "hour_sin", "hour_cos", "dow", "month", "doy_sin", "doy_cos",
    "is_weekend", "is_holiday",
    "lag_2d", "lag_3d", "lag_7d", "lag_14d",
    "roll7d_mean", "roll7d_max", "roll7d_min", "sameblock_4w_mean",
    "temp_c", "temp_sq", "cdh", "hdh", "rh_pct", "temp_rh", "rain_mm",
    "cloud_pct", "apparent_temp_c", "temp_24h_mean",
    "temp_lag_1d", "heat_streak_3d", "cdh_evening",
    "r_2_7", "r_2_14", "ewma_sameblock",
]

# recency half-life: down-weights old regimes so the model tracks Delhi's
# load growth (test window is always the most recent data). Chosen in
# models/model_lab.py — 180 d beat unweighted by ~0.45 pp MAPE.
RECENCY_HALF_LIFE_DAYS = 180
ENSEMBLE_SEEDS = (42, 7, 2026)


def mape(y, p):
    return float(np.mean(np.abs((y - p) / y)) * 100)


def main():
    df = pd.read_parquet(DATA / "model_table.parquet")
    f = build_features(df).dropna(subset=FEATURES + ["load_mw"])
    print(f"Feature table: {f.shape[0]:,} rows, {f.index.min()} -> {f.index.max()}")

    # chronological split: last 6 months = test, 6 months before = validation
    test_start = f.index.max() - pd.DateOffset(months=6)
    val_start = test_start - pd.DateOffset(months=6)
    train = f[f.index < val_start]
    val = f[(f.index >= val_start) & (f.index < test_start)]
    test = f[f.index >= test_start]
    print(f"train {len(train):,} | val {len(val):,} | test {len(test):,}")

    # recency weights: half-life chosen in the model lab (see constant above)
    age_days = np.asarray((train.index.max() - train.index).total_seconds()) / 86400
    weights = 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS)

    # small seed ensemble: averages away tree-growth variance (~0.03 pp MAPE)
    models = []
    for seed in ENSEMBLE_SEEDS:
        m = lgb.LGBMRegressor(
            n_estimators=6000,
            learning_rate=0.015,
            num_leaves=255,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_lambda=2.0,
            random_state=seed,
        )
        m.fit(
            train[FEATURES], train["load_mw"], sample_weight=weights,
            eval_set=[(val[FEATURES], val["load_mw"])],
            eval_metric="mape",
            callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)],
        )
        models.append(m)
        print(f"seed {seed}: best iteration {m.best_iteration_}")

    def predict(part):
        return np.mean([m.predict(part[FEATURES]) for m in models], axis=0)

    lines = []
    for name, part in [("train", train), ("val", val), ("test", test)]:
        p = predict(part)
        y = part["load_mw"].values
        rmse = float(np.sqrt(np.mean((y - p) ** 2)))
        r2 = 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
        lines.append(f"{name:5s}  MAPE {mape(y, p):5.2f}%   RMSE {rmse:6.1f} MW   R2 {r2:.4f}")
    report = "\n".join(lines)
    print(report)
    (OUT / "metrics.txt").write_text(report)

    # save every ensemble member + a meta file the live wrapper reads;
    # model.txt stays = first member for backward compatibility
    import json
    names = []
    for seed, m in zip(ENSEMBLE_SEEDS, models):
        n = f"model_s{seed}.txt"
        m.booster_.save_model(str(OUT / n))
        names.append(n)
    models[0].booster_.save_model(str(OUT / "model.txt"))
    (OUT / "model_meta.json").write_text(json.dumps({
        "ensemble": names, "recency_half_life_days": RECENCY_HALF_LIFE_DAYS,
        "recipe": "model-lab L8: +thermal/growth features, recency weights, "
                  "tuned params, 3-seed ensemble"}, indent=2))

    # --- test predictions + plots ---
    pred = pd.DataFrame({
        "actual_mw": test["load_mw"],
        "predicted_mw": predict(test),
    }, index=test.index)
    pred.to_csv(OUT / "test_predictions.csv")

    plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.3})

    # one recent week, actual vs forecast
    week = pred[pred.index >= pred.index.max() - pd.Timedelta(days=7)]
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(week.index, week["actual_mw"], label="Actual", lw=1.2, color="#33577b")
    ax.plot(week.index, week["predicted_mw"], label="Day-ahead forecast", lw=1.2, color="#e07a30")
    ax.set_title("Delhi load — actual vs day-ahead forecast (last test week)")
    ax.set_ylabel("MW"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "forecast_week.png"); plt.close(fig)

    # error profile by hour of day
    err = (pred["predicted_mw"] - pred["actual_mw"]).abs() / pred["actual_mw"] * 100
    by_hour = err.groupby(pred.index.hour).mean()
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(by_hour.index, by_hour.values, color="#33577b")
    ax.set_title("Mean absolute % error by hour of day (test set)")
    ax.set_xlabel("Hour"); ax.set_ylabel("MAPE %")
    fig.tight_layout(); fig.savefig(OUT / "error_by_hour.png"); plt.close(fig)

    # scatter
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(pred["actual_mw"], pred["predicted_mw"], s=2, alpha=0.15, color="#33577b")
    lim = [pred.min().min(), pred.max().max()]
    ax.plot(lim, lim, "r--", lw=1)
    ax.set_xlabel("Actual MW"); ax.set_ylabel("Predicted MW")
    ax.set_title("Test set: predicted vs actual")
    fig.tight_layout(); fig.savefig(OUT / "scatter.png"); plt.close(fig)

    # feature importance
    imp = pd.Series(models[0].feature_importances_, index=FEATURES).sort_values()
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(imp.index, imp.values, color="#33577b")
    ax.set_title("Feature importance (LightGBM splits)")
    fig.tight_layout(); fig.savefig(OUT / "feature_importance.png"); plt.close(fig)

    print("Saved model, predictions and plots to", OUT)


if __name__ == "__main__":
    main()
