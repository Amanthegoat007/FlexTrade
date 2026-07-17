# FlexTrade — Day-Ahead Load Forecasting & BESS Trading Demo

Load prediction pipeline for the Esyasoft FlexTrade hackathon submission.
Forecasts Delhi load for every 15-min block of the next day (aligned to IEX
DAM time blocks) and turns the forecast + market prices into a BESS
charge/discharge plan.

## Data used (parent folder)

| File | Content |
|---|---|
| `sldc_data.csv.xlsx` | Delhi SLDC 5-min load by DISCOM, Jun 2021 – Jun 2026 (**target**) |
| `Load Forecasting dataset C&I Consumers.csv` | Hourly Delhi weather (Open-Meteo, UTC): temp, humidity, rain, cloud (**features**) |
| `DAM_Market Snapshot.xlsx` | IEX DAM 15-min market-clearing prices (price curve for the trading demo) |

## Pipeline

```
python 01_prepare_data.py    # xlsx -> parquet, clean load, merge weather (once, ~3 min)
python 02_train_model.py     # train LightGBM, evaluate, plots       (~1 min)
python 03_dayahead_trade.py  # day-ahead forecast + BESS trade plan  (seconds)
```

Requires: `pandas lightgbm scikit-learn holidays pyarrow matplotlib openpyxl`

## Modelling approach

- **Target:** Delhi total load (MW), 15-min blocks (96/day), cleaned of
  telemetry drops and spikes.
- **Bid-time validity:** IEX DAM bids close ~noon on day D for delivery on
  D+1, so the model only uses load lags ≥ 48 h (same block on D-1, D-2, D-7,
  D-14, weekly rolling stats). No leakage from the delivery day.
- **Weather:** target-day weather is used as a proxy for a day-ahead weather
  forecast (temp², cooling/heating degrees, temp×humidity discomfort).
- **Calendar:** block-of-day, day-of-week, cyclic hour/season encodings,
  Indian public holidays.
- **Model:** LightGBM gradient boosting, chronological train/val/test split
  (last 6 months held out as test).

## Results (test = Dec 2025 – Jun 2026, never seen in training)

| Split | MAPE | RMSE | R² |
|---|---|---|---|
| Train | 2.19 % | 112 MW | 0.993 |
| Validation | 4.70 % | 338 MW | 0.931 |
| **Test** | **4.98 %** | **262 MW** | **0.957** |

## Outputs (`./output`)

- `model.txt` — trained model
- `test_predictions.csv`, `metrics.txt`
- `forecast_week.png` — actual vs forecast, last test week
- `error_by_hour.png`, `scatter.png`, `feature_importance.png`
- `dayahead_plan.csv` / `dayahead_plan.png` — per-block forecast + DAM MCP +
  BESS charge/discharge schedule (illustrative 20 MW / 40 MWh asset;
  greedy price-sorted dispatch — swap for LP/MILP in the real product)

## Demo story for the pitch

1. At noon, FlexTrade forecasts tomorrow's 96 load blocks (≈5 % MAPE).
2. It overlays the DAM price curve: cheap solar midday, ₹10,000/MWh cap
   in the evening peak.
3. It schedules the BESS: charge midday (~₹1,100/MWh), discharge into the
   evening peak → **~₹2.7 lakh/day arbitrage P&L** for a 20 MW/40 MWh asset.
4. Peak blocks are flagged ahead of time → *predictive* (not reactive)
   peak shaving.
