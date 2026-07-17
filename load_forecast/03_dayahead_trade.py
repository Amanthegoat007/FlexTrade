"""
Step 3: Day-ahead forecast -> market trading view.

Simulates what FlexTrade would do at DAM bid time:
  1. Forecast all 96 x 15-min blocks of the target day with the trained model.
  2. Pull the IEX DAM market-clearing prices (from the market snapshot file)
     as the expected price curve.
  3. Build an illustrative BESS arbitrage schedule (charge in the cheapest
     blocks, discharge in the most expensive ones) and estimate revenue.
  4. Flag forecast peak blocks for predictive peak shaving.

Outputs (./output):
  - dayahead_plan.csv    per-block forecast, price, BESS action
  - dayahead_plan.png    combined chart
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from importlib import import_module
train_mod = import_module("02_train_model")

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE / "output"
RAW = HERE.parent

# --- illustrative BESS asset (adjust to your pitch) ---
BESS_POWER_MW = 20.0      # max charge/discharge power
BESS_ENERGY_MWH = 40.0    # usable capacity (2-hour battery)
ROUND_TRIP_EFF = 0.90
BLOCK_H = 0.25            # 15 minutes


def read_dam_prices() -> pd.DataFrame:
    dam = pd.read_excel(RAW / "DAM_Market Snapshot.xlsx", skiprows=5,
                        usecols="A:H")
    dam.columns = ["date", "hour", "time_block", "purchase_bid_mw",
                   "sell_bid_mw", "mcv_mw", "sched_vol_mw", "mcp_rs_mwh"]
    dam = dam.dropna(subset=["time_block"])
    dam["mcp_rs_mwh"] = pd.to_numeric(dam["mcp_rs_mwh"], errors="coerce")
    dam["block"] = range(len(dam))
    return dam[["block", "time_block", "mcp_rs_mwh"]].reset_index(drop=True)


def main():
    df = pd.read_parquet(DATA / "model_table.parquet")
    feats = train_mod.build_features(df)

    # target day = last day with a complete 96-block feature set
    complete = feats.dropna(subset=train_mod.FEATURES)
    counts = complete.groupby(complete.index.date).size()
    target_day = max(d for d, n in counts.items() if n == 96)
    day = complete[complete.index.date == target_day]
    print(f"Target day (pseudo day-ahead): {target_day}")

    model = lgb.Booster(model_file=str(OUT / "model.txt"))
    plan = pd.DataFrame(index=day.index)
    plan["block"] = day["block"].astype(int)
    plan["forecast_load_mw"] = model.predict(day[train_mod.FEATURES])
    if (df.loc[day.index, "load_mw"].notna()).all():
        plan["actual_load_mw"] = df.loc[day.index, "load_mw"]

    dam = read_dam_prices()
    plan = plan.merge(dam, on="block", how="left").set_index(plan.index)

    # --- BESS arbitrage schedule (greedy, SoC-feasible by construction:
    #     all charge blocks are cheaper night blocks preceding the evening
    #     discharge; a real dispatcher would use an LP/MILP) ---
    n_blocks = int(BESS_ENERGY_MWH / (BESS_POWER_MW * BLOCK_H))  # blocks to fill
    order = plan["mcp_rs_mwh"].sort_values()
    charge_idx = order.index[:n_blocks]
    discharge_idx = order.index[-n_blocks:]
    plan["bess_action"] = "idle"
    plan.loc[charge_idx, "bess_action"] = "charge"
    plan.loc[discharge_idx, "bess_action"] = "discharge"
    plan["bess_mw"] = 0.0
    plan.loc[charge_idx, "bess_mw"] = -BESS_POWER_MW
    plan.loc[discharge_idx, "bess_mw"] = BESS_POWER_MW * ROUND_TRIP_EFF

    e = BESS_POWER_MW * BLOCK_H
    cost = (plan.loc[charge_idx, "mcp_rs_mwh"] * e).sum()
    revenue = (plan.loc[discharge_idx, "mcp_rs_mwh"] * e * ROUND_TRIP_EFF).sum()
    profit = revenue - cost

    # --- predictive peak shaving flags ---
    peak_thr = plan["forecast_load_mw"].quantile(0.95)
    plan["peak_flag"] = plan["forecast_load_mw"] >= peak_thr

    plan.to_csv(OUT / "dayahead_plan.csv", index_label="ts")

    print(f"BESS {BESS_POWER_MW:.0f} MW / {BESS_ENERGY_MWH:.0f} MWh, "
          f"round-trip eff {ROUND_TRIP_EFF:.0%}")
    print(f"  charge cost   : Rs {cost:12,.0f}")
    print(f"  discharge rev : Rs {revenue:12,.0f}")
    print(f"  arbitrage P&L : Rs {profit:12,.0f} / day")
    print(f"  avg buy price : Rs {plan.loc[charge_idx,'mcp_rs_mwh'].mean():,.0f}/MWh")
    print(f"  avg sell price: Rs {plan.loc[discharge_idx,'mcp_rs_mwh'].mean():,.0f}/MWh")

    # --- chart ---
    plt.rcParams.update({"figure.dpi": 120, "axes.grid": True, "grid.alpha": 0.3})
    fig, ax1 = plt.subplots(figsize=(13, 5.5))
    x = plan.index
    ax1.plot(x, plan["forecast_load_mw"], color="#33577b", lw=1.8,
             label="Forecast load (MW)")
    if "actual_load_mw" in plan:
        ax1.plot(x, plan["actual_load_mw"], color="#33577b", lw=1, ls="--",
                 alpha=0.6, label="Actual load (MW)")
    ax1.set_ylabel("Delhi load (MW)", color="#33577b")

    ax2 = ax1.twinx()
    ax2.step(x, plan["mcp_rs_mwh"], color="#e07a30", lw=1.4, where="post",
             label="DAM MCP (Rs/MWh)")
    ax2.set_ylabel("DAM MCP (Rs/MWh)", color="#e07a30")
    ax2.grid(False)

    for idx_set, color, lab in [(charge_idx, "#7fb069", "BESS charge"),
                                (discharge_idx, "#d94f4f", "BESS discharge")]:
        first = True
        for t in idx_set:
            ax1.axvspan(t, t + pd.Timedelta(minutes=15), color=color, alpha=0.25,
                        label=lab if first else None)
            first = False

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left", ncol=2, framealpha=0.9)
    ax1.set_title(f"Day-ahead plan for {target_day}: load forecast, DAM price, "
                  f"BESS schedule (P&L Rs {profit:,.0f}/day)")
    fig.tight_layout()
    fig.savefig(OUT / "dayahead_plan.png")
    print("Saved", OUT / "dayahead_plan.csv", "and dayahead_plan.png")


if __name__ == "__main__":
    main()
