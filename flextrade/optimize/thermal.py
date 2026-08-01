"""Thermal derating — what Delhi's heat does to battery revenue.

Grid batteries derate above ~35 C: cell power is HVAC-limited, and the
HVAC itself is parasitic load that peaks exactly when prices do (hot
evenings). Revenue models that ignore this overstate summer earnings.

Model (indicative LFP container assumptions, all configurable, labelled):
  - available power: 100% at <= 35 C, linear down to 80% at >= 45 C
    (container HVAC keeps cells in range but limits sustained C-rate);
  - auxiliary (HVAC) energy: 2% of throughput when ambient > 35 C,
    1% otherwise, bought at the concurrent price.

Applied to the committed plan for a day using stored hourly temperature:
clip dispatch to derated power, charge the aux load at market price, and
report the revenue delta — "what the heat costs today".
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from ingest import store  # noqa: E402
from optimize.dispatch import BLOCK_H, Bess  # noqa: E402

DERATE_START_C = 35.0
DERATE_FULL_C = 45.0
DERATE_FLOOR = 0.80
AUX_FRAC_HOT = 0.02
AUX_FRAC_COOL = 0.01


def derate_factor(temp_c) -> np.ndarray:
    t = np.asarray(temp_c, dtype=float)
    frac = (t - DERATE_START_C) / (DERATE_FULL_C - DERATE_START_C)
    return 1.0 - np.clip(frac, 0.0, 1.0) * (1.0 - DERATE_FLOOR)


def _day_temp(d, index: pd.DatetimeIndex) -> pd.Series:
    w = store.read("weather")
    w = w[~w.index.duplicated(keep="first")]
    t = w[w["kind"] == "actual"]["temp_c"].combine_first(
        w[w["kind"] == "forecast"]["temp_c"])
    return t.resample("15min").interpolate().reindex(index).ffill().bfill()


def heat_cost(day=None, bess: Bess = Bess()) -> dict:
    """Apply derating + aux load to the committed plan for `day`."""
    from datetime import date, timedelta
    day = day or (date.today() + timedelta(days=1))
    plan_f = HERE.parent / "output" / f"plan_{day}.csv"
    if not plan_f.exists():
        return {"error": f"no plan for {day}"}
    plan = pd.read_csv(plan_f, parse_dates=["ts"], index_col="ts")
    if "forecast_mcp" not in plan or "discharge_mw" not in plan:
        return {"error": "plan lacks dispatch columns"}

    temp = _day_temp(day, plan.index)
    f = derate_factor(temp.values)
    p_avail = bess.power_mw * f

    ch = plan["charge_mw"].values
    dis = plan["discharge_mw"].values
    price = plan["forecast_mcp"].values

    ch_d = np.minimum(ch, p_avail)
    dis_d = np.minimum(dis, p_avail)
    clipped_mwh = float(((ch - ch_d) + (dis - dis_d)).sum() * BLOCK_H)

    aux_frac = np.where(temp.values > DERATE_START_C, AUX_FRAC_HOT, AUX_FRAC_COOL)
    aux_mwh = (ch_d + dis_d) * BLOCK_H * aux_frac
    aux_cost = float((aux_mwh * price).sum())

    rev_ideal = float(((dis - ch) * BLOCK_H * price).sum())
    rev_derated = float(((dis_d - ch_d) * BLOCK_H * price).sum()) - aux_cost

    return {
        "day": str(day),
        "temp_max_c": round(float(temp.max()), 1),
        "hours_above_35c": round(float((temp > DERATE_START_C).sum()) / 4, 1),
        "min_derate_factor": round(float(f.min()), 3),
        "clipped_mwh": round(clipped_mwh, 2),
        "aux_mwh": round(float(aux_mwh.sum()), 2),
        "aux_cost_rs": round(aux_cost, 0),
        "revenue_ideal_rs": round(rev_ideal, 0),
        "revenue_thermal_rs": round(rev_derated, 0),
        "heat_cost_rs": round(rev_ideal - rev_derated, 0),
        "assumptions": (f"linear derate 100%->{DERATE_FLOOR:.0%} over "
                        f"{DERATE_START_C:.0f}-{DERATE_FULL_C:.0f} C; aux "
                        f"{AUX_FRAC_HOT:.0%} of throughput when hot — indicative "
                        f"LFP container values, tune per asset datasheet"),
    }


if __name__ == "__main__":
    r = heat_cost()
    if "error" in r:
        print(r["error"])
    else:
        print(f"Thermal impact on plan for {r['day']}:")
        print(f"  peak temp {r['temp_max_c']} C | {r['hours_above_35c']} h above 35 C "
              f"| worst derate x{r['min_derate_factor']}")
        print(f"  clipped {r['clipped_mwh']} MWh | aux {r['aux_mwh']} MWh "
              f"(Rs {r['aux_cost_rs']:,.0f})")
        print(f"  revenue Rs {r['revenue_ideal_rs']:,.0f} -> Rs {r['revenue_thermal_rs']:,.0f}"
              f"  (heat costs Rs {r['heat_cost_rs']:,.0f})")
