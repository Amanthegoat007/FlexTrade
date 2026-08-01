"""C&I peak shaving under Delhi ToD tariffs — the C&I persona's workspace.

A commercial/industrial consumer in Delhi pays three things FlexTrade can
attack with a behind-the-meter battery:

  1. Demand charge on billed peak demand (Rs/kVA/month) — shave the peak.
  2. Time-of-Day surcharge on energy drawn in peak hours — shift it.
  3. ToD rebate for off-peak drawal — charge the battery then.

Tariff parameters (DERC schedule for HT industrial supply; VERIFY against
the current DERC tariff order before quoting to a customer — they change
every fiscal year and by category):
  energy charge     Rs 7.75/kWh  (HT industrial, FY25-26 order of magnitude)
  demand charge     Rs 250/kVA/month on billed demand
  ToD peak hours    1400-1700 and 2200-0100, +20% surcharge on energy
  ToD off-peak      0400-1000, -20% rebate
  (DERC ToD applies May-September; surcharge/rebate percentages have been
   20% in recent orders. All configurable in Tariff below.)

The load profile is PLUGGABLE: pass any 96-block meter series. The demo
uses a documented synthetic two-shift factory profile (clearly labelled
"illustrative"), because no public C&I meter feed exists — a pilot
customer's meter CSV drops straight in.

LP: minimise [energy cost with ToD multipliers] + [demand charge on the
month-projected peak] subject to battery power/SoC limits and no export
(net-metering rules for C&I storage differ; conservative default).
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pulp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from optimize.dispatch import BLOCK_H, Bess  # noqa: E402


@dataclass
class Tariff:
    energy_rs_kwh: float = 7.75
    demand_rs_kva_month: float = 250.0
    peak_hours: tuple = ((14, 17), (22, 25))   # 25 = 01:00 next day
    offpeak_hours: tuple = ((4, 10),)
    peak_surcharge: float = 0.20
    offpeak_rebate: float = 0.20
    power_factor: float = 0.95                 # kW -> kVA for billing
    note: str = ("DERC HT-industrial schedule, indicative FY25-26 values — "
                 "verify against the current DERC tariff order")

    def tod_multiplier(self, hour: int) -> float:
        h24 = hour % 24
        for a, b in self.peak_hours:
            if a <= hour < b or a <= h24 + 24 < b:
                return 1.0 + self.peak_surcharge
        for a, b in self.offpeak_hours:
            if a <= h24 < b:
                return 1.0 - self.offpeak_rebate
        return 1.0

    def rate_series(self, index: pd.DatetimeIndex) -> pd.Series:
        """Rs/MWh drawn, per block."""
        mult = np.array([self.tod_multiplier(t.hour) for t in index])
        return pd.Series(self.energy_rs_kwh * 1000.0 * mult, index=index)


def factory_profile(day: pd.Timestamp, peak_mw: float = 5.0) -> pd.Series:
    """Illustrative two-shift factory: base load + day shift + a 3-hour
    afternoon process peak (furnace/compressor/HVAC — lands inside DERC's
    14:00-17:00 ToD peak window, as it does for many heat- and
    cooling-heavy sites) + evening shoulder. Deterministic, documented,
    replaceable by a real meter series — peak shaving economics are
    profile-shaped, so a pilot's real meter data is the real test."""
    idx = pd.date_range(day.normalize(), periods=96, freq="15min")
    h = idx.hour + idx.minute / 60.0
    base = 0.30 * peak_mw
    shift1 = 0.45 * peak_mw * ((h >= 8) & (h < 20))
    process = 0.35 * peak_mw * ((h >= 14) & (h < 17))
    lunch = -0.15 * peak_mw * ((h >= 13) & (h < 14))
    evening = 0.25 * peak_mw * ((h >= 20) & (h < 23))
    ramp = 0.05 * peak_mw * np.sin((h / 24.0) * 2 * np.pi)
    load = np.clip(base + shift1 + process + lunch + evening + ramp,
                   0.1 * peak_mw, None)
    return pd.Series(load, index=idx, name="load_mw")


def bill(load_mw: pd.Series, grid_mw: pd.Series, tariff: Tariff) -> dict:
    """One day's bill: ToD energy + month-projected demand charge share."""
    rate = tariff.rate_series(grid_mw.index)
    energy = float((grid_mw * BLOCK_H * rate).sum())
    peak_kva = float(grid_mw.max()) * 1000.0 / tariff.power_factor
    demand = peak_kva * tariff.demand_rs_kva_month / 30.0  # daily share
    return {"energy_rs": round(energy, 0), "demand_rs": round(demand, 0),
            "total_rs": round(energy + demand, 0),
            "billed_peak_mw": round(float(grid_mw.max()), 3)}


def optimize_peak_shave(load_mw: pd.Series, bess: Bess,
                        tariff: Tariff = Tariff()) -> dict:
    rate = tariff.rate_series(load_mw.index).values
    load = load_mw.values.astype(float)
    n = len(load)
    eta = np.sqrt(bess.round_trip_eff)
    soc0 = bess.soc0_frac * bess.energy_mwh
    soc_min = bess.soc_min_frac * bess.energy_mwh

    prob = pulp.LpProblem("peak_shave", pulp.LpMinimize)
    ch = pulp.LpVariable.dicts("ch", range(n), 0, bess.power_mw)
    dis = pulp.LpVariable.dicts("dis", range(n), 0, bess.power_mw)
    soc = pulp.LpVariable.dicts("soc", range(n + 1), soc_min, bess.energy_mwh)
    grid = pulp.LpVariable.dicts("grid", range(n), 0)  # no export
    peak = pulp.LpVariable("peak_mw", 0)

    # daily share of the monthly demand charge, in Rs per MW of peak
    demand_rs_mw = tariff.demand_rs_kva_month * 1000.0 / tariff.power_factor / 30.0

    prob += (pulp.lpSum(BLOCK_H * rate[t] * grid[t]
                        + BLOCK_H * bess.degradation_rs_mwh * (ch[t] + dis[t])
                        for t in range(n))
             + demand_rs_mw * peak)

    prob += soc[0] == soc0
    for t in range(n):
        prob += grid[t] == load[t] - dis[t] + ch[t]
        prob += peak >= grid[t]
        prob += soc[t + 1] == soc[t] + BLOCK_H * (ch[t] * eta - dis[t] / eta)
    prob += soc[n] >= soc0

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"peak-shave LP: {pulp.LpStatus[prob.status]}")

    out = pd.DataFrame(index=load_mw.index)
    out["load_mw"] = load
    out["grid_mw"] = [grid[t].value() for t in range(n)]
    out["bess_mw"] = out["load_mw"] - out["grid_mw"]  # +ve = discharging
    out["soc_mwh"] = [soc[t + 1].value() for t in range(n)]
    out["tod_rate_rs_mwh"] = rate

    base = bill(load_mw, load_mw, tariff)
    opt = bill(load_mw, out["grid_mw"], tariff)
    deg = float(BLOCK_H * bess.degradation_rs_mwh
                * sum(ch[t].value() + dis[t].value() for t in range(n)))
    saving = base["total_rs"] - opt["total_rs"] - deg
    return {
        "schedule": out,
        "baseline": base, "optimized": opt,
        "degradation_rs": round(deg, 0),
        "saving_rs_day": round(saving, 0),
        "saving_pct": round(100 * saving / base["total_rs"], 1),
        "peak_cut_mw": round(base["billed_peak_mw"] - opt["billed_peak_mw"], 3),
        "tariff_note": tariff.note,
    }


if __name__ == "__main__":
    day = pd.Timestamp.today().normalize()
    load = factory_profile(day, peak_mw=5.0)
    cni_bess = Bess(power_mw=2.0, energy_mwh=4.0, degradation_rs_mwh=843.0)
    r = optimize_peak_shave(load, cni_bess)
    b, o = r["baseline"], r["optimized"]
    print("C&I peak shaving — illustrative 5 MW two-shift factory, "
          "2 MW / 4 MWh behind-the-meter BESS")
    print(f"  ({r['tariff_note']})")
    print(f"  baseline  bill Rs {b['total_rs']:>9,.0f}/day  "
          f"(energy {b['energy_rs']:,.0f} + demand {b['demand_rs']:,.0f}; "
          f"peak {b['billed_peak_mw']:.2f} MW)")
    print(f"  optimized bill Rs {o['total_rs']:>9,.0f}/day  "
          f"(energy {o['energy_rs']:,.0f} + demand {o['demand_rs']:,.0f}; "
          f"peak {o['billed_peak_mw']:.2f} MW)")
    print(f"  peak cut {r['peak_cut_mw']:.2f} MW | degradation Rs "
          f"{r['degradation_rs']:,.0f} | NET SAVING Rs {r['saving_rs_day']:,.0f}/day "
          f"({r['saving_pct']}%)  ~ Rs {r['saving_rs_day'] * 365 / 1e5:,.1f} lakh/yr")
