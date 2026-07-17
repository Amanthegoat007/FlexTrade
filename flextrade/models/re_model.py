"""RE generation forecasting + deviation settlement (DSM) module.

Serves the "Renewable Scheduling" market segment from the business model:
RE developers must submit day-ahead generation schedules and pay deviation
settlement penalties when actual output strays outside the tolerance band.

Approach: a physical digital twin of a co-located solar park + wind farm
(PVWatts-style PV model, standard wind power curve) driven by weather.
  - Day-ahead schedule  = twin(live Open-Meteo *forecast* for tomorrow)
  - Actual generation   = twin(analysis/actual weather for the same day)
  - Naive benchmark     = persistence (same block yesterday)
The DSM calculator prices both, so the "penalty saved by FlexTrade
forecasting" is a directly billable number (Forecast-as-a-Service /
revenue-share pitch).

Plant parameters are illustrative and configurable.
"""
import sys
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ingest import store, weather

BLOCK_H = 0.25


@dataclass
class SolarPlant:
    capacity_mw: float = 50.0
    noct_c: float = 45.0          # nominal operating cell temperature
    gamma: float = -0.0035        # power temperature coefficient per °C
    inverter_eff: float = 0.96
    dc_ac_ratio: float = 1.25


@dataclass
class WindFarm:
    capacity_mw: float = 50.0
    cut_in_ms: float = 3.0
    rated_ms: float = 12.0
    cut_out_ms: float = 25.0


def solar_power(ghi: pd.Series, temp_c: pd.Series,
                plant: SolarPlant = SolarPlant()) -> pd.Series:
    cell_t = temp_c + ghi * (plant.noct_c - 20.0) / 800.0
    dc = (plant.capacity_mw * plant.dc_ac_ratio * (ghi / 1000.0)
          * (1 + plant.gamma * (cell_t - 25.0)))
    ac = (dc * plant.inverter_eff).clip(lower=0, upper=plant.capacity_mw)
    return ac.rename("solar_mw")


def wind_power(wind100_kmh: pd.Series, farm: WindFarm = WindFarm()) -> pd.Series:
    v = wind100_kmh / 3.6  # m/s at hub height
    frac = (v**3 - farm.cut_in_ms**3) / (farm.rated_ms**3 - farm.cut_in_ms**3)
    p = farm.capacity_mw * frac.clip(0, 1)
    p[(v < farm.cut_in_ms) | (v > farm.cut_out_ms)] = 0.0
    return p.rename("wind_mw")


def _twin(wx: pd.DataFrame) -> pd.DataFrame:
    wx = wx.resample("15min").interpolate(limit=8)
    out = pd.DataFrame(index=wx.index)
    out["solar_mw"] = solar_power(wx["ghi"], wx["temp_c"])
    out["wind_mw"] = wind_power(wx["wind100_kmh"])
    out["re_total_mw"] = out["solar_mw"] + out["wind_mw"]
    return out


def forecast_day(target: date | None = None) -> pd.DataFrame:
    """Day-ahead RE generation schedule from the live weather forecast."""
    target = target or (date.today() + timedelta(days=1))
    wx, meta = weather.get_re_forecast(days=2)
    gen = _twin(wx)
    day = gen[gen.index.date == target]
    if day.empty:
        raise ValueError(f"no RE weather rows for {target}")
    day.attrs["live"] = meta.get("live", False)
    return day


def actual_day(d: date) -> pd.DataFrame:
    """Twin on analysis (past) weather — the settlement-side generation."""
    wx = store.read("re_weather")
    wx = wx[wx.index.date == d]
    if wx.empty:
        raise ValueError(f"no stored RE weather for {d} (fetch first)")
    return _twin(wx.drop(columns=["kind", "fetched_at"], errors="ignore"))


def dsm_report(schedule_mw: pd.Series, actual_mw: pd.Series,
               capacity_mw: float, band: float = 0.10,
               penalty_rs_mwh: float = 1500.0) -> dict:
    """Deviation settlement: penalty on energy deviating beyond +/-band of
    the *scheduled* output per block (simplified single-slab version of
    the CERC DSM mechanism for RE; band and rate are configurable). A
    floor of 2% of capacity avoids penalizing near-zero night blocks."""
    dev_mw = (actual_mw - schedule_mw).abs()
    tol_mw = (band * schedule_mw).clip(lower=0.02 * capacity_mw)
    penal_mwh = ((dev_mw - tol_mw).clip(lower=0) * BLOCK_H)
    return {
        "energy_scheduled_mwh": float(schedule_mw.sum() * BLOCK_H),
        "energy_actual_mwh": float(actual_mw.sum() * BLOCK_H),
        "mae_mw": float(dev_mw.mean()),
        "blocks_outside_band": int((dev_mw > tol_mw).sum()),
        "penalty_rs": float(penal_mwh.sum() * penalty_rs_mwh),
    }


def fetch_prev_run(past_days: int = 3) -> pd.DataFrame:
    """Open-Meteo Previous Runs API: current analysis alongside what the
    model forecast for the same hours one day earlier (the honest
    day-ahead forecast, `*_previous_day1`)."""
    import requests
    varlist = ["shortwave_radiation", "wind_speed_100m", "temperature_2m"]
    hourly = ",".join(varlist + [v + "_previous_day1" for v in varlist])
    r = requests.get(
        "https://previous-runs-api.open-meteo.com/v1/forecast",
        params=dict(latitude=weather.LAT, longitude=weather.LON, hourly=hourly,
                    past_days=past_days, forecast_days=1,
                    timezone="Asia/Kolkata"),
        timeout=30,
    )
    r.raise_for_status()
    df = pd.DataFrame(r.json()["hourly"])
    df["ts"] = pd.to_datetime(df.pop("time"))
    return df.set_index("ts").sort_index()


def dsm_comparison(d: date | None = None) -> tuple[pd.DataFrame, dict, dict]:
    """FlexTrade day-ahead forecast vs naive persistence for day `d`
    (default: yesterday). Schedule = twin(previous-day NWP run); actual =
    twin(current analysis) — real forecast error, not simulated.
    Returns (per-block df, dsm_flextrade, dsm_naive)."""
    d = d or (date.today() - timedelta(days=1))
    raw = fetch_prev_run(past_days=5)

    def twin_from(ghi_col, wind_col, temp_col):
        wx = raw[[ghi_col, wind_col, temp_col]].copy()
        wx.columns = ["ghi", "wind100_kmh", "temp_c"]
        return _twin(wx)["re_total_mw"]

    actual = twin_from("shortwave_radiation", "wind_speed_100m", "temperature_2m")
    schedule = twin_from("shortwave_radiation_previous_day1",
                         "wind_speed_100m_previous_day1",
                         "temperature_2m_previous_day1")
    naive = actual.shift(96)  # persistence: same block yesterday

    day_mask = actual.index.date == d
    df = pd.DataFrame({
        "schedule_mw": schedule[day_mask],
        "actual_mw": actual[day_mask],
        "naive_mw": naive[day_mask],
    }).dropna()

    cap = SolarPlant().capacity_mw + WindFarm().capacity_mw
    flex = dsm_report(df["schedule_mw"], df["actual_mw"], cap)
    npen = dsm_report(df["naive_mw"], df["actual_mw"], cap)
    return df, flex, npen


if __name__ == "__main__":
    gen = forecast_day()
    print(f"RE forecast for {gen.index[0].date()} (live={gen.attrs['live']}):")
    print(gen.describe().round(1))
    df, flex, naive = dsm_comparison()
    print("\nDSM (yesterday):")
    print("  FlexTrade forecast:", {k: round(v, 1) for k, v in flex.items()})
    print("  naive persistence :", {k: round(v, 1) for k, v in naive.items()})
    print(f"  penalty saved: Rs {naive['penalty_rs'] - flex['penalty_rs']:,.0f}/day")
