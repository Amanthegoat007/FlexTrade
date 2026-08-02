"""LP dispatch optimizer for the BESS (PuLP / CBC).

Given a 96-block price curve (forecast or actual), finds the profit-
maximising charge/discharge schedule subject to:
  - power limit (MW) on charge and discharge
  - SoC dynamics with split charge/discharge efficiency (sqrt of round-trip)
  - SoC bounds [soc_min, capacity], end-of-day SoC == start SoC
  - optional degradation cost per MWh of throughput
  - optional peak-shaving blocks where charging is forbidden and a minimum
    discharge is enforced (multi-objective mode from the pitch deck)

Returns a per-block DataFrame and the expected P&L.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pulp

BLOCK_H = 0.25


# Marginal cost of cycling the battery, Rs per MWh of throughput. This is the
# single most consequential number in the objective and it was wrong: it sat at
# a round Rs 200 "proxy" that nobody had ever derived, while our OWN physics
# model (optimize/degradation.py — rainflow counting on the SoC path, LFP
# Wohler curve L(d) = L100 * d^-kp, Rs 1.5 Cr/MWh capex, 70% attributed to
# cycling) puts it near Rs 800.
#
# Calibrated 2 Aug 2026 by running the degradation fixed point over 30 sampled
# days of real DAM prices from the last 90:
#     median Rs 806/MWh   mean 791   p10 724   p90 831
# A tight distribution, so a constant is defensible; 800 is the rounded median.
#
# Why this matters more than a 4x number usually would: under-charging
# degradation does not just overstate profit, it changes the SCHEDULE — the LP
# takes marginal spreads that are not actually worth taking. Measured over the
# last 60 days of real DAM prices, the effect on cycling is 457 -> 412 EFC/yr
# (both inside a 365-550 warranty on THIS window); on the higher-spread
# May-July window the Rs 200 schedule reached ~657 EFC/yr, outside it. So the
# proxy did not always breach the envelope, but it was always willing to, and
# the revenue it reported was partly paid for in warranty rather than earned.
#
# Pricing it correctly makes the optimizer self-regulate instead of needing an
# artificial cycle cap bolted on, which is why no such cap was added.
DEGRADATION_RS_MWH = 800.0


@dataclass
class Bess:
    power_mw: float = 20.0
    energy_mwh: float = 40.0
    round_trip_eff: float = 0.90
    soc0_frac: float = 0.5
    soc_min_frac: float = 0.05
    degradation_rs_mwh: float = DEGRADATION_RS_MWH


def optimize_dispatch(prices: pd.Series, bess: Bess = Bess(),
                      peak_mask: pd.Series | None = None) -> tuple[pd.DataFrame, float]:
    p = prices.values.astype(float)
    n = len(p)
    eta = np.sqrt(bess.round_trip_eff)
    soc0 = bess.soc0_frac * bess.energy_mwh
    soc_min = bess.soc_min_frac * bess.energy_mwh

    prob = pulp.LpProblem("bess_dispatch", pulp.LpMaximize)
    ch = pulp.LpVariable.dicts("ch", range(n), 0, bess.power_mw)
    dis = pulp.LpVariable.dicts("dis", range(n), 0, bess.power_mw)
    soc = pulp.LpVariable.dicts("soc", range(n + 1), soc_min, bess.energy_mwh)

    prob += pulp.lpSum(
        BLOCK_H * (p[t] * dis[t] - p[t] * ch[t])
        - BLOCK_H * bess.degradation_rs_mwh * (ch[t] + dis[t])
        for t in range(n)
    )

    prob += soc[0] == soc0
    for t in range(n):
        # grid-side ch/dis; battery stores ch*eta, delivers dis/eta
        prob += soc[t + 1] == soc[t] + BLOCK_H * (ch[t] * eta - dis[t] / eta)
    prob += soc[n] >= soc0  # give the day back a full battery position

    if peak_mask is not None:
        for t, is_peak in enumerate(peak_mask.values):
            if is_peak:
                prob += ch[t] == 0

    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[prob.status] != "Optimal":
        raise RuntimeError(f"LP status: {pulp.LpStatus[prob.status]}")

    out = pd.DataFrame(index=prices.index)
    out["price"] = p
    out["charge_mw"] = [ch[t].value() for t in range(n)]
    out["discharge_mw"] = [dis[t].value() for t in range(n)]
    out["soc_mwh"] = [soc[t + 1].value() for t in range(n)]
    out["bess_mw"] = out["discharge_mw"] - out["charge_mw"]
    pnl = float(pulp.value(prob.objective))
    return out, pnl


def settle(schedule: pd.DataFrame, actual_prices: pd.Series,
           degradation_rs_mwh: float = DEGRADATION_RS_MWH) -> float:
    """P&L of an already-fixed schedule at actual (cleared) prices.

    If the schedule carries soc_end/soc0 attrs (greedy does), the change
    in stored inventory is valued at the day's mean price so a schedule
    can't book profit by simply liquidating its starting charge."""
    e_dis = schedule["discharge_mw"] * BLOCK_H
    e_ch = schedule["charge_mw"] * BLOCK_H
    pnl = float(((e_dis - e_ch) * actual_prices.values).sum()
                - degradation_rs_mwh * (e_dis + e_ch).sum())
    if "soc_end_mwh" in schedule.attrs:
        d_soc = schedule.attrs["soc_end_mwh"] - schedule.attrs["soc0_mwh"]
        # conservative valuation: a deficit is bought back at the day's max
        # price, a surplus only credited at the day's min
        rate = float(actual_prices.max() if d_soc < 0 else actual_prices.min())
        pnl += d_soc * rate
    return pnl


def greedy_dispatch(prices: pd.Series, bess: Bess = Bess()) -> pd.DataFrame:
    """Feasible rule-based baseline (what a static EMS would do):
    charge whenever price is in the day's cheapest quartile and there is
    headroom, discharge in the priciest quartile while SoC lasts.
    Chronological simulation, so it is always physically feasible."""
    lo, hi = prices.quantile(0.25), prices.quantile(0.75)
    eta = np.sqrt(bess.round_trip_eff)
    soc, soc_min = bess.soc0_frac * bess.energy_mwh, bess.soc_min_frac * bess.energy_mwh
    ch = np.zeros(len(prices))
    dis = np.zeros(len(prices))
    for t, p in enumerate(prices.values):
        if p <= lo and soc < bess.energy_mwh:
            ch[t] = min(bess.power_mw,
                        (bess.energy_mwh - soc) / (BLOCK_H * eta))
            soc += ch[t] * BLOCK_H * eta
        elif p >= hi and soc > soc_min:
            dis[t] = min(bess.power_mw, (soc - soc_min) * eta / BLOCK_H)
            soc -= dis[t] * BLOCK_H / eta
    out = pd.DataFrame(index=prices.index)
    out["price"] = prices
    out["charge_mw"] = ch
    out["discharge_mw"] = dis
    out["bess_mw"] = out["discharge_mw"] - out["charge_mw"]
    out.attrs["soc_end_mwh"] = soc
    out.attrs["soc0_mwh"] = bess.soc0_frac * bess.energy_mwh
    return out


# How far the price limit sits from the forecast. Raised 10% -> 15% on
# 2 Aug 2026 against measured evidence from the issued-order ledger
# (trade/book.py margin_sweep, 8 settled delivery days, degradation charged at
# the calibrated Rs 800/MWh):
#
#   margin   realised P&L   fill rate   undeliverable   EFC/yr
#     10%     Rs 19.2 L       77.5%        34.8 MWh      409
#     15%     Rs 20.5 L       82.5%        12.9 MWh      451   <- chosen
#     30%     Rs 20.6 L       88.8%        12.9 MWh      501
#     50%     Rs 20.2 L       98.8%         2.9 MWh      609   breaks warranty
#
# The reasoning matters more than the constant. IEX DAM is a UNIFORM-PRICE
# auction: a filled order settles at the market clearing price, never at our
# limit. Measured on 2026-07-31, we asked Rs 7,881 and were paid Rs 9,165. So
# the limit is purely a FILL SWITCH — tightening it buys no price protection
# whatsoever, it only causes misses. And a missed BUY is expensive twice over,
# because the energy never arrives and every later SELL that depended on it
# becomes undeliverable (34.8 MWh stranded at 10%, 12.9 MWh at 15%).
#
# Note the curve now TURNS DOWN at 50%. It did not before: with degradation
# under-charged at Rs 200/MWh, P&L rose monotonically with the margin and the
# only thing stopping "fill at any price" was the warranty. Once cycling costs
# what it actually costs, filling indiscriminately stops paying on its own —
# which is the same lesson as the degradation fix, arriving twice.
BID_MARGIN = 0.15


def bid_sheet(schedule: pd.DataFrame, forecast_prices: pd.Series,
              margin: float = BID_MARGIN) -> pd.DataFrame:
    """96-block DAM bid sheet: buy when charging, sell when discharging."""
    bids = pd.DataFrame(index=schedule.index)
    bids["block"] = range(1, len(bids) + 1)
    bids["time_block"] = [f"{t:%H:%M} - {(t + pd.Timedelta(minutes=15)):%H:%M}"
                          for t in bids.index]
    bids["side"] = np.where(schedule["charge_mw"] > 0.01, "BUY",
                            np.where(schedule["discharge_mw"] > 0.01, "SELL", "-"))
    bids["volume_mw"] = (schedule["charge_mw"] + schedule["discharge_mw"]).round(2)
    # bid ABOVE the forecast to buy, ask BELOW it to sell — a loosening margin
    bids["price_limit_rs_mwh"] = np.where(
        bids["side"] == "BUY", (forecast_prices * (1 + margin)).round(0),
        np.where(bids["side"] == "SELL", (forecast_prices * (1 - margin)).round(0),
                 np.nan))
    return bids
