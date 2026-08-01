"""Deviation Settlement Mechanism (DSM) engine — versioned CERC regulation profiles.

This module is the FlexTrade DSM Module described in
`FlexTrade_DSM_Feature.pdf`: a configurable rule engine (not a fixed
calculator), because the underlying regulation keeps changing. It
implements the two most recent CERC frameworks as separate, independently
citable profiles, plus the specific amendments known to be in force or
scheduled as of this build:

  CERC_2022  — CERC (DSM and Related Matters) Regulations, 2022
               (in force 5 Dec 2022, superseded 2024). Volume-band
               multipliers, NOT frequency-linked (the 2022 regulation
               explicitly de-linked deviation charges from grid
               frequency — a fact confirmed by CERC's own summary of its
               regulatory history). This is the mechanism whose exact
               bands are independently corroborated across multiple
               industry sources (Mercom India, SolarQuarter) and are the
               ones reproduced in Table 2 of the DSM feature spec.

  CERC_2024  — CERC (DSM and Related Matters) Regulations, 2024
               (notified 5 Aug 2024, in force since Sep 2024, amended
               Dec 2024 and again by a draft Third Amendment published
               26 May 2026). Introduces the "Normal Rate" (NR) concept:
               deviation is priced off actual market prices rather than
               a flat rate.

Known, dated amendments layered on top of the 2024 base (all with an
effective date, applied automatically by settlement date so a backtest
over 2025 data and a live run in mid-2026 each get the rule that
actually applied):

  * 2026-04-01 — solar/hybrid tolerance band tightens 10% -> 5%; wind
    tightens 15% -> 10%. New wind/solar projects commissioned on or
    after this date are treated as general sellers, not WS sellers.
  * 2027-04-01 (FY2027-28) — first step of the X-factor glide path
    begins: the WS deviation denominator blends available capacity and
    scheduled generation (X * capacity + (1-X) * schedule), with X
    declining from 100% toward 0% by April 2031. Not yet active as of
    this build (CERC confirmed FY27, i.e. through Mar 2027, unchanged).

WHAT IS NOT INDEPENDENTLY VERIFIED
-----------------------------------------------------------------------
The frequency-linked rate curve in the previous version of this module
came from a CERC stakeholder consultation *explanatory memorandum* for
the 2024 draft — commentary proposing changes, not the gazetted
regulation text itself. No corroborating source confirms a frequency
multiplier survives into the notified 2024 regulation; several sources
say the opposite (frequency-linkage was removed in 2022 and nothing
found says it was reintroduced). It is kept here ONLY as an explicitly
opt-in variant (`freq_linked=True` on the CERC_2024 profile) for
scenario exploration, defaulting OFF, and must not be treated as
confirmed law.

The Third Amendment (May 2026 draft, proposed daily-average NR instead
of block-wise) is similarly implemented as an opt-in
(`nr_aggregation="daily"`), because as of this build its consultation
period had not been confirmed closed.

>>> This module is decision support. Before it prices a single rupee of
>>> real settlement or backs a patent claim, every band, date and rate
>>> below must be checked against the CERC gazette notification text
>>> directly (cercind.gov.in) by counsel, not against secondary
>>> reporting. That is also literally what Section 6 of the DSM feature
>>> spec's own risk table requires ("regulatory version in flux ...
>>> build as a configurable rule-set").
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

BLOCK_H = 0.25


# =========================================================================
# CERC (DSM and Related Matters) Regulations, 2022 — volume-band profile
# =========================================================================
# Sources cross-checked: Mercom India ("CERC Issues DSM Regulations,
# Discourages Over Injection of Wind and Solar Power"), SolarQuarter
# ("No Settlement For Wind and Solar Power Over Injection"), and the DSM
# feature spec's own Table 2. All three agree on these numbers.

@dataclass(frozen=True)
class Band2022:
    lo_pct: float   # inclusive lower bound of |deviation| %, of relevant base
    hi_pct: float | None  # exclusive upper bound; None = unbounded
    rate_frac: float      # charge/credit as a fraction of Normal Rate for
                          # the portion of deviation inside this band


# Wind/solar seller, OVER-injection (actual > schedule)
WS_OVER_2022 = (
    Band2022(0.0, 5.0, 1.00),   # up to 5%: paid at full NR
    Band2022(5.0, 10.0, 0.90),  # 5-10%: tariff on the excess cut 10% -> 90% NR
    Band2022(10.0, None, 0.00),  # beyond 10%: not compensated
)
# Wind/solar seller, UNDER-injection (actual < schedule)
WS_UNDER_2022 = (
    Band2022(0.0, 10.0, 0.00),   # up to 10%: no charge
    Band2022(10.0, None, 0.10),  # beyond 10%: pay 10% of NR on the excess
)
# General seller (non-RE generator), either direction
GEN_SELLER_2022 = (
    Band2022(0.0, 2.0, 0.00),
    Band2022(2.0, 10.0, 1.20),
    Band2022(10.0, None, 1.50),
)
# Buyer / drawee, over-drawal
BUYER_2022 = (
    Band2022(0.0, 10.0, 1.00),
    Band2022(10.0, 15.0, 1.20),
    Band2022(15.0, None, 1.50),
)

# WS tolerance tightening, CERC notification effective 2026-04-01
WS_BAND_PRE_2026_04 = {"solar": 10.0, "hybrid": 10.0, "wind": 15.0}
WS_BAND_POST_2026_04 = {"solar": 5.0, "hybrid": 5.0, "wind": 10.0}
WS_BAND_CUTOVER = date(2026, 4, 1)


def _ws_band_pct(technology: str, settlement_date: date) -> float:
    table = (WS_BAND_POST_2026_04 if settlement_date >= WS_BAND_CUTOVER
             else WS_BAND_PRE_2026_04)
    return table.get(technology, 10.0)


def _apply_bands(dev_pct: float, bands: tuple[Band2022, ...]) -> float:
    """Blended rate (fraction of NR) for a given |deviation| %, by
    integrating across whichever bands it spans -- e.g. an 8% over-
    injection is 5% at 100% NR plus 3% at 90% NR, not one flat rate."""
    if dev_pct <= 0:
        return 0.0
    charged_pct, weighted = 0.0, 0.0
    for b in bands:
        hi = b.hi_pct if b.hi_pct is not None else dev_pct
        span = max(0.0, min(dev_pct, hi) - b.lo_pct)
        if span <= 0:
            continue
        weighted += span * b.rate_frac
        charged_pct += span
        if b.hi_pct is not None and dev_pct <= b.hi_pct:
            break
    return weighted / dev_pct if dev_pct > 0 else 0.0


def settle_2022(actual_mw: pd.Series, scheduled_mw: pd.Series,
                dam_price: pd.Series, rtm_price: pd.Series,
                technology: str = "solar", seller: str = "ws",
                settlement_date: date | None = None) -> pd.DataFrame:
    """Block-wise settlement under the CERC 2022 volume-band mechanism.

    technology: accepted for interface symmetry with settle_2024 and
    recorded on the result, but the 2022 bands do not differentiate by
    technology (that split was introduced later, in the 2024 lineage).
    seller: "ws" (wind/solar generator) | "gen" (general seller /
    standalone ESS) | "buyer" (drawee entity).
    """
    idx = actual_mw.index.intersection(scheduled_mw.index) \
        .intersection(dam_price.index).intersection(rtm_price.index)
    df = pd.DataFrame({
        "actual_mw": actual_mw.loc[idx], "scheduled_mw": scheduled_mw.loc[idx],
        "dam": dam_price.loc[idx], "rtm": rtm_price.loc[idx],
    })
    # 2022 mechanism prices deviation off time-block DAM/RTM directly
    # (no separate ancillary third -- that is a 2024-only concept)
    df["normal_rate"] = (df["dam"] + df["rtm"]) / 2.0
    df["deviation_mw"] = df["actual_mw"] - df["scheduled_mw"]

    sdate = settlement_date or (idx[0].date() if len(idx) else date.today())
    if seller == "ws":
        # Fixed, original 2022 bands (5% / 10% breakpoints). This profile
        # is a frozen historical snapshot of an already-superseded
        # regulation: it predates and is unaffected by the 2026-04-01
        # tolerance tightening and X-factor glide path, both of which are
        # amendments to the CURRENTLY-IN-FORCE 2024 lineage and are
        # modelled only in settle_2024 / _ws_band_pct / x_factor. Do not
        # apply either here -- that was a bug in an earlier version of
        # this file, caught by the module's own test in dsm_selftest.py.
        rate_frac, charge_sign = [], []
        for dev, sch in zip(df["deviation_mw"], df["scheduled_mw"]):
            b = max(abs(sch), 1e-6)
            pct = abs(dev) / b * 100
            if dev >= 0:
                rate_frac.append(_apply_bands(pct, WS_OVER_2022))
                charge_sign.append(-1)  # seller is PAID for over-injection
            else:
                rate_frac.append(_apply_bands(pct, WS_UNDER_2022))
                charge_sign.append(1)   # seller PAYS for under-injection
        df["rate_frac"] = rate_frac
        df["_sign"] = charge_sign
        df["tolerance_pct"] = 10.0  # first band edge common to both directions
    elif seller == "buyer":
        base = df["scheduled_mw"].abs().replace(0, np.nan)
        pct = (df["deviation_mw"].abs() / base * 100).fillna(0)
        df["rate_frac"] = [_apply_bands(p, BUYER_2022) for p in pct]
        df["_sign"] = np.where(df["deviation_mw"] > 0, 1, -1)  # over-draw pays
        df["tolerance_pct"] = 10.0
    else:  # general seller / standalone ESS
        base = df["scheduled_mw"].abs().replace(0, np.nan)
        pct = (df["deviation_mw"].abs() / base * 100).fillna(0)
        df["rate_frac"] = [_apply_bands(p, GEN_SELLER_2022) for p in pct]
        df["_sign"] = np.where(df["deviation_mw"] < 0, 1, -1)  # under-inject pays
        df["tolerance_pct"] = 2.0

    df["charge_rs"] = (df["_sign"] * df["rate_frac"] * df["normal_rate"]
                       * df["deviation_mw"].abs() * BLOCK_H)
    df["outside_band"] = df["rate_frac"] > 0
    df.attrs["profile"] = "CERC_2022"
    df.attrs["seller"] = seller
    df.attrs["technology"] = technology
    return df.drop(columns="_sign")


# =========================================================================
# CERC (DSM and Related Matters) Regulations, 2024 — Normal Rate profile
# =========================================================================

RATE_CURVE_UNCONFIRMED = {  # see module docstring: opt-in only, not verified
    49.88: 115, 49.89: 115, 49.90: 115, 49.91: 114, 49.92: 112, 49.93: 111,
    49.94: 109, 49.95: 108, 49.96: 106, 49.97: 105, 49.98: 103, 49.99: 102,
    50.00: 100, 50.01: 90, 50.02: 80, 50.03: 70, 50.04: 60, 50.05: 50,
    50.06: 0, 50.07: 0, 50.08: 0, 50.09: 0, 50.10: 0,
    50.11: -10, 50.12: -10, 50.13: -10,
}
_FREQS = np.array(sorted(RATE_CURVE_UNCONFIRMED))
_RATES = np.array([RATE_CURVE_UNCONFIRMED[f] for f in _FREQS])

GEN_LIMIT_PCT = 0.10
GEN_LIMIT_MW = 100.0

# X-factor glide path for the WS deviation denominator (available capacity
# vs scheduled blend), by fiscal-year start (1 April). X=1.0 = pure
# available-capacity (today's rule); X=0.0 = pure schedule (2031 endpoint).
X_FACTOR_SCHEDULE = {
    # fiscal year starting  solar/hybrid X   wind X
    date(2026, 4, 1): (1.00, 1.00),   # confirmed unchanged through FY27
    date(2027, 4, 1): (0.90, 0.95),
    date(2028, 4, 1): (0.75, 0.85),
    date(2029, 4, 1): (0.55, 0.65),
    date(2030, 4, 1): (0.30, 0.35),
    date(2031, 4, 1): (0.00, 0.00),
}


def x_factor(technology: str, settlement_date: date) -> float:
    applicable = [d for d in X_FACTOR_SCHEDULE if d <= settlement_date]
    key = max(applicable) if applicable else min(X_FACTOR_SCHEDULE)
    solar_x, wind_x = X_FACTOR_SCHEDULE[key]
    return wind_x if technology == "wind" else solar_x


def rate_pct(frequency_hz: float | pd.Series):
    """Charge as a % of Normal Rate for a frequency — UNCONFIRMED variant,
    see module docstring. Only used if freq_linked=True is passed."""
    f = np.clip(np.asarray(frequency_hz, dtype=float), _FREQS[0], _FREQS[-1])
    idx = np.searchsorted(_FREQS, np.round(f, 2), side="right") - 1
    out = _RATES[np.clip(idx, 0, len(_RATES) - 1)]
    if isinstance(frequency_hz, pd.Series):
        return pd.Series(out, index=frequency_hz.index, name="rate_pct")
    return float(out)


def normal_rate(dam_price: pd.Series, rtm_price: pd.Series,
                ancillary_price: pd.Series | None = None,
                nr_aggregation: str = "block") -> pd.DataFrame:
    """Reg 14 Normal Rate, Rs/MWh, per block (default) or Third-Amendment
    daily-average (opt-in, unconfirmed effective status — see docstring).
    """
    df = pd.DataFrame({"dam": dam_price, "rtm": rtm_price}).dropna()
    proxied = ancillary_price is None
    df["ancillary"] = df["rtm"] if proxied else ancillary_price.reindex(df.index)
    df["ancillary"] = df["ancillary"].fillna(df["rtm"])
    if nr_aggregation == "daily":
        daily = df.groupby(df.index.date)[["dam", "rtm", "ancillary"]].transform("mean")
        df["normal_rate"] = daily.mean(axis=1)
    else:
        df["normal_rate"] = (df["dam"] + df["rtm"] + df["ancillary"]) / 3.0
    df.attrs["ancillary_proxied"] = proxied
    df.attrs["nr_aggregation"] = nr_aggregation
    return df


def ws_deviation_pct(actual_mw: pd.Series, scheduled_mw: pd.Series,
                     available_capacity_mw: float, technology: str = "solar",
                     settlement_date: date | None = None) -> pd.Series:
    """Reg 6(2)-style deviation %, with the X-factor blended denominator
    applied by settlement date (X=1.0 today -> pure available capacity,
    identical to the pre-2027 rule)."""
    sdate = settlement_date or date.today()
    x = x_factor(technology, sdate)
    denom = x * available_capacity_mw + (1 - x) * scheduled_mw.abs()
    denom = denom.replace(0, np.nan) if hasattr(denom, "replace") else max(denom, 1e-6)
    return 100.0 * (actual_mw - scheduled_mw) / denom


def settle_2024(actual_mw: pd.Series, scheduled_mw: pd.Series, frequency_hz,
                dam_price: pd.Series, rtm_price: pd.Series,
                available_capacity_mw: float, seller: str = "ws",
                technology: str = "solar", ancillary_price: pd.Series | None = None,
                freq_linked: bool = False, nr_aggregation: str = "block",
                settlement_date: date | None = None) -> pd.DataFrame:
    """Block-by-block settlement under the CERC 2024 Normal Rate mechanism.

    freq_linked=False (default): rate = 100% of NR for chargeable
    deviation (the confirmed, source-corroborated behaviour). Set True
    only to explore the unconfirmed frequency-curve variant.
    """
    nr = normal_rate(dam_price, rtm_price, ancillary_price, nr_aggregation)
    idx = nr.index.intersection(actual_mw.index).intersection(scheduled_mw.index)
    df = nr.loc[idx].copy()
    df["actual_mw"] = actual_mw.loc[idx]
    df["scheduled_mw"] = scheduled_mw.loc[idx]
    sdate = settlement_date or (idx[0].date() if len(idx) else date.today())

    if freq_linked:
        if isinstance(frequency_hz, pd.Series):
            df["frequency_hz"] = frequency_hz.reindex(idx).ffill().bfill()
        else:
            df["frequency_hz"] = float(frequency_hz)
        df["rate_pct"] = rate_pct(df["frequency_hz"])
    else:
        df["frequency_hz"] = np.nan
        df["rate_pct"] = 100.0
    df["rate_rs_mwh"] = df["normal_rate"] * df["rate_pct"] / 100.0

    dev_mw = df["actual_mw"] - df["scheduled_mw"]
    df["deviation_mw"] = dev_mw

    if seller == "ws":
        df["deviation_pct"] = ws_deviation_pct(df["actual_mw"], df["scheduled_mw"],
                                               available_capacity_mw, technology, sdate)
        band_pct = _ws_band_pct(technology, sdate)
        x = x_factor(technology, sdate)
        denom = x * available_capacity_mw + (1 - x) * df["scheduled_mw"].abs()
        tol_mw = band_pct / 100.0 * denom
    else:
        denom = df["scheduled_mw"].abs().replace(0, np.nan)
        df["deviation_pct"] = (dev_mw / denom * 100).fillna(0)
        tol_mw = np.minimum(GEN_LIMIT_PCT * df["scheduled_mw"].abs(), GEN_LIMIT_MW)

    df["tolerance_mw"] = tol_mw
    charge_mw = (dev_mw.abs() - tol_mw).clip(lower=0) * np.sign(dev_mw)
    df["chargeable_mwh"] = charge_mw * BLOCK_H
    df["outside_band"] = dev_mw.abs() > tol_mw
    df["charge_rs"] = -df["chargeable_mwh"] * df["rate_rs_mwh"]
    df.attrs["ancillary_proxied"] = nr.attrs["ancillary_proxied"]
    df.attrs["seller"] = seller
    df.attrs["technology"] = technology
    df.attrs["freq_linked"] = freq_linked
    df.attrs["nr_aggregation"] = nr_aggregation
    df.attrs["profile"] = "CERC_2024"
    return df


# =========================================================================
# Unified entry point
# =========================================================================

def settle(profile: str, **kwargs) -> pd.DataFrame:
    """Dispatch to settle_2022 or settle_2024. `profile` in
    {"CERC_2022", "CERC_2024"}. See each function's signature for the
    kwargs it accepts."""
    if profile == "CERC_2022":
        keys = {"actual_mw", "scheduled_mw", "dam_price", "rtm_price",
                "technology", "seller", "settlement_date"}
        return settle_2022(**{k: v for k, v in kwargs.items() if k in keys})
    if profile == "CERC_2024":
        keys = {"actual_mw", "scheduled_mw", "frequency_hz", "dam_price",
                "rtm_price", "available_capacity_mw", "seller", "technology",
                "ancillary_price", "freq_linked", "nr_aggregation",
                "settlement_date"}
        return settle_2024(**{k: v for k, v in kwargs.items() if k in keys})
    raise ValueError(f"unknown profile {profile!r}; use CERC_2022 or CERC_2024")


def summarize(settled: pd.DataFrame) -> dict:
    payable = settled["charge_rs"].clip(lower=0).sum()
    receivable = (-settled["charge_rs"].clip(upper=0)).sum()
    out = {
        "profile": settled.attrs.get("profile", "unknown"),
        "blocks": int(len(settled)),
        "blocks_outside_band": int(settled["outside_band"].sum()),
        "mae_mw": float(settled["deviation_mw"].abs().mean()),
        "mean_normal_rate_rs_mwh": float(settled["normal_rate"].mean()),
        "charge_payable_rs": float(payable),
        "credit_receivable_rs": float(receivable),
        "net_dsm_rs": float(settled["charge_rs"].sum()),
    }
    if "rate_frac" in settled:
        out["mean_rate_pct"] = float(settled["rate_frac"].mean() * 100)
    if "rate_pct" in settled:
        out["mean_rate_pct"] = float(settled["rate_pct"].mean())
    if "ancillary_proxied" in settled.attrs:
        out["ancillary_proxied"] = bool(settled.attrs["ancillary_proxied"])
    return out
