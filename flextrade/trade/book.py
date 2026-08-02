"""The trading book — orders, fills, positions and realised P&L.

Everything else in this repo forecasts or optimises. This module is the part
that makes the name honest: it takes the bid sheets we ACTUALLY ISSUED before
each delivery day, clears them against the prices that actually settled, and
reports what the book really earned.

That is a stricter test than the backtest, and it is different in kind. A
backtest re-derives a schedule with hindsight over the whole window; this
replays orders that were written to disk before the gate closed and cannot be
revised. If the two ever disagree, this one is right.

Three things it models that a bid generator does not:

  1. AUCTION CLEARING, NOT WISHFUL FILLING
     IEX DAM is a uniform-price double auction. A SELL limit of L clears if and
     only if the market clears at or above L — and when it clears you are paid
     the MARKET price, not your limit. Symmetrically for BUY. Assuming you get
     your limit price is the single most common way a paper P&L flatters
     itself; here the limit only decides IF you trade, never at what price.

  2. UNFILLED LEGS BREAK THE PHYSICS
     The optimiser's schedule is a chain: charge cheap now so you can discharge
     expensive later. If a BUY does not clear, the energy never arrives, and
     every later SELL that depended on it is undeliverable. A book that settles
     each block independently would happily report revenue on energy the
     battery never had. So the SoC is walked forward over the FILLED blocks
     only, and any discharge the state of charge cannot support is clipped —
     the shortfall is reported as `undeliverable_mwh` rather than earned.

  3. CYCLES ARE A BUDGET
     Throughput is counted in full-cycle equivalents so the book can be checked
     against a warranty envelope, not just a P&L target.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from ingest import store  # noqa: E402
from optimize.dispatch import BLOCK_H, Bess  # noqa: E402

OUT = HERE.parent / "output"

MARKETS = {"dam": "dam_price", "rtm": "rtm_price", "gdam": "gdam_price"}


# --------------------------------------------------------------- clearing ---

def clear(bids: pd.DataFrame, mcp: pd.Series) -> pd.DataFrame:
    """Apply uniform-price auction rules to an issued bid sheet.

    Returns the sheet with `mcp_rs_mwh`, `filled`, `fill_reason` and the
    cash each block would generate IF the physics allows it (see `replay`).
    """
    b = bids.copy()
    b["mcp_rs_mwh"] = mcp.reindex(b.index).astype(float)

    is_sell = b["side"].eq("SELL")
    is_buy = b["side"].eq("BUY")
    lim = b["price_limit_rs_mwh"].astype(float)
    px = b["mcp_rs_mwh"]

    # a sell needs the market at or above its ask; a buy at or below its bid
    b["filled"] = np.where(is_sell, px >= lim,
                           np.where(is_buy, px <= lim, False))
    b.loc[px.isna(), "filled"] = False

    b["fill_reason"] = np.where(
        px.isna(), "no settled price",
        np.where(~(is_sell | is_buy), "no order",
                 np.where(b["filled"], "cleared",
                          np.where(is_sell, "ask above market",
                                   "bid below market"))))
    return b


# ------------------------------------------------------------- settlement ---

@dataclass
class DayResult:
    day: date
    market: str
    rows: pd.DataFrame
    summary: dict


def replay(day: date, bess: Bess = Bess(), market: str = "dam") -> DayResult | None:
    """Settle one delivery day's issued orders against what actually cleared."""
    sheet = OUT / f"bid_sheet_{day}.csv"
    plan_f = OUT / f"plan_{day}.csv"
    if not sheet.exists():
        return None

    bids = pd.read_csv(sheet)
    plan = pd.read_csv(plan_f, parse_dates=["ts"], index_col="ts") \
        if plan_f.exists() else None
    if plan is None or not len(plan):
        return None
    bids.index = plan.index[: len(bids)]

    px = store.read(MARKETS[market])
    px = px[px.index.normalize() == pd.Timestamp(day)]["mcp_rs_mwh"]
    if not len(px):
        return None

    b = clear(bids, px)
    if b["mcp_rs_mwh"].isna().all():
        return None

    # ---- walk the battery forward over FILLED blocks only ----------------
    eta = np.sqrt(bess.round_trip_eff)
    soc = bess.soc0_frac * bess.energy_mwh
    soc_min = bess.soc_min_frac * bess.energy_mwh
    cap = bess.energy_mwh

    charged, discharged, undeliverable, cash = [], [], [], []
    soc_path = []
    for _, r in b.iterrows():
        ch = dis = 0.0
        want = float(r["volume_mw"] or 0.0)
        if r["filled"] and r["side"] == "BUY":
            # cannot charge past full
            room = max(cap - soc, 0.0) / (eta * BLOCK_H)
            ch = min(want, room)
        elif r["filled"] and r["side"] == "SELL":
            # cannot discharge energy that is not in the battery
            avail = max(soc - soc_min, 0.0) * eta / BLOCK_H
            dis = min(want, avail)
        short = (want - dis) if (r["filled"] and r["side"] == "SELL") else 0.0

        soc += BLOCK_H * (ch * eta - dis / eta)
        soc = float(np.clip(soc, 0.0, cap))
        price = float(r["mcp_rs_mwh"]) if pd.notna(r["mcp_rs_mwh"]) else 0.0
        cash.append(BLOCK_H * price * (dis - ch)
                    - BLOCK_H * bess.degradation_rs_mwh * (ch + dis))
        charged.append(ch)
        discharged.append(dis)
        undeliverable.append(short * BLOCK_H)
        soc_path.append(soc)

    b["filled_charge_mw"] = np.round(charged, 3)
    b["filled_discharge_mw"] = np.round(discharged, 3)
    b["undeliverable_mwh"] = np.round(undeliverable, 3)
    b["soc_mwh"] = np.round(soc_path, 2)
    b["cash_rs"] = np.round(cash, 2)

    throughput = float((b["filled_charge_mw"] + b["filled_discharge_mw"]).sum() * BLOCK_H)
    efc = throughput / (2 * bess.energy_mwh) if bess.energy_mwh else 0.0

    ordered = int((b["side"] != "-").sum())
    filled = int(b["filled"].sum())
    # Expected must be NET of the same degradation charge the realised side
    # carries, or "slippage" measures an accounting difference instead of an
    # execution shortfall. (It briefly did: after the degradation rate was
    # recalibrated, realised went net of Rs 800/MWh while expected stayed
    # gross, and slippage read -31.7% for no trading reason at all.)
    expected = None
    if plan is not None and {"charge_mw", "discharge_mw", "forecast_mcp"} <= set(plan.columns):
        gross = float(((plan["discharge_mw"] - plan["charge_mw"])
                       * BLOCK_H * plan["forecast_mcp"]).sum())
        planned_throughput = float(
            (plan["charge_mw"] + plan["discharge_mw"]).sum() * BLOCK_H)
        expected = gross - planned_throughput * bess.degradation_rs_mwh

    realised = float(b["cash_rs"].sum())
    # what a perfect executor would have made on the same day and same asset,
    # i.e. the value of the day itself rather than of our orders
    # DAM is financially firm the moment it clears, but physical delivery of
    # today's later blocks has not happened yet — so today is marked rather
    # than quietly counted as a settled result.
    complete = day < date.today()
    summary = {
        "day": str(day),
        "complete": complete,
        "market": market.upper(),
        "orders": ordered,
        "filled": filled,
        "fill_rate_pct": round(100 * filled / ordered, 1) if ordered else None,
        "unfilled_sell": int(((b["side"] == "SELL") & ~b["filled"]).sum()),
        "unfilled_buy": int(((b["side"] == "BUY") & ~b["filled"]).sum()),
        "sold_mwh": round(float(b["filled_discharge_mw"].sum() * BLOCK_H), 2),
        "bought_mwh": round(float(b["filled_charge_mw"].sum() * BLOCK_H), 2),
        "undeliverable_mwh": round(float(b["undeliverable_mwh"].sum()), 2),
        "realised_pnl_rs": round(realised, 0),
        "expected_pnl_rs": round(expected, 0) if expected is not None else None,
        "slippage_rs": round(realised - expected, 0) if expected is not None else None,
        "throughput_mwh": round(throughput, 2),
        "full_cycle_equivalents": round(efc, 3),
        "mean_sell_price": round(float(
            (b.loc[b["filled_discharge_mw"] > 0, "mcp_rs_mwh"]).mean()), 0)
        if (b["filled_discharge_mw"] > 0).any() else None,
        "mean_buy_price": round(float(
            (b.loc[b["filled_charge_mw"] > 0, "mcp_rs_mwh"]).mean()), 0)
        if (b["filled_charge_mw"] > 0).any() else None,
    }
    return DayResult(day, market, b, summary)


def issued_days() -> list[date]:
    days = []
    for f in sorted(OUT.glob("bid_sheet_*.csv")):
        stem = f.stem.replace("bid_sheet_", "")
        try:
            days.append(datetime.strptime(stem, "%Y-%m-%d").date())
        except ValueError:
            continue           # bid_sheet_latest.csv
    return days


def ledger(bess: Bess = Bess(), market: str = "dam",
           complete_only: bool = True) -> pd.DataFrame:
    """Every delivery day we issued orders for, settled at actual prices.

    complete_only keeps the headline honest: today's DAM has cleared
    financially, but its later blocks have not been delivered yet.
    """
    rows = []
    for d in issued_days():
        try:
            res = replay(d, bess, market)
        except Exception as e:
            print(f"  {d}: settle failed — {type(e).__name__}: {str(e)[:90]}")
            continue
        if res and (res.summary["complete"] or not complete_only):
            rows.append(res.summary)
    return pd.DataFrame(rows)


def book_summary(bess: Bess = Bess(), market: str = "dam") -> dict:
    """Headline numbers for the whole issued book (completed days only)."""
    led = ledger(bess, market, complete_only=True)
    pending = ledger(bess, market, complete_only=False)
    if not len(led):
        return {"error": "no settled delivery days yet"}

    realised = float(led["realised_pnl_rs"].sum())
    expected = float(led["expected_pnl_rs"].dropna().sum())
    days = int(len(led))
    efc = float(led["full_cycle_equivalents"].sum())

    # warranty envelope: a typical 2-hour LFP warranty allows 365-550 equivalent
    # full cycles a year. Reported as a RATE so a short book still says something.
    efc_per_year = efc / days * 365 if days else 0.0

    return {
        "market": market.upper(),
        "settled_days": days,
        "first_day": led["day"].min(),
        "last_day": led["day"].max(),
        "orders": int(led["orders"].sum()),
        "filled": int(led["filled"].sum()),
        "fill_rate_pct": round(100 * led["filled"].sum() / max(led["orders"].sum(), 1), 1),
        "realised_pnl_rs": round(realised, 0),
        "expected_pnl_rs": round(expected, 0),
        "slippage_rs": round(realised - expected, 0),
        "slippage_pct": round((realised - expected) / abs(expected) * 100, 1)
        if expected else None,
        "realised_per_day_rs": round(realised / days, 0),
        "sold_mwh": round(float(led["sold_mwh"].sum()), 2),
        "bought_mwh": round(float(led["bought_mwh"].sum()), 2),
        "undeliverable_mwh": round(float(led["undeliverable_mwh"].sum()), 2),
        "efc_total": round(efc, 2),
        "efc_per_year": round(efc_per_year, 0),
        "warranty_efc_per_year": 550,
        "within_warranty": bool(efc_per_year <= 550),
        "daily": led.replace({np.nan: None}).to_dict("records"),
        "in_progress_days": int(len(pending) - len(led)),
    }


# --------------------------------------------------------- bid margin lab ---
# The bid sheet sets its price limits at forecast x (1 + m) to BUY and
# forecast x (1 - m) to SELL, with m fixed at 10%. That is a LOOSENING margin:
# it makes fills more likely at the cost of buying dearer and selling cheaper
# than the forecast said. It has never been chosen against evidence, and with
# a real order ledger we can now choose it against evidence.
#
# The experiment is honest because nothing is re-optimised: the SCHEDULE stays
# exactly the one we issued (same blocks, same volumes, same forecast). Only
# the price limit moves, and the result is re-cleared against the prices that
# actually settled. So this measures one decision in isolation.
#
# It is genuinely non-linear: a tighter margin means fewer BUYs clear, which
# strands later SELLs (the energy never arrived), so P&L does not move
# monotonically with m.

def _bids_at_margin(plan: pd.DataFrame, margin: float) -> pd.DataFrame:
    """Rebuild the issued bid sheet at a different price margin."""
    b = pd.DataFrame(index=plan.index)
    b["block"] = range(1, len(b) + 1)
    b["time_block"] = [f"{t:%H:%M} - {(t + pd.Timedelta(minutes=15)):%H:%M}"
                       for t in b.index]
    b["side"] = np.where(plan["charge_mw"] > 0.01, "BUY",
                         np.where(plan["discharge_mw"] > 0.01, "SELL", "-"))
    b["volume_mw"] = (plan["charge_mw"] + plan["discharge_mw"]).round(2)
    f = plan["forecast_mcp"]
    b["price_limit_rs_mwh"] = np.where(
        b["side"] == "BUY", (f * (1 + margin)).round(0),
        np.where(b["side"] == "SELL", (f * (1 - margin)).round(0), np.nan))
    return b


def _settle_frame(bids: pd.DataFrame, px: pd.Series, bess: Bess) -> dict:
    """Clear + walk physics for an arbitrary bid frame. Shared with replay()."""
    b = clear(bids, px)
    eta = np.sqrt(bess.round_trip_eff)
    soc = bess.soc0_frac * bess.energy_mwh
    soc_min = bess.soc_min_frac * bess.energy_mwh
    cap = bess.energy_mwh
    cash = thru = undel = 0.0
    filled = 0
    for _, r in b.iterrows():
        ch = dis = 0.0
        want = float(r["volume_mw"] or 0.0)
        if r["filled"] and r["side"] == "BUY":
            ch = min(want, max(cap - soc, 0.0) / (eta * BLOCK_H))
        elif r["filled"] and r["side"] == "SELL":
            dis = min(want, max(soc - soc_min, 0.0) * eta / BLOCK_H)
        if r["filled"]:
            filled += 1
        if r["filled"] and r["side"] == "SELL":
            undel += (want - dis) * BLOCK_H
        soc = float(np.clip(soc + BLOCK_H * (ch * eta - dis / eta), 0.0, cap))
        price = float(r["mcp_rs_mwh"]) if pd.notna(r["mcp_rs_mwh"]) else 0.0
        cash += BLOCK_H * price * (dis - ch) \
            - BLOCK_H * bess.degradation_rs_mwh * (ch + dis)
        thru += BLOCK_H * (ch + dis)
    return {"pnl": cash, "filled": filled,
            "orders": int((b["side"] != "-").sum()),
            "throughput_mwh": thru, "undeliverable_mwh": undel}


def margin_sweep(bess: Bess = Bess(), market: str = "dam",
                 margins=(0.0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.50)) -> dict:
    """Re-clear every issued plan at each candidate margin."""
    days = [d for d in issued_days() if d < date.today()]
    px_all = store.read(MARKETS[market])["mcp_rs_mwh"]

    rows = []
    for m in margins:
        agg = {"pnl": 0.0, "filled": 0, "orders": 0,
               "throughput_mwh": 0.0, "undeliverable_mwh": 0.0, "days": 0}
        for d in days:
            pf = OUT / f"plan_{d}.csv"
            if not pf.exists():
                continue
            plan = pd.read_csv(pf, parse_dates=["ts"], index_col="ts")
            if not {"charge_mw", "discharge_mw", "forecast_mcp"} <= set(plan.columns):
                continue
            px = px_all[px_all.index.normalize() == pd.Timestamp(d)]
            if not len(px):
                continue
            r = _settle_frame(_bids_at_margin(plan, m), px, bess)
            for k in ("pnl", "filled", "orders", "throughput_mwh", "undeliverable_mwh"):
                agg[k] += r[k]
            agg["days"] += 1
        if not agg["days"]:
            continue
        efc_yr = agg["throughput_mwh"] / (2 * bess.energy_mwh) / agg["days"] * 365
        rows.append({
            "margin_pct": round(m * 100, 1),
            "realised_pnl_rs": round(agg["pnl"], 0),
            "per_day_rs": round(agg["pnl"] / agg["days"], 0),
            "fill_rate_pct": round(100 * agg["filled"] / max(agg["orders"], 1), 1),
            "undeliverable_mwh": round(agg["undeliverable_mwh"], 1),
            "efc_per_year": round(efc_yr, 0),
            "within_warranty": bool(efc_yr <= 550),
            "days": agg["days"],
        })
    if not rows:
        return {"error": "no settled days to sweep"}

    live = next((r for r in rows if r["margin_pct"] == 10.0), None)
    # Only consider margins that keep the asset inside its warranty envelope —
    # a P&L that cycles the battery to death is not an improvement.
    legal = [r for r in rows if r["within_warranty"]] or rows
    best = max(legal, key=lambda r: r["realised_pnl_rs"])
    return {
        "market": market.upper(),
        "days": rows[0]["days"],
        "curve": rows,
        "current_margin_pct": 10.0,
        "current_pnl_rs": live["realised_pnl_rs"] if live else None,
        "best_margin_pct": best["margin_pct"],
        "best_pnl_rs": best["realised_pnl_rs"],
        "gain_rs": round(best["realised_pnl_rs"] - live["realised_pnl_rs"], 0)
        if live else None,
        "gain_pct": round((best["realised_pnl_rs"] / live["realised_pnl_rs"] - 1) * 100, 1)
        if live and live["realised_pnl_rs"] else None,
        "caveat": (
            f"Chosen on only {rows[0]['days']} settled delivery days, all in one "
            "season. Treat as a direction to move, not a tuned constant — and "
            "re-run it as the ledger grows."),
    }


if __name__ == "__main__":
    import json
    bess = Bess()
    s = book_summary(bess)
    if "error" in s:
        print(s["error"])
        raise SystemExit(0)

    print(f"FlexTrade book — {s['market']}, {s['settled_days']} settled delivery days "
          f"({s['first_day']} to {s['last_day']})")
    print(f"  orders {s['orders']}  filled {s['filled']}  "
          f"fill rate {s['fill_rate_pct']}%")
    print(f"  bought {s['bought_mwh']:,.1f} MWh   sold {s['sold_mwh']:,.1f} MWh"
          f"   undeliverable {s['undeliverable_mwh']:,.1f} MWh")
    print(f"  REALISED  Rs {s['realised_pnl_rs']:>12,.0f}")
    print(f"  expected  Rs {s['expected_pnl_rs']:>12,.0f}   "
          f"slippage Rs {s['slippage_rs']:,.0f} ({s['slippage_pct']}%)")
    print(f"  cycling   {s['efc_total']} EFC over {s['settled_days']} days "
          f"-> {s['efc_per_year']:,.0f}/yr vs {s['warranty_efc_per_year']} warranty "
          f"({'WITHIN' if s['within_warranty'] else 'OVER'})")
    print()
    led = pd.DataFrame(s["daily"])
    cols = ["day", "orders", "filled", "fill_rate_pct", "sold_mwh",
            "undeliverable_mwh", "realised_pnl_rs", "expected_pnl_rs", "slippage_rs"]
    print(led[cols].to_string(index=False))
    (OUT / "trade_book.json").write_text(json.dumps(s, indent=2, default=float))
