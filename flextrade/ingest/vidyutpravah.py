"""Vidyut PRAVAH (Ministry of Power) — STATE-WISE area clearing price.

This is the one field we had been missing that actually blocks a product.

IEX publishes a single pan-India Market Clearing Price, so "forecast the price
in state X" has had no target: there is only one price. That is why the
multi-state layer could honestly be sold as demand intelligence but not as
trading. The exception is congestion — when a transmission corridor binds, the
market SPLITS and each area clears at its own price. Those are exactly the
hours a battery is worth most, and they are invisible in the national number.

MoP's Vidyut PRAVAH dashboard publishes the per-area clearing price behind an
undocumented controller:

    POST https://vidyutpravah.in/PXDashboard/BindStatePricesFromJS
    -> [{"StateCode":"DL","ACP":5000.92}, ...]   35 areas

It returned an empty list on 2 Aug and started serving on 3 Aug, so it is
intermittent and worth polling rather than trusting on any single call.

What we get, and what we do not:
  * When the market is uniform, all 35 areas print the same number and the
    row is still worth storing — "no congestion" is information, and the
    frequency of splits is what sizes the opportunity.
  * When it splits, the spread between areas is a state-level price signal we
    have no other route to.
  * It is a SNAPSHOT of the current 15-minute block, not a history. There is
    no date parameter, so depth has to be accrued.
"""
from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ingest import store  # noqa: E402

BASE = "https://vidyutpravah.in"
PRICES = f"{BASE}/PXDashboard/BindStatePricesFromJS"
BLOCK = f"{BASE}/PXDashboard/BindCurrentDateTimeForJson"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120",
      "Referer": BASE + "/", "X-Requested-With": "XMLHttpRequest"}

# IEX operates within a regulatory price cap; anything outside this is a parse
# failure, not a market event
PRICE_MIN, PRICE_MAX = 0.0, 20_000.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS area_price (
    fetched_at TEXT, block_from TEXT, block_date TEXT,
    area TEXT, acp_rs_mwh REAL,
    PRIMARY KEY (fetched_at, area));
CREATE INDEX IF NOT EXISTS idx_area_price_area ON area_price(area, fetched_at);
"""


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(UA)
    try:
        s.get(BASE + "/", timeout=60)      # the controllers want a session
    except Exception:
        pass
    return s


def fetch(retries: int = 3, backoff_s: int = 4) -> dict:
    last = None
    for attempt in range(retries):
        try:
            s = _session()
            rows = s.post(PRICES, data={}, timeout=60).json()
            blk = s.post(BLOCK, data={}, timeout=45).json()
            break
        except Exception as e:
            last = e
            if attempt < retries - 1:
                time.sleep(backoff_s * (2 ** attempt))
    else:
        raise last

    if not rows:
        raise ValueError("area-price feed returned an empty list — it is "
                         "intermittent; treat as unavailable, not as zero")

    prices = {}
    for r in rows:
        code, acp = r.get("StateCode"), r.get("ACP")
        if code is None or acp is None:
            continue
        acp = float(acp)
        if not (PRICE_MIN <= acp <= PRICE_MAX):
            raise ValueError(f"area {code} price {acp} outside the plausible "
                             f"{PRICE_MIN}-{PRICE_MAX} Rs/MWh band")
        prices[str(code)] = acp

    b = (blk or [{}])[0]
    distinct = sorted(set(prices.values()))
    return {
        "block_from": b.get("FromTime"),
        "block_date": b.get("CurrentDate"),
        "n_areas": len(prices),
        "prices": prices,
        "uniform": len(distinct) == 1,
        "distinct_prices": len(distinct),
        "spread_rs_mwh": round(max(distinct) - min(distinct), 2) if distinct else 0.0,
    }


def poll() -> dict:
    snap = fetch()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with store.connect() as con:
        con.executescript(_SCHEMA)
        con.executemany(
            "INSERT OR REPLACE INTO area_price "
            "(fetched_at, block_from, block_date, area, acp_rs_mwh) VALUES (?,?,?,?,?)",
            [(now, snap["block_from"], snap["block_date"], a, p)
             for a, p in snap["prices"].items()])
    note = ("uniform" if snap["uniform"]
            else f"SPLIT spread Rs {snap['spread_rs_mwh']}")
    store.log_fetch("vidyutpravah", True, f"{snap['n_areas']} areas, {note}")
    return snap


def congestion_history(days: int = 30) -> dict:
    """How often does the market actually split? That sizes the opportunity."""
    import pandas as pd
    with store.connect() as con:
        try:
            df = pd.read_sql(
                "SELECT fetched_at, area, acp_rs_mwh FROM area_price", con)
        except Exception:
            return {"error": "no area_price history yet"}
    if not len(df):
        return {"error": "no area_price history yet"}
    g = df.groupby("fetched_at")["acp_rs_mwh"]
    spread = (g.max() - g.min())
    return {
        "snapshots": int(len(spread)),
        "split_snapshots": int((spread > 0.01).sum()),
        "split_pct": round(float((spread > 0.01).mean() * 100), 1),
        "max_spread_rs_mwh": round(float(spread.max()), 2),
        "mean_spread_when_split_rs_mwh": round(
            float(spread[spread > 0.01].mean()), 2) if (spread > 0.01).any() else 0.0,
        "note": ("Each snapshot is one 15-minute block. A split means "
                 "transmission congestion separated area prices — the hours a "
                 "battery is worth most, and the only state-level price signal "
                 "available to us."),
    }


if __name__ == "__main__":
    s = poll()
    print(f"Vidyut PRAVAH area prices — block {s['block_from']} "
          f"on {s['block_date']}")
    print(f"  {s['n_areas']} areas | "
          f"{'UNIFORM (no congestion)' if s['uniform'] else 'SPLIT'} | "
          f"spread Rs {s['spread_rs_mwh']}/MWh")
    for a in list(s["prices"])[:8]:
        print(f"    {a:5s} {s['prices'][a]:,.2f}")
    print(f"    ... {s['n_areas'] - 8} more")
    print()
    print("congestion so far:", congestion_history())
