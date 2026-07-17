"""IEX India Day-Ahead Market scraper.

The new iexindia.com server-renders the market snapshot table into the
page HTML, so a plain GET + read_html gives all 96 x 15-min blocks:

  /market-data/day-ahead-market/market-snapshot
      ?interval=ONE_FOURTH_HOUR&dp=TODAY
      ?interval=ONE_FOURTH_HOUR&dp=SELECT_RANGE&fromDate=DD-MM-YYYY&toDate=DD-MM-YYYY

One request returns one delivery day (server-side pagination), so history
is bootstrapped one day per request.
"""
import io
import time
from datetime import date, timedelta

import pandas as pd
import requests

from . import store

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
BASE = "https://www.iexindia.com/market-data/day-ahead-market/market-snapshot"
RTM_BASE = "https://www.iexindia.com/market-data/real-time-market/market-snapshot"


def _parse(html: str) -> pd.DataFrame:
    tables = pd.read_html(io.StringIO(html))
    t = next(t for t in tables if "Time Block" in "".join(map(str, t.columns)))
    t = t.drop(columns=[c for c in t.columns if "Session" in str(c)])
    t.columns = ["date", "hour", "time_block", "purchase_bid_mw", "sell_bid_mw",
                 "mcv_mw", "sched_mw", "mcp_rs_mwh"]
    start = t["time_block"].str.split("-").str[0].str.strip()
    t["ts"] = pd.to_datetime(t["date"] + " " + start, format="%d-%m-%Y %H:%M")
    t = t.set_index("ts").sort_index()
    t = t[~t.index.duplicated(keep="last")]  # RTM lists a row per session
    num = ["purchase_bid_mw", "sell_bid_mw", "mcv_mw", "sched_mw", "mcp_rs_mwh"]
    t[num] = t[num].apply(pd.to_numeric, errors="coerce")
    return t[num]


def fetch_dam(d: date | None = None) -> pd.DataFrame:
    """96-block DAM snapshot for one delivery day (default: today)."""
    if d is None:
        params = {"interval": "ONE_FOURTH_HOUR", "dp": "TODAY"}
    else:
        ds = f"{d:%d-%m-%Y}"
        params = {"interval": "ONE_FOURTH_HOUR", "dp": "SELECT_RANGE",
                  "fromDate": ds, "toDate": ds}
    r = requests.get(BASE, params=params, headers=UA, timeout=40)
    r.raise_for_status()
    return _parse(r.text)


def get_today():
    """Live fetch with cached fallback. Returns (df, meta)."""
    try:
        df = fetch_dam()
        store.upsert("dam_price", df)
        store.log_fetch("iex_dam", True)
        return df, {"live": True, "asof": pd.Timestamp.now()}
    except Exception as e:
        store.log_fetch("iex_dam", False, str(e))
        cached = store.read("dam_price")
        if len(cached):
            last_day = cached.index.normalize().max()
            cached = cached[cached.index.normalize() == last_day]
        return (cached.drop(columns="fetched_at", errors="ignore"),
                {"live": False, "asof": store.last_good_fetch("iex_dam"),
                 "error": str(e)})


def fetch_rtm(d: date | None = None) -> pd.DataFrame:
    """Real-Time Market snapshot (15-min blocks, auctions through the day)."""
    if d is None:
        params = {"interval": "ONE_FOURTH_HOUR", "dp": "TODAY"}
    else:
        ds = f"{d:%d-%m-%Y}"
        params = {"interval": "ONE_FOURTH_HOUR", "dp": "SELECT_RANGE",
                  "fromDate": ds, "toDate": ds}
    r = requests.get(RTM_BASE, params=params, headers=UA, timeout=40)
    r.raise_for_status()
    return _parse(r.text)


def get_rtm_today():
    """Live fetch with cached fallback. Returns (df, meta)."""
    try:
        df = fetch_rtm()
        store.upsert("rtm_price", df)
        store.log_fetch("iex_rtm", True)
        return df, {"live": True, "asof": pd.Timestamp.now()}
    except Exception as e:
        store.log_fetch("iex_rtm", False, str(e))
        cached = store.read("rtm_price")
        if len(cached):
            last_day = cached.index.normalize().max()
            cached = cached[cached.index.normalize() == last_day]
        return (cached.drop(columns="fetched_at", errors="ignore"),
                {"live": False, "asof": store.last_good_fetch("iex_rtm"),
                 "error": str(e)})


def backfill_prices(start: date, end: date | None = None, pause: float = 0.5,
                    verbose: bool = True) -> int:
    """One request per day into dam_price; skips days already stored."""
    end = end or date.today()
    have = store.read("dam_price")
    have_days = set(have.index.date) if len(have) else set()
    total, d = 0, start
    while d <= end:
        if d not in have_days:
            try:
                df = fetch_dam(d)
                total += store.upsert("dam_price", df)
                if verbose:
                    print(f"  iex {d}: {len(df)} blocks")
            except Exception as e:
                print(f"  iex {d}: FAILED ({e})")
            time.sleep(pause)
        d += timedelta(days=1)
    store.log_fetch("iex_hist", True, f"backfilled to {end}")
    return total
