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

import numpy as np
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
    t["ts"] = pd.to_datetime(t["date"].astype(str) + " " + start,
                             format="%d-%m-%Y %H:%M", errors="coerce")
    # Drop rows whose timestamp would not parse. ts is the PRIMARY KEY, but
    # SQLite treats NULLs as distinct, so unparsed rows accumulate silently
    # rather than colliding — 244 NULL-ts rows had piled up in rtm_price this
    # way and broke every join on that table.
    t = t.dropna(subset=["ts"])
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


GDAM_BASE = "https://www.iexindia.com/market-data/green-day-ahead-market/market-snapshot"


def _parse_gdam(html: str) -> pd.DataFrame:
    """GDAM table has two header rows (MultiIndex) and splits sell/MCV
    volumes by fuel — hydro, wind, other-RE, DRE. We keep the totals plus
    the RE fuel split, which is what makes GDAM interesting: it prices
    *green* energy separately from the plain DAM."""
    tables = pd.read_html(io.StringIO(html))
    t = next(t for t in tables if "Time Block" in "".join(map(str, t.columns)))
    flat = []
    for col in t.columns:
        if isinstance(col, tuple):
            top, sub = (str(c).strip() for c in col[:2])
            flat.append(top if top == sub else f"{top} | {sub}")
        else:
            flat.append(str(col).strip())
    t.columns = flat

    def pick(*needles, required=True):
        for c in t.columns:
            low = c.lower()
            if all(n.lower() in low for n in needles):
                return c
        if required:
            raise KeyError(f"GDAM column not found: {needles}")
        return None

    out = pd.DataFrame(index=t.index)
    dates = t[pick("date")]
    start = t[pick("time block")].astype(str).str.split("-").str[0].str.strip()
    out["ts"] = pd.to_datetime(dates + " " + start, format="%d-%m-%Y %H:%M",
                               errors="coerce")
    mapping = {
        "purchase_bid_mw": pick("purchase bid"),
        "sell_bid_mw": pick("sell bid", "total"),
        "mcv_mw": pick("mcv", "total"),
        "sched_mw": pick("final scheduled", "total"),
        "mcp_rs_mwh": pick("mcp"),
        "sell_hydro_mw": pick("sell bid", "hydro", required=False),
        "sell_wind_mw": pick("sell bid", "wind", required=False),
        "sell_other_re_mw": pick("sell bid", "other", required=False),
    }
    for name, col in mapping.items():
        # np.nan, never pd.NA: an absent fuel column assigned pd.NA leaves an
        # OBJECT-dtype column, and sqlite3 cannot bind pd.NA — every historical
        # GDAM day failed with "Error binding parameter 7" until this was
        # float. On days GDAM publishes no wind/other-RE split at all, the
        # column must still be a float column of NaNs.
        out[name] = (pd.to_numeric(t[col], errors="coerce") if col
                     else pd.Series(np.nan, index=t.index, dtype="float64"))
        out[name] = pd.to_numeric(out[name], errors="coerce").astype("float64")
    out = out.dropna(subset=["ts"]).set_index("ts").sort_index()
    return out[~out.index.duplicated(keep="last")]


def fetch_gdam(d: date | None = None) -> pd.DataFrame:
    if d is None:
        params = {"interval": "ONE_FOURTH_HOUR", "dp": "TODAY"}
    else:
        ds = f"{d:%d-%m-%Y}"
        params = {"interval": "ONE_FOURTH_HOUR", "dp": "SELECT_RANGE",
                  "fromDate": ds, "toDate": ds}
    r = requests.get(GDAM_BASE, params=params, headers=UA, timeout=40)
    r.raise_for_status()
    return _parse_gdam(r.text)


def get_gdam_today():
    """Live fetch with cached fallback. Returns (df, meta)."""
    try:
        df = fetch_gdam()
        store.upsert("gdam_price", df)
        store.log_fetch("iex_gdam", True)
        return df, {"live": True, "asof": pd.Timestamp.now()}
    except Exception as e:
        store.log_fetch("iex_gdam", False, str(e))
        cached = store.read("gdam_price")
        if len(cached):
            cached = cached[cached.index.normalize() == cached.index.normalize().max()]
        return (cached.drop(columns="fetched_at", errors="ignore"),
                {"live": False, "asof": store.last_good_fetch("iex_gdam"),
                 "error": str(e)})


# every IEX market is backfillable the same way: its fetcher takes a
# delivery date and the site serves that day's cleared table
MARKETS = {
    "dam": ("dam_price", lambda d: fetch_dam(d)),
    "rtm": ("rtm_price", lambda d: fetch_rtm(d)),
    "gdam": ("gdam_price", lambda d: fetch_gdam(d)),
}


def backfill_prices(start: date, end: date | None = None, pause: float = 0.5,
                    verbose: bool = True, market: str = "dam") -> tuple[int, list]:
    """One request per day into <market>_price; skips days already stored.

    Generalised beyond DAM (28 Jul): RTM and GDAM history is served by the
    same date-ranged endpoint, and we only held 14 / 7 days of them. That
    shortage had two costs — the intraday re-optimizer had to *guess* RTM
    prices from a DAM ratio, and the price model lab had to drop its
    cross-market features as too sparse to use.
    """
    if market not in MARKETS:
        raise ValueError(f"market must be one of {sorted(MARKETS)}")
    table, fetch = MARKETS[market]
    end = end or date.today()
    have = store.read(table)
    have_days = set(have.index.date) if len(have) else set()
    total, failed, d = 0, [], start
    while d <= end:
        if d not in have_days:
            try:
                df = fetch(d)
                total += store.upsert(table, df)
                if verbose:
                    print(f"  {market} {d}: {len(df)} blocks", flush=True)
            except Exception as e:
                failed.append(d)
                print(f"  {market} {d}: FAILED ({str(e)[:90]})", flush=True)
            time.sleep(pause)
        d += timedelta(days=1)
    ok = not failed
    store.log_fetch(f"iex_hist_{market}", ok,
                    f"backfilled to {end}" if ok else f"{len(failed)} days failed")
    return total, failed


def ensure_prices_current(verbose: bool = True):
    """Self-heal DAM price history up to today. Returns a status dict in
    the same shape the other fetchers use, so run_pipeline can print it."""
    stored = store.read("dam_price")
    today = date.today()
    if not len(stored):
        _, failed = backfill_prices(today - timedelta(days=365), today, verbose=verbose)
    else:
        last = stored.index.max().date()
        if last >= today:
            if verbose:
                print(f"  dam_price current through {last}")
            return {"live": True, "asof": last}
        if verbose:
            print(f"  dam_price stale by {(today - last).days} day(s) — backfilling")
        _, failed = backfill_prices(last + timedelta(days=1), today, verbose=verbose)
    return {"live": not failed, "asof": store.read("dam_price").index.max()}
