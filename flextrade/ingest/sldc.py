"""Delhi SLDC live scrapers.

- fetch_realtime():  current Delhi load / schedule / drawl / frequency from
  the real-time monitoring page (updates every few seconds).
- fetch_day_curve(): full 5-min load curve for any past date from
  Loaddata.aspx — same table the historical hackathon dataset came from,
  which lets us backfill right up to yesterday.
"""
import io
import re
from datetime import date, timedelta

import pandas as pd
import requests

from . import store

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
RT_URL = "https://www.delhisldc.org/Redirect.aspx?Loc=0804"
DAY_URL = "https://www.delhisldc.org/Loaddata.aspx?mode={d:%d/%m/%Y}"


def fetch_realtime() -> dict:
    html = requests.get(RT_URL, headers=UA, timeout=25).text
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)

    def grab(label):
        m = re.search(label + r"\s*(-?[\d.]+)", text)
        return float(m.group(1)) if m else None

    snap = {
        "delhi_load": grab(r"Delhi Load"),
        "schedule": grab(r"Schedule"),
        "drawl": grab(r"Drawl"),
        "frequency": grab(r"Frequency"),
        "od_ud": grab(r"OD / UD"),
    }
    if snap["delhi_load"] is None:
        raise ValueError("could not parse Delhi Load from realtime page")
    return snap


def fetch_day_curve(d: date) -> pd.DataFrame:
    """5-min load curve (DELHI + discoms, MW) for one date."""
    html = requests.get(DAY_URL.format(d=d), headers=UA, timeout=30).text
    tables = pd.read_html(io.StringIO(html))
    big = max(tables, key=len)
    big.columns = [str(c).strip().lower() for c in big.iloc[0]]
    big = big.iloc[1:]
    big = big.rename(columns={"timeslot": "slot"})
    big["ts"] = pd.to_datetime(f"{d:%Y-%m-%d} " + big["slot"], errors="coerce")
    big = big.dropna(subset=["ts"]).set_index("ts").drop(columns="slot")
    big = big.apply(pd.to_numeric, errors="coerce")
    big.columns = [c if c != "tpddl" else "ndpl" for c in big.columns]  # renamed discom
    return big[["delhi", "brpl", "bypl", "ndpl", "ndmc", "mes"]]


def get_realtime():
    """Live fetch with cached fallback. Returns (snap_dict, meta)."""
    try:
        snap = fetch_realtime()
        row = pd.DataFrame([snap])
        row.insert(0, "fetched_at", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"))
        with store.connect() as con:
            row.to_sql("rt_snapshot", con, if_exists="append", index=False)
        store.log_fetch("sldc_rt", True)
        return snap, {"live": True, "asof": pd.Timestamp.now()}
    except Exception as e:
        store.log_fetch("sldc_rt", False, str(e))
        with store.connect() as con:
            df = pd.read_sql(
                "SELECT * FROM rt_snapshot ORDER BY fetched_at DESC LIMIT 1", con)
        snap = df.iloc[0].to_dict() if len(df) else {}
        return snap, {"live": False, "asof": snap.get("fetched_at"), "error": str(e)}


def backfill_load(start: date, end: date | None = None, verbose: bool = True) -> int:
    """Pull day curves for [start, end] into the DB. Returns rows written."""
    end = end or (date.today() - timedelta(days=1))
    total, d = 0, start
    while d <= end:
        try:
            df = fetch_day_curve(d)
            total += store.upsert("load_5min", df)
            if verbose:
                print(f"  sldc {d}: {len(df)} rows")
        except Exception as e:
            print(f"  sldc {d}: FAILED ({e})")
        d += timedelta(days=1)
    store.log_fetch("sldc_hist", True, f"backfilled to {end}")
    return total
