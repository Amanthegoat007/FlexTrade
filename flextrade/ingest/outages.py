"""CEA Daily Maintenance Report — unit outages, planned and forced.

The second supply-side feed, and the one that carries the shocks. Coal stock
(ingest/coal.py) describes a slow squeeze; this describes a discontinuity. A
660 MW unit tripping at 03:00 removes supply the market had already priced in,
and no calendar or weather feature can anticipate it.

CEA publishes it as Sub-Report 11 of the Daily Generation Report family, on
the same date-parameterised path as the coal report — so it backfills:

    https://npp.gov.in/public-reports/cea/daily/dgr/DD-MM-YYYY/dgr11-YYYY-MM-DD.xls

    State/System · Power Station · Unit No.
    Planned Maintenance (MW) · Forced Maintenance MAJOR (MW)
    Forced Maintenance MINOR (MW) · Others (MW) · start timestamp

The distinction that matters is PLANNED vs FORCED. Planned outages are known
to the market weeks ahead and are already in the price; forced ones are not,
and are exactly the residual the price model cannot otherwise see. We keep
them in separate columns rather than summing to "MW out", because summing
would blur the only part with information in it.

Same discipline as every other fetcher: a plausibility guard on total outage
MW, because a wrong number is worse than a crash.
"""
from __future__ import annotations

import io
import sys
import time
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ingest import store  # noqa: E402

warnings.filterwarnings("ignore")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120"}

# Column 1 is a merged empty cell, so every index after the state shifts by
# one. Getting this wrong parsed zero rows rather than wrong ones, which is the
# good failure mode — but it is why the map is written out explicitly.
COLS = {"state": 0, "station": 2, "unit": 3, "planned_mw": 4,
        "forced_major_mw": 5, "forced_minor_mw": 6, "others_mw": 7,
        "since": 8, "expected_return": 9, "reason": 10}

# India's coal+lignite fleet is ~224 GW; total simultaneous outage above ~40%
# of it, or a negative total, means the sheet moved under us
OUTAGE_MAX_MW = 90_000.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS unit_outage (
    day TEXT, state TEXT, station TEXT, unit TEXT,
    planned_mw REAL, forced_major_mw REAL, forced_minor_mw REAL,
    others_mw REAL, out_since TEXT, expected_return TEXT, reason TEXT,
    PRIMARY KEY (day, station, unit));
CREATE INDEX IF NOT EXISTS idx_outage_day ON unit_outage(day);
"""


def url_for(d: date) -> str:
    return (f"https://npp.gov.in/public-reports/cea/daily/dgr/"
            f"{d:%d-%m-%Y}/dgr11-{d:%Y-%m-%d}.xls")


def _num(v) -> float:
    if v is None or (isinstance(v, float) and v != v):
        return 0.0
    s = str(v).replace(",", "").strip()
    if not s or s.lower() in ("nan", "-", "na"):
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def fetch(d: date, retries: int = 3, backoff_s: int = 4) -> pd.DataFrame:
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url_for(d), headers=UA, timeout=90)
            if r.status_code == 404:
                return pd.DataFrame()
            r.raise_for_status()
            raw = pd.read_excel(io.BytesIO(r.content), header=None)
            break
        except Exception as e:
            last = e
            if attempt < retries - 1:
                time.sleep(backoff_s * (2 ** attempt))
    else:
        raise last

    rows = []
    for i in range(6, len(raw)):
        r0 = raw.iloc[i]
        state = str(r0[COLS["state"]]).strip()
        station = str(r0[COLS["station"]]).strip()
        if state in ("", "nan") or station in ("", "nan"):
            continue
        if "total" in state.lower() or "total" in station.lower():
            continue
        rows.append({
            "day": str(d), "state": state, "station": station,
            "unit": str(r0[COLS["unit"]]).strip(),
            "planned_mw": _num(r0[COLS["planned_mw"]]),
            "forced_major_mw": _num(r0[COLS["forced_major_mw"]]),
            "forced_minor_mw": _num(r0[COLS["forced_minor_mw"]]),
            "others_mw": _num(r0[COLS["others_mw"]]),
            "out_since": str(r0[COLS["since"]]).strip() or None,
            # forward-looking: CEA publishes when the unit is expected back,
            # which is a scheduled supply return the market can be modelled on
            "expected_return": (str(r0[COLS["expected_return"]]).strip()
                                if COLS["expected_return"] < len(r0) else None),
            "reason": (str(r0[COLS["reason"]]).strip()[:120]
                       if COLS["reason"] < len(r0) else None),
        })

    df = pd.DataFrame(rows)
    if len(df):
        tot = df[["planned_mw", "forced_major_mw",
                  "forced_minor_mw", "others_mw"]].sum().sum()
        if tot < 0 or tot > OUTAGE_MAX_MW:
            raise ValueError(
                f"parsed {tot:,.0f} MW of outages, outside the plausible "
                f"0-{OUTAGE_MAX_MW:,.0f} MW band — sheet layout has moved")
    return df


def store_day(df: pd.DataFrame) -> int:
    if not len(df):
        return 0
    cols = ["day", "state", "station", "unit", "planned_mw", "forced_major_mw",
            "forced_minor_mw", "others_mw", "out_since", "expected_return",
            "reason"]
    with store.connect() as con:
        con.executescript(_SCHEMA)
        con.executemany(
            f"INSERT OR REPLACE INTO unit_outage ({','.join(cols)}) "
            f"VALUES ({','.join(['?'] * len(cols))})",
            df[cols].where(pd.notna(df[cols]), None).itertuples(index=False, name=None))
    return len(df)


def backfill(days: int = 420, end: date | None = None, pause: float = 0.35,
             verbose: bool = True) -> dict:
    end = end or (date.today() - timedelta(days=1))
    with store.connect() as con:
        con.executescript(_SCHEMA)
        have = {r[0] for r in con.execute("SELECT DISTINCT day FROM unit_outage")}
    got = missing = failed = 0
    for k in range(days):
        d = end - timedelta(days=k)
        if str(d) in have:
            continue
        try:
            df = fetch(d)
            if not len(df):
                missing += 1
            else:
                store_day(df)
                got += 1
        except Exception as e:
            failed += 1
            if verbose:
                print(f"  {d}: {type(e).__name__}: {str(e)[:80]}")
        time.sleep(pause)
        if verbose and (k + 1) % 40 == 0:
            print(f"  {k + 1}/{days} — stored {got}, absent {missing}, failed {failed}")
    return {"stored": got, "absent": missing, "failed": failed}


def daily_summary(days: int = 500) -> pd.DataFrame:
    """National outage position per day, planned and forced kept apart."""
    with store.connect() as con:
        try:
            df = pd.read_sql("SELECT * FROM unit_outage", con)
        except Exception:
            return pd.DataFrame()
    if not len(df):
        return pd.DataFrame()
    df["day"] = pd.to_datetime(df["day"])
    df["forced_mw"] = df["forced_major_mw"] + df["forced_minor_mw"]
    g = df.groupby("day")
    out = pd.DataFrame({
        "units_out": g.size(),
        "planned_mw": g["planned_mw"].sum(),
        "forced_mw": g["forced_mw"].sum(),
        "forced_major_mw": g["forced_major_mw"].sum(),
        "others_mw": g["others_mw"].sum(),
    })
    out["total_out_mw"] = (out["planned_mw"] + out["forced_mw"] + out["others_mw"])
    return out.tail(days).reset_index()


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    if n:
        print(f"backfilling {n} days of CEA unit outages")
        print(backfill(days=n))
    else:
        df, d = None, None
        for back in range(1, 8):
            cand = date.today() - timedelta(days=back)
            got = fetch(cand)
            if len(got):
                df, d = got, cand
                break
        if df is None:
            print("no maintenance report published in the last 7 days")
            raise SystemExit(0)
        store_day(df)
        f = df["forced_major_mw"] + df["forced_minor_mw"]
        print(f"CEA unit outages — {d}: {len(df)} units")
        print(f"  planned {df['planned_mw'].sum():,.0f} MW | "
              f"FORCED {f.sum():,.0f} MW | others {df['others_mw'].sum():,.0f} MW")
        big = df.assign(forced=f).nlargest(6, "forced")
        print("\n  largest forced outages:")
        for _, r in big.iterrows():
            if r["forced"] > 0:
                print(f"    {r['state'][:14]:14s} {r['station'][:28]:28s} "
                      f"u{r['unit']:<3s} {r['forced']:>6,.0f} MW  "
                      f"{str(r['reason'])[:34]}")
