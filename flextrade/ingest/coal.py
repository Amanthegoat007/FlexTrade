"""CEA Daily Coal Stock Report — the supply-side shock the price model cannot see.

Every model in this repo looks at demand, weather and price. None of them sees
SUPPLY. That matters because the residual error in the DAM price model is not
demand error — we measured that: day-level demand characteristics (spread,
volatility, cap share) correlate with money lost at +0.03 to -0.06, essentially
zero. What actually moves Indian power prices is thermal availability, and
thermal availability is a coal story.

CEA's Fuel Management Division publishes a plant-by-plant coal position every
day, and unlike almost everything else we have found it is DATE-PARAMETERISED,
so it can be backfilled rather than only accrued:

    https://npp.gov.in/public-reports/cea/daily/fuel/DD-MM-YYYY/dailyCoal1-YYYY-MM-DD.xls

Verified reachable at least 180 days back. ~285 rows per day covering every
major thermal station with:

    state · plant · capacity MW · PLF · daily requirement at 85% PLF
    actual stock (indigenous / import / total) · % of normative
    critical flag · receipt of the day · consumption of the day

The number that matters is DAYS OF STOCK = actual / daily requirement. A plant
under about 7 days is one supply disruption from de-rating, and a fleet running
thin is a market that spikes on any demand surprise. Aggregated to state and
national level this is a genuine leading indicator, and it is the first
supply-side feature the price model will have.

XLS, not XLSX — it is a real BIFF workbook and needs xlrd.
"""
from __future__ import annotations

import io
import sys
import time
import warnings
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ingest import store  # noqa: E402

warnings.filterwarnings("ignore")

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120"}

# column positions in the published sheet (row 3 is the header)
COLS = {
    "sl_state": 0, "mode": 2, "plant": 4, "capacity_mw": 7, "plf": 9,
    "norm_days": 11, "daily_req_kt": 15, "norm_stock_kt": 17,
    "stock_indigenous_kt": 20, "stock_import_kt": 23, "stock_total_kt": 26,
    "pct_of_normative": 28, "critical": 30,
    "receipt_kt": 32, "consumption_kt": 33,
}

# India's thermal fleet is ~240 GW; a parse that lands far outside this has
# picked up the wrong rows
FLEET_MIN_MW, FLEET_MAX_MW = 100_000.0, 350_000.0

_SCHEMA = """
CREATE TABLE IF NOT EXISTS coal_stock (
    day TEXT, plant TEXT, state TEXT, mode TEXT,
    capacity_mw REAL, plf_pct REAL, daily_req_kt REAL,
    stock_total_kt REAL, stock_indigenous_kt REAL, stock_import_kt REAL,
    pct_of_normative REAL, days_of_stock REAL, critical INTEGER,
    receipt_kt REAL, consumption_kt REAL,
    PRIMARY KEY (day, plant));
CREATE INDEX IF NOT EXISTS idx_coal_day ON coal_stock(day);
"""


def url_for(d: date) -> str:
    return (f"https://npp.gov.in/public-reports/cea/daily/fuel/"
            f"{d:%d-%m-%Y}/dailyCoal1-{d:%Y-%m-%d}.xls")


def _num(v) -> float | None:
    if v is None or (isinstance(v, float) and v != v):
        return None
    s = str(v).replace(",", "").replace("%", "").strip()
    if not s or s.lower() in ("nan", "-", "na"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def fetch(d: date, retries: int = 3, backoff_s: int = 4) -> pd.DataFrame:
    """Plant-level coal position for one day."""
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url_for(d), headers=UA, timeout=90)
            if r.status_code == 404:
                return pd.DataFrame()          # not published for this date
            r.raise_for_status()
            raw = pd.read_excel(io.BytesIO(r.content), header=None)
            break
        except Exception as e:
            last = e
            if attempt < retries - 1:
                time.sleep(backoff_s * (2 ** attempt))
    else:
        raise last

    rows, state = [], None
    for i in range(5, len(raw)):
        r0 = raw.iloc[i]
        first = str(r0[COLS["sl_state"]]).strip()
        plant = str(r0[COLS["plant"]]).strip()
        cap = _num(r0[COLS["capacity_mw"]])

        # A state header is a bare label with no plant and no capacity. Section
        # banners ("A. PLANT HAVING COAL LINKAGE...") and utility subtotals
        # ("HPGCL-Total") must not become rows.
        if plant in ("", "nan") or cap is None:
            if (first not in ("", "nan") and not first.isdigit()
                    and "total" not in first.lower()
                    and not first.startswith(("A.", "B.", "C.", "D."))):
                state = first
            continue
        if "total" in plant.lower():
            continue

        daily_req = _num(r0[COLS["daily_req_kt"]])
        stock = _num(r0[COLS["stock_total_kt"]])
        rows.append({
            "day": str(d), "plant": plant, "state": state,
            "mode": str(r0[COLS["mode"]]).strip() or None,
            "capacity_mw": cap,
            "plf_pct": _num(r0[COLS["plf"]]),
            "daily_req_kt": daily_req,
            "stock_total_kt": stock,
            "stock_indigenous_kt": _num(r0[COLS["stock_indigenous_kt"]]),
            "stock_import_kt": _num(r0[COLS["stock_import_kt"]]),
            "pct_of_normative": _num(r0[COLS["pct_of_normative"]]),
            "days_of_stock": (round(stock / daily_req, 2)
                              if stock is not None and daily_req else None),
            "critical": int(str(r0[COLS["critical"]]).strip() not in ("", "nan")),
            "receipt_kt": _num(r0[COLS["receipt_kt"]]),
            "consumption_kt": _num(r0[COLS["consumption_kt"]]),
        })

    df = pd.DataFrame(rows)
    if len(df):
        fleet = df["capacity_mw"].sum()
        if not (FLEET_MIN_MW <= fleet <= FLEET_MAX_MW):
            raise ValueError(
                f"parsed fleet {fleet:,.0f} MW outside the plausible "
                f"{FLEET_MIN_MW:,.0f}-{FLEET_MAX_MW:,.0f} MW band for India — "
                "the sheet layout has probably moved")
    return df


def store_day(df: pd.DataFrame) -> int:
    if not len(df):
        return 0
    with store.connect() as con:
        con.executescript(_SCHEMA)
        cols = ["day", "plant", "state", "mode", "capacity_mw", "plf_pct",
                "daily_req_kt", "stock_total_kt", "stock_indigenous_kt",
                "stock_import_kt", "pct_of_normative", "days_of_stock",
                "critical", "receipt_kt", "consumption_kt"]
        con.executemany(
            f"INSERT OR REPLACE INTO coal_stock ({','.join(cols)}) "
            f"VALUES ({','.join(['?'] * len(cols))})",
            df[cols].where(pd.notna(df[cols]), None).itertuples(index=False, name=None))
    return len(df)


def backfill(days: int = 180, end: date | None = None, pause: float = 0.4,
             verbose: bool = True) -> dict:
    """Walk backwards from `end`, skipping days already stored."""
    end = end or (date.today() - timedelta(days=1))
    with store.connect() as con:
        con.executescript(_SCHEMA)
        have = {r[0] for r in con.execute("SELECT DISTINCT day FROM coal_stock")}

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
        if verbose and (k + 1) % 20 == 0:
            print(f"  {k + 1}/{days} — stored {got}, absent {missing}, failed {failed}")
    return {"stored": got, "absent": missing, "failed": failed}


def daily_summary(days: int = 120) -> pd.DataFrame:
    """National coal position per day — the shape a price model consumes."""
    with store.connect() as con:
        df = pd.read_sql("SELECT * FROM coal_stock", con)
    if not len(df):
        return pd.DataFrame()
    df["day"] = pd.to_datetime(df["day"])
    g = df.groupby("day")
    out = pd.DataFrame({
        "plants": g.size(),
        "fleet_mw": g["capacity_mw"].sum(),
        "stock_kt": g["stock_total_kt"].sum(),
        "daily_req_kt": g["daily_req_kt"].sum(),
        "critical_plants": g["critical"].sum(),
    })
    out["days_of_stock"] = (out["stock_kt"] / out["daily_req_kt"]).round(2)
    crit = df[df["critical"] == 1].groupby("day")["capacity_mw"].sum()
    out["critical_mw"] = crit.reindex(out.index).fillna(0.0)
    out["critical_capacity_pct"] = (out["critical_mw"] / out["fleet_mw"] * 100).round(2)
    return out.tail(days).reset_index()


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    if n:
        print(f"backfilling {n} days of CEA coal stock")
        print(backfill(days=n))
    else:
        # the report lags: T-1 is often not up yet, so walk back to the newest
        # day that actually exists rather than reporting an empty frame
        df, d = None, None
        for back in range(1, 8):
            cand = date.today() - timedelta(days=back)
            got = fetch(cand)
            if len(got):
                df, d = got, cand
                break
        if df is None:
            print("no coal report published in the last 7 days")
            raise SystemExit(0)
        print(f"CEA daily coal stock — {d}: {len(df)} plants, "
              f"{df['capacity_mw'].sum():,.0f} MW fleet")
        store_day(df)
        thin = df[df["days_of_stock"].notna()].nsmallest(8, "days_of_stock")
        print("\n  thinnest coal positions:")
        for _, r in thin.iterrows():
            print(f"    {str(r['state'])[:12]:12s} {r['plant'][:26]:26s} "
                  f"{r['capacity_mw']:>7,.0f} MW  {r['days_of_stock']:>5.1f} days"
                  f"{'  CRITICAL' if r['critical'] else ''}")
        s = daily_summary()
        if len(s):
            print(f"\n  national: {s.iloc[-1]['days_of_stock']:.1f} days of stock, "
                  f"{int(s.iloc[-1]['critical_plants'])} critical plants "
                  f"({s.iloc[-1]['critical_capacity_pct']:.1f}% of fleet)")
