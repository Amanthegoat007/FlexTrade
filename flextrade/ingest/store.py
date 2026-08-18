"""SQLite store shared by all live fetchers.

Every fetcher writes successful pulls here; on a failed live fetch the
caller falls back to the latest cached rows. The dashboard shows a
LIVE / CACHED badge based on the `meta` dict fetchers return.
"""
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_PATH = DATA_DIR / "flextrade.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS weather (
    ts TEXT PRIMARY KEY, temp_c REAL, rh_pct REAL, apparent_temp_c REAL,
    rain_mm REAL, cloud_pct REAL, kind TEXT, fetched_at TEXT);
CREATE TABLE IF NOT EXISTS load_5min (
    ts TEXT PRIMARY KEY, delhi REAL, brpl REAL, bypl REAL,
    ndpl REAL, ndmc REAL, mes REAL, fetched_at TEXT);
CREATE TABLE IF NOT EXISTS rt_snapshot (
    fetched_at TEXT PRIMARY KEY, delhi_load REAL, schedule REAL,
    drawl REAL, frequency REAL, od_ud REAL);
CREATE TABLE IF NOT EXISTS dam_price (
    ts TEXT PRIMARY KEY, purchase_bid_mw REAL, sell_bid_mw REAL,
    mcv_mw REAL, sched_mw REAL, mcp_rs_mwh REAL, fetched_at TEXT);
CREATE TABLE IF NOT EXISTS re_weather (
    ts TEXT PRIMARY KEY, ghi REAL, dni REAL, dhi REAL, wind10_kmh REAL,
    wind100_kmh REAL, temp_c REAL, cloud_pct REAL, kind TEXT, fetched_at TEXT);
CREATE TABLE IF NOT EXISTS rtm_price (
    ts TEXT PRIMARY KEY, purchase_bid_mw REAL, sell_bid_mw REAL,
    mcv_mw REAL, sched_mw REAL, mcp_rs_mwh REAL, fetched_at TEXT);
CREATE TABLE IF NOT EXISTS hpdam_price (
    ts TEXT PRIMARY KEY, purchase_bid_mw REAL, sell_bid_mw REAL,
    mcv_mw REAL, sched_mw REAL, mcp_rs_mwh REAL, fetched_at TEXT);
CREATE TABLE IF NOT EXISTS frequency (
    ts TEXT PRIMARY KEY, frequency_hz REAL, fetched_at TEXT);
CREATE TABLE IF NOT EXISTS gdam_price (
    ts TEXT PRIMARY KEY, purchase_bid_mw REAL, sell_bid_mw REAL,
    mcv_mw REAL, sched_mw REAL, mcp_rs_mwh REAL,
    sell_hydro_mw REAL, sell_wind_mw REAL, sell_other_re_mw REAL,
    fetched_at TEXT);
CREATE TABLE IF NOT EXISTS fetch_log (
    source TEXT, at TEXT, ok INTEGER, note TEXT);
"""


def connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.executescript(_SCHEMA)
    return con


def upsert(table: str, df: pd.DataFrame) -> int:
    """INSERT OR REPLACE df rows (index must be the primary-key ts)."""
    if df.empty:
        return 0
    df = df.copy()
    if isinstance(df.index, pd.DatetimeIndex):
        df.index = df.index.strftime("%Y-%m-%d %H:%M:%S")
        df.index.name = "ts"
        df = df.reset_index()
    df["fetched_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with connect() as con:
        cols = ",".join(df.columns)
        ph = ",".join("?" * len(df.columns))
        con.executemany(
            f"INSERT OR REPLACE INTO {table} ({cols}) VALUES ({ph})",
            df.itertuples(index=False, name=None),
        )
    return len(df)


def read(table: str, since: str | None = None) -> pd.DataFrame:
    q = f"SELECT * FROM {table}"
    if since:
        q += f" WHERE ts >= '{since}'"
    with connect() as con:
        df = pd.read_sql(q, con)
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"])
        df = df.set_index("ts").sort_index()
    return df


def log_fetch(source: str, ok: bool, note: str = "") -> None:
    with connect() as con:
        con.execute(
            "INSERT INTO fetch_log VALUES (?,?,?,?)",
            (source, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), int(ok), note),
        )


def last_good_fetch(source: str) -> str | None:
    with connect() as con:
        row = con.execute(
            "SELECT MAX(at) FROM fetch_log WHERE source=? AND ok=1", (source,)
        ).fetchone()
    return row[0] if row else None
