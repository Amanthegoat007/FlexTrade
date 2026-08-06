"""Merge CI-collected CSV snapshots back into the local SQLite store.

collect_ci.py runs on GitHub Actions every ~15 minutes and appends to
data/collected/*.csv, because CSV merges through git and a SQLite file does
not. This pulls those rows into the same tables the laptop collector writes,
so every model reads one store and does not care which machine captured a
given block.

Idempotent by primary key, so running it repeatedly — or after a git pull that
brought several days of CI rows — inserts each snapshot once.

Timezone: the CI collector stamps UTC, the laptop stamps local (IST). They are
converted to IST here so a single ordered series comes out the other side. That
is not cosmetic: an intraday model keyed on hour-of-day would otherwise learn
two different definitions of "07:00" from the same table.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ingest import store  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent.parent
COLLECTED = ROOT / "data" / "collected"
IST_OFFSET = pd.Timedelta(hours=5, minutes=30)


def _to_ist(series: pd.Series) -> pd.Series:
    return (pd.to_datetime(series, errors="coerce", utc=True).dt.tz_localize(None)
            + IST_OFFSET).dt.strftime("%Y-%m-%d %H:%M:%S")


def _read(prefix: str) -> pd.DataFrame:
    files = sorted(COLLECTED.glob(f"{prefix}-*.csv"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return df.drop_duplicates()


def merge_merit() -> int:
    df = _read("merit_state")
    if not len(df):
        return 0
    df["fetched_at"] = _to_ist(df["fetched_at_utc"])
    rows = df[["fetched_at", "code", "demand_mw", "own_gen_mw", "import_mw"]]
    rows = rows.dropna(subset=["fetched_at", "code"])
    with store.connect() as con:
        con.executemany(
            "INSERT OR IGNORE INTO state_live "
            "(fetched_at, code, demand_mw, own_gen_mw, import_mw) VALUES (?,?,?,?,?)",
            rows.where(pd.notna(rows), None).itertuples(index=False, name=None))
    return len(rows)


def merge_upsldc() -> int:
    df = _read("upsldc")
    if not len(df):
        return 0
    from ingest import upsldc as up
    df["fetched_at"] = _to_ist(df["fetched_at_utc"])
    cols = ["fetched_at", "demand_met_mw", "schedule_mw", "drawal_mw",
            "deviation_signed_mw", "intra_gen_mw", "source_updated"]
    have = [c for c in cols if c in df.columns]
    with store.connect() as con:
        con.executescript(up._SCHEMA)
        con.executemany(
            f"INSERT OR IGNORE INTO up_live ({','.join(have)}) "
            f"VALUES ({','.join(['?'] * len(have))})",
            df[have].where(pd.notna(df[have]), None)
            .itertuples(index=False, name=None))
    return len(df)


def merge_area_price() -> int:
    df = _read("area_price")
    if not len(df):
        return 0
    from ingest import vidyutpravah as vp
    df["fetched_at"] = _to_ist(df["fetched_at_utc"])
    cols = ["fetched_at", "block_from", "block_date", "area", "acp_rs_mwh"]
    have = [c for c in cols if c in df.columns]
    with store.connect() as con:
        con.executescript(vp._SCHEMA)
        con.executemany(
            f"INSERT OR IGNORE INTO area_price ({','.join(have)}) "
            f"VALUES ({','.join(['?'] * len(have))})",
            df[have].where(pd.notna(df[have]), None)
            .itertuples(index=False, name=None))
    return len(df)


def merge_all(verbose: bool = True) -> dict:
    out = {}
    for name, fn in (("state_live", merge_merit), ("up_live", merge_upsldc),
                     ("area_price", merge_area_price)):
        try:
            out[name] = fn()
        except Exception as e:
            out[name] = f"{type(e).__name__}: {str(e)[:90]}"
        if verbose:
            print(f"  {name:12s} {out[name]}")
    return out


if __name__ == "__main__":
    if not COLLECTED.exists():
        print(f"nothing to merge — {COLLECTED} does not exist yet")
        raise SystemExit(0)
    print(f"merging CI snapshots from {COLLECTED}")
    merge_all()
    with store.connect() as con:
        for t in ("state_live", "up_live", "area_price"):
            try:
                n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                print(f"  -> {t} now holds {n:,} rows")
            except Exception:
                pass
