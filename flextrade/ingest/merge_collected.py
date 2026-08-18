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
# The Lambda collector writes one object per source per UTC day under its own
# prefix, so `aws s3 sync` lands them as data/collected_aws/<source>/<date>.csv
# rather than the CI collector's flat <source>-<month>.csv. Same columns either
# way, so both layouts are read and concatenated here and nothing downstream
# needs to know which runner captured a given block.
COLLECTED_AWS = ROOT / "data" / "collected_aws"
IST_OFFSET = pd.Timedelta(hours=5, minutes=30)


def _to_ist(series: pd.Series) -> pd.Series:
    return (pd.to_datetime(series, errors="coerce", utc=True).dt.tz_localize(None)
            + IST_OFFSET).dt.strftime("%Y-%m-%d %H:%M:%S")


def _read(prefix: str) -> pd.DataFrame:
    """Every row for one source, from both collectors, deduped.

    The two runners overlap deliberately — neither going dark should cost a
    block — so the same snapshot can arrive twice. drop_duplicates() collapses
    them, and the INSERT OR IGNORE downstream catches anything it misses.
    """
    files = sorted(COLLECTED.glob(f"{prefix}-*.csv"))
    files += sorted(COLLECTED_AWS.glob(f"{prefix}/*.csv"))
    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(f))
        except Exception as e:
            # a half-synced object must not sink the whole merge
            print(f"    skipped {f.name}: {type(e).__name__}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates()


def _ensure_columns(con, table: str, coldefs: dict) -> None:
    """Idempotent ADD COLUMN. SQLite has no IF NOT EXISTS for columns."""
    have = {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
    for name, decl in coldefs.items():
        if name not in have:
            con.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")


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
    # real-time-summary.json carries fields dynamic-data.json never had:
    # the generation split, frequency, and the published DSM rate
    extra = {"frequency_hz": "REAL", "up_thermal_mw": "REAL",
             "ipp_thermal_mw": "REAL", "up_hydro_mw": "REAL",
             "cogen_cpp_mw": "REAL", "re_solar_mw": "REAL",
             "dsm_rate_paise_kwh": "REAL", "deviation_published_mw": "REAL"}
    cols = ["fetched_at", "demand_met_mw", "schedule_mw", "drawal_mw",
            "deviation_signed_mw", "intra_gen_mw", "source_updated",
            *extra]
    have = [c for c in cols if c in df.columns]
    with store.connect() as con:
        con.executescript(up._SCHEMA)
        _ensure_columns(con, "up_live", extra)
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


_PSTCL_SCHEMA = """
CREATE TABLE IF NOT EXISTS pstcl_live (
    fetched_at TEXT PRIMARY KEY,
    source_updated TEXT, frequency_hz REAL, demand_met_mw REAL,
    schedule_mw REAL, drawal_mw REAL,
    deviation_signed_mw REAL, deviation_published_mw REAL);
"""

_NATIONAL_SCHEMA = """
CREATE TABLE IF NOT EXISTS national_gen (
    ts TEXT PRIMARY KEY,
    demand_met_mw REAL, thermal_mw REAL, hydro_mw REAL, wind_mw REAL,
    solar_mw REAL, nuclear_mw REAL, gas_mw REAL);
"""

_FUEL_COL = {
    "DEMAND MET": "demand_met_mw", "THERMAL GENERATION": "thermal_mw",
    "HYDRO GENERATION": "hydro_mw", "WIND GENERATION": "wind_mw",
    "SOLAR GENERATION": "solar_mw", "NUCLEAR GENERATION": "nuclear_mw",
    "GAS GENERATION": "gas_mw",
}


def merge_pstcl() -> int:
    df = _read("pstcl")
    if not len(df):
        return 0
    df["fetched_at"] = _to_ist(df["fetched_at_utc"])
    cols = ["fetched_at", "source_updated", "frequency_hz", "demand_met_mw",
            "schedule_mw", "drawal_mw", "deviation_signed_mw",
            "deviation_published_mw"]
    have = [c for c in cols if c in df.columns]
    df = df.dropna(subset=["fetched_at"]).drop_duplicates(subset=["fetched_at"])
    with store.connect() as con:
        con.executescript(_PSTCL_SCHEMA)
        con.executemany(
            f"INSERT OR IGNORE INTO pstcl_live ({','.join(have)}) "
            f"VALUES ({','.join(['?'] * len(have))})",
            df[have].where(pd.notna(df[have]), None)
            .itertuples(index=False, name=None))
    return len(df)


def merge_national() -> int:
    """NPP demand + fuel mix, pivoted long -> wide on a 4-minute clock.

    Stored wide because that is how it gets used: a price model wants solar and
    thermal as columns beside the block it is predicting, not six rows to
    reshape at feature-build time.

    Timestamps are floored to the minute before pivoting. The six fuels in one
    reading carry stamps a few seconds apart — 360 rows arrived under 64
    distinct timestamps rather than 60 — so grouping on the raw string would
    scatter one reading across several near-identical rows, each mostly null.
    """
    df = pd.concat([_read("npp_demand"), _read("npp_fuelmix")],
                   ignore_index=True)
    if not len(df):
        return 0
    df["col"] = df["series"].map(_FUEL_COL)
    df = df.dropna(subset=["col", "ts_utc"])
    ts = pd.to_datetime(df["ts_utc"], errors="coerce", utc=True)
    df = df[ts.notna()]
    df["ts"] = ((ts[ts.notna()].dt.tz_localize(None) + IST_OFFSET)
                .dt.floor("min").dt.strftime("%Y-%m-%d %H:%M:%S"))
    wide = (df.pivot_table(index="ts", columns="col", values="value_mw",
                           aggfunc="last").reset_index())
    have = ["ts"] + [c for c in _FUEL_COL.values() if c in wide.columns]
    with store.connect() as con:
        con.executescript(_NATIONAL_SCHEMA)
        con.executemany(
            f"INSERT OR IGNORE INTO national_gen ({','.join(have)}) "
            f"VALUES ({','.join(['?'] * len(have))})",
            wide[have].where(pd.notna(wide[have]), None)
            .itertuples(index=False, name=None))
    return len(wide)


def merge_all(verbose: bool = True) -> dict:
    out = {}
    for name, fn in (("state_live", merge_merit), ("up_live", merge_upsldc),
                     ("area_price", merge_area_price),
                     ("pstcl_live", merge_pstcl),
                     ("national_gen", merge_national)):
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
        for t in ("state_live", "up_live", "area_price",
                  "pstcl_live", "national_gen"):
            try:
                n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                print(f"  -> {t} now holds {n:,} rows")
            except Exception:
                pass
