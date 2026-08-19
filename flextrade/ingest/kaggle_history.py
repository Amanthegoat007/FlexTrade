"""Two verified public datasets that predate our own collection.

Both were cross-checked against our metered Delhi SLDC feed before being
trusted, because a re-upload is only as good as its provenance:

    POSOCO daily state   Max Demand Met vs our 5-min daily peak
                         MAPE 1.10%, median bias -0.51%, corr 0.9946, 1,352 days
    State consumption    Delhi MU vs our metered daily energy
                         MAPE 1.15%, median bias +0.09%, corr 0.9963, 1,039 days

Roughly 1% is rounding plus convention (POSOCO's reported peak against our
5-minute maximum), not disagreement. They are sound.

COVID IS FLAGGED, NOT DELETED

Measured against a 2017-19 monthly baseline in the consumption series itself,
national daily-average consumption ran -5.1% in March 2020, -20.4% in April,
-9.8% in May, -7.4% in June, and +4.8% by July. So the anomaly is four months
wide and closes cleanly. May 2021 sits -3.2% against a year averaging +9.7%,
which is the Delta wave and is much milder.

The rows stay, carrying covid_flag, because deleting them makes the exclusion
invisible to anyone reading the table later and forecloses studying the shock
itself. Models filter on the flag; nothing filters silently.

Note this only matters for these two tables. Our own series start after it:
load_5min at 2021-06-21 and dam_price at 2022-06-01, both already clear.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ingest import store  # noqa: E402

COVID_FROM, COVID_TO = "2020-03-01", "2020-06-30"
DELTA_FROM, DELTA_TO = "2021-04-15", "2021-06-15"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS posoco_state_daily (
    day TEXT NOT NULL, state TEXT NOT NULL,
    max_demand_met_mw REAL, shortage_peak_mw REAL, energy_met_mu REAL,
    drawl_schedule_mu REAL, od_ud_mu REAL, max_od_mw REAL,
    energy_shortage_mu REAL, covid_flag TEXT,
    PRIMARY KEY (day, state));
CREATE TABLE IF NOT EXISTS state_consumption_daily (
    day TEXT NOT NULL, state TEXT NOT NULL,
    consumption_mu REAL, covid_flag TEXT,
    PRIMARY KEY (day, state));
"""


def _flag(days: pd.Series) -> pd.Series:
    f = pd.Series("", index=days.index, dtype=object)
    f[(days >= COVID_FROM) & (days <= COVID_TO)] = "covid_lockdown"
    f[(days >= DELTA_FROM) & (days <= DELTA_TO)] = "covid_delta"
    return f


def load_posoco(csv: str | Path) -> int:
    d = pd.read_csv(csv)
    d["day"] = pd.to_datetime(d["Date"]).dt.strftime("%Y-%m-%d")
    d = d.rename(columns={
        "State": "state", "Max Demand Met": "max_demand_met_mw",
        "Shortage During Peak": "shortage_peak_mw", "Energy Met": "energy_met_mu",
        "Drawl Schedule": "drawl_schedule_mu", "OD(+) / UD(-)": "od_ud_mu",
        "Max OD": "max_od_mw", "Energy Shortage": "energy_shortage_mu"})
    d["covid_flag"] = _flag(pd.to_datetime(d["day"]))
    cols = ["day", "state", "max_demand_met_mw", "shortage_peak_mw",
            "energy_met_mu", "drawl_schedule_mu", "od_ud_mu", "max_od_mw",
            "energy_shortage_mu", "covid_flag"]
    with store.connect() as con:
        con.executescript(_SCHEMA)
        con.executemany(
            f"INSERT OR REPLACE INTO posoco_state_daily ({','.join(cols)}) "
            f"VALUES ({','.join('?' * len(cols))})",
            d[cols].where(pd.notna(d[cols]), None).itertuples(index=False, name=None))
    return len(d)


def load_consumption(csv: str | Path) -> int:
    d = pd.read_csv(csv)
    d["day"] = pd.to_datetime(d["Dates"]).dt.strftime("%Y-%m-%d")
    drop = [c for c in d.columns if c in ("Dates", "day", "Total Consumption")
            or c.startswith("Unnamed")]
    long = d.melt(id_vars=["day"], value_vars=[c for c in d.columns if c not in drop],
                  var_name="state", value_name="consumption_mu").dropna()
    long["covid_flag"] = _flag(pd.to_datetime(long["day"]))
    cols = ["day", "state", "consumption_mu", "covid_flag"]
    with store.connect() as con:
        con.executescript(_SCHEMA)
        con.executemany(
            f"INSERT OR REPLACE INTO state_consumption_daily ({','.join(cols)}) "
            f"VALUES ({','.join('?' * len(cols))})",
            long[cols].itertuples(index=False, name=None))
    return len(long)
