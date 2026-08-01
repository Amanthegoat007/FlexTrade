"""Live telemetry from the BRPL Kilokari BESS — a real, operating asset.

Delhi SLDC publishes the state of BSES Rajdhani's battery at
delhisldc.org/bess.aspx: net power (MW), reactive power (kVAr) and State
of Charge (%). The asset is India's first utility-scale standalone BESS
(20 MW / 40 MWh, COD 1 April 2025, Kilokari 33/11 kV substation,
AmpereHour + IndiGrid, approved under Section 63 of the Electricity Act).

Sign convention on the page (verified empirically 23 Jul 2026, NOT
assumed): the column header is "Consumption/Generation (MW)" and the
value is GENERATION-POSITIVE — during an observed real discharge the
raw value sat at +19.6 MW while SoC fell 81% -> 20%. So `discharge_mw`
= raw value unchanged. An earlier version negated it (guessing
load-positive); stored rows from before the fix were migrated with
`UPDATE bess_telemetry SET discharge_mw = -discharge_mw`.

Why this matters: our reference asset in the optimizer is 20 MW / 40 MWh
— the same spec. So FlexTrade's schedule can be compared block-for-block
against what a real battery actually did. See validate/bess_validate.py.
"""
import re
from datetime import datetime

import pandas as pd
import requests

from . import store

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
URL = "https://www.delhisldc.org/bess.aspx"

# nameplate of the published asset (public sources; see module docstring)
RATED_POWER_MW = 20.0
RATED_ENERGY_MWH = 40.0


def fetch_bess() -> dict:
    """Scrape one instantaneous reading. Raises on parse failure."""
    html = requests.get(URL, headers=UA, timeout=25).text
    cells = re.findall(r"<td[^>]*>\s*(-?[\d.]+)\s*</td>", html)
    if len(cells) < 3:
        raise ValueError(f"expected 3 numeric cells, parsed {len(cells)}")
    net_mw, kvar, soc = (float(c) for c in cells[:3])
    # sanity gate: a mis-parsed cell once stored SoC "89,100,000%" — reject
    # physically impossible readings instead of poisoning the history
    if not (0.0 <= soc <= 100.0) or abs(net_mw) > 25.0:
        raise ValueError(f"implausible reading rejected: net={net_mw} soc={soc}")
    return {
        "ts": datetime.now().replace(microsecond=0).isoformat(sep=" "),
        "net_mw": net_mw,            # raw page value, generation-positive
        "discharge_mw": net_mw,      # +ve = exporting (same as page)
        "kvar": kvar,
        "soc_pct": soc,
        "soc_mwh": soc / 100.0 * RATED_ENERGY_MWH,
    }


def poll_once() -> tuple[dict, dict]:
    """Live fetch with cached fallback. Returns (reading, meta)."""
    try:
        row = fetch_bess()
        with store.connect() as con:
            con.execute("""CREATE TABLE IF NOT EXISTS bess_telemetry (
                ts TEXT PRIMARY KEY, net_mw REAL, discharge_mw REAL,
                kvar REAL, soc_pct REAL, soc_mwh REAL)""")
            con.execute("INSERT OR REPLACE INTO bess_telemetry VALUES "
                        "(:ts,:net_mw,:discharge_mw,:kvar,:soc_pct,:soc_mwh)", row)
        store.log_fetch("bess_brpl", True)
        return row, {"live": True, "asof": pd.Timestamp.now()}
    except Exception as e:
        store.log_fetch("bess_brpl", False, str(e))
        hist = read_history()
        last = hist.iloc[-1].to_dict() if len(hist) else {}
        return last, {"live": False, "asof": hist.index.max() if len(hist) else None,
                      "error": str(e)}


def read_history() -> pd.DataFrame:
    with store.connect() as con:
        try:
            df = pd.read_sql("SELECT * FROM bess_telemetry ORDER BY ts", con,
                             parse_dates=["ts"], index_col="ts")
        except Exception:
            return pd.DataFrame()
    return df


def daily_profile(d=None) -> pd.DataFrame:
    """Telemetry resampled to the 15-min market blocks for one day."""
    hist = read_history()
    if not len(hist):
        return pd.DataFrame()
    if d is not None:
        hist = hist[hist.index.date == d]
    if not len(hist):
        return pd.DataFrame()
    out = hist.resample("15min").agg({"discharge_mw": "mean", "soc_pct": "last",
                                      "soc_mwh": "last"})
    return out.dropna(subset=["discharge_mw"])


if __name__ == "__main__":
    row, meta = poll_once()
    state = ("discharging" if row["discharge_mw"] > 0.05 else
             "charging" if row["discharge_mw"] < -0.05 else "idle")
    print(f"BRPL Kilokari BESS ({RATED_POWER_MW:.0f} MW / {RATED_ENERGY_MWH:.0f} MWh)")
    print(f"  live={meta['live']}  {state}")
    print(f"  net {row['net_mw']:+.3f} MW  |  SoC {row['soc_pct']:.0f}% "
          f"({row['soc_mwh']:.1f} MWh)  |  {row['kvar']:+.3f} kVAr")
    print(f"  stored readings: {len(read_history())}")
