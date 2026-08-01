"""Frequency-response readiness — monetizing our sampled frequency data.

India's SRAS/TRAS ancillary markets are operationalizing, and batteries
are the ideal fast-response provider — but there is no public price feed
yet (settled inside NLDC). What CAN be shown today, honestly: using the
grid-frequency history we sample from Delhi SLDC (no public archive
exists — this dataset exists because our poller built it), simulate a
droop-controlled battery and report how often and how deeply it would
have been called. That is the "readiness report" an operator needs
before committing capacity to the ancillary market on day one.

Droop model (configurable, IEGC-informed):
  dead band +/-0.03 Hz around 50.00; response ramps linearly to full
  power at +/-0.15 Hz. Under-frequency -> discharge, over -> charge.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from ingest import store  # noqa: E402

DEAD_BAND_HZ = 0.03
FULL_RESPONSE_HZ = 0.15


def droop_response(freq_hz, power_mw: float = 20.0) -> np.ndarray:
    """Signed MW response (+ = discharge) for a frequency series."""
    dev = 50.0 - np.asarray(freq_hz, dtype=float)   # + = under-frequency
    mag = (np.abs(dev) - DEAD_BAND_HZ) / (FULL_RESPONSE_HZ - DEAD_BAND_HZ)
    return power_mw * np.clip(mag, 0.0, 1.0) * np.sign(dev)


def readiness(power_mw: float = 20.0) -> dict:
    with store.connect() as con:
        try:
            f = pd.read_sql("SELECT ts, frequency_hz FROM frequency ORDER BY ts",
                            con, parse_dates=["ts"], index_col="ts")["frequency_hz"]
        except Exception:
            return {"error": "no frequency data"}
    f = f.dropna()
    if not len(f):
        return {"error": "no frequency data"}

    resp = droop_response(f.values, power_mw)
    called = np.abs(resp) > 0.01
    n_days = max(len(np.unique(f.index.date)), 1)
    # samples are ~5-min; energy per sample = MW * 1/12 h
    energy_mwh = float(np.abs(resp).sum() / 12.0)

    by_hour = pd.Series(np.abs(resp), index=f.index).groupby(f.index.hour).mean()
    return {
        "samples": int(len(f)),
        "days_sampled": n_days,
        "span": {"from": str(f.index.min()), "to": str(f.index.max())},
        "freq_mean_hz": round(float(f.mean()), 3),
        "freq_min_hz": round(float(f.min()), 2),
        "freq_max_hz": round(float(f.max()), 2),
        "pct_samples_called": round(float(called.mean() * 100), 1),
        "pct_under_frequency": round(float((f < 50.0 - DEAD_BAND_HZ).mean() * 100), 1),
        "mean_response_when_called_mw": round(float(np.abs(resp[called]).mean()), 2)
        if called.any() else 0.0,
        "max_response_mw": round(float(np.abs(resp).max()), 2),
        "energy_mwh_per_day": round(energy_mwh / n_days, 2),
        "busiest_hours": [int(h) for h in by_hour.nlargest(3).index],
        "droop": {"dead_band_hz": DEAD_BAND_HZ, "full_response_hz": FULL_RESPONSE_HZ,
                  "power_mw": power_mw},
        "note": ("no public ancillary price exists (NLDC-internal) — this is a "
                 "readiness/duty-cycle report on our sampled frequency history, "
                 "not a revenue claim; the dataset itself is proprietary because "
                 "SLDC serves no archive"),
    }


if __name__ == "__main__":
    r = readiness()
    if "error" in r:
        print(r["error"])
    else:
        print(f"Frequency-response readiness ({r['droop']['power_mw']:.0f} MW droop, "
              f"±{DEAD_BAND_HZ} Hz dead band):")
        print(f"  {r['samples']} samples over {r['days_sampled']} days "
              f"({r['span']['from']} -> {r['span']['to']})")
        print(f"  grid frequency mean {r['freq_mean_hz']} Hz, range "
              f"{r['freq_min_hz']}-{r['freq_max_hz']}")
        print(f"  battery would be called on {r['pct_samples_called']}% of samples "
              f"(under-frequency {r['pct_under_frequency']}% of the time)")
        print(f"  mean response when called {r['mean_response_when_called_mw']} MW | "
              f"max {r['max_response_mw']} MW | ~{r['energy_mwh_per_day']} MWh/day")
        print(f"  busiest hours: {r['busiest_hours']}")
        print(f"  ({r['note']})")
