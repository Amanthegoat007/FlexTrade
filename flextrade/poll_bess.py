"""Continuous poller for the BRPL Kilokari BESS telemetry.

    python poll_bess.py [interval_seconds]

Delhi SLDC publishes only the *instantaneous* state of the battery — there
is no historical endpoint — so the only way to build a dispatch history is
to sample it ourselves. Run this alongside the platform; every reading
lands in the bess_telemetry table and feeds validate/bess_validate.py.
"""
import sys
import time
from datetime import datetime

from ingest import bess

INTERVAL = int(sys.argv[1]) if len(sys.argv) > 1 else 300


def main():
    print(f"polling BRPL BESS every {INTERVAL}s — Ctrl-C to stop")
    misses = 0
    while True:
        try:
            row, meta = bess.poll_once()
            if meta["live"]:
                misses = 0
                state = ("DISCHARGE" if row["discharge_mw"] > 0.05 else
                         "CHARGE" if row["discharge_mw"] < -0.05 else "idle")
                print(f"{row['ts']}  {state:9s} {row['discharge_mw']:+7.2f} MW  "
                      f"SoC {row['soc_pct']:5.1f}%  "
                      f"({len(bess.read_history())} readings stored)",
                      flush=True)
            else:
                misses += 1
                print(f"{datetime.now():%Y-%m-%d %H:%M:%S}  fetch failed "
                      f"({misses} in a row): {meta.get('error', '')[:80]}",
                      flush=True)
        except KeyboardInterrupt:
            print("\nstopped")
            return
        except Exception as e:  # keep the loop alive across anything transient
            print(f"{datetime.now():%Y-%m-%d %H:%M:%S}  loop error: {e}", flush=True)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
