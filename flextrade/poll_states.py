"""The 15-minute collector — every first-party feed, one process.

This exists because the feeds were drifting apart. MERIT ran on the 15-minute
StatesPoller and had accrued 31,557 rows; UP and Karnataka were only touched by
the daily pipeline and had FIVE rows each. A first-party feed that is polled
once a day is not a data asset, it is a screenshot.

Everything here is snapshot-only upstream — there is no history endpoint for
any of it — so depth exists only if we collect it. Every block missed is a
block that cannot be bought back, which is why each source is isolated: one
failing source must never stop the others from being written.
"""
from __future__ import annotations

import sys
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _merit():
    from ingest import states
    snap, meta = states.get_india_snapshot()
    n = snap["national"].get("demand_met_mw")
    return f"{len(snap['states'])} states, {n:,.0f} MW national" if n else \
        f"{len(snap['states'])} states"


def _upsldc():
    from ingest import upsldc
    s = upsldc.poll()
    return (f"demand {s['demand_met_mw']:,.0f} MW, "
            f"deviation {s.get('deviation_signed_mw', 0):+,.0f} MW")


def _kptcl():
    from ingest import kptcl
    s = kptcl.poll()
    return (f"demand {s['demand_mw']:,.0f} MW "
            f"(MERIT cross-check {s.get('crosscheck_pct')}%)")


def _coal():
    # daily report with a publication lag; cheap to re-check and it self-skips
    from datetime import date, timedelta
    from ingest import coal
    for back in range(1, 5):
        d = date.today() - timedelta(days=back)
        df = coal.fetch(d)
        if len(df):
            coal.store_day(df)
            return (f"{d} {len(df)} plants, "
                    f"{df['days_of_stock'].median():.1f} median days of stock")
    return "no report published in the last 4 days"


def _area_price():
    from ingest import vidyutpravah
    s = vidyutpravah.poll()
    return (f"{s['n_areas']} areas, "
            + ("uniform" if s["uniform"] else f"SPLIT spread Rs {s['spread_rs_mwh']}"))


SOURCES = [
    ("merit", _merit),          # 23 states, the breadth layer
    ("upsldc", _upsldc),        # UP: schedule/drawal/deviation, largest load
    ("kptcl", _kptcl),          # Karnataka: demand/drawal/frequency
    ("area_price", _area_price),  # state-wise clearing price when the market splits
    ("coal", _coal),            # supply side: the strongest exogenous signal found
]


def main() -> int:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ok = 0
    for name, fn in SOURCES:
        try:
            print(f"{stamp}  {name:11s} OK   {fn()}")
            ok += 1
        except Exception as e:
            # one dead source must not cost us the others
            print(f"{stamp}  {name:11s} FAIL {type(e).__name__}: {str(e)[:140]}")
    print(f"{stamp}  ---> {ok}/{len(SOURCES)} sources collected")
    # non-zero only if EVERYTHING failed, which usually means no network
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
