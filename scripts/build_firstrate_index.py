"""Build the FirstRate pointer index (step 2.2).

Goes through every FirstRate zip (futures, FX, ETFs) WITHOUT extracting anything
and records one pointer per file inside: zip, member, symbol, frequency,
adjustment, contract, first and last timestamp, row count, size and checksum.
Zips are only opened for reading. Output goes to data/index/.

Set the options below and run the file (F5 in Spyder). No arguments needed.

Needs: pandas, pyyaml (pyarrow optional, for the parquet copy).
"""

from __future__ import annotations

import sys
from pathlib import Path

# =============================================================================
# SETTINGS - edit these, then run the file
# =============================================================================

READ_LAST_LINE = True   # True  = first AND last timestamp + row count. Has to decompress
                        #         every file (~62 GB of zips); expect a long run, it resumes
                        #         where it stopped if interrupted.
                        # False = first timestamp only, a few minutes. Can be upgraded later
                        #         by running again with True.

GROUPS = {              # which FirstRate folders (names from config/sources.yaml)
    "futures": "firstrate_futures",
    "fx": "firstrate_fx",
    "etf": "firstrate_etf",
}

ONLY_ZIPS = []          # [] = all zips; or parts of zip names to test, e.g. ["fut_1day", "fx_full_1day"]

WORKERS = 4             # zips read at the same time

FORCE = False           # True = re-read zips even if their size and date did not change

# =============================================================================

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pricedata.firstrate_index import build  # noqa: E402
from pricedata.settings import load_sources  # noqa: E402


def main():
    sources = load_sources(REPO)
    build(sources, REPO, GROUPS, read_last=READ_LAST_LINE, workers=WORKERS,
          only_zips=ONLY_ZIPS, force=FORCE)


if __name__ == "__main__":
    main()
