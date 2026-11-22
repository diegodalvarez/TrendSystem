"""Build the series index (step 2.3).

One row per series from both sources: every Bloomberg series in data/store/bbg
and every FirstRate pointer in data/index/firstrate_pointers.csv.gz, linked to
the catalogue (instrument_id, asset class), with date span and the default flag
(longest span; overrides in config/preferences.yaml).

Only reads what is already in data/. Output: data/index/series_index.{csv.gz,parquet,xlsx}

Set the options below and run the file (F5 in Spyder). No arguments needed.
Close series_index.xlsx in Excel before running.
"""

from __future__ import annotations

import sys
from pathlib import Path

# =============================================================================
# SETTINGS - edit these, then run the file
# =============================================================================

WRITE_XLSX = True       # False = only the .csv.gz / .parquet (faster)

# =============================================================================

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pricedata.series_index import build  # noqa: E402
from pricedata.settings import load_yaml  # noqa: E402


def main():
    prefs = load_yaml(REPO / "config" / "preferences.yaml")
    build(REPO, prefs, write_xlsx_view=WRITE_XLSX)


if __name__ == "__main__":
    main()
