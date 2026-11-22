"""Build the Bloomberg part of the price store (step 2.1).

Reads every Bloomberg origin file switched on in config/bbg_store.yaml and writes
one series per ticker per origin file to data/store/bbg/... as parquet +
manifest.json. Origin files are only read. Nothing is merged across files.

Set the options below and run the file (F5 in Spyder). No arguments needed.

Needs: pandas, pyarrow, pyyaml, openpyxl (only for Excel origins).
"""

from __future__ import annotations

import sys
from pathlib import Path

# =============================================================================
# SETTINGS - edit these, then run the file
# =============================================================================

DRY_RUN = False     # True  = read everything, write only the log in data/store/_logs
                    # False = write the store to data/store/bbg

ONLY = []           # [] = every origin switched on in config/bbg_store.yaml
                    # or a few file names (or parts of names) to test, e.g. ["Prices1", "FuturesPX"]

FORCE = False       # True = re-read origins even if their size and date did not change

LIST_ONLY = False   # True = just print the origins from the config and stop

# =============================================================================

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from pricedata.bbg_store import build, expand_origins, origin_key  # noqa: E402
from pricedata.settings import load_bbg_store_config, load_sources  # noqa: E402


def main():
    cfg = load_bbg_store_config(REPO)
    sources = load_sources(REPO)

    if LIST_ONLY:
        for o in expand_origins(cfg, sources):
            flag = "in " if o.get("include", True) else "out"
            note = o.get("_problem", "")
            print(f"{flag} {o['source']:<22} {o['file']:<45} {o.get('dataset', ''):<12} "
                  f"{origin_key(o['source'], o['file'])} {note}")
        return

    build(cfg, sources, REPO, dry_run=DRY_RUN, only=ONLY, force=FORCE)


if __name__ == "__main__":
    main()
