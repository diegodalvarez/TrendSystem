"""Convert Bloomberg origin files into the versioned store.

    data/store/bbg/<dataset>/1day/<TICKER>/<origin_key>/v0001/data.parquet
                                                          /manifest.json

One series = one ticker from one origin file. Nothing is merged across files.
Origin files are only opened for reading.

Versioning
    * An origin whose size and modified time match the last build is skipped.
    * If it changed, it is re-read. Each series whose content changed gets the
      next version (v0002, ...); series whose content is identical keep their
      current version. Old versions are never deleted.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import BUILDER_VERSION
from .bbg_readers import read_origin
from .naming import folder_name, order_fields, parse_contract, parse_generic

ADJUSTMENT = "back_adjusted (method unknown, per origin)"
FREQUENCY = "1day"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:                     # read-only
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def frame_hash(df: pd.DataFrame) -> str:
    h = hashlib.sha256()
    h.update("|".join(map(str, df.columns)).encode())
    h.update(pd.util.hash_pandas_object(df, index=False).values.tobytes())
    return h.hexdigest()


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    df.to_parquet(path, index=False)


def read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


def origin_key(source: str, rel_path: str, sheet: str | None = None) -> str:
    """bbg_adhoc_futures + Prices1.parquet -> bbg_adhoc_futures__Prices1_parquet"""
    base = f"{source}__{rel_path}".replace("\\", "/").replace("/", "__").replace(".", "_")
    if sheet:
        base += f"__{sheet}"
    return folder_name(base)


def _inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Origins from the config
# ---------------------------------------------------------------------------

def expand_origins(cfg: dict, sources: dict) -> list[dict]:
    """Turn config entries (file names or globs) into concrete origin files.

    The first entry that matches a file wins, so specific names go above globs.
    """
    out, seen = [], set()
    for entry in cfg.get("origins", []):
        src_name = entry["source"]
        src = sources.get(src_name)
        if not src or not src.get("root"):
            out.append({**entry, "_problem": f"source {src_name} has no root in sources.yaml"})
            continue
        root = Path(src["root"])
        pattern = entry["file"]
        if any(ch in pattern for ch in "*?["):
            if not root.exists():
                out.append({**entry, "_problem": f"root not found: {root}"})
                continue
            matches = sorted(p for p in root.iterdir() if p.is_file() and fnmatch.fnmatch(p.name, pattern))
        else:
            matches = [root / pattern]
        for p in matches:
            key = (src_name, p.name)
            if key in seen:
                continue
            seen.add(key)
            item = {**entry, "file": p.name, "_path": p, "_root": root}
            if not p.exists():
                item["_problem"] = f"file not found: {p}"
            out.append(item)
    return out


# ---------------------------------------------------------------------------
# Series splitting
# ---------------------------------------------------------------------------

def split_series(long: pd.DataFrame, entry: dict) -> list[dict]:
    """Long frame -> list of series dicts {dataset, vendor_ticker, frame, ...}."""
    dataset = entry.get("dataset", "index_px")
    # Only futures files are searched for generic tickers: strategy-index names
    # such as 'GIN6 Index' or 'EG00 Index' look like generics but are not.
    detect = entry.get("detect_generics", dataset == "fut_generic")
    series = []

    if dataset == "fut_contract":
        parsed = {t: parse_contract(t) for t in long["ticker"].unique()}
        long = long.copy()
        long["root"] = long["ticker"].map(
            lambda t: f"{parsed[t][0]} {parsed[t][3]}" if parsed[t] else "_unparsed"
        )
        for root, g in long.groupby("root", sort=True):
            wide = g.pivot(index=["date", "ticker"], columns="field", values="value")
            wide = wide[order_fields(wide.columns)].reset_index().rename(columns={"ticker": "contract"})
            wide.columns.name = None
            wide = wide.sort_values(["contract", "date"]).reset_index(drop=True)
            series.append({
                "dataset": "fut_contract", "vendor_ticker": root, "root": root,
                "generic_n": None, "frame": wide,
                "raw_fields": sorted(g["raw_field"].unique()),
                "contracts": int(g["ticker"].nunique()),
            })
        return series

    for ticker, g in long.groupby("ticker", sort=True):
        ds, root, n = dataset, None, None
        gen = parse_generic(ticker) if detect else None
        if gen:
            ds, root, n = "fut_generic", f"{gen[0]} {gen[2]}", gen[1]
        wide = g.pivot(index="date", columns="field", values="value")
        wide = wide[order_fields(wide.columns)].sort_index().reset_index()
        wide.columns.name = None
        series.append({
            "dataset": ds, "vendor_ticker": ticker, "root": root, "generic_n": n,
            "frame": wide, "raw_fields": sorted(g["raw_field"].unique()),
        })
    return series


def field_ranges(frame: pd.DataFrame) -> dict:
    out = {}
    for col in frame.columns:
        if col in ("date", "contract"):
            continue
        s = frame.loc[frame[col].notna(), "date"]
        if s.empty:
            continue
        out[col] = {"first": s.min().strftime("%Y-%m-%d"), "last": s.max().strftime("%Y-%m-%d"), "n": int(len(s))}
    return out


# ---------------------------------------------------------------------------
# Writing one series
# ---------------------------------------------------------------------------

def series_dir(store: Path, s: dict, okey: str) -> Path:
    return store / s["dataset"] / FREQUENCY / folder_name(s["vendor_ticker"]) / okey


def latest_version(sdir: Path):
    if not sdir.exists():
        return 0, None
    versions = sorted(int(p.name[1:]) for p in sdir.iterdir() if p.is_dir() and p.name.startswith("v") and p.name[1:].isdigit())
    if not versions:
        return 0, None
    v = versions[-1]
    m = sdir / f"v{v:04d}" / "manifest.json"
    return v, (read_json(m) if m.exists() else None)


def write_series(store: Path, s: dict, okey: str, origin_info: dict, read_stats: dict, dry_run: bool) -> dict:
    frame = s["frame"]
    content = frame_hash(frame)
    sdir = series_dir(store, s, okey)
    v, last = latest_version(sdir)
    dates = frame["date"]
    row = {
        "dataset": s["dataset"], "vendor_ticker": s["vendor_ticker"], "root": s["root"],
        "generic_n": s["generic_n"], "origin_key": okey, "origin_file": origin_info["rel_path"],
        "fields": ";".join(c for c in frame.columns if c not in ("date", "contract")),
        "n_rows": int(len(frame)),
        "first_date": dates.min().strftime("%Y-%m-%d") if len(frame) else None,
        "last_date": dates.max().strftime("%Y-%m-%d") if len(frame) else None,
        "path": str(sdir),
    }
    if last and last.get("content_hash") == content:
        row.update(status="unchanged", version=v)
        return row
    new_v = v + 1
    row.update(status=("new" if v == 0 else "updated"), version=new_v)
    if dry_run:
        row["status"] = "would_be_" + row["status"]
        return row

    manifest = {
        "series_id": f"bbg|{s['dataset']}|{FREQUENCY}|{s['vendor_ticker']}|{okey}",
        "source": "bbg",
        "dataset": s["dataset"],
        "frequency": FREQUENCY,
        "adjustment": ADJUSTMENT,
        "vendor_ticker": s["vendor_ticker"],
        "root": s["root"],
        "generic_n": s["generic_n"],
        "contracts": s.get("contracts"),
        "version": new_v,
        "previous_version": v or None,
        "n_rows": row["n_rows"],
        "first_date": row["first_date"],
        "last_date": row["last_date"],
        "fields": field_ranges(frame),
        "raw_fields": s["raw_fields"],
        "content_hash": content,
        "origin": origin_info,
        "read": {k: read_stats.get(k) for k in (
            "layout", "ticker_column", "field_column", "fixed_field", "sheets",
            "rows_read", "rows_kept", "dropped_bad_date", "dropped_empty_value",
            "dropped_non_numeric", "dropped_duplicates", "field_clashes")},
        "cleaning": "none (straight conversion)",
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "builder_version": BUILDER_VERSION,
    }
    final = sdir / f"v{new_v:04d}"
    tmp = sdir / f".tmp_v{new_v:04d}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    write_parquet(frame, tmp / "data.parquet")
    write_json(manifest, tmp / "manifest.json")
    os.replace(tmp, final)                         # version appears only when complete
    return row


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build(cfg: dict, sources: dict, repo_root: Path, dry_run: bool = True,
          only: list | None = None, force: bool = False, log=print) -> pd.DataFrame:
    store = (repo_root / cfg.get("store", "data/store/bbg")).resolve()
    data_dir = (repo_root / "data").resolve()
    if not _inside(store, data_dir):
        raise RuntimeError(f"store {store} must be inside {data_dir}")
    for src in sources.values():                   # never write next to the origins
        if src.get("root") and _inside(store, Path(src["root"])):
            raise RuntimeError(f"store {store} is inside a source folder {src['root']}")

    state_dir = store / "_origins"
    log_dir = repo_root / cfg.get("log_dir", "data/store/_logs")
    origins = expand_origins(cfg, sources)
    only = [o.lower() for o in (only or [])]

    rows, origin_rows = [], []
    t_all = time.time()
    todo = [o for o in origins if o.get("include", True)]
    log(f"{'DRY RUN - nothing is written' if dry_run else 'WRITING'} | store: {store}")
    log(f"origins in config: {len(origins)}, included: {len(todo)}")

    for i, entry in enumerate(todo, 1):
        name = f"{entry['source']}/{entry['file']}"
        okey = origin_key(entry["source"], entry["file"])
        if only and not any(o in name.lower() or o == okey.lower() for o in only):
            continue
        orow = {"origin": name, "origin_key": okey, "dataset": entry.get("dataset"), "status": None}
        if entry.get("_problem"):
            orow["status"] = "problem"
            orow["note"] = entry["_problem"]
            log(f"[{i}/{len(todo)}] {name}: {entry['_problem']}")
            origin_rows.append(orow)
            continue

        path: Path = entry["_path"]
        st = path.stat()
        state_file = state_dir / f"{okey}.json"
        state = read_json(state_file) if state_file.exists() else None
        if state and not force and state.get("size") == st.st_size and state.get("mtime") == st.st_mtime:
            orow.update(status="unchanged_origin", n_series=state.get("n_series"))
            log(f"[{i}/{len(todo)}] {name}: unchanged since {state.get('built_at')}, skipped")
            origin_rows.append(orow)
            continue

        t0 = time.time()
        try:
            long, stats = read_origin(path, entry)
            series = split_series(long, entry)
        except Exception as exc:                    # keep going; report at the end
            orow.update(status="error", note=f"{type(exc).__name__}: {exc}")
            log(f"[{i}/{len(todo)}] {name}: ERROR {orow['note']}")
            origin_rows.append(orow)
            continue

        digest = sha256_file(path)
        origin_info = {
            "source": entry["source"], "rel_path": entry["file"], "abs_path": str(path),
            "size": st.st_size, "mtime": st.st_mtime,
            "mtime_text": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
            "sha256": digest,
        }
        counts = {}
        for s in series:
            r = write_series(store, s, okey, origin_info, stats, dry_run)
            counts[r["status"]] = counts.get(r["status"], 0) + 1
            rows.append(r)
        orow.update(
            status="read", n_series=len(series), rows_read=stats.get("rows_read"),
            rows_kept=stats.get("rows_kept"), dropped_non_numeric=stats.get("dropped_non_numeric"),
            dropped_duplicates=stats.get("dropped_duplicates"), dropped_bad_date=stats.get("dropped_bad_date"),
            layout=stats.get("layout"), raw_fields=";".join(stats.get("raw_fields", [])),
            field_clashes=";".join(stats.get("field_clashes", [])),
            seconds=round(time.time() - t0, 1),
            **{f"series_{k}": v for k, v in counts.items()},
        )
        log(f"[{i}/{len(todo)}] {name}: {stats.get('layout')}, {len(series)} series, "
            f"{stats.get('rows_kept'):,} values kept of {stats.get('rows_read'):,} read, {counts} "
            f"({orow['seconds']}s)")
        if stats.get("dropped_non_numeric"):
            log(f"      non-numeric values dropped: {stats['dropped_non_numeric']} e.g. {stats.get('non_numeric_examples')}")
        if not dry_run:
            state_dir.mkdir(parents=True, exist_ok=True)
            write_json({**origin_info, "n_series": len(series), "built_at": datetime.now().isoformat(timespec="seconds")}, state_file)
        origin_rows.append(orow)

    series_log = pd.DataFrame(rows)
    origin_log = pd.DataFrame(origin_rows)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir.mkdir(parents=True, exist_ok=True)
    tag = "dryrun" if dry_run else "build"
    series_log.to_csv(log_dir / f"bbg_{tag}_{stamp}_series.csv", index=False)
    origin_log.to_csv(log_dir / f"bbg_{tag}_{stamp}_origins.csv", index=False)
    log(f"done in {time.time() - t_all:.0f}s | logs: {log_dir / f'bbg_{tag}_{stamp}_*.csv'}")
    if len(series_log):
        log(series_log.groupby(["dataset", "status"]).size().to_string())
    problems = origin_log[origin_log["status"].isin(["error", "problem"])] if len(origin_log) else origin_log
    if len(problems):
        log(f"{len(problems)} origin(s) with problems:")
        log(problems[["origin", "status", "note"]].to_string(index=False))
    return series_log
