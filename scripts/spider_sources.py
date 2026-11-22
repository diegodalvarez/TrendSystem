"""Crawl every folder listed in config/sources.yaml and save what is in it.

It never copies, moves, extracts or changes any file. What it reads depends on
each source's `mode` in sources.yaml:

  every source   folder and file names, sizes, modified dates
                 + for .zip files: the zip's table of contents (member names
                   and sizes). Nothing is extracted.
  mode: full     + for .parquet / .xlsx / .xlsm / .csv: structure only -
                   column names, types, row counts, the date range, and the
                   names in identifier columns (security / ticker / field /
                   variable) or Excel header rows. Price values are never read.
  mode: names_only   nothing beyond the first two lines above.

Which sources run, and how many files get opened, comes from the `spider:`
block in sources.yaml (groups, max_files_per_source, sample). Edit that block
to work through the sources step by step, or override it on the command line.

Run from anywhere (Anaconda prompt):

    python A:\\GitHub\\TrendSystem\\scripts\\spider_sources.py --list       # show what WOULD run, crawl nothing
    python A:\\GitHub\\TrendSystem\\scripts\\spider_sources.py              # run with the yaml settings
    python A:\\GitHub\\TrendSystem\\scripts\\spider_sources.py --limit 10   # open 10 files per source
    python A:\\GitHub\\TrendSystem\\scripts\\spider_sources.py --group firstrate --limit 0
    python A:\\GitHub\\TrendSystem\\scripts\\spider_sources.py firstrate_fx  # one or more sources by name
    ... --all-groups    ignore the groups setting
    ... --sample first  pick the first N files instead of spreading
    ... --no-zip        skip zip tables of contents
    ... --no-inspect    skip file structure even for mode: full

Output goes to a new timestamped folder each run, so nothing is overwritten:

    data/catalog/inventory/<YYYYMMDD_HHMMSS>/
        sources.csv                 one row per source: root, status, counts, size, time taken
        folders.csv                 one row per folder: file count, size, extensions, date range
        run_settings.csv            the groups / limit / sample used for this run
        files_<source>.csv          one row per file: path, name, extension, size, modified, opened (yes/blank)
        zip_contents_<source>.csv   one row per file inside each zip (if the source has zips)
        structure_<source>.csv      one row per file (or Excel sheet): columns, rows, date range
        identifiers_<source>.csv    one row per file x identifier name (ticker / field / header)
        errors.csv                  anything that could not be read (only if there were any)

Needs: Python 3.8+, pyyaml. For mode: full also pandas, pyarrow, openpyxl (all in Anaconda).
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import fnmatch
import os
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config" / "sources.yaml"

# Column names treated as identifiers (their distinct values are recorded) or dates.
ID_COLUMNS = {"security", "ticker", "variable", "field", "symbol", "name", "file", "window"}
DATE_COLUMNS = {"date", "datetime", "timestamp", "time", "trade_date"}
MAX_HEADER_CELLS = 2000        # Excel header cells recorded per sheet
MAX_IDS_PER_COLUMN = 50000     # safety cap on distinct identifiers per column

INSPECT_EXT = {"parquet", "xlsx", "xlsm", "csv"}


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------

def fmt_time(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def fmt_size(n: float) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024 or unit == "TB":
            return "{:.1f} {}".format(n, unit) if unit != "B" else "{} B".format(int(n))
        n /= 1024.0
    return str(n)


def matches_any(rel: str, patterns) -> bool:
    return any(fnmatch.fnmatch(rel, p) for p in patterns)


def dir_excluded(rel_dir: str, patterns) -> bool:
    """A folder is skipped if a pattern names it, e.g. 'FX/**' or 'FX'."""
    for p in patterns:
        base = p[:-3] if p.endswith("/**") else p
        if fnmatch.fnmatch(rel_dir, base) or fnmatch.fnmatch(rel_dir + "/", p):
            return True
    return False


def load_sources(config_path: Path):
    """-> (sources dict, spider settings dict)"""
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return raw.get("sources", {}) or {}, raw.get("spider", {}) or {}


def short_str(v) -> str:
    if v is None:
        return ""
    s = str(v)
    return s[:19] if len(s) > 19 and s[4:5] == "-" else s   # trim timestamps to seconds


class CsvSink:
    """Opens a CSV lazily so empty outputs are not written."""

    def __init__(self, path: Path, header):
        self.path, self.header = path, header
        self._f = None
        self._w = None
        self.rows = 0

    def write(self, row):
        if self._w is None:
            self._f = open(self.path, "w", newline="", encoding="utf-8")
            self._w = csv.writer(self._f)
            self._w.writerow(self.header)
        self._w.writerow(row)
        self.rows += 1

    def close(self):
        if self._f is not None:
            self._f.close()


# -----------------------------------------------------------------------------
# readers: zip table of contents and file structure (never price values)
# -----------------------------------------------------------------------------

def read_zip_toc(path: Path):
    out = []
    with zipfile.ZipFile(path) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            try:
                mod = dt.datetime(*info.date_time).strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                mod = ""
            ext = os.path.splitext(info.filename)[1].lower().lstrip(".")
            out.append((info.filename, ext, info.file_size, info.compress_size, mod))
    return out


def inspect_parquet(path: Path):
    """-> list of structure dicts, list of (column, value, count)"""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(str(path))
    schema = pf.schema_arrow
    cols = list(schema.names)
    types = [str(schema.field(c).type) for c in cols]
    lower = {c.lower(): c for c in cols}

    date_col = next((lower[c] for c in lower if c in DATE_COLUMNS), "")
    id_cols = [lower[c] for c in lower if c in ID_COLUMNS]

    min_d = max_d = ""
    ids = []
    read_cols = ([date_col] if date_col else []) + id_cols
    if read_cols:
        # ignore_metadata: keep every column as a plain column, even if pandas
        # originally saved it as the index (e.g. a 'Date' index)
        df = pf.read(columns=read_cols).to_pandas(ignore_metadata=True)
        if date_col:
            s = df[date_col].dropna()
            if len(s):
                min_d, max_d = short_str(s.min()), short_str(s.max())
        for c in id_cols:
            vc = df[c].astype(str).value_counts()
            for val, n in list(vc.items())[:MAX_IDS_PER_COLUMN]:
                ids.append((c, val, int(n)))

    struct = [{
        "sheet": "",
        "n_rows": pf.metadata.num_rows,
        "n_columns": len(cols),
        "columns": "; ".join(cols),
        "dtypes": "; ".join(types),
        "layout": guess_layout(cols),
        "date_column": date_col,
        "min_date": min_d,
        "max_date": max_d,
        "id_columns": "; ".join(id_cols),
        "n_ids": sum(1 for x in ids),
    }]
    return struct, ids


def inspect_excel(path: Path):
    import openpyxl

    wb = openpyxl.load_workbook(str(path), read_only=True, data_only=True)
    structs, ids = [], []
    try:
        for ws in wb.worksheets:
            rows = ws.iter_rows(values_only=True)
            header = next(rows, None) or ()
            header = [("" if h is None else str(h)) for h in header]
            # first column = dates in Bloomberg exports; read only that column
            n_rows, min_d, max_d = 0, None, None
            for r in ws.iter_rows(min_row=2, min_col=1, max_col=1, values_only=True):
                v = r[0] if r else None
                if v is None:
                    continue
                n_rows += 1
                if isinstance(v, (dt.date, dt.datetime)):
                    min_d = v if min_d is None or v < min_d else min_d
                    max_d = v if max_d is None or v > max_d else max_d
            heads = [h for h in header[1:] if h][:MAX_HEADER_CELLS]
            for h in heads:
                ids.append(("{}!header".format(ws.title), h, 1))
            structs.append({
                "sheet": ws.title,
                "n_rows": n_rows,
                "n_columns": len(header),
                "columns": "; ".join(header[:20]) + (" ..." if len(header) > 20 else ""),
                "dtypes": "",
                "layout": "wide" if len(header) > 2 else guess_layout(header),
                "date_column": header[0] if header else "",
                "min_date": short_str(min_d),
                "max_date": short_str(max_d),
                "id_columns": "header row",
                "n_ids": len(heads),
            })
    finally:
        wb.close()
    return structs, ids


def inspect_csv(path: Path):
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, []) or []
        n_rows = sum(1 for _ in reader)
    return [{
        "sheet": "",
        "n_rows": n_rows,
        "n_columns": len(header),
        "columns": "; ".join(header),
        "dtypes": "",
        "layout": guess_layout(header),
        "date_column": next((h for h in header if h.lower() in DATE_COLUMNS), ""),
        "min_date": "",
        "max_date": "",
        "id_columns": "",
        "n_ids": 0,
    }], []


def guess_layout(cols) -> str:
    low = {str(c).lower() for c in cols}
    if {"field", "value"} <= low:
        return "long_by_field"
    if {"variable", "value"} <= low:
        return "melted"
    if {"open", "high", "low", "close"} <= low:
        return "ohlc"
    if "px_last" in low and ("security" in low or "ticker" in low):
        return "long_px_last"
    return ""


INSPECTORS = {"parquet": inspect_parquet, "xlsx": inspect_excel, "xlsm": inspect_excel, "csv": inspect_csv}

STRUCT_COLS = ["sheet", "n_rows", "n_columns", "columns", "dtypes", "layout",
               "date_column", "min_date", "max_date", "id_columns", "n_ids"]


# -----------------------------------------------------------------------------
# choosing which files to open when there is a limit
# -----------------------------------------------------------------------------

def spread_order(n: int):
    """0, n-1, middle, quarters, ... : any prefix is spread across the list,
    and a longer prefix always contains a shorter one."""
    if n <= 0:
        return []
    order, seen = [0], {0}
    if n > 1:
        order.append(n - 1)
        seen.add(n - 1)
    queue = [(0, n - 1)]
    while queue:
        nxt = []
        for a, b in queue:
            if b - a < 2:
                continue
            m = (a + b) // 2
            if m not in seen:
                order.append(m)
                seen.add(m)
            nxt.extend([(a, m), (m, b)])
        queue = nxt
    order.extend(i for i in range(n) if i not in seen)
    return order


def pick_files(candidates, limit: int, sample: str):
    """candidates: list of (rel, ext, path) sorted by rel. Returns set of rel."""
    if limit <= 0 or len(candidates) <= limit:
        return {c[0] for c in candidates}
    if sample == "first":
        return {c[0] for c in candidates[:limit]}
    by_ext = {}
    for c in candidates:
        by_ext.setdefault(c[1], []).append(c)
    queues = {e: [lst[i] for i in spread_order(len(lst))] for e, lst in sorted(by_ext.items())}
    picked = []
    while len(picked) < limit:
        for e in sorted(queues):
            if queues[e] and len(picked) < limit:
                picked.append(queues[e].pop(0))
    return {c[0] for c in picked}


# -----------------------------------------------------------------------------
# crawl one source
# -----------------------------------------------------------------------------

def crawl(name: str, spec: dict, out_dir: Path, do_zip: bool, do_inspect: bool,
          limit: int, sample: str):
    root_str = (spec.get("root") or "").strip()
    include = list(spec.get("include") or [])
    exclude = list(spec.get("exclude") or [])
    mode = spec.get("mode", "full")
    inspect = do_inspect and mode == "full"

    summary = {
        "source": name, "group": spec.get("group", ""), "root": root_str,
        "vendor": spec.get("vendor", ""), "category": spec.get("category", ""),
        "mode": mode, "status": "",
        "n_folders": 0, "n_files": 0, "total_bytes": 0, "total_size": "",
        "first_modified": "", "last_modified": "", "extensions": "",
        "n_openable": 0, "limit": limit if limit > 0 else "all", "n_opened": 0,
        "n_zip_members": 0, "n_inspected": 0, "seconds": 0.0, "n_errors": 0,
    }

    if not root_str:
        summary["status"] = "empty (no root)"
        return summary, [], []
    root = Path(root_str)
    if not root.is_dir():
        summary["status"] = "MISSING (folder not found)"
        return summary, [], []

    t0 = time.time()
    folder_rows, error_rows, file_rows = [], [], []
    all_ext = Counter()
    min_mt = max_mt = None

    # ---- pass 1: list every folder and file (names, sizes, dates only) ----
    stack = [(root, "", 0)]
    while stack:
        path, rel_dir, depth = stack.pop()
        n_files = n_sub = size = 0
        ext_counts = Counter()
        f_min = f_max = None

        try:
            with os.scandir(path) as it:
                entries = sorted(it, key=lambda e: e.name.lower())
        except OSError as exc:
            error_rows.append({"source": name, "path": rel_dir or ".", "stage": "list folder", "error": repr(exc)})
            continue

        subdirs = []
        for e in entries:
            rel = (rel_dir + "/" + e.name) if rel_dir else e.name
            try:
                if e.is_dir(follow_symlinks=False):
                    if not dir_excluded(rel, exclude):
                        subdirs.append((Path(e.path), rel, depth + 1))
                        n_sub += 1
                    continue
                if not e.is_file(follow_symlinks=False):
                    continue
                if include and not matches_any(rel, include):
                    continue
                if exclude and matches_any(rel, exclude):
                    continue
                st = e.stat(follow_symlinks=False)
            except OSError as exc:
                error_rows.append({"source": name, "path": rel, "stage": "stat", "error": repr(exc)})
                continue

            ext = os.path.splitext(e.name)[1].lower().lstrip(".")
            file_rows.append([name, rel, rel_dir or ".", e.name, ext, st.st_size,
                              fmt_time(st.st_mtime), Path(e.path)])
            n_files += 1
            size += st.st_size
            ext_counts[ext or "(none)"] += 1
            f_min = st.st_mtime if f_min is None else min(f_min, st.st_mtime)
            f_max = st.st_mtime if f_max is None else max(f_max, st.st_mtime)

        stack.extend(reversed(subdirs))   # depth-first, alphabetical

        folder_rows.append({
            "source": name, "folder": rel_dir or ".", "depth": depth,
            "n_files": n_files, "n_subfolders": n_sub, "total_bytes": size,
            "total_size": fmt_size(size),
            "extensions": "; ".join("{}:{}".format(k, v) for k, v in ext_counts.most_common()),
            "first_modified": fmt_time(f_min) if f_min else "",
            "last_modified": fmt_time(f_max) if f_max else "",
        })
        summary["n_files"] += n_files
        summary["total_bytes"] += size
        all_ext.update(ext_counts)
        if f_min is not None:
            min_mt = f_min if min_mt is None else min(min_mt, f_min)
            max_mt = f_max if max_mt is None else max(max_mt, f_max)

    # ---- choose which files to open ----
    def openable(row):
        ext, fname = row[4], row[3]
        if do_zip and ext == "zip":
            return True
        return inspect and ext in INSPECT_EXT and not fname.startswith("~$")

    candidates = sorted(((r[1], r[4], r[7]) for r in file_rows if openable(r)), key=lambda c: c[0].lower())
    chosen = pick_files(candidates, limit, sample)
    summary["n_openable"] = len(candidates)
    summary["n_opened"] = len(chosen)

    # ---- pass 2: open the chosen files (zip contents / structure only) ----
    files_out = CsvSink(out_dir / "files_{}.csv".format(name),
                        ["source", "rel_path", "folder", "file_name", "extension", "size_bytes",
                         "modified", "opened"])
    zip_out = CsvSink(out_dir / "zip_contents_{}.csv".format(name),
                      ["source", "zip_rel_path", "member", "member_ext", "uncompressed_bytes",
                       "compressed_bytes", "member_modified"])
    struct_out = CsvSink(out_dir / "structure_{}.csv".format(name),
                         ["source", "rel_path", "file_type"] + STRUCT_COLS)
    ids_out = CsvSink(out_dir / "identifiers_{}.csv".format(name),
                      ["source", "rel_path", "column", "identifier", "n_rows"])
    try:
        for i, r in enumerate(file_rows):
            rel, ext, fpath = r[1], r[4], r[7]
            opened = rel in chosen
            files_out.write(r[:7] + ["yes" if opened else ""])
            if not opened:
                continue
            if ext == "zip":
                try:
                    for member in read_zip_toc(fpath):
                        zip_out.write([name, rel, *member])
                        summary["n_zip_members"] += 1
                except Exception as exc:  # noqa: BLE001
                    error_rows.append({"source": name, "path": rel, "stage": "zip contents", "error": repr(exc)})
            else:
                try:
                    structs, ids = INSPECTORS[ext](fpath)
                    for s_ in structs:
                        struct_out.write([name, rel, ext] + [s_[c] for c in STRUCT_COLS])
                    for col, val, n in ids:
                        ids_out.write([name, rel, col, val, n])
                    summary["n_inspected"] += 1
                except Exception as exc:  # noqa: BLE001
                    error_rows.append({"source": name, "path": rel, "stage": "structure", "error": repr(exc)})
            if summary["n_inspected"] and summary["n_inspected"] % 250 == 0:
                print(".", end="", flush=True)
    finally:
        for sink in (files_out, zip_out, struct_out, ids_out):
            sink.close()

    summary.update({
        "status": "ok" if not error_rows else "ok with errors",
        "n_folders": len(folder_rows),
        "total_size": fmt_size(summary["total_bytes"]),
        "first_modified": fmt_time(min_mt) if min_mt else "",
        "last_modified": fmt_time(max_mt) if max_mt else "",
        "extensions": "; ".join("{}:{}".format(k, v) for k, v in all_ext.most_common()),
        "seconds": round(time.time() - t0, 1),
        "n_errors": len(error_rows),
    })
    return summary, folder_rows, error_rows


def write_csv(path: Path, rows, columns):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="List folders and files in every configured source.")
    ap.add_argument("sources", nargs="*", help="source names to crawl (overrides groups)")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="path to sources.yaml")
    ap.add_argument("--out", default=str(REPO_ROOT / "data" / "catalog" / "inventory"),
                    help="folder for the output (a timestamped subfolder is made inside)")
    ap.add_argument("--list", action="store_true", help="show the sources and settings, crawl nothing")
    ap.add_argument("--group", nargs="+", help="groups to crawl (overrides spider.groups)")
    ap.add_argument("--all-groups", action="store_true", help="ignore spider.groups; crawl every source")
    ap.add_argument("--limit", type=int, help="files opened per source; 0 = all (overrides spider.max_files_per_source)")
    ap.add_argument("--sample", choices=["spread", "first"], help="how files are picked (overrides spider.sample)")
    ap.add_argument("--no-zip", action="store_true", help="do not read zip tables of contents")
    ap.add_argument("--no-inspect", action="store_true", help="do not read file structure")
    args = ap.parse_args(argv)

    sources, settings = load_sources(Path(args.config))

    limit = args.limit if args.limit is not None else int(settings.get("max_files_per_source", 0) or 0)
    sample = args.sample or settings.get("sample", "spread") or "spread"
    groups = [] if args.all_groups else (args.group or list(settings.get("groups") or []))

    unknown = [s for s in args.sources if s not in sources]
    if unknown:
        print("Unknown source(s): {}\nKnown: {}".format(", ".join(unknown), ", ".join(sources)))
        return 1
    known_groups = {s.get("group", "") for s in sources.values()}
    bad_groups = [g for g in groups if g not in known_groups]
    if bad_groups:
        print("Unknown group(s): {}\nKnown: {}".format(", ".join(bad_groups), ", ".join(sorted(known_groups))))
        return 1

    if args.sources:
        names = list(args.sources)
    else:
        names = [n for n, s in sources.items()
                 if (s.get("root") or "").strip() and (not groups or s.get("group") in groups)]

    print("Groups : {}".format(", ".join(groups) if groups and not args.sources else "(named sources)" if args.sources else "all"))
    print("Limit  : {} file(s) opened per source ({})".format(limit if limit > 0 else "all", sample))
    print("Sources: {}\n".format(len(names)))

    if args.list:
        print("{:<34} {:<10} {:<14} {:<11} {:<8} {:<4} {}".format("source", "group", "vendor", "mode", "status", "run", "root"))
        for name, spec in sources.items():
            root = (spec.get("root") or "").strip()
            status = "empty" if not root else ("ok" if Path(root).is_dir() else "MISSING")
            print("{:<34} {:<10} {:<14} {:<11} {:<8} {:<4} {}".format(
                name, spec.get("group", ""), spec.get("vendor", ""), spec.get("mode", ""),
                status, "yes" if name in names else "", root))
        return 0

    out_dir = Path(args.out) / dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries, all_folders, all_errors = [], [], []
    for name in names:
        print("Crawling {:<34}".format(name), end="", flush=True)
        summary, folders, errors = crawl(name, sources[name], out_dir,
                                         do_zip=not args.no_zip, do_inspect=not args.no_inspect,
                                         limit=limit, sample=sample)
        summaries.append(summary)
        all_folders.extend(folders)
        all_errors.extend(errors)
        print(" {:<26} {:>8} files {:>10}  opened {:>5}/{:<5} {:>8} in zips {:>4} errors  {}s".format(
            summary["status"], summary["n_files"], summary["total_size"],
            summary["n_opened"], summary["n_openable"], summary["n_zip_members"],
            summary["n_errors"], summary["seconds"]))

    run_info = [{"setting": "groups", "value": ", ".join(groups) or "all"},
                {"setting": "named_sources", "value": ", ".join(args.sources)},
                {"setting": "max_files_per_source", "value": limit if limit > 0 else "all"},
                {"setting": "sample", "value": sample},
                {"setting": "zip_contents", "value": not args.no_zip},
                {"setting": "inspect_structure", "value": not args.no_inspect},
                {"setting": "started", "value": out_dir.name}]
    write_csv(out_dir / "run_settings.csv", run_info, ["setting", "value"])
    write_csv(out_dir / "sources.csv", summaries, list(summaries[0].keys()) if summaries else ["source"])
    write_csv(out_dir / "folders.csv", all_folders,
              ["source", "folder", "depth", "n_files", "n_subfolders", "total_bytes",
               "total_size", "extensions", "first_modified", "last_modified"])
    if all_errors:
        write_csv(out_dir / "errors.csv", all_errors, ["source", "path", "stage", "error"])
        print("\n{} problem(s) written to errors.csv".format(len(all_errors)))

    print("\nSaved to {}".format(out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
