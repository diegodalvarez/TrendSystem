"""FirstRate pointer index (step 2.2).

FirstRate data stays where it is (~62 GB of zips). This module records one
POINTER per file inside each zip: which zip, which member, what it holds
(symbol, frequency, adjustment, contract) and its date range. Readers later
open the member straight from the zip; nothing is extracted or copied.

Member names (from the inventory):
    futures continuous  ES_full_1min_continuous_UNadjusted.txt   (also ratio_adjusted, absolute_adjusted)
    futures contracts   ES_H24_1min.txt                           (fut_contract_<freq>_archive.zip)
    fx                  EURUSD_full_1min.txt
    etf                 SPY_full_1min_adjsplitdiv.txt             (also adjsplit, UNADJUSTED)

File contents: no header, comma separated: date, open, high, low, close, volume.
Times are US Eastern, bar START time (FirstRate FAQ).

Output (data/index/):
    firstrate_pointers.csv.gz       one row per member (always written)
    firstrate_pointers.parquet      same rows (written if pyarrow is installed)
    firstrate_summary.csv           counts and date ranges per dataset / frequency / adjustment
    _firstrate_zips/<zip>.csv.gz    per-zip results, so an interrupted run resumes and
    _firstrate_zips/<zip>.json      unchanged zips (same size and date) are not re-read
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import BUILDER_VERSION

COLUMNS = ["date", "open", "high", "low", "close", "volume"]
TIMEZONE = "US/Eastern"
BAR_TIME = "start"

_FREQ = r"(?P<freq>\d+(?:min|hour|day))"
PATTERNS = [
    ("fut_continuous", re.compile(
        rf"^(?P<symbol>.+?)_full_{_FREQ}_continuous_(?P<adj>UNadjusted|ratio_adjusted|absolute_adjusted)\.txt$", re.I)),
    ("fut_contract", re.compile(
        rf"^(?P<symbol>.+?)_(?P<month>[FGHJKMNQUVXZ])(?P<yy>\d{{2}})_{_FREQ}\.txt$")),
    ("etf", re.compile(
        rf"^(?P<symbol>.+?)_full_{_FREQ}_(?P<adj>adjsplitdiv|adjsplit|UNADJUSTED)\.txt$", re.I)),
    ("fx_spot", re.compile(rf"^(?P<symbol>[A-Z0-9]+)_full_{_FREQ}\.txt$")),
]
ADJ_NAMES = {
    "unadjusted": "unadjusted",            # futures continuous and ETF UNADJUSTED
    "ratio_adjusted": "ratio",
    "absolute_adjusted": "absolute",
    "adjsplit": "split",
    "adjsplitdiv": "splitdiv",
}
SET_ASIDE = {("fut_continuous", "ratio"), ("fut_continuous", "absolute")}   # step 3 decision

CHUNK = 8 << 20


def parse_member(name: str, group: str) -> dict:
    """Member file name -> dataset, symbol, frequency, adjustment, contract."""
    base = name.rsplit("/", 1)[-1]
    for dataset, rx in PATTERNS:
        if group == "fx" and dataset != "fx_spot":
            continue
        if group == "etf" and dataset != "etf":
            continue
        if group == "futures" and dataset not in ("fut_continuous", "fut_contract"):
            continue
        m = rx.match(base)
        if not m:
            continue
        d = m.groupdict()
        adj = ADJ_NAMES.get((d.get("adj") or "").lower(), "none")
        if dataset == "fut_contract":
            adj = "unadjusted"
        out = {
            "dataset": dataset,
            "symbol": d["symbol"],
            "frequency": d["freq"],
            "adjustment": adj,
            "contract": f"{d['month']}{d['yy']}" if dataset == "fut_contract" else None,
            "contract_month": d.get("month"),
            "contract_year": (2000 + int(d["yy"])) if d.get("yy") else None,
        }
        out["status"] = "set_aside" if (dataset, adj) in SET_ASIDE else "active"
        return out
    return {"dataset": "unknown", "symbol": None, "frequency": None, "adjustment": None,
            "contract": None, "contract_month": None, "contract_year": None, "status": "unknown"}


def _first_field(line: str):
    return line.split(",", 1)[0].strip() if line else None


def scan_member(zf: zipfile.ZipFile, info: zipfile.ZipInfo, read_last: bool) -> dict:
    """First line always; last line and row count only when read_last (needs a full decompress)."""
    out = {"first_line": None, "last_line": None, "n_rows": None, "n_columns": None, "error": None}
    try:
        with zf.open(info) as f:
            if not read_last:
                head = f.read(65536)
                first = head.split(b"\n", 1)[0]
                out["first_line"] = first.decode("utf-8", "replace").strip()
            else:
                first = None
                tail = b""
                newlines = 0
                while True:
                    block = f.read(CHUNK)
                    if not block:
                        break
                    if first is None:
                        first = block.split(b"\n", 1)[0]
                    newlines += block.count(b"\n")
                    tail = (tail + block)[-4096:]
                lines = tail.rstrip(b"\r\n").split(b"\n")
                last = lines[-1] if lines else b""
                ends_with_newline = tail.endswith(b"\n")
                out["first_line"] = (first or b"").decode("utf-8", "replace").strip()
                out["last_line"] = last.decode("utf-8", "replace").strip()
                out["n_rows"] = newlines + (0 if ends_with_newline or not tail else 1)
        if out["first_line"]:
            out["n_columns"] = out["first_line"].count(",") + 1
    except Exception as exc:                        # damaged member: record and keep going
        out["error"] = f"{type(exc).__name__}: {exc}"
    out["first_ts"] = _first_field(out["first_line"])
    out["last_ts"] = _first_field(out["last_line"])
    return out


def zip_key(group: str, rel: str) -> str:
    return f"{group}__" + re.sub(r"[^A-Za-z0-9_.-]", "_", rel.replace("\\", "/").replace("/", "__"))


def index_zip(group: str, source: str, root: Path, zpath: Path, read_last: bool) -> pd.DataFrame:
    st = zpath.stat()
    rel = str(zpath.relative_to(root))
    rows = []
    with zipfile.ZipFile(zpath) as zf:              # opened for reading only
        for info in zf.infolist():
            if info.is_dir():
                continue
            meta = parse_member(info.filename, group)
            scan = scan_member(zf, info, read_last)
            rows.append({
                "source": "firstrate",
                "group": group,
                **meta,
                "member": info.filename,
                "zip_rel_path": rel,
                "zip_path": str(zpath),
                "source_name": source,
                "uncompressed_bytes": info.file_size,
                "compressed_bytes": info.compress_size,
                "member_crc": f"{info.CRC:08x}",
                "member_modified": datetime(*info.date_time).isoformat(sep=" "),
                "zip_size": st.st_size,
                "zip_mtime": st.st_mtime,
                **scan,
                "timezone": TIMEZONE if meta["dataset"] != "unknown" else None,
                "bar_time": BAR_TIME,
                "columns": ",".join(COLUMNS),
                "read_last_line": read_last,
            })
    return pd.DataFrame(rows)


def _summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    d = df.copy()
    d["first_date"] = d["first_ts"].astype(str).str[:10]
    d["last_date"] = d["last_ts"].astype(str).str[:10]
    g = d.groupby(["group", "dataset", "frequency", "adjustment", "status"], dropna=False)
    return g.agg(
        members=("member", "size"),
        symbols=("symbol", "nunique"),
        earliest=("first_date", "min"),
        latest=("last_date", lambda s: s[s != "None"].max() if (s != "None").any() else None),
        uncompressed_gb=("uncompressed_bytes", lambda s: round(s.sum() / 1e9, 2)),
        errors=("error", lambda s: int(s.notna().sum())),
    ).reset_index()


def build(sources: dict, repo_root: Path, groups: dict, read_last: bool = True, workers: int = 4,
          only_zips: list | None = None, force: bool = False, log=print) -> pd.DataFrame:
    """groups: {'futures': 'firstrate_futures', 'fx': 'firstrate_fx', 'etf': 'firstrate_etf'}"""
    out_dir = repo_root / "data" / "index"
    part_dir = out_dir / "_firstrate_zips"
    part_dir.mkdir(parents=True, exist_ok=True)
    only_zips = [o.lower() for o in (only_zips or [])]

    jobs = []
    for group, src_name in groups.items():
        src = sources.get(src_name) or {}
        root = Path(src.get("root") or "")
        if not src.get("root") or not root.exists():
            log(f"{group}: source {src_name} root not found ({src.get('root')}), skipped")
            continue
        for z in sorted(root.rglob("*.zip")):
            rel = str(z.relative_to(root))
            if only_zips and not any(o in rel.lower() for o in only_zips):
                continue
            jobs.append((group, src_name, root, z))
    total_bytes = sum(j[3].stat().st_size for j in jobs)
    log(f"{len(jobs)} zips, {total_bytes / 1e9:.1f} GB | read last line: {read_last} | workers: {workers}")

    lock = threading.Lock()
    done = {"bytes": 0, "n": 0}
    t0 = time.time()
    status_rows = []

    def run(job):
        group, src_name, root, z = job
        st = z.stat()
        key = zip_key(group, str(z.relative_to(root)))
        part, state_file = part_dir / f"{key}.csv.gz", part_dir / f"{key}.json"
        state = json.loads(state_file.read_text()) if state_file.exists() else None
        reuse = (state and not force and part.exists() and state["size"] == st.st_size
                 and state["mtime"] == st.st_mtime and (state["read_last_line"] or not read_last))
        if reuse:
            status = "unchanged"
        else:
            ts = time.time()
            df = index_zip(group, src_name, root, z, read_last)
            tmp = part.with_suffix(".tmp")
            df.to_csv(tmp, index=False, compression="gzip")
            os.replace(tmp, part)
            state_file.write_text(json.dumps({"zip": str(z), "size": st.st_size, "mtime": st.st_mtime,
                                              "read_last_line": read_last, "members": len(df),
                                              "seconds": round(time.time() - ts, 1),
                                              "built_at": datetime.now().isoformat(timespec="seconds")}))
            status = "indexed"
        with lock:
            done["bytes"] += st.st_size
            done["n"] += 1
            el = time.time() - t0
            eta = el / max(done["bytes"], 1) * (total_bytes - done["bytes"])
            log(f"[{done['n']}/{len(jobs)}] {group}/{z.name} {status} "
                f"({done['bytes'] / 1e9:.1f}/{total_bytes / 1e9:.1f} GB, {el / 60:.1f} min, ~{eta / 60:.0f} min left)")
        return {"group": group, "zip": str(z), "key": key, "status": status}

    # biggest zips first so the long ones are not left for the end
    jobs.sort(key=lambda j: j[3].stat().st_size, reverse=True)
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = [ex.submit(run, j) for j in jobs]
        for f in as_completed(futs):
            try:
                status_rows.append(f.result())
            except Exception as exc:
                status_rows.append({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
                log(f"ERROR {type(exc).__name__}: {exc}")

    parts = [pd.read_csv(part_dir / f"{r['key']}.csv.gz", dtype=str, keep_default_na=False)
             for r in status_rows if r.get("key") and (part_dir / f"{r['key']}.csv.gz").exists()]
    if not parts:
        log("nothing indexed")
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True).replace({"": None})
    for c in ("uncompressed_bytes", "compressed_bytes", "zip_size", "n_rows", "n_columns", "contract_year"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.insert(0, "pointer_id", df["group"] + "|" + df["zip_rel_path"] + "|" + df["member"])
    df["indexed_at"] = datetime.now().isoformat(timespec="seconds")
    df["builder_version"] = BUILDER_VERSION

    df.to_csv(out_dir / "firstrate_pointers.csv.gz", index=False, compression="gzip")
    try:
        df.to_parquet(out_dir / "firstrate_pointers.parquet", index=False)
    except Exception as exc:                        # parquet is a convenience copy
        log(f"parquet copy not written ({type(exc).__name__}); the .csv.gz has the same rows")
    summ = _summary(df)
    summ.to_csv(out_dir / "firstrate_summary.csv", index=False)
    pd.DataFrame(status_rows).to_csv(out_dir / "_firstrate_zips" / "_last_run.csv", index=False)

    log(f"done in {(time.time() - t0) / 60:.1f} min | {len(df):,} pointers -> {out_dir / 'firstrate_pointers.csv.gz'}")
    log(summ.to_string(index=False))
    bad = df[df["dataset"].eq("unknown") | df["error"].notna()]
    if len(bad):
        log(f"{len(bad)} members with unknown names or read errors, e.g.:")
        log(bad[["zip_rel_path", "member", "error"]].head(10).to_string(index=False))
    return df
