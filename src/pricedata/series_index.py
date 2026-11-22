"""Series index (step 2.3): one row per series, both sources.

Inputs (all already in the repo's data folder):
    data/store/bbg/**/vNNNN/manifest.json      Bloomberg series (latest version of each)
    data/index/firstrate_pointers.csv.gz       FirstRate pointers (step 2.2)
    data/catalog/price_catalog.xlsx            Instruments sheet: instrument_id, asset_class
    config/preferences.yaml                    default rule, overrides, instrument aliases

Outputs (data/index/):
    series_index.csv.gz      every series (always written)
    series_index.parquet     same rows (if pyarrow is installed)
    series_index.xlsx        human view: Bloomberg, FirstRate futures/FX, contract and ETF
                             summaries, one row per instrument
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd

from . import BUILDER_VERSION
from .naming import parse_generic

# FirstRate line layouts (confirmed 2026-11-14 from the first/last line of every file)
FR_LAYOUTS = {
    "std6": "datetime,open,high,low,close,volume",
    "fut_daily_oi7": "date,open,high,low,close,volume,open_interest",
    "fx_daily6": "yyyymmdd,open,high,low,close,volume",
    "fx_datetime7": "yyyymmdd,hhmmss,open,high,low,close,volume",
}


# ---------------------------------------------------------------------------
# Catalogue
# ---------------------------------------------------------------------------

def load_catalogue(path: Path) -> pd.DataFrame:
    inst = pd.read_excel(path, sheet_name="Instruments", dtype=str)
    return inst


def catalogue_maps(inst: pd.DataFrame, aliases: dict) -> dict:
    fut = inst[inst["kind"] == "future"]
    by_root = {}
    for r in fut.dropna(subset=["bbg_root"]).itertuples():
        by_root[f"{r.bbg_root} {r.bbg_yellow_key}"] = r.instrument_id
    for k, v in (aliases or {}).items():
        by_root[k] = v
    return {
        "bbg_root": by_root,
        "bbg_front": dict(zip(inst["bbg_front"].dropna(), inst.loc[inst["bbg_front"].notna(), "instrument_id"])),
        "fr_future": dict(zip(fut["firstrate_symbol"].dropna(), fut.loc[fut["firstrate_symbol"].notna(), "instrument_id"])),
        "fr_fx": {s: f"fx.{s}" for s in inst.loc[inst["kind"] == "fx_spot", "firstrate_symbol"].dropna()},
        "fr_etf": {s: f"etf.{s}" for s in inst.loc[inst["kind"] == "etf", "firstrate_symbol"].dropna()},
        "asset_class": dict(zip(inst["instrument_id"], inst["asset_class"])),
        "name": dict(zip(inst["instrument_id"], inst["name"])),
    }


# ---------------------------------------------------------------------------
# Bloomberg
# ---------------------------------------------------------------------------

def bloomberg_rows(store: Path, repo: Path, maps: dict) -> pd.DataFrame:
    rows = []
    for sdir in sorted(p for p in store.glob("*/*/*/*") if p.is_dir() and not p.name.startswith("_")):
        versions = sorted(p for p in sdir.glob("v[0-9][0-9][0-9][0-9]") if (p / "manifest.json").exists())
        if not versions:
            continue
        m = json.loads((versions[-1] / "manifest.json").read_text(encoding="utf-8"))
        fields = m.get("fields", {}) or {}
        close = fields.get("close") or {}
        ticker = m["vendor_ticker"]
        dataset = m["dataset"]
        root = m.get("root")
        if dataset == "fut_contract":
            root = ticker
        inst_id = None
        if root:
            inst_id = maps["bbg_root"].get(root)
        if inst_id is None:
            inst_id = maps["bbg_front"].get(ticker)
        if inst_id is None and dataset == "fut_generic":
            g = parse_generic(ticker)
            if g:
                inst_id = maps["bbg_root"].get(f"{g[0]} {g[2]}")
        rows.append({
            "series_id": m["series_id"],
            "source": "bbg",
            "dataset": dataset,
            "frequency": m["frequency"],
            "adjustment": m["adjustment"],
            "status": "active",
            "instrument_id": inst_id,
            "vendor_ticker": ticker,
            "root": root,
            "generic_n": m.get("generic_n"),
            "contract": None,
            "n_contracts": m.get("contracts"),
            "fields": ";".join(fields),
            "first_date": m.get("first_date"),
            "last_date": m.get("last_date"),
            "close_first": close.get("first"),
            "close_last": close.get("last"),
            "n_rows": m.get("n_rows"),
            "origin": m["origin"]["rel_path"].rsplit(".", 1)[0],
            "origin_file": m["origin"]["rel_path"],
            "origin_source": m["origin"]["source"],
            "origin_size": m["origin"]["size"],
            "origin_mtime": m["origin"].get("mtime_text"),
            "origin_hash": m["origin"]["sha256"],
            "storage": "copied",
            "location": str((versions[-1] / "data.parquet").relative_to(repo)),
            "member": None,
            "layout": "parquet_wide",
            "version": m["version"],
            "built_at": m["built_at"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# FirstRate
# ---------------------------------------------------------------------------

def _fr_layout(dataset: str, freq: str, n_columns) -> str:
    if dataset == "fx_spot":
        return "fx_daily6" if freq == "1day" else "fx_datetime7"
    if dataset in ("fut_continuous", "fut_contract") and freq == "1day" and n_columns == 7:
        return "fut_daily_oi7"
    return "std6"


def _fr_ts(line, layout):
    if not isinstance(line, str) or not line:
        return pd.NaT
    parts = line.split(",")
    try:
        if layout == "fx_datetime7":
            return pd.to_datetime(parts[0] + " " + parts[1], format="%Y%m%d %H:%M:%S")
        if layout == "fx_daily6":
            return pd.to_datetime(parts[0], format="%Y%m%d")
        return pd.to_datetime(parts[0])
    except (ValueError, IndexError):
        return pd.NaT


def firstrate_rows(pointers: Path, maps: dict) -> pd.DataFrame:
    p = pd.read_csv(pointers, dtype=str, keep_default_na=False).replace({"": None})
    p["n_columns"] = pd.to_numeric(p["n_columns"], errors="coerce")
    p["layout"] = [_fr_layout(d, f, n) for d, f, n in zip(p["dataset"], p["frequency"], p["n_columns"])]
    first = [_fr_ts(l, lay) for l, lay in zip(p["first_line"], p["layout"])]
    last = [_fr_ts(l, lay) for l, lay in zip(p["last_line"], p["layout"])]
    p["first_ts"] = pd.to_datetime(pd.Series(first, index=p.index))
    p["last_ts"] = pd.to_datetime(pd.Series(last, index=p.index))

    def inst(row):
        if row.dataset in ("fut_continuous", "fut_contract"):
            return maps["fr_future"].get(row.symbol)
        if row.dataset == "fx_spot":
            return maps["fr_fx"].get(row.symbol, f"fx.{row.symbol}")
        if row.dataset == "etf":
            return maps["fr_etf"].get(row.symbol, f"etf.{row.symbol}")
        return None

    out = pd.DataFrame({
        "series_id": "firstrate|" + p["pointer_id"],
        "source": "firstrate",
        "dataset": p["dataset"],
        "frequency": p["frequency"],
        "adjustment": p["adjustment"],
        "status": p["status"],
        "instrument_id": [inst(r) for r in p.itertuples()],
        "vendor_ticker": p["symbol"],
        "root": p["symbol"],
        "generic_n": None,
        "contract": p["contract"],
        "n_contracts": None,
        "fields": ["open;high;low;close;volume;open_interest" if lay == "fut_daily_oi7" else "open;high;low;close;volume"
                   for lay in p["layout"]],
        "first_date": p["first_ts"].dt.strftime("%Y-%m-%d"),
        "last_date": p["last_ts"].dt.strftime("%Y-%m-%d"),
        "first_ts": p["first_ts"].astype(str).replace("NaT", None),
        "last_ts": p["last_ts"].astype(str).replace("NaT", None),
        "close_first": p["first_ts"].dt.strftime("%Y-%m-%d"),
        "close_last": p["last_ts"].dt.strftime("%Y-%m-%d"),
        "n_rows": pd.to_numeric(p["n_rows"], errors="coerce"),
        "origin": p["zip_rel_path"].str.replace(r"\.zip$", "", regex=True),
        "origin_file": p["zip_rel_path"],
        "origin_source": p["source_name"],
        "origin_size": pd.to_numeric(p["zip_size"], errors="coerce"),
        "origin_mtime": pd.to_datetime(pd.to_numeric(p["zip_mtime"], errors="coerce"), unit="s").dt.strftime("%Y-%m-%d %H:%M:%S"),
        "origin_hash": "crc32:" + p["member_crc"].fillna(""),
        "storage": "pointer",
        "location": p["zip_path"],
        "member": p["member"],
        "layout": p["layout"],
        "version": None,
        "built_at": p["indexed_at"],
    })
    return out


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

def add_span_and_defaults(df: pd.DataFrame, prefs: dict) -> pd.DataFrame:
    df = df.copy()
    f = pd.to_datetime(df["close_first"], errors="coerce")
    l = pd.to_datetime(df["close_last"], errors="coerce")
    df["span_years"] = ((l - f).dt.days / 365.25).round(2)
    df["has_close"] = df["fields"].fillna("").str.split(";").map(lambda x: "close" in x)
    df["has_ohlc"] = df["fields"].fillna("").str.split(";").map(lambda x: {"open", "high", "low", "close"} <= set(x))
    df["instrument_key"] = df["instrument_id"].fillna(df["source"] + ":" + df["vendor_ticker"].fillna(""))

    # what competes for "the default": same source, kind of series, instrument and
    # generic number / contract / frequency / adjustment. Bloomberg origins compete;
    # FirstRate has one file per combination.
    gn = df["generic_n"].astype("string").fillna("")
    df["default_group"] = (df["source"] + "|" + df["dataset"] + "|" + df["instrument_key"] + "|g" + gn + "|"
                           + df["contract"].fillna("") + "|" + df["frequency"].fillna("") + "|"
                           + df["adjustment"].fillna(""))
    cand = df[df["has_close"] & df["span_years"].notna() & (df["status"] == "active")].copy()
    cand["_last"] = pd.to_datetime(cand["close_last"])
    cand = cand.sort_values(["default_group", "span_years", "n_rows", "_last"], ascending=[True, False, False, False])
    winners = set(cand.drop_duplicates("default_group")["series_id"])

    overrides = prefs.get("overrides") or {}
    df["default_reason"] = None
    for key, origin_part in overrides.items():
        hit = (df["vendor_ticker"] == key) | (df["instrument_id"] == key)
        groups = df.loc[hit, "default_group"].unique()
        for g in groups:
            members = df["default_group"] == g
            chosen = df.loc[members & df["origin"].str.contains(str(origin_part), case=False, regex=False)]
            if len(chosen):
                winners -= set(df.loc[members, "series_id"])
                winners.add(chosen["series_id"].iloc[0])
                df.loc[df["series_id"] == chosen["series_id"].iloc[0], "default_reason"] = "override"
    df["is_default"] = df["series_id"].isin(winners)
    df.loc[df["is_default"] & df["default_reason"].isna(), "default_reason"] = "longest_span"
    df["n_alternatives"] = df.groupby("default_group")["series_id"].transform("size") - 1
    return df


# ---------------------------------------------------------------------------
# Excel view
# ---------------------------------------------------------------------------

def write_xlsx(df: pd.DataFrame, path: Path, maps: dict) -> None:
    keep = ["instrument_id", "name", "asset_class", "vendor_ticker", "generic_n", "dataset", "frequency",
            "adjustment", "status", "origin", "fields", "first_date", "last_date", "span_years", "n_rows",
            "is_default", "n_alternatives", "default_reason", "layout", "location", "member", "series_id"]
    d = df.copy()
    d["name"] = d["instrument_id"].map(maps["name"])

    bbg = d[d["source"] == "bbg"].sort_values(["dataset", "instrument_key", "generic_n", "span_years"],
                                               ascending=[True, True, True, False])
    fr_fut = d[(d["source"] == "firstrate") & (d["dataset"] == "fut_continuous")].sort_values(
        ["vendor_ticker", "frequency", "adjustment"])
    fr_fx = d[(d["source"] == "firstrate") & (d["dataset"] == "fx_spot")].sort_values(["vendor_ticker", "frequency"])

    ct = d[(d["source"] == "firstrate") & (d["dataset"] == "fut_contract")]
    fr_ct = ct.groupby(["instrument_id", "vendor_ticker", "frequency"], dropna=False).agg(
        contracts=("contract", "nunique"), first_date=("first_date", "min"), last_date=("last_date", "max"),
        rows=("n_rows", "sum"), layout=("layout", lambda s: ";".join(sorted(set(s))))).reset_index()

    etf = d[(d["source"] == "firstrate") & (d["dataset"] == "etf")].copy()
    etf["col"] = etf["frequency"] + "_" + etf["adjustment"]
    fr_etf = etf.groupby("vendor_ticker").agg(
        instrument_id=("instrument_id", "first"), first_date=("first_date", "min"),
        last_date=("last_date", "max"), files=("series_id", "size"),
        available=("col", lambda s: ", ".join(sorted(s)))).reset_index()

    # one row per instrument: defaults from each source
    bd = d[d["is_default"] & (d["source"] == "bbg") & (d["dataset"] == "fut_generic") & (d["generic_n"] == 1)]
    fd = d[d["is_default"] & (d["source"] == "firstrate") & (d["dataset"] == "fut_continuous")
           & (d["frequency"] == "1day") & (d["adjustment"] == "unadjusted")]
    snaps = d[(d["source"] == "bbg") & (d["dataset"] == "fut_generic") & (d["generic_n"] == 1) & d["has_close"]]
    inst = pd.DataFrame({"instrument_id": sorted(set(bd["instrument_id"].dropna()) | set(fd["instrument_id"].dropna()))})
    inst["name"] = inst["instrument_id"].map(maps["name"])
    inst["asset_class"] = inst["instrument_id"].map(maps["asset_class"])
    b = bd.drop_duplicates("instrument_id").set_index("instrument_id")
    f = fd.drop_duplicates("instrument_id").set_index("instrument_id")
    inst["bbg_ticker"] = inst["instrument_id"].map(b["vendor_ticker"])
    inst["bbg_default_origin"] = inst["instrument_id"].map(b["origin"])
    inst["bbg_close_from"] = inst["instrument_id"].map(b["first_date"])
    inst["bbg_close_to"] = inst["instrument_id"].map(b["last_date"])
    inst["bbg_span_years"] = inst["instrument_id"].map(b["span_years"])
    inst["bbg_close_snapshots"] = inst["instrument_id"].map(snaps.groupby("instrument_id").size()).fillna(0).astype(int)
    inst["fr_symbol"] = inst["instrument_id"].map(f["vendor_ticker"])
    inst["fr_1day_from"] = inst["instrument_id"].map(f["first_date"])
    inst["fr_1day_to"] = inst["instrument_id"].map(f["last_date"])

    readme = pd.DataFrame({"TrendSystem series index": [
        f"Built {datetime.now():%Y-%m-%d %H:%M} by scripts/build_series_index.py (pricedata {BUILDER_VERSION}). Generated: edits here are overwritten.",
        "Full detail (every series incl. every FirstRate contract and ETF file): data/index/series_index.csv.gz / .parquet.",
        "Instruments: one row per futures instrument with the default Bloomberg generic-1 close and the FirstRate 1day unadjusted continuous.",
        "Bloomberg: every stored series. Filter instrument_id (or vendor_ticker) to see all snapshots of one ticker; is_default marks the longest close span.",
        "FR_Futures / FR_FX: FirstRate continuous futures and FX, one row per file. FR_Contracts / FR_ETF: summaries.",
        "span_years = close last date - close first date. default_reason: longest_span or override (config/preferences.yaml).",
        "FirstRate layout: std6 = datetime,o,h,l,c,v | fut_daily_oi7 = date,o,h,l,c,v,open_interest | fx_daily6 = yyyymmdd,o,h,l,c,v | fx_datetime7 = yyyymmdd,hhmmss,o,h,l,c,v.",
        "Bloomberg adjustment: back-adjusted, method unknown and possibly different per origin file. Never mix origins.",
    ]})
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        readme.to_excel(xw, sheet_name="ReadMe", index=False)
        inst.to_excel(xw, sheet_name="Instruments", index=False)
        bbg[[c for c in keep if c in bbg]].to_excel(xw, sheet_name="Bloomberg", index=False)
        fr_fut[[c for c in keep if c in fr_fut]].to_excel(xw, sheet_name="FR_Futures", index=False)
        fr_fx[[c for c in keep if c in fr_fx]].to_excel(xw, sheet_name="FR_FX", index=False)
        fr_ct.to_excel(xw, sheet_name="FR_Contracts", index=False)
        fr_etf.to_excel(xw, sheet_name="FR_ETF", index=False)
        for ws in xw.book.worksheets:
            ws.freeze_panes = "A2"
            if ws.max_row > 1:
                ws.auto_filter.ref = ws.dimensions
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in list(col)[:200])
                ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 60)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build(repo: Path, prefs: dict, write_xlsx_view: bool = True, log=print) -> pd.DataFrame:
    idx = repo / "data" / "index"
    idx.mkdir(parents=True, exist_ok=True)
    cat = repo / "data" / "catalog" / "price_catalog.xlsx"
    inst = load_catalogue(cat)
    maps = catalogue_maps(inst, prefs.get("instrument_aliases"))
    log(f"catalogue: {len(inst):,} instruments")

    bbg = bloomberg_rows(repo / "data" / "store" / "bbg", repo, maps)
    log(f"bloomberg: {len(bbg):,} series from manifests")
    ptr = idx / "firstrate_pointers.csv.gz"
    fr = firstrate_rows(ptr, maps) if ptr.exists() else pd.DataFrame()
    log(f"firstrate: {len(fr):,} pointers")

    df = pd.concat([bbg, fr], ignore_index=True)
    df["asset_class"] = df["instrument_id"].map(maps["asset_class"])
    df.loc[df["asset_class"].isna() & (df["dataset"] == "index_px"), "asset_class"] = "index"
    df.loc[df["asset_class"].isna() & (df["dataset"] == "fx_spot"), "asset_class"] = "currency"
    df.loc[df["asset_class"].isna() & (df["dataset"] == "etf"), "asset_class"] = "etf"
    df = add_span_and_defaults(df, prefs)
    df["index_built_at"] = datetime.now().isoformat(timespec="seconds")

    df.to_csv(idx / "series_index.csv.gz", index=False, compression="gzip")
    try:
        df.to_parquet(idx / "series_index.parquet", index=False)
    except Exception as exc:
        log(f"parquet copy not written ({type(exc).__name__}); the .csv.gz has the same rows")
    if write_xlsx_view:
        write_xlsx(df, idx / "series_index.xlsx", maps)
        log(f"excel view: {idx / 'series_index.xlsx'}")

    log(f"series index: {len(df):,} rows -> {idx / 'series_index.csv.gz'}")
    log(df.groupby(["source", "dataset"]).agg(series=("series_id", "size"),
                                              mapped=("instrument_id", lambda s: int(s.notna().sum())),
                                              defaults=("is_default", "sum")).to_string())
    unmapped = df[(df["source"] == "bbg") & df["dataset"].isin(["fut_generic", "fut_contract"]) & df["instrument_id"].isna()]
    log(f"Bloomberg futures series without a catalogue instrument: {len(unmapped)} "
        f"({unmapped['vendor_ticker'].nunique()} tickers)")
    return df
