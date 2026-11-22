"""Readers for the Bloomberg file layouts found by the spider.

Every reader returns the same long frame:
    date (datetime64), ticker (str), field (standard name), value (float64)
plus a stats dict. No cleaning happens here beyond what is needed to line the
layouts up; anything dropped is counted in the stats.

Layouts (detected from the columns unless the config names one):
    long_by_field   ticker, date, field, value              (Prices2/3, Volume1, OHCL1, IndividualContracts, NYMEX, Osaka, FX)
    melted          date, two id columns, value             (Prices1, EquityFutures, LeftOverFutures, FXFutures, RawFactorPX)
                    one id column is the ticker, the other the field; which is which is
                    worked out from the values (the one holding PX_LAST, OPEN_INT, ... is the field)
    melted_single   date, one id column, value              (Barclays, FX2): the id is the ticker, field from config (default PX_LAST)
    wide_fields     date, security, one column per field    (1*Px exchange files, FuturesPX, Combined\\PX files)
    excel_sheet_field   Excel, each sheet is a field, columns are tickers   (FXFutures.xlsx)
    excel_sheet_ticker  Excel, each sheet is a ticker, columns are fields   (LeftOverFutures.xlsx)
    excel_wide          Excel, one sheet, columns are tickers, field from config (FX.xlsx)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .naming import clean_ticker, known_field, standard_field

INDEX_COLUMNS = {"__index_level_0__", "index", "Unnamed: 0"}
DATE_NAMES = ("date", "Date", "DATE", "dates", "Dates")


# ---------------------------------------------------------------------------
# Loading (kept in one place so tests can swap it)
# ---------------------------------------------------------------------------

def load_parquet(path: Path) -> pd.DataFrame:
    """Read a parquet file without restoring any saved pandas index."""
    import pyarrow.parquet as pq

    table = pq.read_table(str(path))
    return table.to_pandas(ignore_metadata=True)


def load_excel_sheets(path: Path, sheets=None) -> dict:
    """Return {sheet_name: raw frame with header=None}."""
    book = pd.read_excel(path, sheet_name=None if not sheets else list(sheets), header=None)
    return book


LOADERS = {"parquet": load_parquet, "excel": load_excel_sheets}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _date_col(df: pd.DataFrame, wanted=None) -> str:
    if wanted:
        return wanted
    for name in DATE_NAMES:
        if name in df.columns:
            return name
    raise ValueError(f"no date column found in {list(df.columns)}")


def _to_dates(s: pd.Series) -> pd.Series:
    out = pd.to_datetime(s, errors="coerce")
    if getattr(out.dt, "tz", None) is not None:
        out = out.dt.tz_localize(None)
    return out.astype("datetime64[ns]")


def _finish(long: pd.DataFrame, stats: dict) -> tuple[pd.DataFrame, dict]:
    """Common last step: types, drop empty values, map fields, report what was dropped."""
    stats["rows_read"] = int(len(long))
    long = long.copy()
    long["date"] = _to_dates(long["date"])
    bad_date = long["date"].isna()
    stats["dropped_bad_date"] = int(bad_date.sum())
    long = long[~bad_date]

    raw_value = long["value"]
    value = pd.to_numeric(raw_value, errors="coerce")
    was_blank = raw_value.isna() | (raw_value.astype(str).str.strip().isin(["", "nan", "NaN", "None", "#N/A N/A", "#N/A"]))
    stats["dropped_empty_value"] = int((value.isna() & was_blank).sum())
    stats["dropped_non_numeric"] = int((value.isna() & ~was_blank).sum())
    if stats["dropped_non_numeric"]:
        examples = raw_value[value.isna() & ~was_blank].astype(str).unique()[:5]
        stats["non_numeric_examples"] = list(examples)
    long["value"] = value.astype("float64")
    long = long[long["value"].notna()]

    long["ticker"] = long["ticker"].astype(str).map(clean_ticker)

    # Map raw field names to standard names per ticker. If two raw names land on
    # the same standard name for one ticker (EquityFutures has FUT_AGGT_OPEN and
    # AGG_FUT_OPEN), the second keeps its raw name so nothing is merged silently.
    pairs = long[["ticker", "field"]].drop_duplicates()
    pairs["std"] = pairs["field"].map(standard_field)
    pairs = pairs.sort_values(["ticker", "field"])
    clash = pairs.duplicated(["ticker", "std"], keep="first")
    pairs.loc[clash, "std"] = pairs.loc[clash, "std"] + "__" + pairs.loc[clash, "field"].astype(str).str.lower()
    stats["field_clashes"] = [f"{t}:{f}" for t, f in pairs.loc[clash, ["ticker", "field"]].itertuples(index=False)]
    long = long.merge(pairs, on=["ticker", "field"], how="left")
    long["raw_field"] = long["field"].astype(str)
    long["field"] = long.pop("std")

    dup = long.duplicated(["ticker", "date", "field"], keep="last")
    stats["dropped_duplicates"] = int(dup.sum())
    long = long[~dup]

    stats["rows_kept"] = int(len(long))
    stats["tickers"] = int(long["ticker"].nunique())
    stats["raw_fields"] = sorted(long["raw_field"].unique().tolist())
    long = long[["date", "ticker", "field", "value", "raw_field"]].reset_index(drop=True)
    return long, stats


# ---------------------------------------------------------------------------
# Layout detection
# ---------------------------------------------------------------------------

def detect_layout(df: pd.DataFrame) -> str:
    cols = [c for c in df.columns if c not in INDEX_COLUMNS]
    lower = {c.lower() for c in cols}
    if {"ticker", "date", "field", "value"} <= lower:
        return "long_by_field"
    if "value" in lower:
        ids = [c for c in cols if c.lower() not in ("value",) and c not in DATE_NAMES]
        if len(ids) == 2:
            return "melted"
        if len(ids) == 1:
            return "melted_single"
    if "security" in lower:
        return "wide_fields"
    raise ValueError(f"cannot detect layout from columns {cols}; set 'reader' in config/bbg_store.yaml")


def _field_share(s: pd.Series) -> float:
    vals = pd.Series(s.dropna().astype(str).unique())
    if vals.empty:
        return 0.0
    return float(vals.map(known_field).mean())


# ---------------------------------------------------------------------------
# Readers for frames already loaded
# ---------------------------------------------------------------------------

def from_long_by_field(df: pd.DataFrame, spec: dict) -> tuple[pd.DataFrame, dict]:
    cols = {c.lower(): c for c in df.columns}
    long = pd.DataFrame({
        "date": df[spec.get("date") or cols["date"]],
        "ticker": df[spec.get("ticker") or cols["ticker"]],
        "field": df[spec.get("field_column") or cols["field"]],
        "value": df[spec.get("value") or cols["value"]],
    })
    return _finish(long, {"layout": "long_by_field"})


def from_melted(df: pd.DataFrame, spec: dict) -> tuple[pd.DataFrame, dict]:
    date = _date_col(df, spec.get("date"))
    value = spec.get("value") or next(c for c in df.columns if c.lower() == "value")
    ids = [c for c in df.columns if c not in INDEX_COLUMNS and c not in (date, value)]
    field_col = spec.get("field_column")
    ticker_col = spec.get("ticker")
    if not (field_col and ticker_col):
        shares = {c: _field_share(df[c]) for c in ids}
        ranked = sorted(shares, key=shares.get, reverse=True)
        if shares[ranked[0]] < 0.5 or shares[ranked[1]] >= 0.5:
            raise ValueError(
                f"melted file: cannot tell which of {ids} holds the field names "
                f"(share of known fields {shares}); set ticker/field_column in config"
            )
        field_col = field_col or ranked[0]
        ticker_col = ticker_col or ranked[1]
    long = pd.DataFrame({"date": df[date], "ticker": df[ticker_col], "field": df[field_col], "value": df[value]})
    return _finish(long, {"layout": "melted", "ticker_column": ticker_col, "field_column": field_col})


def from_melted_single(df: pd.DataFrame, spec: dict) -> tuple[pd.DataFrame, dict]:
    date = _date_col(df, spec.get("date"))
    value = spec.get("value") or next(c for c in df.columns if c.lower() == "value")
    ids = [c for c in df.columns if c not in INDEX_COLUMNS and c not in (date, value)]
    ticker_col = spec.get("ticker") or ids[0]
    field = spec.get("field", "PX_LAST")
    long = pd.DataFrame({"date": df[date], "ticker": df[ticker_col], "field": field, "value": df[value]})
    return _finish(long, {"layout": "melted_single", "ticker_column": ticker_col, "fixed_field": field})


def from_wide_fields(df: pd.DataFrame, spec: dict) -> tuple[pd.DataFrame, dict]:
    date = _date_col(df, spec.get("date"))
    ticker_col = spec.get("ticker") or next(c for c in df.columns if c.lower() == "security")
    fields = [c for c in df.columns if c not in INDEX_COLUMNS and c not in (date, ticker_col)]
    if spec.get("fields"):
        fields = [c for c in fields if c in spec["fields"]]
    long = df[[date, ticker_col] + fields].melt(
        id_vars=[date, ticker_col], var_name="field", value_name="value"
    ).rename(columns={date: "date", ticker_col: "ticker"})
    return _finish(long, {"layout": "wide_fields", "ticker_column": ticker_col, "field_columns": fields})


# ---- Excel ---------------------------------------------------------------

def _excel_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Find the header row (the row whose first cell is 'Date') and return a proper frame."""
    header_row = None
    for i in range(min(len(raw), 15)):
        first = str(raw.iat[i, 0]).strip().lower()
        if first in ("date", "dates"):
            header_row = i
            break
    if header_row is None:
        raise ValueError("Excel sheet: no row starting with 'Date' in the first 15 rows")
    table = raw.iloc[header_row + 1:].copy()
    table.columns = [str(c) for c in raw.iloc[header_row]]
    table = table.rename(columns={table.columns[0]: "date"})
    table = table.loc[:, [c for c in table.columns if c and c.lower() != "nan"]]
    return table


def from_excel(book: dict, spec: dict, layout: str) -> tuple[pd.DataFrame, dict]:
    parts = []
    for sheet, raw in book.items():
        table = _excel_table(raw)
        cols = [c for c in table.columns if c != "date"]
        melted = table.melt(id_vars="date", value_vars=cols, var_name="col", value_name="value")
        if layout == "excel_sheet_field":          # sheet = field, column = ticker
            melted["ticker"] = melted["col"]
            melted["field"] = sheet
        elif layout == "excel_sheet_ticker":       # sheet = ticker, column = field
            melted["ticker"] = spec.get("ticker_map", {}).get(sheet, sheet)
            melted["field"] = melted["col"]
        elif layout == "excel_wide":               # one sheet, column = ticker, fixed field
            melted["ticker"] = melted["col"]
            melted["field"] = spec.get("field", "PX_LAST")
        else:
            raise ValueError(layout)
        parts.append(melted[["date", "ticker", "field", "value"]])
    long = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["date", "ticker", "field", "value"])
    return _finish(long, {"layout": layout, "sheets": list(book)})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

FRAME_READERS = {
    "long_by_field": from_long_by_field,
    "melted": from_melted,
    "melted_single": from_melted_single,
    "wide_fields": from_wide_fields,
}
EXCEL_LAYOUTS = ("excel_sheet_field", "excel_sheet_ticker", "excel_wide")


def read_origin(path: Path, spec: dict) -> tuple[pd.DataFrame, dict]:
    """Read one origin file into the standard long frame."""
    path = Path(path)
    layout = spec.get("reader", "auto")
    if path.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        if layout not in EXCEL_LAYOUTS:
            raise ValueError(f"{path.name}: Excel files need reader: one of {EXCEL_LAYOUTS}")
        book = LOADERS["excel"](path, spec.get("sheets"))
        return from_excel(book, spec, layout)
    df = LOADERS["parquet"](path)
    if layout == "auto":
        layout = detect_layout(df)
    return FRAME_READERS[layout](df, spec)
