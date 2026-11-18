"""Field names, ticker cleaning and ticker parsing shared by every reader."""

from __future__ import annotations

import re

# Bloomberg field (or Excel column label) -> standard field name.
# Matching ignores case and treats spaces/underscores alike.
FIELD_MAP = {
    "PX_LAST": "close",
    "LAST_PX": "close",
    "LAST_PRICE": "close",
    "PX_SETTLE": "settle",
    "SETTLE_PX": "settle",
    "PX_OPEN": "open",
    "PX_HIGH": "high",
    "PX_LOW": "low",
    "PX_BID": "bid",
    "PX_ASK": "ask",
    "PX_VOLUME": "volume",
    "VOLUME": "volume",
    "OPEN_INT": "open_interest",
    "FUT_AGGTE_OPEN_INT": "agg_open_interest",
    "FUT_AGGT_OPEN_INT": "agg_open_interest",
    "FUT_AGGT_OPEN": "agg_open_interest",
    "AGG_FUT_OPEN": "agg_open_interest",
    "FUT_AGG_OPEN": "agg_open_interest",
}

# Order of columns in a stored series. Fields not listed here are kept after these.
STANDARD_FIELDS = [
    "open", "high", "low", "close", "settle", "bid", "ask",
    "volume", "open_interest", "agg_open_interest",
]

_GENERIC_RE = re.compile(
    r"^(?P<root>[A-Z][A-Z ]{0,3}?)(?P<n>\d{1,2})(?P<comb> COMB)? (?P<yk>Comdty|Index|Curncy)$"
)
_CONTRACT_RE = re.compile(
    r"^(?P<root>[A-Z0-9][A-Z0-9 ]*?)(?P<month>[FGHJKMNQUVXZ])(?P<year>\d{1,2}) (?P<yk>Comdty|Index|Curncy)$"
)
_DECORATION_RE = re.compile(r"\s*\([^()]*\)\s*$")
_UNSAFE_RE = re.compile(r'[\\/:*?"<>|]')


def _key(raw: str) -> str:
    return re.sub(r"[\s_]+", "_", str(raw).strip()).upper()


def known_field(raw) -> bool:
    return _key(raw) in FIELD_MAP


def standard_field(raw) -> str:
    """PX_LAST -> close. Unknown fields come back lower-case with underscores."""
    k = _key(raw)
    return FIELD_MAP.get(k, k.lower())


def clean_ticker(raw) -> str:
    """Strip Excel decorations: 'CC1 COMB Curncy  (R1)' -> 'CC1 COMB Curncy'."""
    s = str(raw)
    prev = None
    while prev != s:                       # remove one or more trailing (...) groups
        prev = s
        s = _DECORATION_RE.sub("", s)
    return re.sub(r"\s+", " ", s).strip()


def parse_generic(ticker: str):
    """'ES1 Index' -> ('ES', 1, 'Index'); 'C 2 Comdty' -> ('C', 2, 'Comdty'); else None.

    Roots are 1-4 characters, so index names such as 'BNPICT10 Index' are not generics.
    """
    m = _GENERIC_RE.match(ticker)
    if not m or int(m.group("n")) == 0:
        return None
    return m.group("root").strip(), int(m.group("n")), m.group("yk")


def parse_contract(ticker: str):
    """'ESZ24 Index' -> ('ES', 'Z', '24', 'Index'); 'G U16 Comdty' -> ('G', 'U', '16', 'Comdty')."""
    m = _CONTRACT_RE.match(ticker)
    if not m:
        return None
    return m.group("root").strip(), m.group("month"), m.group("year"), m.group("yk")


def folder_name(text: str) -> str:
    """Make a safe folder name: spaces -> '_', Windows-illegal characters -> '-'."""
    s = _UNSAFE_RE.sub("-", str(text).strip())
    return re.sub(r"\s+", "_", s)


def order_fields(fields) -> list:
    fields = list(fields)
    known = [f for f in STANDARD_FIELDS if f in fields]
    extra = sorted(f for f in fields if f not in STANDARD_FIELDS)
    return known + extra
