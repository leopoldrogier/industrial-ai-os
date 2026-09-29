"""Display formatting helpers (pure functions, no Streamlit dependency)."""
from __future__ import annotations

import math

import pandas as pd


def _ok(x) -> bool:
    try:
        return x is not None and not (isinstance(x, float) and math.isnan(x)) and not pd.isna(x)
    except (TypeError, ValueError):
        return x is not None


def fmt_eur(value, compact: bool = False, unit: str | None = None, decimals: int | None = None) -> str:
    """€1,234 / €168k / €1.2M ; with unit -> €2,467/t."""
    if not _ok(value):
        return "—"
    v = float(value)
    sign = "-" if v < 0 else ""
    a = abs(v)
    if compact:
        if a >= 1_000_000:
            s = f"€{a / 1_000_000:.1f}M"
        elif a >= 1_000:
            s = f"€{a / 1_000:.0f}k"
        else:
            s = f"€{a:.0f}"
    else:
        d = decimals if decimals is not None else (2 if a < 100 and unit else 0)
        s = f"€{a:,.{d}f}"
    return f"{sign}{s}" + (f"/{unit}" if unit else "")


def fmt_pct(value, decimals: int = 1, signed: bool = False) -> str:
    if not _ok(value):
        return "—"
    return f"{float(value):+.{decimals}f}%" if signed else f"{float(value):.{decimals}f}%"


def fmt_qty(value, unit: str = "", decimals: int | None = None) -> str:
    if not _ok(value):
        return "—"
    v = float(value)
    if decimals is None:
        decimals = 0 if (unit == "pcs" or abs(v) >= 100) else 1
    return f"{v:,.{decimals}f} {unit}".strip()


def fmt_days(value, decimals: int = 1) -> str:
    if not _ok(value):
        return "—"
    if math.isinf(float(value)):
        return "∞"
    return f"{float(value):.{decimals}f} d"


def fmt_date(value) -> str:
    if not _ok(value):
        return "—"
    return pd.Timestamp(value).strftime("%d %b %Y")


def severity_icon(severity: str) -> str:
    return {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}.get(severity, "⚪")
