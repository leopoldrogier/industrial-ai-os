"""Quality analytics: PPM, trends, spikes, cost of non-quality."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def compute_ppm(defects: float, inspected: float) -> float:
    """Parts per million = defects / quantity inspected x 1,000,000."""
    if inspected is None or inspected <= 0:
        return float("nan")
    return float(defects) / float(inspected) * 1_000_000


def ppm_of(df: pd.DataFrame) -> float:
    return compute_ppm(df["defects"].sum(), df["quantity_inspected"].sum()) if len(df) else float("nan")


def monthly_ppm(quality: pd.DataFrame, supplier_id: str | None = None) -> pd.DataFrame:
    q = quality if supplier_id is None else quality[quality["supplier_id"] == supplier_id]
    q = q.assign(month=q["date"].dt.to_period("M").dt.to_timestamp())
    g = q.groupby(["month", "supplier_id"], as_index=False)[["defects", "quantity_inspected"]].sum()
    g["ppm"] = [compute_ppm(d, i) for d, i in zip(g["defects"], g["quantity_inspected"])]
    return g


def quality_trends(quality: pd.DataFrame, products: pd.DataFrame, as_of: pd.Timestamp,
                   recent_days: int = config.QUALITY_RECENT_DAYS) -> pd.DataFrame:
    """Baseline vs recent PPM per supplier x material."""
    cut = as_of - pd.Timedelta(days=recent_days)
    rows = []
    for (sid, mid), g in quality.groupby(["supplier_id", "material_id"]):
        base, recent = g[g["date"] <= cut], g[g["date"] > cut]
        rec_def = recent[recent["defects"] > 0]
        top = rec_def.groupby("defect_type")["defects"].sum().sort_values(ascending=False)
        rows.append({
            "supplier_id": sid,
            "material_id": mid,
            "inspections": int(len(g)),
            "ppm_baseline": ppm_of(base),
            "ppm_recent": ppm_of(recent),
            "recent_inspections": int(len(recent)),
            "recent_nonconforming_lots": int((recent["quality_status"] != "Accepted").sum()),
            "recent_defects": float(recent["defects"].sum()),
            "recent_inspected": float(recent["quantity_inspected"].sum()),
            "top_defect_recent": str(top.index[0]) if len(top) else "None",
            "top_defect_share_pct": float(top.iloc[0] / top.sum() * 100) if len(top) else 0.0,
        })
    out = pd.DataFrame(rows).merge(products[["material_id", "quality_ppm_limit", "unit", "standard_cost"]],
                                   on="material_id", how="left")
    out["ppm_ratio"] = out["ppm_recent"] / out["ppm_baseline"].replace(0, np.nan)
    return out


def detect_quality_spikes(trends: pd.DataFrame, ratio: float = config.QUALITY_SPIKE_RATIO) -> pd.DataFrame:
    """Recent PPM at least `ratio` x baseline AND above the material PPM limit."""
    if trends.empty:
        return trends
    m = (trends["ppm_ratio"] >= ratio) & (trends["ppm_recent"] > trends["quality_ppm_limit"])
    return trends[m].sort_values("ppm_ratio", ascending=False).reset_index(drop=True)


def cost_of_non_quality(recent_defects: float, unit: str, unit_price: float, recent_days: int,
                        handling_rate: float = config.NON_QUALITY_HANDLING_RATE) -> float:
    """Annualised cost of non-conforming material (sorting, re-melt, dross loss).

    Defects are recorded in kg for tonnage materials and in pieces otherwise.
    """
    qty = recent_defects / 1000.0 if unit == "t" else recent_defects
    return qty * unit_price * handling_rate * (365.0 / recent_days)
