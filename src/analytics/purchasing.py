"""Purchasing analytics: spend, price evolution, price anomalies, contract leakage."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def price_change_pct(old: float, new: float) -> float:
    """Relative price change in %."""
    if old is None or not np.isfinite(old) or old == 0:
        return float("nan")
    return (new - old) / old * 100.0


def potential_savings(price_gap: float, annual_volume: float) -> float:
    """Price Gap x Annual Volume (never negative)."""
    if not np.isfinite(price_gap) or not np.isfinite(annual_volume):
        return 0.0
    return max(price_gap, 0.0) * max(annual_volume, 0.0)


def with_value(po: pd.DataFrame) -> pd.DataFrame:
    return po.assign(po_value=po["ordered_quantity"] * po["unit_price"])


def window(po: pd.DataFrame, as_of: pd.Timestamp, days: int, date_col: str = "order_date") -> pd.DataFrame:
    start = as_of - pd.Timedelta(days=days)
    return po[(po[date_col] > start) & (po[date_col] <= as_of)]


def weighted_price(df: pd.DataFrame) -> float:
    q = df["ordered_quantity"].sum()
    return float((df["ordered_quantity"] * df["unit_price"]).sum() / q) if q > 0 else float("nan")


def spend_summary(po: pd.DataFrame, as_of: pd.Timestamp, days: int = config.SPEND_WINDOW_DAYS) -> dict:
    w = with_value(window(po, as_of, days))
    return {
        "total_spend": float(w["po_value"].sum()),
        "po_count": int(len(w)),
        "active_suppliers": int(w["supplier_id"].nunique()),
        "materials": int(w["material_id"].nunique()),
    }


def spend_by(po: pd.DataFrame, suppliers: pd.DataFrame, products: pd.DataFrame,
             as_of: pd.Timestamp, by: str, days: int = config.SPEND_WINDOW_DAYS) -> pd.DataFrame:
    """Spend by 'supplier' or 'material' over the window, sorted descending."""
    w = with_value(window(po, as_of, days))
    key = "supplier_id" if by == "supplier" else "material_id"
    out = (w.groupby(key)
           .agg(spend=("po_value", "sum"), volume=("ordered_quantity", "sum"), orders=("po_id", "count"))
           .reset_index())
    if by == "supplier":
        out = out.merge(suppliers[["supplier_id", "supplier_name", "material"]], on="supplier_id", how="left")
    else:
        out = out.merge(products[["material_id", "material_name", "unit", "category"]], on="material_id", how="left")
    out["share_pct"] = out["spend"] / out["spend"].sum() * 100
    return out.sort_values("spend", ascending=False).reset_index(drop=True)


def monthly_spend(po: pd.DataFrame, as_of: pd.Timestamp, days: int = config.SPEND_WINDOW_DAYS) -> pd.DataFrame:
    w = with_value(window(po, as_of, days))
    w = w.assign(month=w["order_date"].dt.to_period("M").dt.to_timestamp())
    return w.groupby(["month", "material_id"], as_index=False)["po_value"].sum()


def monthly_prices(po: pd.DataFrame, material_id: str) -> pd.DataFrame:
    """Volume-weighted monthly price per supplier for one material."""
    m = po[po["material_id"] == material_id]
    m = m.assign(month=m["order_date"].dt.to_period("M").dt.to_timestamp(),
                 v=m["ordered_quantity"] * m["unit_price"])
    g = m.groupby(["month", "supplier_id"], as_index=False)[["v", "ordered_quantity"]].sum()
    g["avg_price"] = g["v"] / g["ordered_quantity"]
    return g[["month", "supplier_id", "avg_price", "ordered_quantity"]]


def price_stats(po: pd.DataFrame, suppliers: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Per supplier x material: baseline price, current price, evolution, volume and peer benchmark."""
    year = window(po, as_of, config.SPEND_WINDOW_DAYS)
    base_end = as_of - pd.Timedelta(days=config.SPEND_WINDOW_DAYS - config.PRICE_BASELINE_DAYS)
    rows = []
    for (sid, mid), g in year.groupby(["supplier_id", "material_id"]):
        base = g[g["order_date"] <= base_end]
        recent = g[g["order_date"] > as_of - pd.Timedelta(days=config.PRICE_RECENT_DAYS)]
        rows.append({
            "supplier_id": sid,
            "material_id": mid,
            "baseline_price": weighted_price(base) if len(base) else float("nan"),
            "current_price": weighted_price(recent) if len(recent) else float("nan"),
            "avg_price_12m": weighted_price(g),
            "annual_volume": float(g["ordered_quantity"].sum()),
            "annual_spend": float((g["ordered_quantity"] * g["unit_price"]).sum()),
            "orders": int(len(g)),
        })
    st = pd.DataFrame(rows)
    if st.empty:
        return st
    st = st.merge(suppliers[["supplier_id", "supplier_name", "pricing_model", "contracted_price"]],
                  on="supplier_id", how="left")
    st["price_change_pct"] = [price_change_pct(b, c) for b, c in zip(st["baseline_price"], st["current_price"])]

    # Peer benchmark: other non-spot suppliers of the same material
    bench, peer_chg = [], []
    for _, r in st.iterrows():
        peers = st[(st["material_id"] == r["material_id"]) & (st["supplier_id"] != r["supplier_id"])
                   & (st["pricing_model"] != "Spot")]
        bench.append(float(peers["current_price"].median()) if len(peers) else float("nan"))
        peer_chg.append(float(peers["price_change_pct"].median()) if len(peers) else float("nan"))
    st["benchmark_price"] = bench
    st["peer_change_pct"] = peer_chg
    st["excess_change_pp"] = st["price_change_pct"] - st["peer_change_pct"]
    st["gap_vs_benchmark"] = st["current_price"] - st["benchmark_price"]
    st["gap_vs_benchmark_pct"] = st["gap_vs_benchmark"] / st["benchmark_price"] * 100
    return st


def detect_price_anomalies(stats: pd.DataFrame,
                           drift_pp: float = config.PRICE_DRIFT_THRESHOLD_PP,
                           gap_pct: float = config.BENCHMARK_GAP_THRESHOLD_PCT) -> pd.DataFrame:
    """Suppliers whose price rose abnormally vs. their own history AND vs. peers."""
    if stats.empty:
        return stats
    mask = ((stats["pricing_model"] != "Spot")
            & (stats["excess_change_pp"] >= drift_pp)
            & (stats["gap_vs_benchmark_pct"] >= gap_pct))
    out = stats[mask].copy()
    out["potential_savings"] = [potential_savings(g, v) for g, v in zip(out["gap_vs_benchmark"], out["annual_volume"])]
    return out.sort_values("potential_savings", ascending=False).reset_index(drop=True)


def contract_deviations(stats: pd.DataFrame,
                        threshold_pct: float = config.CONTRACT_DEVIATION_THRESHOLD_PCT) -> pd.DataFrame:
    """Fixed-price contracts invoiced above the contracted price (price leakage)."""
    if stats.empty:
        return stats
    fx = stats[stats["pricing_model"] == "Fixed annual"].copy()
    fx["contract_gap"] = fx["current_price"] - fx["contracted_price"]
    fx["contract_gap_pct"] = fx["contract_gap"] / fx["contracted_price"] * 100
    fx = fx[fx["contract_gap_pct"] >= threshold_pct]
    fx["potential_savings"] = [potential_savings(g, v) for g, v in zip(fx["contract_gap"], fx["annual_volume"])]
    return fx.sort_values("potential_savings", ascending=False).reset_index(drop=True)
