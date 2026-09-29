"""Inventory analytics: coverage, stock projection, stockout risk, overstock."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src import config


def days_of_coverage(stock: float, daily_consumption: float) -> float:
    """Days of Coverage = Current Stock / Daily Consumption."""
    if daily_consumption is None or daily_consumption <= 0:
        return float("inf")
    return max(stock, 0.0) / daily_consumption


def estimate_stockout_date(stock: float, daily_consumption: float, as_of: pd.Timestamp) -> pd.Timestamp | None:
    """First day on which end-of-day stock becomes negative, with no inbound delivery."""
    if daily_consumption is None or daily_consumption <= 0:
        return None
    days = math.floor(max(stock, 0.0) / daily_consumption) + 1
    return as_of + pd.Timedelta(days=days)


def open_inbound(po: pd.DataFrame, logistics: pd.DataFrame, as_of: pd.Timestamp,
                 material_id: str | None = None, view: str = "ai") -> pd.DataFrame:
    """Open purchase orders with their expected arrival date.

    view="erp": the ERP promised date (what the system of record believes)
    view="ai" : carrier ETA from logistics tracking when available (cross-source view)
    """
    o = po[po["actual_delivery_date"].isna()]
    if material_id is not None:
        o = o[o["material_id"] == material_id]
    o = o.merge(logistics[["po_id", "current_eta", "shipment_status", "transport_mode"]], on="po_id", how="left")
    o = o.assign(open_quantity=o["ordered_quantity"] - o["quantity_received"])
    if view == "ai":
        expected = o["current_eta"].fillna(o["promised_date"])
    else:
        expected = o["promised_date"]
    expected = expected.where(expected > as_of, as_of + pd.Timedelta(days=1))
    o = o.assign(expected_date=expected, delay_vs_promise=(expected - o["promised_date"]).dt.days)
    return o.sort_values("expected_date").reset_index(drop=True)


def project_stock(stock: float, daily_consumption: float, inbound: pd.DataFrame, as_of: pd.Timestamp,
                  horizon: int = config.PROJECTION_HORIZON_DAYS) -> pd.DataFrame:
    """Day-by-day projected end-of-day stock (receipts arrive before consumption).

    Negative values represent unmet demand (shortage).
    """
    dates = pd.date_range(as_of + pd.Timedelta(days=1), periods=horizon, freq="D")
    inb = (inbound.groupby("expected_date")["open_quantity"].sum()
           if len(inbound) else pd.Series(dtype=float))
    receipts = np.array([float(inb.get(d, 0.0)) for d in dates])
    projected = stock + np.cumsum(receipts) - daily_consumption * np.arange(1, horizon + 1)
    return pd.DataFrame({"date": dates, "inbound": receipts, "projected_stock": projected})


def analyse_projection(proj: pd.DataFrame, safety_stock: float, until: pd.Timestamp | None = None) -> dict:
    """Stockout / shortage metrics on a projection.

    ``until`` limits the analysis to the replenishment plan horizon: beyond the last
    known inbound delivery (or the supplier lead time), a new order can still be placed,
    so a negative projection there is not a risk yet.
    """
    if until is not None:
        proj = proj[proj["date"] <= until]
    neg = proj[proj["projected_stock"] < 0]
    below_ss = proj[proj["projected_stock"] < safety_stock]
    shortage = float(-proj["projected_stock"].min()) if len(neg) else 0.0
    return {
        "stockout_date": neg["date"].iloc[0] if len(neg) else None,
        "shortage_qty": shortage,
        "days_without_stock": int(len(neg)),
        "ss_breach_date": below_ss["date"].iloc[0] if len(below_ss) else None,
        "recovery_date": (proj.loc[(proj["date"] > neg["date"].iloc[-1]), "date"].iloc[0]
                          if len(neg) and (proj["date"] > neg["date"].iloc[-1]).any() else None),
    }


def average_consumption(inventory: pd.DataFrame, material_id: str, as_of: pd.Timestamp,
                        days: int = config.CONSUMPTION_WINDOW_DAYS) -> float:
    m = inventory[(inventory["material_id"] == material_id)
                  & (inventory["date"] > as_of - pd.Timedelta(days=days)) & (inventory["date"] <= as_of)]
    return float(m["daily_consumption"].mean()) if len(m) else 0.0


def risk_horizon(inbound: pd.DataFrame, as_of: pd.Timestamp, min_lead_time_days: int) -> pd.Timestamp:
    """End of the period that the current replenishment plan must cover."""
    end = as_of + pd.Timedelta(days=int(min_lead_time_days))
    if len(inbound):
        end = max(end, inbound["expected_date"].max())
    return end


def inventory_status(inventory: pd.DataFrame, products: pd.DataFrame, po: pd.DataFrame,
                     logistics: pd.DataFrame, as_of: pd.Timestamp,
                     suppliers: pd.DataFrame | None = None) -> pd.DataFrame:
    """Snapshot per material with coverage, projections (ERP vs cross-source view) and risk."""
    snap = inventory[inventory["date"] == as_of].set_index("material_id")
    hist90 = inventory[inventory["date"] > as_of - pd.Timedelta(days=90)]
    rows = []
    for _, p in products.iterrows():
        mid = p["material_id"]
        if mid not in snap.index:
            continue
        s = snap.loc[mid]
        stock, ss = float(s["stock_quantity"]), float(s["safety_stock"])
        cons = average_consumption(inventory, mid, as_of)
        cov = days_of_coverage(stock, cons)
        unit_value = float(s["stock_value"]) / stock if stock > 0 else float(p["standard_cost"])
        inb_ai = open_inbound(po, logistics, as_of, mid, "ai")
        inb_erp = open_inbound(po, logistics, as_of, mid, "erp")
        lt = 7
        if suppliers is not None:
            sup = suppliers[(suppliers["material"] == mid) & (suppliers["pricing_model"] != "Spot")]
            lt = int(sup["standard_lead_time_days"].min()) if len(sup) else lt
        until = risk_horizon(inb_ai, as_of, lt)
        pr_ai = analyse_projection(project_stock(stock, cons, inb_ai, as_of), ss, until)
        pr_erp = analyse_projection(project_stock(stock, cons, inb_erp, as_of), ss,
                                    risk_horizon(inb_erp, as_of, lt))
        h = hist90[hist90["material_id"] == mid]
        excess_qty = max(stock - config.TARGET_COVERAGE_DAYS * cons, 0.0) if cov > config.OVERSTOCK_COVERAGE_DAYS else 0.0

        if pr_ai["stockout_date"] is not None:
            risk = "STOCKOUT RISK"
        elif stock < ss:
            risk = "BELOW SAFETY STOCK"
        elif cov > config.OVERSTOCK_COVERAGE_DAYS:
            risk = "OVERSTOCK"
        elif pr_ai["ss_breach_date"] is not None and (pr_ai["ss_breach_date"] - as_of).days <= 7:
            risk = "WATCH"
        else:
            risk = "OK"

        rows.append({
            "material_id": mid,
            "material_name": p["material_name"],
            "category": p["category"],
            "criticality": p["criticality"],
            "unit": p["unit"],
            "warehouse": s["warehouse"],
            "stock": stock,
            "risk_horizon": until,
            "stock_value": float(s["stock_value"]),
            "unit_value": unit_value,
            "safety_stock": ss,
            "daily_consumption": cons,
            "coverage_days": cov,
            "safety_stock_days": ss / cons if cons > 0 else float("inf"),
            "open_po_qty": float(inb_ai["open_quantity"].sum()),
            "next_inbound_erp": inb_erp["expected_date"].iloc[0] if len(inb_erp) else None,
            "next_inbound_ai": inb_ai["expected_date"].iloc[0] if len(inb_ai) else None,
            "next_inbound_supplier": inb_ai["supplier_id"].iloc[0] if len(inb_ai) else None,
            "stockout_date": pr_ai["stockout_date"],
            "stockout_date_erp_view": pr_erp["stockout_date"],
            "shortage_qty": pr_ai["shortage_qty"],
            "days_without_stock": pr_ai["days_without_stock"],
            "recovery_date": pr_ai["recovery_date"],
            "ss_breach_date": pr_ai["ss_breach_date"],
            "ss_breach_days_90d": int((h["stock_quantity"] < h["safety_stock"]).sum()),
            "below_safety_stock": stock < ss,
            "excess_qty": excess_qty,
            "excess_value": excess_qty * unit_value,
            "carrying_cost_saving": excess_qty * unit_value * config.INVENTORY_CARRYING_RATE,
            "risk": risk,
        })
    order = {"STOCKOUT RISK": 0, "BELOW SAFETY STOCK": 1, "WATCH": 2, "OVERSTOCK": 3, "OK": 4}
    out = pd.DataFrame(rows)
    return out.sort_values(by="risk", key=lambda c: c.map(order)).reset_index(drop=True)


def stock_history(inventory: pd.DataFrame, material_id: str, days: int = 180,
                  as_of: pd.Timestamp | None = None) -> pd.DataFrame:
    m = inventory[inventory["material_id"] == material_id]
    as_of = as_of or m["date"].max()
    return m[m["date"] > as_of - pd.Timedelta(days=days)][["date", "stock_quantity", "safety_stock", "daily_consumption"]]


def production_impact(production: pd.DataFrame, material_id: str, shortage_qty: float,
                      daily_consumption: float, as_of: pd.Timestamp,
                      margin_per_t: float = config.CONTRIBUTION_MARGIN_EUR_PER_T,
                      restart_cost: float = config.LINE_RESTART_COST_EUR) -> dict:
    """Estimated production impact of a material shortage (no mitigation)."""
    recent = production[(production["material_id"] == material_id)
                        & (production["date"] > as_of - pd.Timedelta(days=config.CONSUMPTION_WINDOW_DAYS))]
    lines = sorted(recent["production_line"].unique().tolist())
    cons = float(recent["consumption_quantity"].sum())
    yield_ = float(recent["production_quantity"].sum()) / cons if cons > 0 else 0.0
    lost_output = shortage_qty * yield_
    stop_days = shortage_qty / daily_consumption if daily_consumption > 0 else 0.0
    restart = restart_cost * len(lines) if shortage_qty > 0 else 0.0
    return {
        "lines": lines,
        "yield": yield_,
        "lost_output_t": lost_output,
        "line_stop_days": stop_days,
        "lost_margin": lost_output * margin_per_t,
        "restart_cost": restart,
        "total_impact": lost_output * margin_per_t + restart,
    }
