"""Rule-based recommendation engine (V1).

Maps each detected issue type to an action playbook and writes diagnoses from
computed metrics only. This module is deliberately deterministic: a future LLM
layer (see ``src/ai/llm.py``) may rephrase or enrich the output, but the facts
and numbers always come from the analytics layer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.engine.detection import Issue

PLAYBOOKS: dict[str, list[str]] = {
    "stockout_risk": [
        "Contact {supplier} today to confirm a firm shipping date and request expedited delivery.",
        "Check qualified alternative supplier capacity for a spot quantity of {shortage}.",
        "Evaluate an emergency shipment (express road) and its premium freight cost.",
        "Simulate a temporary production adjustment on lines {lines}.",
        "Inform production planning and customer service of the potential impact.",
    ],
    "below_safety_stock": [
        "Review open purchase orders and expedite the next delivery.",
        "Check the safety stock parameter against current consumption.",
    ],
    "supplier_deterioration": [
        "Review supplier delivery performance with {supplier} (monthly OTD, delays, root causes).",
        "Check all open purchase orders with {supplier} and request confirmed dates.",
        "Evaluate alternative supplier capacity for {material}.",
        "Launch a supplier recovery action plan with measurable targets (OTD, PPM, lead time).",
        "Simulate a volume reallocation to reduce the dependency ({share:.0f}% of volume today).",
    ],
    "price_drift": [
        "Investigate the price deviation with {supplier}: index formula, premium, surcharges.",
        "Prepare the negotiation using benchmark prices of comparable suppliers ({benchmark}).",
        "Quantify volume reallocation scenarios as negotiation leverage.",
        "Align price validation rules in the ERP (tolerance vs contract / index).",
    ],
    "quality_spike": [
        "Open an 8D / supplier corrective action request with {supplier}.",
        "Reinforce incoming inspection and quarantine suspect lots ({defect}).",
        "Assess the impact on production quality (melts, scrap rate) and customer risk.",
        "Temporarily shift volume to the alternative supplier if the trend continues.",
    ],
    "contract_deviation": [
        "Check whether a contract amendment exists for the new price.",
        "Claim credit notes for the over-invoiced purchase orders.",
        "Lock the contract price in the ERP purchasing info record.",
    ],
    "overstock": [
        "Pause replenishment until coverage returns to target.",
        "Review reorder point and lot size parameters in the ERP.",
        "Negotiate consignment stock or buy-back with the supplier.",
    ],
    "premium_freight": [
        "Root-cause each express shipment (supplier delay vs internal planning).",
        "Track premium freight as a monthly KPI per supplier.",
    ],
}


def _fmt_money(x: float) -> str:
    return f"€{x:,.0f}"


def actions_for(issue: Issue) -> list[str]:
    m = issue.metrics
    ctx = {
        "supplier": issue.entity_name,
        "material": m.get("material_id", ""),
        "shortage": f"{m.get('shortage_qty', 0):,.0f} t",
        "lines": ", ".join(m.get("impact_lines", []) or []) or "concerned",
        "share": float(m.get("volume_share_pct", 0) or 0),
        "benchmark": _fmt_money(float(m.get("benchmark_price", 0) or 0)) + "/t",
        "defect": m.get("top_defect_recent", "main defect"),
    }
    if issue.issue_type == "stockout_risk":
        ctx["supplier"] = m.get("next_inbound_supplier_name", "the supplier")
    return [a.format(**ctx) for a in PLAYBOOKS.get(issue.issue_type, [])]


def attach_actions(issues: list[Issue]) -> list[Issue]:
    for i in issues:
        i.actions = actions_for(i)
    return issues


def diagnose_supplier(row: pd.Series) -> str:
    """Plain-language diagnosis of a supplier scorecard row (numbers from analytics only)."""
    name = row["supplier_name"]
    w = config.RECENT_WINDOW_DAYS // 30
    sentences = []
    otd_b, otd_r = row["otd_baseline"], row["otd_recent"]
    if np.isfinite(otd_b) and np.isfinite(otd_r):
        drop = otd_b - otd_r
        if drop >= config.OTD_DROP_ALERT_PP:
            sentences.append(
                f"{name} shows a significant deterioration in delivery performance over the last {w} months. "
                f"OTD decreased from {otd_b:.1f}% to {otd_r:.1f}%, while average delay increased from "
                f"{row['avg_delay_baseline']:.1f} to {row['avg_delay_recent']:.1f} days "
                f"({int(row['late_deliveries_recent'])} late deliveries out of {int(row['deliveries_recent'])}, "
                f"{row['delay_when_late_recent']:.1f} days late on average vs "
                f"{row['delay_when_late_baseline']:.1f} previously).")
        elif otd_r < 90:
            sentences.append(f"{name} delivery performance is below target: OTD {otd_r:.1f}% over the last {w} months.")
        else:
            sentences.append(f"{name} delivery performance is stable: OTD {otd_r:.1f}% over the last {w} months "
                             f"(previously {otd_b:.1f}%).")
    lt_inc = row.get("lead_time_increase_pct", 0.0)
    if np.isfinite(lt_inc) and lt_inc >= config.LEAD_TIME_INCREASE_ALERT_PCT:
        sentences.append(f"Actual lead time increased from {row['avg_lead_time_baseline']:.1f} to "
                         f"{row['avg_lead_time_recent']:.1f} days (+{lt_inc:.0f}%).")
    pb, pr, lim = row["ppm_baseline"], row["ppm_recent"], row["ppm_limit"]
    if np.isfinite(pb) and np.isfinite(pr) and pb > 0:
        if pr / pb >= config.PPM_RATIO_ALERT and pr > lim:
            sentences.append(f"Quality is degrading: PPM rose from {pb:,.0f} to {pr:,.0f}, above the {lim:,.0f} limit.")
        else:
            sentences.append(f"Quality within expectations: PPM {pr:,.0f} (limit {lim:,.0f}).")
    gap = row.get("price_gap_pct", 0.0)
    if np.isfinite(gap) and gap >= config.BENCHMARK_GAP_THRESHOLD_PCT:
        ref = "the contract price" if row["pricing_model"] == "Fixed annual" else "comparable suppliers"
        sentences.append(f"Price is {gap:+.1f}% above {ref} (price evolution {row['price_evolution_pct']:+.1f}% "
                         f"over the period).")
    sentences.append(f"Overall risk score: {row['risk_score']:.0f}/100 ({row['risk_level']}).")
    return " ".join(sentences)


def supplier_actions(row: pd.Series, material_name: str | None = None) -> list[str]:
    acts = []
    if np.isfinite(row["otd_recent"]) and (row["otd_drop_pp"] >= config.OTD_DROP_ALERT_PP or row["otd_recent"] < 90):
        acts += PLAYBOOKS["supplier_deterioration"]
    if np.isfinite(row["ppm_recent"]) and row["ppm_recent"] > row["ppm_limit"]:
        acts += [a for a in PLAYBOOKS["quality_spike"][:2]]
    if np.isfinite(row.get("price_gap_pct", np.nan)) and row["price_gap_pct"] >= config.BENCHMARK_GAP_THRESHOLD_PCT:
        acts += PLAYBOOKS["price_drift"][:2] if row["pricing_model"] != "Fixed annual" else PLAYBOOKS["contract_deviation"][:2]
    if not acts:
        acts = ["No corrective action required — keep standard quarterly business review."]
    ctx = {"supplier": row["supplier_name"], "material": material_name or row["material_id"], "share": row["volume_share_pct"],
           "benchmark": "peer prices", "defect": "main defect"}
    seen, out = set(), []
    for a in acts:
        t = a.format(**ctx)
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out
