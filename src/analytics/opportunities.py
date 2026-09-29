"""Cost & opportunity engine: turns analytics results into quantified opportunities.

Every financial value is computed here from the data and the documented
assumptions in ``src/config.py``. Each opportunity carries its formula so that
the number can be audited in the UI.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src import config
from src.analytics.quality import cost_of_non_quality
from src.utils.formatting import fmt_eur, fmt_qty

ANNUAL = "Annual savings"
RISK = "Risk exposure"
CASH = "One-off cash release"


@dataclass
class Opportunity:
    opp_id: str
    category: str            # Supplier price | Inventory | Logistics | Supplier performance | Quality
    title: str
    entity: str
    criticality: str         # Critical | High | Medium | Low
    value: float             # EUR
    value_type: str          # ANNUAL | RISK | CASH
    formula: str
    explanation: str
    recommendation: str
    metrics: dict = field(default_factory=dict)


def _crit_from_value(value: float) -> str:
    if value >= 150_000:
        return "Critical"
    if value >= 30_000:
        return "High"
    if value >= 5_000:
        return "Medium"
    return "Low"


def price_opportunities(anomalies: pd.DataFrame, products: pd.DataFrame) -> list[Opportunity]:
    unit = products.set_index("material_id")["unit"]
    out = []
    for _, r in anomalies.iterrows():
        u = unit.get(r["material_id"], "t")
        out.append(Opportunity(
            opp_id=f"OPP-PRICE-{r['supplier_id']}",
            category="Supplier price",
            title=f"Supplier price drift — {r['supplier_name']}",
            entity=r["supplier_name"],
            criticality=_crit_from_value(r["potential_savings"]),
            value=float(r["potential_savings"]),
            value_type=ANNUAL,
            formula=(f"({fmt_eur(r['current_price'], unit=u)} − {fmt_eur(r['benchmark_price'], unit=u)}) "
                     f"× {fmt_qty(r['annual_volume'], u)} = {fmt_eur(r['potential_savings'])}"),
            explanation=(f"Current price {fmt_eur(r['current_price'], unit=u)} (last {config.PRICE_RECENT_DAYS} days) vs "
                         f"{fmt_eur(r['baseline_price'], unit=u)} at the start of the period "
                         f"({r['price_change_pct']:+.1f}%), while comparable suppliers moved "
                         f"{r['peer_change_pct']:+.1f}%. Gap vs peer benchmark: {r['gap_vs_benchmark_pct']:+.1f}%."),
            recommendation=("Investigate the price deviation (index formula, premium, surcharges) and renegotiate "
                            "using benchmark supplier prices; prepare volume reallocation as leverage."),
            metrics=r.to_dict(),
        ))
    return out


def contract_opportunities(deviations: pd.DataFrame, products: pd.DataFrame) -> list[Opportunity]:
    unit = products.set_index("material_id")["unit"]
    out = []
    for _, r in deviations.iterrows():
        u = unit.get(r["material_id"], "t")
        out.append(Opportunity(
            opp_id=f"OPP-CONTRACT-{r['supplier_id']}",
            category="Supplier price",
            title=f"Invoiced above contract — {r['supplier_name']}",
            entity=r["supplier_name"],
            criticality=_crit_from_value(r["potential_savings"]),
            value=float(r["potential_savings"]),
            value_type=ANNUAL,
            formula=(f"({fmt_eur(r['current_price'], unit=u)} − {fmt_eur(r['contracted_price'], unit=u)} contract) "
                     f"× {fmt_qty(r['annual_volume'], u)} = {fmt_eur(r['potential_savings'])}"),
            explanation=(f"Fixed annual contract at {fmt_eur(r['contracted_price'], unit=u)} but recent purchase orders "
                         f"are priced at {fmt_eur(r['current_price'], unit=u)} ({r['contract_gap_pct']:+.1f}%)."),
            recommendation="Check contract amendments; claim credit notes for over-invoiced orders; block price in ERP info record.",
            metrics=r.to_dict(),
        ))
    return out


def spot_opportunity(po: pd.DataFrame, suppliers: pd.DataFrame, price_stats: pd.DataFrame,
                     as_of: pd.Timestamp) -> list[Opportunity]:
    """Premium paid on emergency spot purchases vs. the regular peer price at the time."""
    spot_ids = suppliers.loc[suppliers["pricing_model"] == "Spot", "supplier_id"]
    year = po[(po["order_date"] > as_of - pd.Timedelta(days=365)) & (po["order_date"] <= as_of)]
    spot = year[year["supplier_id"].isin(spot_ids)]
    if spot.empty:
        return []
    regular = year[~year["supplier_id"].isin(spot_ids)]
    premium, vol = 0.0, 0.0
    for _, r in spot.iterrows():
        ref = regular[(regular["material_id"] == r["material_id"])
                      & ((regular["order_date"] - r["order_date"]).abs() <= pd.Timedelta(days=30))]
        if ref.empty:
            continue
        ref_price = float((ref["ordered_quantity"] * ref["unit_price"]).sum() / ref["ordered_quantity"].sum())
        premium += (r["unit_price"] - ref_price) * r["ordered_quantity"]
        vol += r["ordered_quantity"]
    if premium <= 0:
        return []
    return [Opportunity(
        opp_id="OPP-SPOT",
        category="Supplier price",
        title="Emergency spot purchases (primary aluminium)",
        entity=", ".join(suppliers.loc[suppliers["supplier_id"].isin(spot["supplier_id"]), "supplier_name"]),
        criticality=_crit_from_value(premium),
        value=premium,
        value_type=ANNUAL,
        formula=f"Σ (spot price − regular price ±30 days) × quantity = {fmt_eur(premium)} on {fmt_qty(vol, 't')}",
        explanation=(f"{len(spot)} spot purchase orders placed in the last 12 months to avoid shortages, "
                     "priced above regular supplier prices at the same period."),
        recommendation="Reduce emergency buys by fixing replenishment parameters and supplier reliability upstream.",
        metrics={"spot_orders": int(len(spot)), "spot_volume": vol},
    )]


def inventory_opportunities(inv: pd.DataFrame) -> list[Opportunity]:
    out = []
    for _, r in inv[inv["excess_qty"] > 0].iterrows():
        out.append(Opportunity(
            opp_id=f"OPP-INV-{r['material_id']}",
            category="Inventory",
            title=f"Overstock — {r['material_name']}",
            entity=r["material_name"],
            criticality=_crit_from_value(r["carrying_cost_saving"]),
            value=float(r["carrying_cost_saving"]),
            value_type=ANNUAL,
            formula=(f"Excess {fmt_qty(r['excess_qty'], r['unit'])} × {fmt_eur(r['unit_value'], unit=r['unit'])} "
                     f"= {fmt_eur(r['excess_value'])} tied up × {config.INVENTORY_CARRYING_RATE:.0%} carrying rate "
                     f"= {fmt_eur(r['carrying_cost_saving'])}/year"),
            explanation=(f"Coverage of {r['coverage_days']:.0f} days vs a {config.TARGET_COVERAGE_DAYS}-day target. "
                         f"Reducing to target releases {fmt_eur(r['excess_value'])} of working capital (one-off)."),
            recommendation="Pause replenishment, review reorder point / lot size in the ERP, negotiate consignment stock.",
            metrics={"excess_value": float(r["excess_value"]), "coverage_days": float(r["coverage_days"])},
        ))
    return out


def logistics_opportunity(express: pd.DataFrame) -> list[Opportunity]:
    if express.empty or express["premium"].sum() <= 0:
        return []
    total = float(express["premium"].sum())
    detail = ", ".join(f"{r.supplier_name}: {r.express_shipments} shipments" for r in express.itertuples())
    return [Opportunity(
        opp_id="OPP-LOG-EXPRESS",
        category="Logistics",
        title="Premium freight (express road shipments)",
        entity="Inbound logistics",
        criticality=_crit_from_value(total),
        value=total,
        value_type=ANNUAL,
        formula=(f"Σ express cost × (1 − 1/{config.EXPRESS_COST_MULTIPLIER}) = {fmt_eur(total)} premium over 12 months"),
        explanation=f"Express shipments used to recover late supply: {detail}.",
        recommendation="Treat premium freight as a KPI: root-cause each express order (supplier delay vs planning).",
        metrics={"express_shipments": int(express["express_shipments"].sum())},
    )]


def supplier_performance_opportunities(deteriorating: pd.DataFrame, inv: pd.DataFrame,
                                       impacts: dict[str, dict]) -> list[Opportunity]:
    out = []
    for _, r in deteriorating.iterrows():
        mat = inv[inv["material_id"] == r["material_id"]]
        exposure = float(impacts.get(r["material_id"], {}).get("total_impact", 0.0))
        out.append(Opportunity(
            opp_id=f"OPP-PERF-{r['supplier_id']}",
            category="Supplier performance",
            title=f"Reduce dependency on {r['supplier_name']}",
            entity=r["supplier_name"],
            criticality="Critical" if exposure > 0 else "High",
            value=exposure,
            value_type=RISK,
            formula=(f"Production impact of the current {mat['material_name'].iloc[0] if len(mat) else ''} shortage "
                     f"= {fmt_eur(exposure)} (no mitigation)"),
            explanation=(f"{r['supplier_name']} carries {r['volume_share_pct']:.0f}% of the volume while OTD fell from "
                         f"{r['otd_baseline']:.1f}% to {r['otd_recent']:.1f}% and PPM rose from "
                         f"{r['ppm_baseline']:.0f} to {r['ppm_recent']:.0f}."),
            recommendation="Launch a supplier recovery plan and simulate volume reallocation to qualified alternatives.",
            metrics=r.to_dict(),
        ))
    return out


def quality_opportunities(spikes: pd.DataFrame, suppliers: pd.DataFrame, price_stats: pd.DataFrame) -> list[Opportunity]:
    names = suppliers.set_index("supplier_id")["supplier_name"]
    out = []
    for _, r in spikes.iterrows():
        ps = price_stats[(price_stats["supplier_id"] == r["supplier_id"]) & (price_stats["material_id"] == r["material_id"])]
        price = float(ps["avg_price_12m"].iloc[0]) if len(ps) else float(r["standard_cost"])
        cost = cost_of_non_quality(r["recent_defects"], r["unit"], price, config.QUALITY_RECENT_DAYS)
        name = names.get(r["supplier_id"], r["supplier_id"])
        out.append(Opportunity(
            opp_id=f"OPP-QUAL-{r['supplier_id']}",
            category="Quality",
            title=f"Quality drift — {name}",
            entity=name,
            criticality="High" if r["ppm_ratio"] >= 3 else "Medium",
            value=cost,
            value_type=ANNUAL,
            formula=(f"{r['recent_defects'] / (1000 if r['unit'] == 't' else 1):,.2f} {r['unit']} non-conforming in "
                     f"{config.QUALITY_RECENT_DAYS} days × {fmt_eur(price, unit=r['unit'])} × "
                     f"{config.NON_QUALITY_HANDLING_RATE:.0%} handling, annualised = {fmt_eur(cost)}"),
            explanation=(f"PPM rose from {r['ppm_baseline']:.0f} to {r['ppm_recent']:.0f} (×{r['ppm_ratio']:.1f}); "
                         f"main defect: {r['top_defect_recent']}; {r['recent_nonconforming_lots']} non-conforming lots."),
            recommendation="Open an 8D with the supplier, reinforce incoming inspection, quarantine suspect lots.",
            metrics=r.to_dict(),
        ))
    return out


def build_opportunities(**kw) -> list[Opportunity]:
    opps = (price_opportunities(kw["price_anomalies"], kw["products"])
            + contract_opportunities(kw["contract_deviations"], kw["products"])
            + spot_opportunity(kw["po"], kw["suppliers"], kw["price_stats"], kw["as_of"])
            + inventory_opportunities(kw["inventory"])
            + logistics_opportunity(kw["express"])
            + supplier_performance_opportunities(kw["deteriorating"], kw["inventory"], kw["impacts"])
            + quality_opportunities(kw["quality_spikes"], kw["suppliers"], kw["price_stats"]))
    order = config.SEVERITY_ORDER
    return sorted(opps, key=lambda o: (order.get(o.criticality, 9), -o.value))


def total_annual_savings(opps: list[Opportunity]) -> float:
    return float(np.sum([o.value for o in opps if o.value_type == ANNUAL]))


def opportunities_frame(opps: list[Opportunity]) -> pd.DataFrame:
    return pd.DataFrame([{
        "Category": o.category, "Opportunity": o.title, "Entity": o.entity, "Criticality": o.criticality,
        "Value (€)": round(o.value), "Value type": o.value_type,
    } for o in opps])
