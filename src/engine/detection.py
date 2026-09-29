"""Issue detection: turns analytics outputs into prioritised, evidence-backed issues.

DETECTION -> DIAGNOSIS -> IMPACT. Recommended actions are attached afterwards by
``src.ai.recommendations`` (kept separate so that a future LLM/agent layer can
replace or enrich them without touching the detection rules).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src import config
from src.utils.formatting import fmt_date, fmt_eur, fmt_pct, fmt_qty


@dataclass
class Issue:
    issue_id: str
    issue_type: str          # stockout_risk | supplier_deterioration | price_drift | quality_spike | ...
    severity: str            # Critical | High | Medium | Low
    category: str
    title: str
    entity_id: str
    entity_name: str
    summary: str
    impact: float = 0.0
    impact_label: str = ""
    metrics: dict = field(default_factory=dict)
    sources: list[str] = field(default_factory=list)
    page: str = "dashboard"
    actions: list[str] = field(default_factory=list)


def detect_stockout_risks(inv: pd.DataFrame, impacts: dict[str, dict], data) -> list[Issue]:
    out = []
    for _, r in inv[inv["risk"] == "STOCKOUT RISK"].iterrows():
        imp = impacts.get(r["material_id"], {})
        sup = data.supplier_name(r["next_inbound_supplier"]) if r["next_inbound_supplier"] else "n/a"
        sev = "Critical" if r["criticality"] in ("Critical", "High") else "High"
        out.append(Issue(
            issue_id=f"ISS-STOCK-{r['material_id']}",
            issue_type="stockout_risk",
            severity=sev,
            category="Supply risk",
            title=f"Stockout risk — {r['material_name']}",
            entity_id=r["material_id"],
            entity_name=r["material_name"],
            summary=(f"Stock {fmt_qty(r['stock'], r['unit'])} covers {r['coverage_days']:.1f} days at "
                     f"{fmt_qty(r['daily_consumption'], r['unit'])}/day. Next delivery ({sup}) now expected "
                     f"{fmt_date(r['next_inbound_ai'])} vs {fmt_date(r['next_inbound_erp'])} promised in the ERP. "
                     f"Estimated stockout {fmt_date(r['stockout_date'])}, shortage {fmt_qty(r['shortage_qty'], r['unit'])}."),
            impact=float(imp.get("total_impact", 0.0)),
            impact_label="Production impact if no action",
            metrics={**r.to_dict(), **{f"impact_{k}": v for k, v in imp.items()},
                     "next_inbound_supplier_name": sup},
            sources=["inventory", "purchase_orders", "logistics", "production"],
            page="inventory",
        ))
    for _, r in inv[inv["risk"] == "BELOW SAFETY STOCK"].iterrows():
        out.append(Issue(
            issue_id=f"ISS-SS-{r['material_id']}", issue_type="below_safety_stock", severity="High",
            category="Supply risk", title=f"Below safety stock — {r['material_name']}",
            entity_id=r["material_id"], entity_name=r["material_name"],
            summary=f"Stock {fmt_qty(r['stock'], r['unit'])} below safety stock {fmt_qty(r['safety_stock'], r['unit'])}.",
            metrics=r.to_dict(), sources=["inventory"], page="inventory",
        ))
    return out


def detect_supplier_deterioration(deteriorating: pd.DataFrame, products: pd.DataFrame) -> list[Issue]:
    crit = products.set_index("material_id")["criticality"]
    out = []
    for _, r in deteriorating.iterrows():
        critical = r["volume_share_pct"] >= 20 and crit.get(r["material_id"]) in ("Critical", "High")
        parts = [f"OTD {r['otd_baseline']:.1f}% → {r['otd_recent']:.1f}%",
                 f"avg delay {r['avg_delay_baseline']:.1f} → {r['avg_delay_recent']:.1f} days"]
        if r["signal_lead_time"]:
            parts.append(f"lead time {r['avg_lead_time_baseline']:.1f} → {r['avg_lead_time_recent']:.1f} days")
        if r["signal_ppm"]:
            parts.append(f"PPM {r['ppm_baseline']:.0f} → {r['ppm_recent']:.0f}")
        out.append(Issue(
            issue_id=f"ISS-PERF-{r['supplier_id']}",
            issue_type="supplier_deterioration",
            severity="Critical" if critical else "High",
            category="Supplier performance",
            title=f"Supplier performance deterioration — {r['supplier_name']}",
            entity_id=r["supplier_id"],
            entity_name=r["supplier_name"],
            summary=(f"Last {config.RECENT_WINDOW_DAYS} days vs previous 9 months: " + ", ".join(parts)
                     + f". Supplier carries {r['volume_share_pct']:.0f}% of the material volume."),
            impact=float(r["spend"]),
            impact_label="Annual spend exposed",
            metrics=r.to_dict(),
            sources=["purchase_orders", "quality", "suppliers"],
            page="suppliers",
        ))
    return out


def detect_price_drift(anomalies: pd.DataFrame) -> list[Issue]:
    out = []
    for _, r in anomalies.iterrows():
        out.append(Issue(
            issue_id=f"ISS-PRICE-{r['supplier_id']}",
            issue_type="price_drift",
            severity="Critical" if r["potential_savings"] >= 100_000 else "High",
            category="Purchasing",
            title=f"Abnormal price increase — {r['supplier_name']}",
            entity_id=r["supplier_id"],
            entity_name=r["supplier_name"],
            summary=(f"Price {fmt_eur(r['baseline_price'], unit='t')} → {fmt_eur(r['current_price'], unit='t')} "
                     f"({fmt_pct(r['price_change_pct'], signed=True)}) while comparable suppliers moved "
                     f"{fmt_pct(r['peer_change_pct'], signed=True)}. Current gap vs benchmark "
                     f"{fmt_eur(r['benchmark_price'], unit='t')}: {fmt_pct(r['gap_vs_benchmark_pct'], signed=True)}."),
            impact=float(r["potential_savings"]),
            impact_label="Potential savings / year",
            metrics=r.to_dict(),
            sources=["purchase_orders", "suppliers"],
            page="purchasing",
        ))
    return out


def detect_quality_issues(spikes: pd.DataFrame, suppliers: pd.DataFrame, exclude: set[str],
                          quality_costs: dict[str, float]) -> list[Issue]:
    names = suppliers.set_index("supplier_id")["supplier_name"]
    out = []
    for _, r in spikes.iterrows():
        if r["supplier_id"] in exclude:
            continue  # already covered by a supplier deterioration issue
        name = names.get(r["supplier_id"], r["supplier_id"])
        out.append(Issue(
            issue_id=f"ISS-QUAL-{r['supplier_id']}",
            issue_type="quality_spike",
            severity="Critical" if r["ppm_ratio"] >= 3 else "High",
            category="Quality",
            title=f"Quality spike — {name}",
            entity_id=r["supplier_id"],
            entity_name=name,
            summary=(f"PPM {r['ppm_baseline']:,.0f} → {r['ppm_recent']:,.0f} (×{r['ppm_ratio']:.1f}) over the last "
                     f"{config.QUALITY_RECENT_DAYS} days, limit {r['quality_ppm_limit']:,.0f}. Main defect: "
                     f"{r['top_defect_recent']} ({r['top_defect_share_pct']:.0f}% of defects); "
                     f"{r['recent_nonconforming_lots']} non-conforming lots."),
            impact=float(quality_costs.get(r["supplier_id"], 0.0)),
            impact_label="Cost of non-quality / year",
            metrics=r.to_dict(),
            sources=["quality", "purchase_orders"],
            page="suppliers",
        ))
    return out


def detect_contract_deviations(dev: pd.DataFrame) -> list[Issue]:
    return [Issue(
        issue_id=f"ISS-CONTRACT-{r['supplier_id']}", issue_type="contract_deviation",
        severity="High" if r["potential_savings"] >= 50_000 else "Medium",
        category="Purchasing", title=f"Invoiced above contract — {r['supplier_name']}",
        entity_id=r["supplier_id"], entity_name=r["supplier_name"],
        summary=(f"Recent price {fmt_eur(r['current_price'], unit='t')} vs contract "
                 f"{fmt_eur(r['contracted_price'], unit='t')} ({fmt_pct(r['contract_gap_pct'], signed=True)})."),
        impact=float(r["potential_savings"]), impact_label="Potential savings / year",
        metrics=r.to_dict(), sources=["purchase_orders", "suppliers"], page="purchasing",
    ) for _, r in dev.iterrows()]


def detect_overstock(inv: pd.DataFrame) -> list[Issue]:
    return [Issue(
        issue_id=f"ISS-OVER-{r['material_id']}", issue_type="overstock", severity="Medium",
        category="Inventory", title=f"Overstock — {r['material_name']}",
        entity_id=r["material_id"], entity_name=r["material_name"],
        summary=(f"Coverage {r['coverage_days']:.0f} days (> {config.OVERSTOCK_COVERAGE_DAYS}). "
                 f"Excess stock value {fmt_eur(r['excess_value'])}."),
        impact=float(r["carrying_cost_saving"]), impact_label="Carrying cost / year",
        metrics=r.to_dict(), sources=["inventory"], page="inventory",
    ) for _, r in inv[inv["risk"] == "OVERSTOCK"].iterrows()]


def detect_premium_freight(express: pd.DataFrame) -> list[Issue]:
    if express.empty:
        return []
    total = float(express["premium"].sum())
    return [Issue(
        issue_id="ISS-LOG-EXPRESS", issue_type="premium_freight", severity="Medium",
        category="Logistics", title="Recurring premium freight",
        entity_id="LOGISTICS", entity_name="Inbound logistics",
        summary=f"{int(express['express_shipments'].sum())} express shipments in 12 months, premium {fmt_eur(total)}.",
        impact=total, impact_label="Premium paid / year",
        metrics={"by_supplier": express.to_dict("records")}, sources=["logistics"], page="opportunities",
    )]


TYPE_PRIORITY = {"stockout_risk": 0, "below_safety_stock": 1, "supplier_deterioration": 2, "price_drift": 3,
                 "quality_spike": 4, "contract_deviation": 5, "overstock": 6, "premium_freight": 7}


def sort_issues(issues: list[Issue]) -> list[Issue]:
    """Severity first; operational continuity (supply) before cost topics; then impact."""
    return sorted(issues, key=lambda i: (config.SEVERITY_ORDER.get(i.severity, 9),
                                         TYPE_PRIORITY.get(i.issue_type, 9), -i.impact))
