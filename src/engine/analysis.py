"""Analysis pipeline: DATA -> DETECTION -> DIAGNOSIS -> IMPACT -> OPTIONS -> RECOMMENDATION.

``run_analysis`` computes everything once; the UI and the copilot only read the
resulting ``AnalysisResult`` (single source of computed truth).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.ai.recommendations import attach_actions
from src.analytics import inventory as inv_a
from src.analytics import logistics as log_a
from src.analytics import opportunities as opp_a
from src.analytics import purchasing as pur_a
from src.analytics import quality as q_a
from src.analytics import suppliers as sup_a
from src.data_loader import DataBundle
from src.engine import detection as det
from src.engine.scenarios import ScenarioSet, build_stockout_scenarios


@dataclass
class AnalysisResult:
    data: DataBundle
    as_of: pd.Timestamp
    spend: dict
    spend_by_supplier: pd.DataFrame
    spend_by_material: pd.DataFrame
    monthly_spend: pd.DataFrame
    price_stats: pd.DataFrame
    price_anomalies: pd.DataFrame
    contract_deviations: pd.DataFrame
    scorecard: pd.DataFrame
    deteriorating: pd.DataFrame
    inventory: pd.DataFrame
    impacts: dict
    quality_trends: pd.DataFrame
    quality_spikes: pd.DataFrame
    express: pd.DataFrame
    logistics_modes: pd.DataFrame
    delayed_shipments: pd.DataFrame
    issues: list[det.Issue]
    opportunities: list[opp_a.Opportunity]
    scenarios: dict[str, ScenarioSet]
    kpis: dict = field(default_factory=dict)

    def issue(self, issue_type: str) -> list[det.Issue]:
        return [i for i in self.issues if i.issue_type == issue_type]

    def supplier_row(self, supplier_id: str) -> pd.Series | None:
        r = self.scorecard[self.scorecard["supplier_id"] == supplier_id]
        return r.iloc[0] if len(r) else None


def run_analysis(data: DataBundle) -> AnalysisResult:
    as_of = data.as_of
    po, q, sup, prod = data.purchase_orders, data.quality, data.suppliers, data.products

    # --- Purchasing
    spend = pur_a.spend_summary(po, as_of)
    pstats = pur_a.price_stats(po, sup, as_of)
    anomalies = pur_a.detect_price_anomalies(pstats)
    deviations = pur_a.contract_deviations(pstats)

    # --- Suppliers & quality
    scorecard = sup_a.supplier_scorecard(po, q, sup, prod, as_of)
    deteriorating = sup_a.detect_deteriorating_suppliers(scorecard)
    qtrends = q_a.quality_trends(q, prod, as_of)
    qspikes = q_a.detect_quality_spikes(qtrends)

    # --- Inventory & production impact
    inventory = inv_a.inventory_status(data.inventory, prod, po, data.logistics, as_of, sup)
    impacts = {}
    for _, r in inventory[inventory["shortage_qty"] > 0].iterrows():
        impacts[r["material_id"]] = inv_a.production_impact(
            data.production, r["material_id"], r["shortage_qty"], r["daily_consumption"], as_of)

    # --- Logistics
    express = log_a.express_premium(data.logistics, sup, as_of)
    modes = log_a.cost_by_mode(data.logistics, as_of)
    delayed = log_a.delayed_open_shipments(data.logistics)

    # --- Opportunities
    opps = opp_a.build_opportunities(
        price_anomalies=anomalies, contract_deviations=deviations, po=po, suppliers=sup, products=prod,
        price_stats=pstats, as_of=as_of, inventory=inventory, express=express, deteriorating=deteriorating,
        impacts=impacts, quality_spikes=qspikes)
    quality_costs = {o.metrics.get("supplier_id"): o.value for o in opps if o.category == "Quality"}

    # --- Issues (detection) + recommended actions
    issues = (det.detect_stockout_risks(inventory, impacts, data)
              + det.detect_supplier_deterioration(deteriorating, prod)
              + det.detect_price_drift(anomalies)
              + det.detect_quality_issues(qspikes, sup, set(deteriorating["supplier_id"]), quality_costs)
              + det.detect_contract_deviations(deviations)
              + det.detect_overstock(inventory)
              + det.detect_premium_freight(express))
    issues = attach_actions(det.sort_issues(issues))

    # --- Options / scenarios for each stockout risk
    scenarios = {}
    for _, r in inventory[inventory["risk"] == "STOCKOUT RISK"].iterrows():
        scenarios[r["material_id"]] = build_stockout_scenarios(r, impacts[r["material_id"]], data, pstats,
                                                               scorecard, as_of)

    at_risk_materials = int(inventory["risk"].isin(["STOCKOUT RISK", "BELOW SAFETY STOCK", "WATCH"]).sum())
    kpis = {
        "inventory_value": float(inventory["stock_value"].sum()),
        "purchasing_spend": spend["total_spend"],
        "active_suppliers": spend["active_suppliers"],
        "critical_issues": sum(1 for i in issues if i.severity == "Critical"),
        "total_issues": len(issues),
        "opportunities": len(opps),
        "potential_savings": opp_a.total_annual_savings(opps),
        "supply_risks": at_risk_materials + int((scorecard["risk_level"] == "High").sum()),
        "stable_areas": int((scorecard["risk_level"] == "Low").sum()) + int((inventory["risk"] == "OK").sum()),
        "materials_at_risk": at_risk_materials,
        "high_risk_suppliers": int((scorecard["risk_level"] == "High").sum()),
    }

    return AnalysisResult(
        data=data, as_of=as_of, spend=spend,
        spend_by_supplier=pur_a.spend_by(po, sup, prod, as_of, "supplier"),
        spend_by_material=pur_a.spend_by(po, sup, prod, as_of, "material"),
        monthly_spend=pur_a.monthly_spend(po, as_of),
        price_stats=pstats, price_anomalies=anomalies, contract_deviations=deviations,
        scorecard=scorecard, deteriorating=deteriorating, inventory=inventory, impacts=impacts,
        quality_trends=qtrends, quality_spikes=qspikes, express=express, logistics_modes=modes,
        delayed_shipments=delayed, issues=issues, opportunities=opps, scenarios=scenarios, kpis=kpis,
    )
