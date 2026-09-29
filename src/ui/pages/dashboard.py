"""Dashboard — INDUSTRIAL AI OS home page."""
from __future__ import annotations

import streamlit as st

from src.ui import charts, components as ui, theme
from src.ui.state import get_result
from src.utils.formatting import fmt_eur


def render() -> None:
    r = get_result()
    k = r.kpis
    ui.page_header("INDUSTRIAL AI OS", "Intelligent Industrial Operations Platform", r.as_of,
                   ("DATA", "DETECTION", "DIAGNOSIS", "IMPACT"))
    ui.roles_strip()
    st.write("")

    ui.kpi_row([
        ui.kpi("Inventory value", fmt_eur(k["inventory_value"], compact=True), f"{len(r.inventory)} materials tracked"),
        ui.kpi("Purchasing spend", fmt_eur(k["purchasing_spend"], compact=True), f"last 12 months · {r.spend['po_count']} POs"),
        ui.kpi("Active suppliers", str(k["active_suppliers"]), f"{k['high_risk_suppliers']} high-risk"),
        ui.kpi("Supply risks", str(k["supply_risks"]),
               f"{k['materials_at_risk']} material(s) · {k['high_risk_suppliers']} supplier(s)",
               accent=theme.STATUS["serious"] if k["supply_risks"] else None),
    ])
    st.write("")
    ui.kpi_row([
        ui.kpi("🔴 Critical issues", str(k["critical_issues"]), f"{k['total_issues']} issues detected in total",
               accent=theme.STATUS["critical"]),
        ui.kpi("🟠 Opportunities", str(k["opportunities"]), "quantified by the opportunity engine",
               accent=theme.STATUS["serious"]),
        ui.kpi("🟢 Stable areas", str(k["stable_areas"]), "low-risk suppliers + materials OK", accent=theme.STATUS["good"]),
        ui.kpi("💰 Potential savings", f"{fmt_eur(k['potential_savings'], compact=True)}/year",
               "annual savings, computed from data", accent=theme.EMPHASIS),
    ])

    st.write("")
    left, right = st.columns([1.55, 1], gap="large")
    with left:
        st.subheader("Priority issues")
        for issue in r.issues:
            ui.issue_card(issue)
        c1, c2, c3 = st.columns(3)
        with c1:
            ui.link("inventory", "Inventory intelligence", ":material/inventory_2:")
        with c2:
            ui.link("suppliers", "Supplier performance", ":material/factory:")
        with c3:
            ui.link("copilot", "Ask the AI Copilot", ":material/smart_toy:")
    with right:
        st.subheader("Trends")
        names = dict(zip(r.data.products["material_id"], r.data.products["material_name"]))
        st.plotly_chart(charts.monthly_spend(r.monthly_spend, names, "Monthly purchasing spend"),
                        width="stretch", config={"displayModeBar": False})
        top = r.scorecard.head(6)
        st.plotly_chart(charts.hbar(top.assign(label=top["supplier_name"]), "risk_score", "label",
                                    "Top supplier risk scores", "supplier_id", top["supplier_id"].iloc[0],
                                    fmt="%{x:.0f}/100", height=260),
                        width="stretch", config={"displayModeBar": False})
    ui.disclaimer()
