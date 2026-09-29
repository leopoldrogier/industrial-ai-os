"""Cost & Opportunity Engine page."""
from __future__ import annotations

import streamlit as st

from src import config
from src.analytics.opportunities import ANNUAL, RISK, opportunities_frame, total_annual_savings
from src.ui import charts, components as ui, theme
from src.ui.state import get_result
from src.utils.formatting import fmt_eur, severity_icon

ICONS = {"Supplier price": "💰", "Inventory": "📦", "Logistics": "🚚", "Supplier performance": "🏭", "Quality": "🔬"}


def render() -> None:
    r = get_result()
    opps = r.opportunities
    ui.page_header("Cost & Opportunity Engine", "Opportunities detected automatically and quantified from the data",
                   r.as_of, ("DETECTION", "IMPACT", "RECOMMENDATION"))
    total = total_annual_savings(opps)
    risk = sum(o.value for o in opps if o.value_type == RISK)
    cash = float(r.inventory["excess_value"].sum())
    ui.kpi_row([
        ui.kpi("💰 Annual savings potential", fmt_eur(total, compact=True), f"{sum(o.value_type == ANNUAL for o in opps)} opportunities",
               accent=theme.EMPHASIS),
        ui.kpi("Working capital release", fmt_eur(cash, compact=True), "one-off, excess inventory"),
        ui.kpi("Risk exposure", fmt_eur(risk, compact=True), "production impact avoidable", accent=theme.STATUS["critical"]),
        ui.kpi("Opportunities", str(len(opps)), f"{len({o.category for o in opps})} categories"),
    ])
    st.write("")

    df = opportunities_frame(opps)
    c1, c2 = st.columns([1, 1.4], gap="large")
    with c1:
        st.plotly_chart(charts.opportunity_bars(df[df["Value type"] == ANNUAL]), width="stretch",
                        config={"displayModeBar": False})
    with c2:
        st.dataframe(df, hide_index=True, width="stretch", column_config={
            "Value (€)": st.column_config.NumberColumn(format="localized")})

    cats = ["All"] + sorted(df["Category"].unique())
    cat = st.segmented_control("Category", cats, default="All")
    st.caption(f"Assumptions: inventory carrying rate {config.INVENTORY_CARRYING_RATE:.0%}/year, non-quality handling "
               f"{config.NON_QUALITY_HANDLING_RATE:.0%} of material value, express freight = "
               f"{config.EXPRESS_COST_MULTIPLIER}× standard road. Editable in src/config.py.")
    for o in opps:
        if cat not in (None, "All") and o.category != cat:
            continue
        head = f"{ICONS.get(o.category, '•')} {o.title} — {fmt_eur(o.value)}" + (
            "/year" if o.value_type == ANNUAL else f" ({o.value_type.lower()})")
        with st.expander(f"{severity_icon(o.criticality)} {head}", expanded=o.criticality == "Critical"):
            a, b = st.columns([1.4, 1])
            with a:
                st.markdown(f"**Category:** {o.category} · **Criticality:** {o.criticality} · **Entity:** {o.entity}")
                st.markdown(f"**Explanation.** {o.explanation}")
                st.markdown(f"**Recommendation.** {o.recommendation}")
            with b:
                st.markdown("**Calculation**")
                st.code(o.formula, language=None)
                st.caption(o.value_type)
