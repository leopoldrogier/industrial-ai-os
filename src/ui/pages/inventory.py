"""Inventory Intelligence page."""
from __future__ import annotations

import numpy as np
import streamlit as st

from src import config
from src.analytics.inventory import open_inbound, project_stock, stock_history
from src.ui import charts, components as ui, theme
from src.ui.state import get_result
from src.utils.formatting import fmt_date, fmt_days, fmt_eur, fmt_qty

CHART_CFG = {"displayModeBar": False}


def render() -> None:
    r = get_result()
    inv = r.inventory
    ui.page_header("Inventory Intelligence", "Coverage, projected stock and stockout risk — WMS × ERP × carrier tracking",
                   r.as_of, ("DATA", "DETECTION", "DIAGNOSIS", "IMPACT"))

    finite_cov = inv.loc[np.isfinite(inv["coverage_days"]), "coverage_days"]
    ui.kpi_row([
        ui.kpi("Total inventory value", fmt_eur(inv["stock_value"].sum(), compact=True), f"{len(inv)} materials"),
        ui.kpi("Materials at risk", str(int(inv["risk"].isin(["STOCKOUT RISK", "BELOW SAFETY STOCK", "WATCH"]).sum())),
               "stockout / below safety stock", accent=theme.STATUS["critical"]),
        ui.kpi("Materials overstocked", str(int((inv["risk"] == "OVERSTOCK").sum())),
               f"> {config.OVERSTOCK_COVERAGE_DAYS} days · {fmt_eur(inv['excess_value'].sum(), compact=True)} excess",
               accent=theme.EMPHASIS),
        ui.kpi("Median stock coverage", fmt_days(finite_cov.median(), 0), "days of consumption"),
        ui.kpi("Safety stock breaches", str(int(inv["ss_breach_days_90d"].sum())), "material-days, last 90 days"),
    ])
    st.write("")

    for _, m in inv[inv["risk"] == "STOCKOUT RISK"].iterrows():
        imp = r.impacts.get(m["material_id"], {})
        sup = r.data.supplier_name(m["next_inbound_supplier"])
        facts = [("Estimated stockout", fmt_date(m["stockout_date"])),
                 ("Supplier delivery (carrier ETA)", fmt_date(m["next_inbound_ai"])),
                 ("ERP promised date", fmt_date(m["next_inbound_erp"])),
                 ("Shortage quantity", fmt_qty(m["shortage_qty"], m["unit"])),
                 ("Coverage", fmt_days(m["coverage_days"])),
                 ("Lines impacted", ", ".join(imp.get("lines", []))),
                 ("Lost output", fmt_qty(imp.get("lost_output_t", 0), "t")),
                 ("Estimated production impact", fmt_eur(imp.get("total_impact", 0)))]
        facts_html = "".join(f'<div class="f"><div class="l">{ui.esc(l)}</div><div class="v">{ui.esc(v)}</div></div>'
                             for l, v in facts)
        st.markdown(
            f'<div class="alert-critical"><div class="h">🔴 STOCKOUT RISK — {ui.esc(m["material_name"])}</div>'
            f'<div style="font-size:.88rem;margin-top:4px">Stock is above safety stock today and the ERP shows no '
            f'problem. Crossing open orders with carrier tracking shows the next {ui.esc(sup)} delivery is held at '
            f'origin: stock runs out before it arrives.</div><div class="facts">{facts_html}</div></div>',
            unsafe_allow_html=True)
        ui.link("copilot", "Why is production at risk? → options & recommendation", ":material/smart_toy:")

    table = inv[["material_name", "stock", "unit", "daily_consumption", "coverage_days", "safety_stock",
                 "stock_value", "next_inbound_ai", "risk"]].copy()
    icon = {"STOCKOUT RISK": "🔴", "BELOW SAFETY STOCK": "🟠", "WATCH": "🟡", "OVERSTOCK": "🔵", "OK": "🟢"}
    table["risk"] = [f"{icon[x]} {x}" for x in table["risk"]]
    table["stock_value"] = table["stock_value"].round(0)
    st.dataframe(table, hide_index=True, width="stretch", column_config={
        "material_name": st.column_config.TextColumn("Material", width="medium"),
        "stock": st.column_config.NumberColumn("Stock", format="%.1f"),
        "unit": "Unit",
        "daily_consumption": st.column_config.NumberColumn("Daily consumption", format="%.2f"),
        "coverage_days": st.column_config.NumberColumn("Coverage (days)", format="%.1f"),
        "safety_stock": st.column_config.NumberColumn("Safety stock", format="%.1f"),
        "stock_value": st.column_config.NumberColumn("Value (€)", format="localized"),
        "next_inbound_ai": st.column_config.DateColumn("Next inbound (ETA)", format="DD MMM YYYY"),
        "risk": "Risk",
    })

    st.subheader("Stock evolution vs safety stock")
    ids = inv["material_id"].tolist()
    names = dict(zip(inv["material_id"], inv["material_name"]))
    c1, c2 = st.columns([3, 1])
    mid = c1.selectbox("Material", ids, format_func=lambda x: names[x])
    days = c2.segmented_control("History", [60, 180, 365], default=60, format_func=lambda d: f"{d} days") or 60
    m = inv.set_index("material_id").loc[mid]
    hist = stock_history(r.data.inventory, mid, days, r.as_of)
    until = m["risk_horizon"]
    proj_ai = project_stock(m["stock"], m["daily_consumption"],
                            open_inbound(r.data.purchase_orders, r.data.logistics, r.as_of, mid, "ai"), r.as_of)
    proj_erp = project_stock(m["stock"], m["daily_consumption"],
                             open_inbound(r.data.purchase_orders, r.data.logistics, r.as_of, mid, "erp"), r.as_of)
    proj_ai, proj_erp = proj_ai[proj_ai["date"] <= until], proj_erp[proj_erp["date"] <= until]
    st.plotly_chart(charts.stock_vs_safety(hist, proj_ai, proj_erp, m["unit"],
                                           f"{names[mid]} — last {days} days + projection to {fmt_date(until)}",
                                           m["stockout_date"] if m["risk"] == "STOCKOUT RISK" else None),
                    width="stretch", config=CHART_CFG)
    st.caption("Projection = current stock − average daily consumption (last 30 days) + open purchase orders. "
               "ERP view uses promised dates; AI OS view uses carrier ETAs from logistics tracking.")

    over = inv[inv["risk"] == "OVERSTOCK"]
    over = over.assign(excess_value=over["excess_value"].round(0),
                       carrying_cost_saving=over["carrying_cost_saving"].round(0))
    if len(over):
        st.subheader("Overstock")
        st.dataframe(over[["material_name", "coverage_days", "excess_qty", "unit", "excess_value", "carrying_cost_saving"]],
                     hide_index=True, width="stretch", column_config={
                         "material_name": "Material",
                         "coverage_days": st.column_config.NumberColumn("Coverage (days)", format="%.0f"),
                         "excess_qty": st.column_config.NumberColumn(f"Excess vs {config.TARGET_COVERAGE_DAYS}-day target",
                                                                     format="%.1f"),
                         "unit": "Unit",
                         "excess_value": st.column_config.NumberColumn("Excess value (€)", format="localized"),
                         "carrying_cost_saving": st.column_config.NumberColumn("Carrying cost / year (€)", format="localized"),
                     })
