"""Purchasing Intelligence page."""
from __future__ import annotations

import streamlit as st

from src import config
from src.analytics.purchasing import monthly_prices
from src.ui import charts, components as ui, theme
from src.ui.state import get_result
from src.utils.formatting import fmt_eur

CHART_CFG = {"displayModeBar": False}


def render() -> None:
    r = get_result()
    ui.page_header("Purchasing Intelligence", "Spend, price evolution and price anomalies — ERP purchase orders",
                   r.as_of, ("DATA", "DETECTION", "DIAGNOSIS", "IMPACT"))
    ss, sm = r.spend_by_supplier, r.spend_by_material
    anomalies = len(r.price_anomalies) + len(r.contract_deviations)
    price_savings = float(r.price_anomalies.get("potential_savings", []).sum() if len(r.price_anomalies) else 0) + \
        float(r.contract_deviations["potential_savings"].sum() if len(r.contract_deviations) else 0)
    top = ss.iloc[0]
    ui.kpi_row([
        ui.kpi("Total purchasing spend", fmt_eur(r.spend["total_spend"], compact=True), "last 12 months"),
        ui.kpi("Purchase orders", f"{r.spend['po_count']:,}", f"{r.spend['materials']} materials"),
        ui.kpi("Top supplier", top["supplier_name"], f"{top['share_pct']:.0f}% of spend"),
        ui.kpi("Price anomalies", str(anomalies), "drift vs peers / above contract", accent=theme.STATUS["critical"]),
        ui.kpi("Price savings potential", f"{fmt_eur(price_savings, compact=True)}/year", "gap × annual volume",
               accent=theme.EMPHASIS),
    ])
    st.write("")

    focus = r.price_anomalies["supplier_id"].iloc[0] if len(r.price_anomalies) else None
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.plotly_chart(charts.hbar(ss.head(10), "spend", "supplier_name", "Spend by supplier (top 10)",
                                    "supplier_id", focus), width="stretch", config=CHART_CFG)
    with c2:
        st.plotly_chart(charts.hbar(sm, "spend", "material_name", "Spend by material"), width="stretch", config=CHART_CFG)

    st.subheader("Price evolution")
    mats = r.price_stats.groupby("material_id")["annual_spend"].sum().sort_values(ascending=False).index.tolist()
    mid = st.selectbox("Material", mats, format_func=r.data.material_name)
    names = dict(zip(r.data.suppliers["supplier_id"], r.data.suppliers["supplier_name"]))
    anomalous = r.price_anomalies[r.price_anomalies["material_id"] == mid]
    f = anomalous["supplier_id"].iloc[0] if len(anomalous) else None
    unit = r.data.material_unit(mid)
    st.plotly_chart(charts.price_evolution(monthly_prices(r.data.purchase_orders, mid), names, f,
                                           f"Monthly average price — {r.data.material_name(mid)} (€/{unit})", unit),
                    width="stretch", config=CHART_CFG)

    ps = r.price_stats[r.price_stats["material_id"] == mid][
        ["supplier_name", "pricing_model", "baseline_price", "current_price", "price_change_pct", "peer_change_pct",
         "benchmark_price", "gap_vs_benchmark_pct", "annual_volume", "annual_spend"]].copy()
    ps["annual_spend"] = ps["annual_spend"] / 1000
    st.dataframe(ps, hide_index=True, width="stretch", column_config={
        "supplier_name": "Supplier", "pricing_model": "Pricing",
        "baseline_price": st.column_config.NumberColumn(f"Price first {config.PRICE_BASELINE_DAYS}d", format="€%.1f"),
        "current_price": st.column_config.NumberColumn(f"Price last {config.PRICE_RECENT_DAYS}d", format="€%.1f"),
        "price_change_pct": st.column_config.NumberColumn("Evolution", format="%+.1f%%"),
        "peer_change_pct": st.column_config.NumberColumn("Peers evolution", format="%+.1f%%"),
        "benchmark_price": st.column_config.NumberColumn("Peer benchmark", format="€%.1f"),
        "gap_vs_benchmark_pct": st.column_config.NumberColumn("Gap vs benchmark", format="%+.1f%%"),
        "annual_volume": st.column_config.NumberColumn("Volume 12m", format="%.0f"),
        "annual_spend": st.column_config.NumberColumn("Spend 12m (k€)", format="%.0f"),
    })

    st.subheader("Price anomalies detected")
    st.caption(f"Rule: supplier price change exceeds peers by ≥ {config.PRICE_DRIFT_THRESHOLD_PP:.0f} pp AND current "
               f"price ≥ {config.BENCHMARK_GAP_THRESHOLD_PCT:.0f}% above peer benchmark; fixed-price contracts: invoiced "
               f"≥ {config.CONTRACT_DEVIATION_THRESHOLD_PCT:.0f}% above contract.")
    for i in r.issue("price_drift") + r.issue("contract_deviation"):
        ui.issue_card(i)
    for o in [o for o in r.opportunities if o.opp_id.startswith(("OPP-PRICE", "OPP-CONTRACT"))]:
        st.markdown(f"**{o.entity}** — `{o.formula}`")
