"""Supplier Performance page."""
from __future__ import annotations

import numpy as np
import streamlit as st

from src import config
from src.ai.recommendations import diagnose_supplier, supplier_actions
from src.analytics.purchasing import monthly_prices
from src.analytics.quality import monthly_ppm
from src.analytics.suppliers import monthly_delivery
from src.ui import charts, components as ui, theme
from src.ui.state import get_result
from src.utils.formatting import fmt_eur, fmt_pct, fmt_qty

CHART_CFG = {"displayModeBar": False}


def render() -> None:
    r = get_result()
    sc = r.scorecard
    ui.page_header("Supplier Performance", "Delivery, quality, price and risk — computed from ERP, QMS and carrier data",
                   r.as_of, ("DATA", "DETECTION", "DIAGNOSIS", "RECOMMENDATION"))

    ui.kpi_row([
        ui.kpi("Suppliers evaluated", str(len(sc)), "last 12 months"),
        ui.kpi("Weighted OTD", fmt_pct((sc["otd"] * sc["deliveries"]).sum() / sc["deliveries"].sum()),
               "all deliveries due"),
        ui.kpi("High-risk suppliers", str(int((sc["risk_level"] == "High").sum())), "risk score ≥ "
               f"{config.RISK_LEVEL_THRESHOLDS['High']:.0f}", accent=theme.STATUS["critical"]),
        ui.kpi("Deteriorating", str(len(r.deteriorating) + len(r.issue("quality_spike"))),
               f"last {config.RECENT_WINDOW_DAYS} days vs baseline", accent=theme.STATUS["serious"]),
    ])
    st.write("")

    table = sc[["supplier_name", "material_id", "spend", "otd", "otd_recent", "avg_delay", "ppm", "avg_price",
                "price_evolution_pct", "risk_score", "risk_level"]].copy()
    table["spend"] = table["spend"] / 1000
    table["risk_level"] = table["risk_level"].map({"High": "🔴 High", "Medium": "🟡 Medium", "Low": "🟢 Low"})
    st.dataframe(
        table, hide_index=True, width="stretch",
        column_config={
            "supplier_name": st.column_config.TextColumn("Supplier", width="medium"),
            "material_id": "Main material",
            "spend": st.column_config.NumberColumn("Spend 12m (k€)", format="%.0f"),
            "otd": st.column_config.ProgressColumn("OTD 12m", format="%.1f%%", min_value=0, max_value=100),
            "otd_recent": st.column_config.NumberColumn("OTD 90d", format="%.1f%%"),
            "avg_delay": st.column_config.NumberColumn("Avg delay (d)", format="%.2f"),
            "ppm": st.column_config.NumberColumn("PPM", format="%.0f"),
            "avg_price": st.column_config.NumberColumn("Avg price (€/unit)", format="€%.2f"),
            "price_evolution_pct": st.column_config.NumberColumn("Price evolution", format="%+.1f%%"),
            "risk_score": st.column_config.ProgressColumn("Risk score", format="%.0f", min_value=0, max_value=100),
            "risk_level": "Risk level",
        })

    c1, c2 = st.columns(2, gap="large")
    ids = sc["supplier_id"].tolist()
    default = st.session_state.get("selected_supplier", ids[0])
    with c1:
        st.plotly_chart(charts.risk_bars(sc), width="stretch", config=CHART_CFG)
    with c2:
        st.plotly_chart(charts.risk_scatter(sc, focus=r.data.supplier_name(default)), width="stretch", config=CHART_CFG)

    st.divider()
    names = dict(zip(sc["supplier_id"], sc["supplier_name"]))
    sid = st.selectbox("Select a supplier", ids, index=ids.index(default) if default in ids else 0,
                       format_func=lambda s: f"{names[s]}  ·  risk {sc.set_index('supplier_id').at[s, 'risk_score']:.0f}")
    st.session_state["selected_supplier"] = sid
    row = r.supplier_row(sid)
    render_supplier_detail(r, row)


def _delta(new, old, suffix="", invert=False) -> str:
    if not (np.isfinite(new) and np.isfinite(old)):
        return ""
    d = new - old
    arrow = "▲" if d > 0 else "▼" if d < 0 else "="
    return f"{arrow} {abs(d):.1f}{suffix} vs baseline"


def render_supplier_detail(r, row) -> None:
    sid = row["supplier_id"]
    mat = row["material_id"]
    unit = r.data.material_unit(mat)
    st.subheader(f"Supplier Overview — {row['supplier_name']}")
    st.markdown(f"{row['country']} · {r.data.material_name(mat)} · {row['pricing_model']} pricing · "
                f"status **{row['supplier_status']}** &nbsp; "
                + ui.pill(f"Risk {row['risk_score']:.0f}/100 · {row['risk_level']}", theme.RISK_COLOR[row["risk_level"]]),
                unsafe_allow_html=True)
    ui.kpi_row([
        ui.kpi("Spend (12m)", fmt_eur(row["spend"], compact=True), f"{row['volume_share_pct']:.0f}% of material volume"),
        ui.kpi("Volume (12m)", fmt_qty(row["volume"], unit), f"{row['deliveries']} deliveries due"),
        ui.kpi("OTD — last 90 days", fmt_pct(row["otd_recent"]), _delta(row["otd_recent"], row["otd_baseline"], " pp")),
        ui.kpi("PPM — last 90 days", f"{row['ppm_recent']:,.0f}" if np.isfinite(row["ppm_recent"]) else "—",
               f"limit {row['ppm_limit']:,.0f} · baseline {row['ppm_baseline']:,.0f}"),
        ui.kpi("Current price", fmt_eur(row["current_price"], unit=unit),
               f"{row['price_evolution_pct']:+.1f}% vs start of period" if np.isfinite(row["price_evolution_pct"]) else ""),
    ])

    md = monthly_delivery(r.data.purchase_orders, sid)
    mq = monthly_ppm(r.data.quality, sid)
    mp = monthly_prices(r.data.purchase_orders, mat)
    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.plotly_chart(charts.monthly_line(md, "month", "otd_rolling_3m", "Delivery performance — OTD, rolling 3 months", "%",
                                            ref=95, ref_label="target 95%"), width="stretch", config=CHART_CFG)
    with c2:
        st.plotly_chart(charts.monthly_bars(mq, "month", "ppm", "Quality performance — monthly PPM", " ppm",
                                            ref=row["ppm_limit"], ref_label="limit"), width="stretch", config=CHART_CFG)
    with c3:
        st.plotly_chart(charts.monthly_line(md, "month", "avg_lead_time", "Actual lead time (days)", " d"),
                        width="stretch", config=CHART_CFG)
    names = dict(zip(r.data.suppliers["supplier_id"], r.data.suppliers["supplier_name"]))
    st.plotly_chart(charts.price_evolution(mp, names, sid, f"Price evolution — {r.data.material_name(mat)} "
                                           f"(highlighted: {row['supplier_name']})", unit),
                    width="stretch", config=CHART_CFG)

    a, b = st.columns([1.3, 1], gap="large")
    with a:
        st.markdown("#### 🧠 AI Diagnosis")
        st.markdown(f'<div class="answer">{ui.esc(diagnose_supplier(row))}</div>', unsafe_allow_html=True)
        st.caption("Generated by the rule-based engine from computed metrics only (no external AI service).")
    with b:
        st.markdown("#### Recommended Actions")
        for i, act in enumerate(supplier_actions(row, r.data.material_name(mat)), 1):
            st.markdown(f"{i}. {act}")
        if row["risk_level"] == "High":
            ui.link("copilot", "Simulate options in the AI Copilot", ":material/smart_toy:")
