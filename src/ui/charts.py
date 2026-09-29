"""Plotly figures (all data comes from the analytics layer)."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from src.ui import theme
from src.ui.theme import base_layout


def emphasis_colors(ids, focus) -> list[str]:
    return [theme.EMPHASIS if i == focus else theme.MUTED_SERIES for i in ids]


def hbar(df: pd.DataFrame, x: str, y: str, title: str, focus_col: str | None = None, focus=None,
         fmt: str = "€%{x:,.0f}", height: int = 320) -> go.Figure:
    d = df.sort_values(x)
    colors = emphasis_colors(d[focus_col], focus) if focus_col else theme.EMPHASIS
    fig = go.Figure(go.Bar(x=d[x], y=d[y], orientation="h", marker=dict(color=colors, cornerradius=4),
                           hovertemplate=f"%{{y}}<br>{fmt}<extra></extra>"))
    fig.update_xaxes(showgrid=True, gridcolor=theme.GRID)
    fig.update_yaxes(showgrid=False)
    return base_layout(fig, height, title)


def monthly_spend(ms: pd.DataFrame, names: dict, title: str) -> go.Figure:
    """Stacked monthly spend: top-2 materials + 'Other'."""
    tot = ms.groupby("material_id")["po_value"].sum().sort_values(ascending=False)
    keep = list(tot.index[:2])
    ms = ms.assign(group=ms["material_id"].where(ms["material_id"].isin(keep), "Other"))
    g = ms.groupby(["month", "group"], as_index=False)["po_value"].sum()
    fig = go.Figure()
    for i, grp in enumerate(keep + ["Other"]):
        s = g[g["group"] == grp]
        fig.add_bar(x=s["month"], y=s["po_value"], name=names.get(grp, grp),
                    marker=dict(color=theme.SERIES[i] if grp != "Other" else theme.MUTED_SERIES,
                                line=dict(color=theme.SURFACE, width=1)),
                    hovertemplate="%{x|%b %Y}<br>" + names.get(grp, grp) + ": €%{y:,.0f}<extra></extra>")
    fig.update_layout(barmode="stack")
    fig.update_yaxes(tickprefix="€", tickformat="~s")
    return base_layout(fig, 320, title)


def price_evolution(mp: pd.DataFrame, names: dict, focus: str | None, title: str, unit: str = "t") -> go.Figure:
    fig = go.Figure()
    order = sorted(mp["supplier_id"].unique())
    for i, sid in enumerate(order):
        s = mp[mp["supplier_id"] == sid].sort_values("month")
        is_focus = focus is None or sid == focus
        color = theme.SERIES[i % len(theme.SERIES)] if focus is None else (theme.EMPHASIS if sid == focus else theme.MUTED_SERIES)
        fig.add_scatter(x=s["month"], y=s["avg_price"], name=names.get(sid, sid), mode="lines+markers",
                        line=dict(width=2.5 if (focus and sid == focus) else 2, color=color),
                        marker=dict(size=7), opacity=1 if is_focus else 0.9,
                        hovertemplate=names.get(sid, sid) + "<br>%{x|%b %Y}: €%{y:,.0f}/" + unit + "<extra></extra>")
    fig.update_yaxes(tickprefix="€")
    fig.update_layout(hovermode="x unified")
    return base_layout(fig, 340, title)


def stock_vs_safety(hist: pd.DataFrame, proj_ai: pd.DataFrame | None, proj_erp: pd.DataFrame | None,
                    unit: str, title: str, stockout_date=None) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=hist["date"], y=hist["stock_quantity"], name="Stock (actual)", mode="lines",
                    line=dict(color=theme.SERIES[0], width=2),
                    hovertemplate="%{x|%d %b}: %{y:,.0f} " + unit + "<extra>Actual</extra>")
    if proj_erp is not None and len(proj_erp):
        fig.add_scatter(x=proj_erp["date"], y=proj_erp["projected_stock"], name="Projection — ERP promised dates",
                        mode="lines", line=dict(color=theme.MUTED, width=2, dash="dot"),
                        hovertemplate="%{x|%d %b}: %{y:,.0f} " + unit + "<extra>ERP view</extra>")
    if proj_ai is not None and len(proj_ai):
        fig.add_scatter(x=proj_ai["date"], y=proj_ai["projected_stock"], name="Projection — carrier ETAs (AI OS)",
                        mode="lines", line=dict(color=theme.SERIES[1], width=2.5, dash="dash"),
                        hovertemplate="%{x|%d %b}: %{y:,.0f} " + unit + "<extra>AI OS view</extra>")
    ss_x = list(hist["date"]) + (list(proj_ai["date"]) if proj_ai is not None else [])
    ss_y = list(hist["safety_stock"]) + ([hist["safety_stock"].iloc[-1]] * len(proj_ai) if proj_ai is not None else [])
    fig.add_scatter(x=ss_x, y=ss_y, name="Safety stock", mode="lines",
                    line=dict(color=theme.STATUS["critical"], width=1.5),
                    hovertemplate="Safety stock: %{y:,.0f} " + unit + "<extra></extra>")
    fig.add_hline(y=0, line=dict(color=theme.AXIS, width=1))
    if proj_ai is not None and len(proj_ai):
        fig.add_vline(x=hist["date"].iloc[-1], line=dict(color=theme.AXIS, width=1))
        fig.add_annotation(x=hist["date"].iloc[-1], y=1, yref="paper", text="today", showarrow=False,
                           font=dict(color=theme.MUTED, size=11), xanchor="left", yanchor="top")
    if stockout_date is not None:
        fig.add_annotation(x=stockout_date, y=0, text=f"Stockout {pd.Timestamp(stockout_date):%d %b}",
                           showarrow=True, arrowhead=0, ax=-70, ay=-50, bgcolor="white", font=dict(color=theme.STATUS["critical"], size=12))
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(title_text=unit)
    return base_layout(fig, 380, title)


def monthly_line(df: pd.DataFrame, x: str, y: str, title: str, suffix: str = "", ref: float | None = None,
                 ref_label: str = "", color: str | None = None, height: int = 260) -> go.Figure:
    fig = go.Figure(go.Scatter(x=df[x], y=df[y], mode="lines+markers", line=dict(color=color or theme.EMPHASIS, width=2),
                               marker=dict(size=8), hovertemplate="%{x|%b %Y}: %{y:,.1f}" + suffix + "<extra></extra>"))
    if ref is not None:
        fig.add_hline(y=ref, line=dict(color=theme.STATUS["critical"], width=1),
                      annotation_text=ref_label, annotation_position="top left",
                      annotation_font=dict(color=theme.MUTED, size=11))
    return base_layout(fig, height, title)


def monthly_bars(df: pd.DataFrame, x: str, y: str, title: str, suffix: str = "", ref: float | None = None,
                 ref_label: str = "", height: int = 260) -> go.Figure:
    fig = go.Figure(go.Bar(x=df[x], y=df[y], marker=dict(color=theme.EMPHASIS, cornerradius=4),
                           hovertemplate="%{x|%b %Y}: %{y:,.0f}" + suffix + "<extra></extra>"))
    if ref is not None:
        fig.add_hline(y=ref, line=dict(color=theme.STATUS["critical"], width=1),
                      annotation_text=ref_label, annotation_position="top left",
                      annotation_font=dict(color=theme.MUTED, size=11))
    return base_layout(fig, height, title)


def risk_scatter(sc: pd.DataFrame, focus: str | None = None) -> go.Figure:
    """OTD recent vs PPM ratio-to-limit; bubble = spend; colour = risk level (status, with label)."""
    d = sc[sc["otd_recent"].notna()].copy()
    d["ppm_vs_limit"] = d["ppm_recent"].fillna(d["ppm"]) / d["ppm_limit"] * 100
    fig = go.Figure()
    for lvl in ["High", "Medium", "Low"]:
        s = d[d["risk_level"] == lvl]
        fig.add_scatter(
            x=s["otd_recent"], y=s["ppm_vs_limit"], mode="markers+text", name=f"{lvl} risk",
            text=[n if (lvl != "Low" or n == focus) else "" for n in s["supplier_name"]],
            textposition="top center", textfont=dict(size=11, color=theme.INK_2),
            marker=dict(size=(s["spend"] / d["spend"].max()) ** 0.5 * 42 + 8, color=theme.RISK_COLOR[lvl],
                        opacity=0.85, line=dict(color=theme.SURFACE, width=2)),
            customdata=s[["supplier_name", "risk_score", "spend"]],
            hovertemplate="%{customdata[0]}<br>OTD %{x:.1f}% · PPM %{y:.0f}% of limit<br>"
                          "Risk %{customdata[1]:.0f}/100 · spend €%{customdata[2]:,.0f}<extra></extra>")
    fig.add_hline(y=100, line=dict(color=theme.AXIS, width=1), annotation_text="PPM limit",
                  annotation_font=dict(color=theme.MUTED, size=11), annotation_position="top left")
    fig.update_xaxes(title_text="OTD — last 90 days (%)", autorange="reversed")
    fig.update_yaxes(title_text="Recent PPM as % of limit")
    fig.update_layout(legend=dict(y=-0.22))
    return base_layout(fig, 380, "Supplier risk map (bubble size = spend)")


def risk_bars(sc: pd.DataFrame) -> go.Figure:
    d = sc.sort_values("risk_score")
    fig = go.Figure(go.Bar(x=d["risk_score"], y=d["supplier_name"], orientation="h",
                           marker=dict(color=[theme.RISK_COLOR[l] for l in d["risk_level"]], cornerradius=4),
                           text=[f"{s:.0f} · {l}" for s, l in zip(d["risk_score"], d["risk_level"])],
                           textposition="outside", textfont=dict(color=theme.INK_2, size=11),
                           hovertemplate="%{y}: %{x:.0f}/100<extra></extra>"))
    fig.update_xaxes(range=[0, 100], showgrid=True, gridcolor=theme.GRID)
    return base_layout(fig, 420, "Supplier risk score (0–100)")


def opportunity_bars(df: pd.DataFrame) -> go.Figure:
    g = df.groupby("Category", as_index=False)["Value (€)"].sum().sort_values("Value (€)")
    fig = go.Figure(go.Bar(x=g["Value (€)"], y=g["Category"], orientation="h",
                           marker=dict(color=theme.EMPHASIS, cornerradius=4),
                           text=[f"€{v / 1000:,.0f}k" for v in g["Value (€)"]], textposition="outside",
                           textfont=dict(color=theme.INK_2), hovertemplate="%{y}: €%{x:,.0f}/year<extra></extra>"))
    fig.update_xaxes(showgrid=True, gridcolor=theme.GRID, tickprefix="€", tickformat="~s")
    return base_layout(fig, 260, "Annual savings by category")


def scenario_bars(options, baseline: float) -> go.Figure:
    labels = [f"{o.code} — {o.name}" for o in options] + ["Do nothing"]
    cost = [o.cost for o in options] + [0]
    resid = [o.residual_impact for o in options] + [baseline]
    fig = go.Figure()
    fig.add_bar(y=labels, x=cost, orientation="h", name="Option cost", marker=dict(color=theme.SERIES[0], cornerradius=4),
                hovertemplate="%{y}<br>Option cost €%{x:,.0f}<extra></extra>")
    fig.add_bar(y=labels, x=resid, orientation="h", name="Residual production impact",
                marker=dict(color=theme.SERIES[1], cornerradius=4),
                hovertemplate="%{y}<br>Residual impact €%{x:,.0f}<extra></extra>")
    fig.update_layout(barmode="stack")
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickprefix="€", tickformat="~s", showgrid=True, gridcolor=theme.GRID)
    return base_layout(fig, 250, "Total expected cost per option")
