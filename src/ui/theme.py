"""Visual tokens and the Plotly chart template (light theme)."""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

# Categorical slots in validated fixed order (never cycled, colour follows entity)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MUTED_SERIES = "#c3c2b7"      # "the rest" when one entity is emphasised
EMPHASIS = SERIES[0]

# Status colours are reserved for state (always shown with icon + label)
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
SEVERITY_COLOR = {"Critical": STATUS["critical"], "High": STATUS["serious"], "Medium": STATUS["warning"],
                  "Low": STATUS["good"]}
RISK_COLOR = {"High": STATUS["critical"], "Medium": STATUS["warning"], "Low": STATUS["good"]}
INVENTORY_RISK_COLOR = {"STOCKOUT RISK": STATUS["critical"], "BELOW SAFETY STOCK": STATUS["serious"],
                        "WATCH": STATUS["warning"], "OVERSTOCK": "#2a78d6", "OK": STATUS["good"]}

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
PAGE = "#f4f5f3"
NAVY = "#14283c"

_template = go.layout.Template()
_template.layout = go.Layout(
    font=dict(family='system-ui, -apple-system, "Segoe UI", sans-serif', size=12, color=INK_2),
    paper_bgcolor=SURFACE,
    plot_bgcolor=SURFACE,
    colorway=SERIES,
    margin=dict(l=8, r=8, t=36, b=8),
    title=dict(font=dict(size=14, color=INK), x=0, xanchor="left", y=0.98),
    xaxis=dict(showgrid=False, linecolor=AXIS, tickcolor=AXIS, ticks="outside", ticklen=3, zeroline=False,
               tickfont=dict(color=MUTED)),
    yaxis=dict(gridcolor=GRID, gridwidth=1, linecolor=AXIS, zeroline=False, tickfont=dict(color=MUTED)),
    legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="left", x=0, font=dict(size=11, color=INK_2),
                bgcolor="rgba(0,0,0,0)"),
    hoverlabel=dict(bgcolor="white", bordercolor=AXIS, font=dict(color=INK, size=12)),
    bargap=0.35,
)
pio.templates["industrial"] = _template
pio.templates.default = "industrial"


def base_layout(fig: go.Figure, height: int = 320, title: str | None = None) -> go.Figure:
    fig.update_layout(height=height, template="industrial")
    if title:
        fig.update_layout(title_text=title)
    return fig


CSS = f"""
<style>
.stApp {{ background: {PAGE}; }}
.block-container {{ padding-top: 3.2rem; padding-bottom: 3rem; max-width: 1400px; }}
h1, h2, h3 {{ color: {INK}; letter-spacing: -0.01em; }}
[data-testid="stSidebar"] {{ background: {NAVY}; }}
[data-testid="stSidebar"] * {{ color: #e8ecf1; }}
[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"] span {{ color: #e8ecf1 !important; }}
[data-testid="stSidebar"] a[aria-current="page"] {{ background: rgba(255,255,255,0.10) !important; }}
.aios-title {{ font-size: 1.9rem; font-weight: 750; color: {INK}; margin: 0; letter-spacing: 0.02em; }}
.aios-sub {{ color: {INK_2}; font-size: 0.98rem; margin: 0.1rem 0 0.2rem 0; }}
.aios-meta {{ color: {MUTED}; font-size: 0.8rem; }}
.kpi {{ background: {SURFACE}; border: 1px solid rgba(11,11,11,0.08); border-radius: 10px; padding: 14px 16px 12px;
        height: 100%; box-shadow: 0 1px 2px rgba(0,0,0,0.03); }}
.kpi .lbl {{ color: {INK_2}; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; }}
.kpi .val {{ color: {INK}; font-size: 1.65rem; font-weight: 700; line-height: 1.25; margin-top: 2px; }}
.kpi .sub {{ color: {MUTED}; font-size: 0.8rem; margin-top: 2px; }}
.kpi.accent {{ border-left: 4px solid var(--accent); }}
.card {{ background: {SURFACE}; border: 1px solid rgba(11,11,11,0.08); border-radius: 10px; padding: 14px 18px;
         margin-bottom: 10px; }}
.issue {{ border-left: 4px solid var(--sev); }}
.issue .head {{ display:flex; justify-content: space-between; gap: 12px; align-items: baseline; }}
.issue .t {{ font-weight: 650; color: {INK}; font-size: 0.98rem; }}
.issue .imp {{ color: {INK}; font-weight: 650; white-space: nowrap; font-size: 0.92rem; }}
.issue .imp small {{ color: {MUTED}; font-weight: 400; display:block; text-align:right; font-size: 0.72rem; }}
.issue .s {{ color: {INK_2}; font-size: 0.87rem; margin-top: 4px; }}
.pill {{ display:inline-block; padding: 1px 9px; border-radius: 999px; font-size: 0.72rem; font-weight: 650;
         border: 1px solid var(--c); color: {INK}; background: color-mix(in srgb, var(--c) 14%, white); }}
.pipeline {{ display:flex; flex-wrap: wrap; gap: 6px; align-items:center; margin: 6px 0 4px; }}
.pipeline .st {{ font-size: 0.72rem; font-weight: 650; letter-spacing: 0.05em; padding: 4px 9px; border-radius: 6px;
                 background: #e9ecef; color: {INK_2}; }}
.pipeline .st.on {{ background: {NAVY}; color: white; }}
.pipeline .ar {{ color: {MUTED}; font-size: 0.75rem; }}
.roles {{ display:grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }}
.role {{ background: {SURFACE}; border: 1px solid rgba(11,11,11,0.08); border-radius: 10px; padding: 10px 14px; }}
.role .n {{ font-weight: 700; color: {INK}; font-size: 0.92rem; }}
.role .d {{ color: {INK_2}; font-size: 0.8rem; }}
.role.mid {{ border: 1.5px solid {NAVY}; }}
.alert-critical {{ background: #fff5f5; border: 1px solid #f1c4c4; border-left: 5px solid {STATUS['critical']};
                   border-radius: 10px; padding: 14px 18px; margin-bottom: 12px; }}
.alert-critical .h {{ font-weight: 750; color: {STATUS['critical']}; letter-spacing: 0.04em; font-size: 0.95rem; }}
.facts {{ display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 10px; margin-top: 10px; }}
.facts .f .l {{ color: {MUTED}; font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.04em; }}
.facts .f .v {{ color: {INK}; font-weight: 700; font-size: 1.05rem; }}
.answer {{ background: {SURFACE}; border: 1px solid rgba(11,11,11,0.08); border-left: 4px solid {NAVY};
           border-radius: 10px; padding: 16px 20px; }}
.disclaimer {{ color: {MUTED}; font-size: 0.75rem; }}
@media (max-width: 900px) {{ .roles, .facts {{ grid-template-columns: 1fr 1fr; }} }}
</style>
"""
