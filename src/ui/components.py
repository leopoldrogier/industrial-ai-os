"""Reusable Streamlit UI components."""
from __future__ import annotations

import html

import streamlit as st

from src import config
from src.data_loader import SOURCE_SYSTEMS
from src.ui import theme
from src.utils.formatting import fmt_date, fmt_eur, severity_icon

PIPELINE = ["DATA", "DETECTION", "DIAGNOSIS", "IMPACT", "OPTIONS", "RECOMMENDATION", "HUMAN VALIDATION", "ACTION"]

# Page registry filled by app.py (key -> st.Page) for cross-page links
PAGES: dict = {}


def esc(x) -> str:
    return html.escape(str(x))


def inject_css() -> None:
    st.markdown(theme.CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str, as_of=None, active_steps: tuple[str, ...] = ()) -> None:
    meta = f"{config.COMPANY_NAME} · {config.COMPANY_SITE}"
    if as_of is not None:
        meta += f" · data as of {fmt_date(as_of)}"
    st.markdown(f'<div class="aios-title">{esc(title)}</div><div class="aios-sub">{esc(subtitle)}</div>'
                f'<div class="aios-meta">{esc(meta)}</div>', unsafe_allow_html=True)
    if active_steps:
        pipeline(active_steps)


def pipeline(active: tuple[str, ...]) -> None:
    parts = []
    for i, s in enumerate(PIPELINE):
        parts.append(f'<span class="st {"on" if s in active else ""}">{s}</span>')
        if i < len(PIPELINE) - 1:
            parts.append('<span class="ar">→</span>')
    st.markdown(f'<div class="pipeline">{"".join(parts)}</div>', unsafe_allow_html=True)


def kpi(label: str, value: str, sub: str = "", accent: str | None = None) -> str:
    cls = "kpi accent" if accent else "kpi"
    style = f' style="--accent:{accent}"' if accent else ""
    return (f'<div class="{cls}"{style}><div class="lbl">{esc(label)}</div><div class="val">{esc(value)}</div>'
            f'<div class="sub">{esc(sub)}</div></div>')


def kpi_row(cards: list[str]) -> None:
    cols = st.columns(len(cards))
    for c, html_card in zip(cols, cards):
        c.markdown(html_card, unsafe_allow_html=True)


def pill(text: str, color: str) -> str:
    return f'<span class="pill" style="--c:{color}">{esc(text)}</span>'


def severity_pill(sev: str) -> str:
    return pill(f"{severity_icon(sev)} {sev}", theme.SEVERITY_COLOR.get(sev, theme.MUTED))


def issue_card(issue) -> None:
    color = theme.SEVERITY_COLOR.get(issue.severity, theme.MUTED)
    impact = (f'<div class="imp">{esc(fmt_eur(issue.impact))}<small>{esc(issue.impact_label)}</small></div>'
              if issue.impact else "")
    st.markdown(
        f'<div class="card issue" style="--sev:{color}"><div class="head"><div>{severity_pill(issue.severity)} '
        f'&nbsp;<span class="t">{esc(issue.title)}</span></div>{impact}</div>'
        f'<div class="s">{esc(issue.summary)}</div></div>', unsafe_allow_html=True)


def roles_strip() -> None:
    st.markdown(
        '<div class="roles">'
        '<div class="role"><div class="n">ERP · MES · WMS · QMS</div><div class="d">System of Record — source of '
        'operational truth. Unchanged.</div></div>'
        '<div class="role mid"><div class="n">Industrial AI OS</div><div class="d">Intelligence Layer — analysis, '
        'reasoning, simulation, recommendations.</div></div>'
        '<div class="role"><div class="n">Human</div><div class="d">Decision Maker — validates every action. '
        'Nothing is executed automatically.</div></div></div>', unsafe_allow_html=True)


def link(page_key: str, label: str, icon: str | None = None) -> None:
    page = PAGES.get(page_key)
    if page is not None:
        st.page_link(page, label=label, icon=icon)


def sidebar(result) -> None:
    with st.sidebar:
        st.markdown(f"**{config.COMPANY_NAME}**  \n<span style='font-size:0.8rem;opacity:.75'>"
                    f"{config.COMPANY_SITE} · aluminium casting</span>", unsafe_allow_html=True)
        st.markdown(f"<span style='font-size:0.8rem;opacity:.75'>Data as of {fmt_date(result.as_of)}</span>",
                    unsafe_allow_html=True)
        st.divider()
        st.markdown("<span style='font-size:0.78rem;letter-spacing:.05em;opacity:.7'>CONNECTED SOURCES (V1: CSV)</span>",
                    unsafe_allow_html=True)
        counts = result.data.row_counts()
        rows = "".join(f"<div style='font-size:0.78rem;display:flex;justify-content:space-between'>"
                       f"<span>● {esc(SOURCE_SYSTEMS[k])}</span><span style='opacity:.6'>{v:,}</span></div>"
                       for k, v in counts.items())
        st.markdown(rows, unsafe_allow_html=True)
        st.divider()
        st.markdown(f"<div style='font-size:0.72rem;opacity:.6'>{esc(config.COMPANY_DISCLAIMER)}<br><br>"
                    "Recommendations only — no action is executed in the ERP.</div>", unsafe_allow_html=True)


def disclaimer() -> None:
    st.markdown(f'<p class="disclaimer">{esc(config.COMPANY_DISCLAIMER)}</p>', unsafe_allow_html=True)
