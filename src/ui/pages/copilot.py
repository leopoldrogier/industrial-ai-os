"""AI Industrial Copilot page: question -> grounded answer -> options -> human validation."""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from src.ai.copilot import SUGGESTED_QUESTIONS, Copilot
from src.data_loader import SOURCE_SYSTEMS
from src.engine.scenarios import scenario_frame
from src.ui import charts, components as ui
from src.ui.state import get_result
from src.utils.formatting import fmt_date, fmt_eur


def render() -> None:
    r = get_result()
    bot = Copilot(r)
    ui.page_header("AI Industrial Copilot", f"Ask about supply risks, suppliers, prices, savings — engine: {bot.narrator.name}",
                   r.as_of, ("DATA", "DETECTION", "DIAGNOSIS", "IMPACT", "OPTIONS", "RECOMMENDATION", "HUMAN VALIDATION"))

    st.markdown("**Suggested questions**")
    cols = st.columns(3)
    for i, q in enumerate(SUGGESTED_QUESTIONS):
        if cols[i % 3].button(q, key=f"sq{i}", width="stretch"):
            st.session_state["copilot_q"] = q
    with st.form("ask", clear_on_submit=False, border=False):
        c1, c2 = st.columns([5, 1])
        typed = c1.text_input("Your question", placeholder="e.g. Why is production at risk?", label_visibility="collapsed")
        if c2.form_submit_button("Ask", type="primary", width="stretch") and typed.strip():
            st.session_state["copilot_q"] = typed.strip()

    q = st.session_state.get("copilot_q")
    if not q:
        st.info("Select a question or type your own. Answers are built only from figures computed on the plant data.")
        return

    ans = bot.ask(q)
    st.markdown(f"#### ❓ {q}")
    left, right = st.columns([1.7, 1], gap="large")
    with left:
        with st.container(border=True):
            st.markdown(ans.answer)
    with right:
        if ans.facts:
            with st.container(border=True):
                st.markdown("**Key figures**")
                for label, value in ans.facts:
                    st.markdown(f"<div style='display:flex;justify-content:space-between'><span style='color:#52514e'>"
                                f"{ui.esc(label)}</span><b>{ui.esc(value)}</b></div>", unsafe_allow_html=True)
        if ans.sources:
            st.markdown("**Data sources crossed**")
            st.markdown("  \n".join(f"✓ {SOURCE_SYSTEMS.get(s, s)}" for s in ans.sources))
        if ans.reasoning:
            with st.expander("How this answer was built"):
                for step in ans.reasoning:
                    st.markdown(f"- {step}")

    if ans.actions:
        st.markdown("**Recommended actions**")
        for i, a in enumerate(ans.actions, 1):
            st.markdown(f"{i}. {a}")

    if ans.scenario is not None:
        render_scenarios(ans.scenario)


def render_scenarios(s) -> None:
    st.divider()
    st.subheader(f"Scenario simulation — {s.material_name}")
    st.caption(f"Shortage {s.shortage_qty:,.0f} t from {fmt_date(s.stockout_date)} · impact if no action "
               f"{fmt_eur(s.baseline_impact)} · costs computed from purchase, logistics and production data + demo assumptions.")
    cols = st.columns(len(s.options))
    for c, o in zip(cols, s.options):
        with c, st.container(border=True):
            badge = " ⭐ Recommended" if o.code == s.recommended else ""
            st.markdown(f"**Option {o.code}**{badge}  \n{o.name}")
            value = fmt_eur(o.cost) if o.code != "C" else fmt_eur(o.residual_impact)
            st.metric("Estimated impact" if o.code == "C" else "Cost", value)
            st.markdown(f"<span style='font-size:.85rem'>{ui.esc(o.description)}</span>", unsafe_allow_html=True)
            st.markdown(f"Arrival: **{fmt_date(o.arrival_date) if o.arrival_date is not None else 'n/a'}** · "
                        f"Before stockout: **{'Yes' if o.feasible else 'No'}** · Confidence: **{o.confidence}**")
            with st.expander("Cost breakdown"):
                for k, v in o.cost_breakdown.items():
                    st.markdown(f"- {k}: {fmt_eur(v) if '(t)' not in k else f'{v:,.1f} t'}")
                if o.notes:
                    st.caption(o.notes)
    a, b = st.columns([1, 1], gap="large")
    with a:
        st.plotly_chart(charts.scenario_bars(s.options, s.baseline_impact), width="stretch",
                        config={"displayModeBar": False})
    with b:
        st.dataframe(scenario_frame(s), hide_index=True, width="stretch", column_config={
            c: st.column_config.NumberColumn(format="localized")
            for c in ["Cost (€)", "Residual production impact (€)", "Total expected cost (€)"]})
    st.success(f"**Recommendation — Option {s.recommended}.** {s.rationale}")

    st.subheader("Human validation")
    st.caption("Industrial AI OS recommends. The decision belongs to a human. V1 prepares a draft — nothing is sent to "
               "the ERP or to suppliers.")
    labels = {o.code: f"Option {o.code} — {o.name}" for o in s.options}
    c1, c2 = st.columns([2, 1])
    choice = c1.radio("Decision", list(labels), format_func=labels.get, index=list(labels).index(s.recommended),
                      horizontal=True)
    comment = c1.text_input("Comment (optional)", key="decision_comment")
    if c2.button("Validate decision & prepare action", type="primary"):
        opt = next(o for o in s.options if o.code == choice)
        st.session_state.setdefault("decisions", []).append({
            "time": datetime.now().strftime("%H:%M:%S"), "material": s.material_name, "decision": labels[choice],
            "cost": fmt_eur(opt.total_cost), "comment": comment, "status": "Draft prepared — awaiting execution in ERP"})
        st.session_state["last_draft"] = _draft(s, opt)
    if st.session_state.get("last_draft"):
        st.markdown("**Prepared action (draft, not executed)**")
        st.code(st.session_state["last_draft"], language=None)
    if st.session_state.get("decisions"):
        st.markdown("**Decision log (this session)**")
        st.dataframe(pd.DataFrame(st.session_state["decisions"]), hide_index=True, width="stretch")


def _draft(s, o) -> str:
    qty = f"{s.order_qty:,.0f} t (shortage {s.shortage_qty:,.0f} t rounded up to full trucks)"
    if o.code == "A":
        return (f"TO: supplier account manager ({o.name.split('(')[-1].rstrip(')')})\n"
                f"SUBJECT: Expedite request — {s.material_name}\n\n"
                f"Please release {qty} from our open orders for express road shipment, arrival no later than "
                f"{fmt_date(s.stockout_date - pd.Timedelta(days=1))}. Premium freight accepted up to {fmt_eur(o.cost)}.\n"
                f"[DRAFT — to be reviewed and sent by the buyer]")
    if o.code == "B":
        return (f"ERP PURCHASE REQUISITION (draft)\nMaterial: {s.material_name}\nQuantity: {qty}\n"
                f"Vendor: {o.name.split('(')[-1].rstrip(')')}\nRequested delivery: before {fmt_date(s.stockout_date)}\n"
                f"Linked change: reduce open order of the delayed supplier by the same quantity\n"
                f"Estimated extra cost: {fmt_eur(o.cost)}\n[DRAFT — requires buyer approval in the ERP]")
    return (f"PRODUCTION PLANNING CHANGE (draft)\nReduce throughput on primary aluminium lines until the delayed "
            f"delivery arrives; load spare capacity on the recycling line.\nEstimated impact: {fmt_eur(o.residual_impact)}\n"
            f"[DRAFT — requires production manager approval]")
