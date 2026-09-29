"""AI Industrial Copilot — V1 rule-based engine.

Routes a question to an intent, then composes the answer exclusively from the
``AnalysisResult`` (numbers computed by Python). No external API is needed.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from src import config
from src.ai.llm import Narrator, build_fact_sheet, get_narrator
from src.ai.recommendations import diagnose_supplier, supplier_actions
from src.engine.analysis import AnalysisResult
from src.engine.scenarios import ScenarioSet
from src.utils.formatting import fmt_date, fmt_eur, fmt_qty

SUGGESTED_QUESTIONS = [
    "Why is production at risk?",
    "Why are we at risk of a stockout?",
    "What are the most critical issues?",
    "Which supplier should I investigate?",
    "Which suppliers are deteriorating?",
    "Where are the biggest purchasing savings?",
]

INTENT_KEYWORDS = {
    "production_risk": ["production", "line", "lines", "produce", "output"],
    "stockout": ["stockout", "stock-out", "stock out", "rupture", "shortage", "run out", "running out"],
    "deteriorating": ["deteriorat", "degrad", "getting worse", "declin", "worsening"],
    "investigate_supplier": ["which supplier", "investigate", "worst supplier", "riskiest", "supplier risk"],
    "savings": ["saving", "économ", "cost reduction", "reduce cost", "opportunit", "money"],
    "critical": ["critical", "priorit", "urgent", "most important", "issues", "problems", "alert"],
    "quality": ["quality", "ppm", "defect", "qualité", "non-conform"],
    "price": ["price", "prix", "expensive", "increase"],
    "inventory": ["overstock", "inventory", "stock level", "surstock", "working capital"],
}


@dataclass
class CopilotAnswer:
    question: str
    intent: str
    answer: str
    facts: list[tuple[str, str]] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    reasoning: list[str] = field(default_factory=list)
    scenario: ScenarioSet | None = None
    page: str | None = None


class Copilot:
    def __init__(self, result: AnalysisResult, narrator: Narrator | None = None):
        self.r = result
        self.narrator = narrator or get_narrator()

    # ------------------------------------------------------------------ routing
    def detect_intent(self, question: str) -> tuple[str, str | None]:
        q = question.lower()
        for _, s in self.r.data.suppliers.iterrows():
            first = s["supplier_name"].split()[0].lower()
            if s["supplier_name"].lower() in q or re.search(rf"\b{re.escape(first)}\b", q):
                if not any(k in q for k in ("price", "prix")):
                    return "supplier", s["supplier_id"]
        scores = {intent: sum(1 for k in kws if k in q) for intent, kws in INTENT_KEYWORDS.items()}
        # "why ... production ... risk" beats "stockout" (it is the broader, cross-source question)
        if scores["production_risk"] and ("risk" in q or "danger" in q):
            return "production_risk", None
        best = max(scores, key=scores.get)
        return (best, None) if scores[best] > 0 else ("help", None)

    def ask(self, question: str) -> CopilotAnswer:
        intent, entity = self.detect_intent(question)
        handler = getattr(self, f"_a_{intent}")
        ans: CopilotAnswer = handler(entity) if intent == "supplier" else handler()
        ans.question, ans.intent = question, intent
        ans.answer = self.narrator.rephrase(question, ans.answer, build_fact_sheet(self.r))
        return ans

    # ------------------------------------------------------------------ helpers
    def _stockout(self):
        iss = self.r.issue("stockout_risk")
        return (iss[0], self.r.scenarios.get(iss[0].entity_id)) if iss else (None, None)

    # ------------------------------------------------------------------ intents
    def _a_stockout(self) -> CopilotAnswer:
        issue, scen = self._stockout()
        if issue is None:
            return CopilotAnswer("", "", "No material is projected to run out before its next confirmed delivery.",
                                 sources=["inventory", "purchase_orders", "logistics"])
        m = issue.metrics
        u = m["unit"]
        sup = m["next_inbound_supplier_name"]
        text = (f"The main stockout risk concerns **{m['material_name']}** supplied by **{sup}**.\n\n"
                f"- Current stock: **{fmt_qty(m['stock'], u)}** (safety stock {fmt_qty(m['safety_stock'], u)})\n"
                f"- Daily consumption: **{fmt_qty(m['daily_consumption'], u, 1)}/day** "
                f"(last {config.CONSUMPTION_WINDOW_DAYS} days)\n"
                f"- Stock coverage: **{m['coverage_days']:.1f} days** → estimated stockout **{fmt_date(m['stockout_date'])}**\n"
                f"- ERP promised delivery: {fmt_date(m['next_inbound_erp'])} — carrier tracking now shows "
                f"**{fmt_date(m['next_inbound_ai'])}**\n"
                f"- Potential shortage: **{fmt_qty(m['shortage_qty'], u)}** "
                f"({m['days_without_stock']} days without stock until {fmt_date(m['recovery_date'])})\n\n"
                f"The ERP alone does not show this risk: with the promised dates, the stock never runs out. "
                f"The risk appears only when purchase orders are crossed with logistics tracking.")
        return CopilotAnswer(
            "", "", text,
            facts=[("Coverage", f"{m['coverage_days']:.1f} days"), ("Stockout date", fmt_date(m["stockout_date"])),
                   ("Shortage", fmt_qty(m["shortage_qty"], u)),
                   ("Impact if no action", fmt_eur(issue.impact))],
            actions=issue.actions[:4], sources=issue.sources, scenario=scen, page="inventory",
            reasoning=self._reasoning(issue, scen))

    def _a_production_risk(self) -> CopilotAnswer:
        issue, scen = self._stockout()
        if issue is None:
            return CopilotAnswer("", "", "No production line is currently exposed to a material shortage.",
                                 sources=["inventory", "production"])
        m = issue.metrics
        u = m["unit"]
        sup_id = m["next_inbound_supplier"]
        row = self.r.supplier_row(sup_id)
        open_po = self.r.data.purchase_orders
        open_po = open_po[(open_po["supplier_id"] == sup_id) & open_po["actual_delivery_date"].isna()
                          & (open_po["material_id"] == m["material_id"])]
        lg = self.r.data.logistics.set_index("po_id")
        po_lines = "\n".join(
            f"  - {p.po_id}: {fmt_qty(p.ordered_quantity, u)}, promised {fmt_date(p.promised_date)} → "
            f"carrier ETA {fmt_date(lg.at[p.po_id, 'current_eta'])} ({lg.at[p.po_id, 'shipment_status']})"
            for p in open_po.itertuples())
        lines = ", ".join(m["impact_lines"])
        text = (f"Production on **{lines}** is at risk because **{m['material_name']}** will run out on "
                f"**{fmt_date(m['stockout_date'])}**, before the next delivery.\n\n"
                f"**1. Supplier** — {m['next_inbound_supplier_name']} supplies {row['volume_share_pct']:.0f}% of this "
                f"material; its OTD fell from {row['otd_baseline']:.1f}% to {row['otd_recent']:.1f}% "
                f"over the last {config.RECENT_WINDOW_DAYS} days.\n\n"
                f"**2. Purchase orders & logistics** — open orders are held at origin:\n{po_lines}\n\n"
                f"**3. Inventory** — {fmt_qty(m['stock'], u)} in stock, {fmt_qty(m['daily_consumption'], u, 1)}/day "
                f"consumed → {m['coverage_days']:.1f} days of coverage. No other supplier delivers before "
                f"{fmt_date(m['recovery_date'])}.\n\n"
                f"**4. Production impact** — shortage of {fmt_qty(m['shortage_qty'], u)} ≈ "
                f"{m['impact_line_stop_days']:.1f} days of full stop on {lines}, "
                f"{fmt_qty(m['impact_lost_output_t'], 't')} of lost output, "
                f"**{fmt_eur(issue.impact)}** impact (contribution margin + line restarts, demo assumptions).\n\n"
                f"**5. Options** — {len(scen.options)} scenarios simulated below. Recommended: **Option {scen.recommended}**. "
                f"{scen.rationale}")
        return CopilotAnswer(
            "", "", text,
            facts=[("Lines at risk", lines), ("Stockout date", fmt_date(m["stockout_date"])),
                   ("Shortage", fmt_qty(m["shortage_qty"], u)), ("Impact if no action", fmt_eur(issue.impact))],
            actions=issue.actions, sources=["suppliers", "purchase_orders", "logistics", "inventory", "production"],
            scenario=scen, page="inventory", reasoning=self._reasoning(issue, scen))

    def _a_critical(self) -> CopilotAnswer:
        crit = [i for i in self.r.issues if i.severity == "Critical"]
        body = "\n".join(f"{n}. **{i.title}** — {i.summary} *{i.impact_label}: {fmt_eur(i.impact)}*"
                         for n, i in enumerate(crit, 1))
        others = len(self.r.issues) - len(crit)
        text = (f"**{len(crit)} critical issues** detected as of {fmt_date(self.r.as_of)}:\n\n{body}\n\n"
                f"{others} further issues of lower severity are listed on the dashboard.")
        return CopilotAnswer("", "", text, facts=[("Critical issues", str(len(crit))),
                                                  ("Total issues", str(len(self.r.issues)))],
                             actions=[f"{i.title}: {i.actions[0]}" for i in crit if i.actions],
                             sources=sorted({s for i in crit for s in i.sources}), page="dashboard",
                             reasoning=["DATA: 7 datasets loaded", "DETECTION: rule-based detectors on each module",
                                        "DIAGNOSIS: evidence computed per issue",
                                        "IMPACT: financial / production impact computed", "PRIORITISATION: severity then impact"])

    def _a_investigate_supplier(self) -> CopilotAnswer:
        sc = self.r.scorecard
        top = sc.iloc[0]
        nxt = sc.iloc[1:3]
        text = (f"Investigate **{top['supplier_name']}** first — highest risk score "
                f"**{top['risk_score']:.0f}/100 ({top['risk_level']})**.\n\n{diagnose_supplier(top)}\n\n"
                f"Spend over 12 months: {fmt_eur(top['spend'])}. "
                + "Next on the watch list: " + ", ".join(f"{r.supplier_name} ({r.risk_score:.0f}/100)" for r in nxt.itertuples())
                + ".")
        comps = ", ".join(f"{k} {top[f'risk_{k}'] * 100:.0f}%" for k in config.RISK_WEIGHTS)
        return CopilotAnswer("", "", text,
                             facts=[("Risk score", f"{top['risk_score']:.0f}/100"), ("OTD recent", f"{top['otd_recent']:.1f}%"),
                                    ("PPM recent", f"{top['ppm_recent']:,.0f}"), ("Price gap", f"{top['price_gap_pct']:+.1f}%")],
                             actions=supplier_actions(top), sources=["purchase_orders", "quality", "suppliers"],
                             page="suppliers",
                             reasoning=[f"Risk score components for {top['supplier_name']}: {comps}",
                                        "Weights: " + ", ".join(f"{k} {v:.0%}" for k, v in config.RISK_WEIGHTS.items())])

    def _a_deteriorating(self) -> CopilotAnswer:
        det = self.r.deteriorating
        if det.empty:
            return CopilotAnswer("", "", "No supplier shows a significant deterioration over the recent period.",
                                 sources=["purchase_orders", "quality"])
        parts = [f"- **{r['supplier_name']}**: {diagnose_supplier(r)}" for _, r in det.iterrows()]
        q_only = self.r.issue("quality_spike")
        if q_only:
            parts += [f"- **{i.entity_name}** (quality only): {i.summary}" for i in q_only]
        text = (f"Comparison of the last {config.RECENT_WINDOW_DAYS} days vs the previous 9 months:\n\n" + "\n".join(parts))
        return CopilotAnswer("", "", text, facts=[("Deteriorating suppliers", str(len(det) + len(q_only)))],
                             actions=supplier_actions(det.iloc[0]), sources=["purchase_orders", "quality", "suppliers"],
                             page="suppliers")

    def _a_savings(self) -> CopilotAnswer:
        from src.analytics.opportunities import ANNUAL, total_annual_savings
        opps = sorted([o for o in self.r.opportunities if o.value_type == ANNUAL], key=lambda o: -o.value)
        total = total_annual_savings(self.r.opportunities)
        body = "\n".join(f"{n}. **{o.title}** — {fmt_eur(o.value)}/year  \n   `{o.formula}`"
                         for n, o in enumerate(opps[:5], 1))
        top_share = opps[0].value / total * 100 if total else 0
        text = (f"Total identified savings: **{fmt_eur(total)}/year** across {len(opps)} opportunities. "
                f"The largest one represents {top_share:.0f}% of the total.\n\n{body}")
        return CopilotAnswer("", "", text, facts=[("Potential savings", f"{fmt_eur(total, compact=True)}/year")],
                             actions=[opps[0].recommendation] if opps else [],
                             sources=["purchase_orders", "suppliers", "inventory", "logistics", "quality"],
                             page="opportunities")

    def _a_quality(self) -> CopilotAnswer:
        sp = self.r.quality_spikes
        if sp.empty:
            return CopilotAnswer("", "", "No abnormal quality trend detected.", sources=["quality"])
        names = self.r.data.suppliers.set_index("supplier_id")["supplier_name"]
        body = "\n".join(
            f"- **{names[r['supplier_id']]}** ({self.r.data.material_name(r['material_id'])}): PPM "
            f"{r['ppm_baseline']:,.0f} → {r['ppm_recent']:,.0f} (×{r['ppm_ratio']:.1f}), main defect "
            f"*{r['top_defect_recent']}*, {r['recent_nonconforming_lots']} non-conforming lots"
            for _, r in sp.iterrows())
        return CopilotAnswer("", "", f"Quality spikes over the last {config.QUALITY_RECENT_DAYS} days:\n\n{body}",
                             facts=[("Suppliers with quality spike", str(len(sp)))],
                             actions=["Open an 8D with each supplier concerned.",
                                      "Reinforce incoming inspection and quarantine suspect lots."],
                             sources=["quality", "purchase_orders"], page="suppliers")

    def _a_price(self) -> CopilotAnswer:
        an = self.r.price_anomalies
        dev = self.r.contract_deviations
        if an.empty and dev.empty:
            return CopilotAnswer("", "", "No abnormal price evolution detected.", sources=["purchase_orders"])
        lines = [f"- **{r['supplier_name']}**: {fmt_eur(r['baseline_price'], unit='t')} → {fmt_eur(r['current_price'], unit='t')} "
                 f"({r['price_change_pct']:+.1f}%) vs peers {r['peer_change_pct']:+.1f}%; gap to benchmark "
                 f"{r['gap_vs_benchmark_pct']:+.1f}% → **{fmt_eur(r['potential_savings'])}/year**"
                 for _, r in an.iterrows()]
        lines += [f"- **{r['supplier_name']}**: invoiced {fmt_eur(r['current_price'], unit='t')} vs contract "
                  f"{fmt_eur(r['contracted_price'], unit='t')} → {fmt_eur(r['potential_savings'])}/year"
                  for _, r in dev.iterrows()]
        return CopilotAnswer("", "", "Price anomalies detected:\n\n" + "\n".join(lines),
                             facts=[("Price anomalies", str(len(an) + len(dev)))],
                             actions=["Investigate the price deviation and renegotiate using benchmark prices."],
                             sources=["purchase_orders", "suppliers"], page="purchasing")

    def _a_inventory(self) -> CopilotAnswer:
        inv = self.r.inventory
        over = inv[inv["risk"] == "OVERSTOCK"]
        risk = inv[inv["risk"].isin(["STOCKOUT RISK", "BELOW SAFETY STOCK", "WATCH"])]
        text = (f"Inventory value: **{fmt_eur(inv['stock_value'].sum())}** across {len(inv)} materials.\n\n"
                f"- At risk: {', '.join(risk['material_name']) or 'none'}\n"
                f"- Overstocked: " + (", ".join(f"{r.material_name} ({r.coverage_days:.0f} days, excess {fmt_eur(r.excess_value)})"
                                                for r in over.itertuples()) or "none"))
        return CopilotAnswer("", "", text, facts=[("Materials at risk", str(len(risk))), ("Overstocked", str(len(over)))],
                             sources=["inventory"], page="inventory")

    def _a_supplier(self, supplier_id: str) -> CopilotAnswer:
        row = self.r.supplier_row(supplier_id)
        text = f"**{row['supplier_name']}** — {diagnose_supplier(row)}"
        return CopilotAnswer("", "", text,
                             facts=[("Spend 12m", fmt_eur(row["spend"])), ("OTD recent", f"{row['otd_recent']:.1f}%"),
                                    ("PPM recent", f"{row['ppm_recent']:,.0f}" if np.isfinite(row["ppm_recent"]) else "—"),
                                    ("Risk", f"{row['risk_score']:.0f}/100")],
                             actions=supplier_actions(row), sources=["purchase_orders", "quality", "suppliers"],
                             page="suppliers")

    def _a_help(self) -> CopilotAnswer:
        text = ("I can answer questions about supply risks, suppliers, purchasing prices, savings, quality and "
                "inventory, using the plant data only. Try one of:\n\n" + "\n".join(f"- {q}" for q in SUGGESTED_QUESTIONS))
        return CopilotAnswer("", "", text)

    # ------------------------------------------------------------------ trace
    @staticmethod
    def _reasoning(issue, scen: ScenarioSet | None) -> list[str]:
        m = issue.metrics
        steps = [
            "DATA — crossed: supplier master, purchase orders (ERP), carrier tracking, stock levels (WMS), production (MES)",
            f"DETECTION — projected stock < 0 on {fmt_date(m['stockout_date'])} using carrier ETAs "
            f"(ERP view: {'no stockout' if m.get('stockout_date_erp_view') is None else fmt_date(m['stockout_date_erp_view'])})",
            f"DIAGNOSIS — coverage {m['coverage_days']:.1f} d = {m['stock']:.1f} / {m['daily_consumption']:.2f}; "
            f"next inbound delayed {fmt_date(m['next_inbound_erp'])} → {fmt_date(m['next_inbound_ai'])}",
            f"IMPACT — shortage {m['shortage_qty']:.1f} t × yield {m['impact_yield']:.3f} × "
            f"€{config.CONTRIBUTION_MARGIN_EUR_PER_T:.0f}/t + restart costs = {fmt_eur(issue.impact)}",
        ]
        if scen:
            steps.append(f"OPTIONS — {len(scen.options)} scenarios costed (expedite / alternative supplier / production adjustment)")
            steps.append(f"RECOMMENDATION — Option {scen.recommended} (lowest total expected cost among feasible options)")
        steps.append("HUMAN VALIDATION — decision required; no action is executed in the ERP")
        return steps
