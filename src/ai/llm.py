"""LLM integration point (prepared for V2, not required in V1).

V1 answers are produced by the deterministic rule-based copilot. A V2 provider
(OpenAI, Anthropic, local model...) plugs in here and may only *rephrase* a
draft answer using a fact sheet computed by Python — it must never compute or
invent figures. The app runs without any API key.

To add a provider in V2:
1. implement ``Narrator`` (``rephrase``)
2. register it in ``get_narrator`` (e.g. via the ``INDUSTRIAL_AI_OS_LLM`` env variable)
3. pass ``build_fact_sheet(result)`` as grounding context + ``SYSTEM_PROMPT``
"""
from __future__ import annotations

import json
import os
from typing import Protocol

SYSTEM_PROMPT = (
    "You are an industrial operations copilot for a manufacturing plant. "
    "Answer ONLY with facts and numbers present in the FACTS JSON. Never estimate, extrapolate "
    "or invent figures. If a fact is missing, say it is not available. You recommend; a human decides. "
    "You never execute actions in the ERP."
)


class Narrator(Protocol):
    name: str

    def rephrase(self, question: str, draft_answer: str, facts: dict) -> str:
        ...


class RuleBasedNarrator:
    """Default V1 narrator: returns the deterministic answer unchanged."""

    name = "Rule-based engine (V1, offline)"

    def rephrase(self, question: str, draft_answer: str, facts: dict) -> str:
        return draft_answer


def get_narrator() -> Narrator:
    provider = os.getenv("INDUSTRIAL_AI_OS_LLM", "rule_based").lower()
    if provider != "rule_based":
        # V2: return AnthropicNarrator() / OpenAINarrator() here.
        # V1 intentionally falls back to the offline engine.
        pass
    return RuleBasedNarrator()


def build_fact_sheet(result) -> dict:
    """JSON-serialisable facts computed by the analytics layer (LLM grounding context)."""
    def clean(d: dict) -> dict:
        return json.loads(json.dumps(d, default=str))

    return clean({
        "as_of": result.as_of,
        "kpis": result.kpis,
        "issues": [{"severity": i.severity, "title": i.title, "summary": i.summary,
                    "impact_eur": round(i.impact), "impact_label": i.impact_label, "actions": i.actions}
                   for i in result.issues],
        "opportunities": [{"title": o.title, "category": o.category, "value_eur": round(o.value),
                           "value_type": o.value_type, "formula": o.formula} for o in result.opportunities],
        "scenarios": {m: {"recommended": s.recommended, "rationale": s.rationale,
                          "options": [{"code": o.code, "name": o.name, "cost_eur": round(o.cost),
                                       "residual_impact_eur": round(o.residual_impact),
                                       "feasible": o.feasible} for o in s.options]}
                      for m, s in result.scenarios.items()},
    })
