"""Integration tests: the hidden problems planted in the demo dataset must be discovered,
and every financial figure must be reproducible from the data."""
import math

import numpy as np
import pandas as pd
import pytest

from src import config
from src.ai.copilot import SUGGESTED_QUESTIONS, Copilot
from src.analytics.opportunities import ANNUAL, total_annual_savings
from src.analytics.purchasing import weighted_price, window
from src.utils.formatting import fmt_eur

MAS, ECOMETAL, P1020 = "S001", "S006", "RM-P1020"


def test_four_critical_issues(result):
    crit = [i for i in result.issues if i.severity == "Critical"]
    assert result.kpis["critical_issues"] == 4
    assert {i.issue_type for i in crit} == {"stockout_risk", "supplier_deterioration", "price_drift", "quality_spike"}
    assert result.issues[0].issue_type == "stockout_risk"      # supply continuity first


def test_problem1_stockout_risk(result, data):
    inv = result.inventory.set_index("material_id")
    m = inv.loc[P1020]
    assert m["risk"] == "STOCKOUT RISK"
    assert (inv["risk"] == "STOCKOUT RISK").sum() == 1
    # enough stock today (above safety stock) ...
    assert m["stock"] > m["safety_stock"]
    # ... but coverage ends before the delayed delivery
    assert m["coverage_days"] == pytest.approx(m["stock"] / m["daily_consumption"])
    assert m["stockout_date"] < m["next_inbound_ai"]
    assert m["next_inbound_supplier"] == MAS
    assert m["next_inbound_ai"] > m["next_inbound_erp"]          # carrier ETA later than ERP promise
    assert m["stockout_date_erp_view"] is None                   # ERP alone does not see it
    assert m["shortage_qty"] > 0
    # stockout date consistent with coverage (no inbound before it)
    assert m["stockout_date"] == result.as_of + pd.Timedelta(days=math.floor(m["coverage_days"]) + 1)
    imp = result.impacts[P1020]
    assert imp["lines"] == ["CL1", "CL2"] and imp["total_impact"] > 0


def test_problem2_price_drift(result, data):
    an = result.price_anomalies
    assert an["supplier_id"].tolist() == [MAS]
    r = an.iloc[0]
    # recompute independently from the purchase orders
    po = window(data.purchase_orders, result.as_of, 365)
    mas = po[(po.supplier_id == MAS) & (po.material_id == P1020)]
    recent = mas[mas.order_date > result.as_of - pd.Timedelta(days=config.PRICE_RECENT_DAYS)]
    assert r["current_price"] == pytest.approx(weighted_price(recent))
    assert r["annual_volume"] == pytest.approx(mas["ordered_quantity"].sum())
    assert r["potential_savings"] == pytest.approx((r["current_price"] - r["benchmark_price"]) * r["annual_volume"])
    assert r["price_change_pct"] - r["peer_change_pct"] >= config.PRICE_DRIFT_THRESHOLD_PP


def test_problem3_supplier_deterioration(result):
    det = result.deteriorating
    assert det["supplier_id"].tolist() == [MAS]
    r = result.supplier_row(MAS)
    assert r["otd_baseline"] - r["otd_recent"] >= config.OTD_DROP_ALERT_PP
    assert r["ppm_recent"] > 2 * r["ppm_baseline"]
    assert r["avg_lead_time_recent"] > r["avg_lead_time_baseline"]
    assert r["avg_delay_recent"] > r["avg_delay_baseline"]
    assert r["risk_level"] == "High"
    assert result.scorecard.iloc[0]["supplier_id"] == MAS           # riskiest supplier


def test_problem4_quality_spike(result):
    sp = result.quality_spikes
    eco = sp[sp["supplier_id"] == ECOMETAL].iloc[0]
    assert eco["ppm_ratio"] >= 3
    assert eco["top_defect_recent"].startswith("Contamination")
    assert eco["material_id"] == "RM-SCRAP"
    # MAS quality drift is reported inside its deterioration issue, not twice
    assert [i.entity_id for i in result.issue("quality_spike")] == [ECOMETAL]


def test_no_false_alarms_on_stable_suppliers(result):
    stable = {"S002", "S004", "S007", "S009", "S010", "S011", "S013"}
    flagged = {i.entity_id for i in result.issues}
    assert not (stable & flagged)


def test_potential_savings_is_sum_of_annual_opportunities(result):
    total = sum(o.value for o in result.opportunities if o.value_type == ANNUAL)
    assert result.kpis["potential_savings"] == pytest.approx(total)
    assert total_annual_savings(result.opportunities) == pytest.approx(total)
    assert all(o.value >= 0 and np.isfinite(o.value) for o in result.opportunities)


def test_kpis_are_computed_not_nan(result):
    for k, v in result.kpis.items():
        assert v is not None and not (isinstance(v, float) and math.isnan(v)), k
    assert result.kpis["active_suppliers"] == 14


def test_financials_reproducible(data):
    from src.engine.analysis import run_analysis
    a, b = run_analysis(data), run_analysis(data)
    assert a.kpis == b.kpis
    assert [o.value for o in a.opportunities] == [o.value for o in b.opportunities]


def test_scenarios(result):
    s = result.scenarios[P1020]
    codes = {o.code for o in s.options}
    assert codes == {"A", "B", "C"}
    rec = next(o for o in s.options if o.code == s.recommended)
    assert rec.feasible
    feasible = [o for o in s.options if o.feasible]
    assert rec.total_cost == min(o.total_cost for o in feasible)
    assert rec.total_cost < s.baseline_impact
    assert s.order_qty >= s.shortage_qty and s.order_qty % config.TRUCK_LOAD_T == 0


def test_copilot_uses_computed_numbers(result):
    bot = Copilot(result)
    m = result.inventory.set_index("material_id").loc[P1020]
    ans = bot.ask("Why are we at risk of a stockout?")
    assert ans.intent == "stockout"
    assert f"{m['coverage_days']:.1f} days" in ans.answer
    assert "MAS Aluminium" in ans.answer
    prod = bot.ask("Why is production at risk?")
    assert prod.intent == "production_risk" and prod.scenario is not None
    assert fmt_eur(result.issues[0].impact) in prod.answer
    sav = bot.ask("Where are the biggest purchasing savings?")
    assert fmt_eur(result.kpis["potential_savings"]) in sav.answer


def test_copilot_routes_all_suggested_questions(result):
    bot = Copilot(result)
    intents = [bot.ask(q).intent for q in SUGGESTED_QUESTIONS]
    assert intents == ["production_risk", "stockout", "critical", "investigate_supplier", "deteriorating", "savings"]
    assert bot.ask("Tell me about EcoMetal Iberia").intent == "supplier"
    assert bot.ask("blabla").intent == "help"
