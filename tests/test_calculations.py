"""Unit tests of the core calculations on small, hand-checkable inputs."""
import math

import pandas as pd
import pytest

from src.analytics.inventory import (analyse_projection, days_of_coverage, estimate_stockout_date,
                                     production_impact, project_stock)
from src.analytics.purchasing import (contract_deviations, detect_price_anomalies, potential_savings,
                                      price_change_pct)
from src.analytics.quality import compute_ppm, cost_of_non_quality, detect_quality_spikes
from src.analytics.suppliers import compute_risk_score, delivery_metrics, risk_level
from src.utils.formatting import fmt_eur, fmt_pct, fmt_qty
from tests.conftest import make_po

AS_OF = pd.Timestamp("2026-09-28")


# --------------------------------------------------------------------------- OTD / OTIF
def test_otd_otif_delay_lead_time():
    po = make_po([
        ("2026-01-01", "2026-01-10", "2026-01-10", 100, 100),   # on time, in full
        ("2026-01-01", "2026-01-10", "2026-01-09", 100, 90),    # early, short -> not in full
        ("2026-01-01", "2026-01-10", "2026-01-14", 100, 100),   # 4 days late
        ("2026-01-01", "2026-01-10", "2026-01-12", 100, 100),   # 2 days late
    ])
    m = delivery_metrics(po)
    assert m["deliveries"] == 4
    assert m["otd"] == pytest.approx(50.0)
    assert m["otif"] == pytest.approx(25.0)
    assert m["avg_delay"] == pytest.approx((0 + 0 + 4 + 2) / 4)
    assert m["avg_delay_when_late"] == pytest.approx(3.0)
    assert m["avg_lead_time"] == pytest.approx((9 + 8 + 13 + 11) / 4)


def test_otd_counts_overdue_open_orders_as_late():
    po = make_po([
        ("2026-09-01", "2026-09-10", "2026-09-10", 10, 10),
        ("2026-09-01", "2026-09-20", None, 10, 0),              # overdue, not delivered
        ("2026-09-01", "2026-10-05", None, 10, 0),              # not yet due -> ignored
    ])
    m = delivery_metrics(po, as_of=AS_OF)
    assert m["deliveries"] == 2
    assert m["otd"] == pytest.approx(50.0)


def test_otd_tolerance():
    po = make_po([("2026-01-01", "2026-01-10", "2026-01-11", 1, 1)])
    assert delivery_metrics(po, tolerance_days=0)["otd"] == 0
    assert delivery_metrics(po, tolerance_days=1)["otd"] == 100


def test_otd_empty():
    m = delivery_metrics(make_po([]))
    assert m["deliveries"] == 0 and math.isnan(m["otd"])


# --------------------------------------------------------------------------- PPM
def test_ppm():
    assert compute_ppm(43, 100_000) == pytest.approx(430.0)
    assert compute_ppm(0, 5000) == 0.0
    assert math.isnan(compute_ppm(3, 0))


def test_quality_spike_detection():
    trends = pd.DataFrame({
        "supplier_id": ["A", "B", "C"],
        "ppm_baseline": [2500, 180, 200],
        "ppm_recent": [9000, 300, 900],
        "ppm_ratio": [3.6, 1.67, 4.5],
        "quality_ppm_limit": [5000, 350, 1000],
    })
    spikes = detect_quality_spikes(trends)
    # A: ratio and limit exceeded; B: ratio too low; C: ratio high but below limit
    assert spikes["supplier_id"].tolist() == ["A"]


def test_cost_of_non_quality():
    # 2,000 kg defects in 60 days, 1,500 €/t, 35% handling, annualised
    assert cost_of_non_quality(2000, "t", 1500, 60, 0.35) == pytest.approx(2 * 1500 * 0.35 * 365 / 60)


# --------------------------------------------------------------------------- coverage / stockout
def test_days_of_coverage():
    assert days_of_coverage(420, 180) == pytest.approx(2.333, rel=1e-3)
    assert days_of_coverage(-5, 10) == 0
    assert math.isinf(days_of_coverage(100, 0))


def test_estimate_stockout_date():
    # 207 t at 23.3 t/day: end of day 8 = 20.6 t, end of day 9 negative
    assert estimate_stockout_date(207, 23.3, AS_OF) == AS_OF + pd.Timedelta(days=9)
    # exactly 8 days of stock: stock hits 0 on day 8 (not negative) -> day 9
    assert estimate_stockout_date(80, 10, AS_OF) == AS_OF + pd.Timedelta(days=9)
    assert estimate_stockout_date(80, 0, AS_OF) is None


def test_projection_shortage_and_recovery():
    inbound = pd.DataFrame({"expected_date": [AS_OF + pd.Timedelta(days=6)], "open_quantity": [100.0]})
    proj = project_stock(30, 10, inbound, AS_OF, horizon=10)
    # day 1..3 -> 20,10,0 ; day 4 -> -10 ; day 5 -> -20 ; day 6 -> +70
    assert proj["projected_stock"].tolist()[:6] == [20, 10, 0, -10, -20, 70]
    a = analyse_projection(proj, safety_stock=15)
    assert a["stockout_date"] == AS_OF + pd.Timedelta(days=4)
    assert a["shortage_qty"] == pytest.approx(20)
    assert a["days_without_stock"] == 2
    assert a["recovery_date"] == AS_OF + pd.Timedelta(days=6)
    assert a["ss_breach_date"] == AS_OF + pd.Timedelta(days=2)


def test_projection_horizon_ignores_unplanned_future():
    proj = project_stock(50, 10, pd.DataFrame(columns=["expected_date", "open_quantity"]), AS_OF, horizon=20)
    assert analyse_projection(proj, 0, until=AS_OF + pd.Timedelta(days=4))["stockout_date"] is None
    assert analyse_projection(proj, 0)["stockout_date"] == AS_OF + pd.Timedelta(days=6)


def test_production_impact():
    prod = pd.DataFrame({
        "date": [AS_OF] * 2, "material_id": ["M"] * 2, "production_line": ["L1", "L2"],
        "production_quantity": [95.0, 95.0], "consumption_quantity": [100.0, 100.0],
    })
    imp = production_impact(prod, "M", shortage_qty=50, daily_consumption=25, as_of=AS_OF,
                            margin_per_t=600, restart_cost=10_000)
    assert imp["lines"] == ["L1", "L2"]
    assert imp["lost_output_t"] == pytest.approx(47.5)
    assert imp["line_stop_days"] == pytest.approx(2.0)
    assert imp["total_impact"] == pytest.approx(47.5 * 600 + 2 * 10_000)


# --------------------------------------------------------------------------- price / savings
def test_price_variance_and_savings():
    assert price_change_pct(2250, 2520) == pytest.approx(12.0)
    assert math.isnan(price_change_pct(0, 10))
    assert potential_savings(140, 1200) == pytest.approx(168_000)
    assert potential_savings(-20, 1200) == 0.0


def _stats():
    return pd.DataFrame({
        "supplier_id": ["S1", "S2", "S3", "S4"],
        "supplier_name": ["Drifting", "Peer", "Spot", "Fixed"],
        "material_id": ["M", "M", "M", "X"],
        "pricing_model": ["LME-indexed", "LME-indexed", "Spot", "Fixed annual"],
        "baseline_price": [2250.0, 2300.0, 2400.0, 2050.0],
        "current_price": [2520.0, 2380.0, 2600.0, 2235.0],
        "price_change_pct": [12.0, 3.48, 8.3, 9.0],
        "peer_change_pct": [3.48, 12.0, 7.7, float("nan")],
        "excess_change_pp": [8.52, -8.52, 0.6, float("nan")],
        "benchmark_price": [2380.0, 2520.0, 2450.0, float("nan")],
        "gap_vs_benchmark": [140.0, -140.0, 150.0, float("nan")],
        "gap_vs_benchmark_pct": [5.88, -5.56, 6.1, float("nan")],
        "annual_volume": [1200.0, 900.0, 50.0, 52.0],
        "contracted_price": [2255.0, 2300.0, 0.0, 2050.0],
    })


def test_price_anomaly_detection_and_savings():
    an = detect_price_anomalies(_stats())
    assert an["supplier_id"].tolist() == ["S1"]          # spot and peer excluded
    assert an["potential_savings"].iloc[0] == pytest.approx(140 * 1200)


def test_contract_deviation():
    dev = contract_deviations(_stats())
    assert dev["supplier_id"].tolist() == ["S4"]
    assert dev["potential_savings"].iloc[0] == pytest.approx((2235 - 2050) * 52)


# --------------------------------------------------------------------------- risk score
def test_risk_score_bounds_and_levels():
    worst = compute_risk_score(50, 30, 2000, 350, 4, 20, 1.0, "Critical", 60)
    best = compute_risk_score(100, 0, 0, 350, 1, -5, 0.0, "Low", 0)
    assert worst["score"] == pytest.approx(100.0)
    assert best["score"] == pytest.approx(0.0)
    assert worst["level"] == "High" and best["level"] == "Low"
    assert risk_level(30) == "Medium" and risk_level(29.9) == "Low" and risk_level(55) == "High"


def test_risk_score_monotonic_in_otd():
    a = compute_risk_score(95, 0, 100, 350, 1, 0, 0.3, "High")["score"]
    b = compute_risk_score(85, 0, 100, 350, 1, 0, 0.3, "High")["score"]
    assert b > a


def test_risk_score_handles_nan():
    s = compute_risk_score(float("nan"), float("nan"), float("nan"), 350, float("nan"), float("nan"), 0, "Low")
    assert 0 <= s["score"] <= 100


# --------------------------------------------------------------------------- formatting
def test_formatting():
    assert fmt_eur(168000) == "€168,000"
    assert fmt_eur(168000, compact=True) == "€168k"
    assert fmt_eur(1_234_567, compact=True) == "€1.2M"
    assert fmt_eur(2520, unit="t") == "€2,520/t"
    assert fmt_eur(float("nan")) == "—"
    assert fmt_pct(12.345, signed=True) == "+12.3%"
    assert fmt_qty(95.04, "t") == "95.0 t"
