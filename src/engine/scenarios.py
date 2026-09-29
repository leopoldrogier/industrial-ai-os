"""Scenario simulation for a stockout risk: options, costs, feasibility, recommendation.

The V1 recommends only. No action is ever executed in the ERP.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src import config
from src.utils.formatting import fmt_date, fmt_eur, fmt_qty


@dataclass
class ScenarioOption:
    code: str
    name: str
    description: str
    cost: float                 # EUR, total expected cost of the option
    arrival_date: pd.Timestamp | None
    feasible: bool              # covers the need before the stockout date
    residual_impact: float      # EUR of production impact remaining
    confidence: str             # High | Medium | Low
    cost_breakdown: dict = field(default_factory=dict)
    notes: str = ""

    @property
    def total_cost(self) -> float:
        return self.cost + self.residual_impact


@dataclass
class ScenarioSet:
    material_id: str
    material_name: str
    supplier_id: str | None
    shortage_qty: float
    stockout_date: pd.Timestamp
    baseline_impact: float
    options: list[ScenarioOption]
    recommended: str
    rationale: str
    order_qty: float = 0.0     # shortage rounded up to full trucks


def _lane_cost_per_t(logistics: pd.DataFrame, supplier_id: str, mode: str | None = None) -> float:
    lg = logistics[(logistics["supplier_id"] == supplier_id) & (logistics["quantity"] > 0)]
    if mode:
        lg = lg[lg["transport_mode"] == mode]
    return float(lg["logistics_cost"].sum() / lg["quantity"].sum()) if len(lg) else float("nan")


def _spot_premium_pct(po: pd.DataFrame, alt_id: str, material_id: str, spot_ids: list[str]) -> float:
    """Historical premium of a supplier vs regular suppliers' prices within ±30 days."""
    alt = po[(po["supplier_id"] == alt_id) & (po["material_id"] == material_id)]
    reg = po[(po["material_id"] == material_id) & (~po["supplier_id"].isin(spot_ids + [alt_id]))]
    prem = []
    for _, r in alt.iterrows():
        ref = reg[(reg["order_date"] - r["order_date"]).abs() <= pd.Timedelta(days=30)]
        if len(ref):
            ref_p = (ref["ordered_quantity"] * ref["unit_price"]).sum() / ref["ordered_quantity"].sum()
            prem.append(r["unit_price"] / ref_p - 1)
    return float(np.mean(prem)) if prem else 0.05


def build_stockout_scenarios(material: pd.Series, impact: dict, data, price_stats: pd.DataFrame,
                             scorecard: pd.DataFrame, as_of: pd.Timestamp) -> ScenarioSet:
    """Options to bridge a projected shortage on one material."""
    mid = material["material_id"]
    shortage = float(material["shortage_qty"])
    stockout = pd.Timestamp(material["stockout_date"])
    qty = math.ceil(shortage / config.TRUCK_LOAD_T) * config.TRUCK_LOAD_T if material["unit"] == "t" else shortage
    po, lg, sup = data.purchase_orders, data.logistics, data.suppliers

    # Supplier whose delayed deliveries cause the gap = first open order in the plan
    main_sup = material.get("next_inbound_supplier")
    main_name = data.supplier_name(main_sup) if main_sup else "the supplier"
    ps_main = price_stats[(price_stats["supplier_id"] == main_sup) & (price_stats["material_id"] == mid)]
    main_price = float(ps_main["current_price"].iloc[0]) if len(ps_main) else float(material["unit_value"])
    sc_main = scorecard[scorecard["supplier_id"] == main_sup]
    main_otd = float(sc_main["otd_recent"].iloc[0]) if len(sc_main) else float("nan")

    baseline = float(impact["total_impact"])
    options: list[ScenarioOption] = []

    # ---- Option A: expedite a partial shipment from the current supplier (express road)
    road = _lane_cost_per_t(lg, main_sup, "Road")
    road = road if np.isfinite(road) else 45.0
    premium_freight = qty * road * (config.EXPRESS_COST_MULTIPLIER - 1)
    arr_a = as_of + pd.Timedelta(days=config.EXPRESS_TRANSIT_DAYS + 1)
    cost_a = premium_freight + config.EXPEDITE_FIXED_FEE_EUR
    options.append(ScenarioOption(
        code="A", name=f"Expedite shipment ({main_name})",
        description=(f"Ask {main_name} to release {fmt_qty(qty, 't')} from its open orders immediately and ship by "
                     f"express road ({config.EXPRESS_TRANSIT_DAYS} days transit)."),
        cost=cost_a, arrival_date=arr_a, feasible=arr_a < stockout, residual_impact=0.0 if arr_a < stockout else baseline,
        confidence="Medium" if (np.isfinite(main_otd) and main_otd < 90) else "High",
        cost_breakdown={"Express freight premium": premium_freight, "Expedite fee": config.EXPEDITE_FIXED_FEE_EUR},
        notes=(f"Depends on {main_name} capacity to ship — recent OTD {main_otd:.1f}%, shipments currently held at origin."
               if np.isfinite(main_otd) else ""),
    ))

    # ---- Option B: buy from the fastest qualified alternative supplier
    alts = scorecard[(scorecard["material_id"] == mid) & (scorecard["supplier_id"] != main_sup)]
    lt_rows = []
    for _, a in alts.iterrows():
        rec = po[(po["supplier_id"] == a["supplier_id"]) & po["actual_delivery_date"].notna()]
        if len(rec):
            lt_rows.append((float((rec["actual_delivery_date"] - rec["order_date"]).dt.days.mean()), a))
    if lt_rows:
        lt, alt = sorted(lt_rows, key=lambda x: x[0])[0]
        alt_id, alt_name = alt["supplier_id"], alt["supplier_name"]
        spot_ids = sup.loc[sup["pricing_model"] == "Spot", "supplier_id"].tolist()
        bench = price_stats.loc[(price_stats["material_id"] == mid) & (price_stats["supplier_id"] == main_sup),
                                "benchmark_price"]
        bench = float(bench.iloc[0]) if len(bench) and pd.notna(bench.iloc[0]) else main_price
        prem = _spot_premium_pct(po, alt_id, mid, spot_ids)
        alt_price = bench * (1 + prem)
        freight = _lane_cost_per_t(lg, alt_id)
        freight = freight if np.isfinite(freight) else road * config.EXPRESS_COST_MULTIPLIER
        price_delta = (alt_price - main_price) * qty
        cost_b = price_delta + freight * qty
        arr_b = as_of + pd.Timedelta(days=math.ceil(lt))
        options.append(ScenarioOption(
            code="B", name=f"Alternative supplier ({alt_name})",
            description=(f"Spot purchase of {fmt_qty(qty, 't')} from {alt_name} (historical lead time {lt:.1f} days, "
                         f"OTD {alt['otd']:.0f}%) and reduce the open {main_name} order by the same quantity."),
            cost=cost_b, arrival_date=arr_b, feasible=arr_b < stockout,
            residual_impact=0.0 if arr_b < stockout else baseline,
            confidence="High" if alt["otd"] >= 95 else "Medium",
            cost_breakdown={"Estimated spot price": alt_price, f"Price delta vs {main_name}": price_delta,
                            "Freight": freight * qty},
            notes=(f"Spot price estimated as peer benchmark {fmt_eur(bench, unit='t')} + historical spot premium "
                   f"{prem * 100:.1f}%. Price delta is negative when the current supplier is priced above the spot market."),
        ))

    # ---- Option C: temporary production adjustment (slow down primary lines, load the scrap line)
    prod = data.production[data.production["date"] > as_of - pd.Timedelta(days=config.CONSUMPTION_WINDOW_DAYS)]
    other = prod[prod["material_id"] != mid].groupby("production_line").agg(
        cap=("capacity", "first"), out=("production_quantity", "mean"))
    spare_per_day = float((other["cap"] - other["out"]).clip(lower=0).sum()) if len(other) else 0.0
    days_short = int(material["days_without_stock"])
    recovered = min(impact["lost_output_t"], spare_per_day * days_short)
    lost_c = (impact["lost_output_t"] - recovered) * config.CONTRIBUTION_MARGIN_EUR_PER_T
    options.append(ScenarioOption(
        code="C", name="Temporary production adjustment",
        description=(f"Reduce throughput on {', '.join(impact['lines'])} from today to stretch the stock until the "
                     f"delayed delivery, and load spare capacity on the recycling line "
                     f"({spare_per_day:.1f} t/day available over {days_short} days)."),
        cost=0.0, arrival_date=None, feasible=True, residual_impact=lost_c, confidence="High",
        cost_breakdown={"Lost output (t)": impact["lost_output_t"] - recovered,
                        "Recovered on recycling line (t)": recovered,
                        "Lost contribution margin": lost_c},
        notes="Planned slowdown avoids a hard stop and restart costs; customer deliveries must be re-sequenced.",
    ))

    feasible = [o for o in options if o.feasible]
    best = min(feasible, key=lambda o: o.total_cost) if feasible else min(options, key=lambda o: o.total_cost)
    rationale = (f"Option {best.code} has the lowest total expected cost ({fmt_eur(best.total_cost)}) among options "
                 f"that secure supply before the stockout date ({fmt_date(stockout)}), versus "
                 f"{fmt_eur(baseline)} production impact if nothing is done.")
    if best.confidence != "High":
        rationale += " Confidence is limited: confirm feasibility with the supplier before committing."
    return ScenarioSet(mid, material["material_name"], main_sup, shortage, stockout, baseline, options, best.code, rationale, qty)


def scenario_frame(s: ScenarioSet) -> pd.DataFrame:
    return _frame(s).round({"Cost (€)": 0, "Residual production impact (€)": 0, "Total expected cost (€)": 0})


def _frame(s: ScenarioSet) -> pd.DataFrame:
    return pd.DataFrame([{
        "Option": f"{o.code} — {o.name}",
        "Cost (€)": o.cost,
        "Residual production impact (€)": o.residual_impact,
        "Total expected cost (€)": o.total_cost,
        "Arrival": fmt_date(o.arrival_date) if o.arrival_date is not None else "n/a",
        "Before stockout": "Yes" if o.feasible else "No",
        "Confidence": o.confidence,
    } for o in s.options] + [{
        "Option": "Do nothing",
        "Cost (€)": 0.0,
        "Residual production impact (€)": s.baseline_impact,
        "Total expected cost (€)": s.baseline_impact,
        "Arrival": "—", "Before stockout": "No", "Confidence": "—",
    }])
