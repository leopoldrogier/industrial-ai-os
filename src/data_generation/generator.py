"""Synthetic dataset generator for the fictitious company Méditerranée Aluminium.

The generator is deterministic (fixed seed) and produces 12 months of
consistent operational data:

* production (MES)          -> drives daily raw material consumption
* purchase orders (ERP)     -> replenishment lanes per supplier, with delays
* inventory (WMS)           -> simulated day by day: receipts - consumption
* logistics (TMS / carrier) -> one shipment per purchase order
* quality (QMS)             -> goods-receipt inspection per delivery

Hidden problems are injected on purpose (see ``SCENARIO`` notes below) and
must be *discovered* by the analytics layer — they are never flagged in the data.

Run:  python -m src.data_generation.generator
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
START = pd.Timestamp("2025-09-29")
AS_OF = pd.Timestamp("2026-09-28")
DAYS = pd.date_range(START, AS_OF, freq="D")
RECENT_START = AS_OF - pd.Timedelta(days=90)
QUALITY_SPIKE_START = AS_OF - pd.Timedelta(days=60)
MAS_PRICE_RAMP_START = pd.Timestamp("2026-03-16")

# --------------------------------------------------------------------------- #
# Master data
# --------------------------------------------------------------------------- #
# material_id, name, category, unit, criticality, ppm_limit, warehouse,
# base_price, safety_stock_days, target_stock_days, lot, consumption_per_t_output
PRODUCTS = [
    ("RM-P1020", "Primary Aluminium Ingot P1020A", "Raw material", "t", "Critical", 350,
     "Fos - Raw Material Yard", 2280.0, 5, 9, 24.0, None),
    ("RM-SCRAP", "Aluminium Scrap 6063 (sorted)", "Raw material", "t", "High", 5000,
     "Fos - Raw Material Yard", 1600.0, 6, 13, 12.0, None),
    ("RM-MG", "Magnesium Ingot 99.9%", "Alloying element", "t", "High", 800,
     "Fos - Alloy Store", 3300.0, 15, 190, 1.0, 0.0066),
    ("RM-SI", "Silicon Metal 553", "Alloying element", "t", "High", 800,
     "Fos - Alloy Store", 2050.0, 15, 30, 1.0, 0.0050),
    ("RM-TIB", "AlTi5B1 Grain Refiner Rod", "Alloying element", "t", "High", 600,
     "Fos - Alloy Store", 4200.0, 20, 40, 0.5, 0.0012),
    ("CON-FLUX", "Degassing & Cleaning Flux", "Consumable", "t", "Medium", 1000,
     "Fos - Consumables Store", 1900.0, 15, 35, 0.5, 0.0021),
    ("CON-FILTER", "Ceramic Foam Filter 20ppi", "Consumable", "pcs", "Medium", 3000,
     "Fos - Consumables Store", 27.5, 12, 30, 100.0, 0.60),
    ("CON-REFRACT", "Refractory Castable (furnace lining)", "Maintenance", "t", "Medium", 2000,
     "Fos - Consumables Store", 1450.0, 20, 160, 1.0, 0.0040),
    ("PKG-PALLET", "Heavy-duty Wooden Pallet 1200x1000", "Packaging", "pcs", "Low", 10000,
     "Fos - Packaging Store", 13.8, 6, 14, 200.0, 0.95),
]

# supplier_id, name, country, main material, payment terms, pricing model, status, incoterm
SUPPLIERS = [
    ("S001", "MAS Aluminium", "Spain", "RM-P1020", "60 days EOM", "LME-indexed", "Strategic", "DAP Fos"),
    ("S002", "Nordic Metals AS", "Norway", "RM-P1020", "45 days", "LME-indexed", "Strategic", "FCA Mosjøen"),
    ("S003", "Gulf Aluminium Trading", "United Arab Emirates", "RM-P1020", "30 days", "LME-indexed", "Approved", "FOB Jebel Ali"),
    ("S004", "Balkan Primary Metals", "Slovenia", "RM-P1020", "45 days", "LME-indexed", "Approved", "FCA Ljubljana"),
    ("S005", "Recyclage Provence", "France", "RM-SCRAP", "30 days", "Scrap-indexed", "Strategic", "DAP Fos"),
    ("S006", "EcoMetal Iberia", "Spain", "RM-SCRAP", "45 days", "Scrap-indexed", "Approved", "DAP Fos"),
    ("S007", "Magnesium Europe GmbH", "Germany", "RM-MG", "60 days", "Fixed annual", "Approved", "DAP Fos"),
    ("S008", "SiliMetal France", "France", "RM-SI", "60 days EOM", "Fixed annual", "Approved", "DAP Fos"),
    ("S009", "Réfractaires du Sud", "France", "CON-REFRACT", "60 days", "Fixed annual", "Approved", "DAP Fos"),
    ("S010", "CastTech Consumables", "United Kingdom", "CON-FLUX", "60 days", "Fixed annual", "Approved", "DAP Fos"),
    ("S011", "Master Alloys BV", "Netherlands", "RM-TIB", "45 days", "Fixed annual", "Approved", "DAP Fos"),
    ("S012", "Emballages Rhône", "France", "PKG-PALLET", "30 days", "Fixed annual", "Approved", "DAP Fos"),
    ("S013", "Ceramic Filters Italia", "Italy", "CON-FILTER", "60 days", "Fixed annual", "Approved", "DAP Fos"),
    ("S014", "Alumetal Trading SA", "Switzerland", "RM-P1020", "Prepayment", "Spot", "Approved - spot", "DAP Fos"),
]

# Replenishment lanes: material -> (supplier, share, cadence_days, lead_time_days, mode, first_offset)
LANES = {
    "RM-P1020": [("S001", 0.46, 7, 8, "Road", 2), ("S002", 0.24, 28, 24, "Sea", 11),
                 ("S003", 0.18, 42, 38, "Sea", 25), ("S004", 0.12, 28, 14, "Rail", 18)],
    "RM-SCRAP": [("S005", 0.60, 4, 2, "Road", 1), ("S006", 0.40, 7, 6, "Road", 3)],
    "RM-MG": [("S007", 1.0, 30, 30, "Road", 12)],
    "RM-SI": [("S008", 1.0, 30, 10, "Road", 6)],
    "RM-TIB": [("S011", 1.0, 30, 21, "Road", 20)],
    "CON-FLUX": [("S010", 1.0, 30, 14, "Road", 9)],
    "CON-FILTER": [("S013", 1.0, 21, 18, "Road", 4)],
    "CON-REFRACT": [("S009", 1.0, 45, 20, "Road", 15)],
    "PKG-PALLET": [("S012", 1.0, 7, 5, "Road", 2)],
}

# Delivery behaviour: late rate (hist, recent), delay range, partial delivery rate (hist, recent)
DELIVERY_PROFILE = {
    "S001": dict(late=(0.04, 0.17), delay=(1, 3), delay_recent=(3, 9), partial=(0.02, 0.12)),
    "S002": dict(late=(0.07, 0.07), delay=(2, 5), partial=(0.02, 0.02)),
    "S003": dict(late=(0.12, 0.12), delay=(3, 8), partial=(0.03, 0.03)),
    "S004": dict(late=(0.05, 0.05), delay=(1, 4), partial=(0.02, 0.02)),
    "S005": dict(late=(0.04, 0.04), delay=(1, 2), partial=(0.02, 0.02)),
    "S006": dict(late=(0.08, 0.09), delay=(1, 4), partial=(0.03, 0.03)),
    "S007": dict(late=(0.10, 0.10), delay=(2, 6), partial=(0.0, 0.0)),
    "S008": dict(late=(0.05, 0.05), delay=(1, 3), partial=(0.0, 0.0)),
    "S009": dict(late=(0.10, 0.10), delay=(2, 7), partial=(0.0, 0.0)),
    "S010": dict(late=(0.05, 0.05), delay=(1, 4), partial=(0.0, 0.0)),
    "S011": dict(late=(0.06, 0.06), delay=(2, 5), partial=(0.0, 0.0)),
    "S012": dict(late=(0.06, 0.06), delay=(1, 2), partial=(0.04, 0.04)),
    "S013": dict(late=(0.08, 0.08), delay=(2, 5), partial=(0.02, 0.02)),
    "S014": dict(late=(0.0, 0.0), delay=(1, 1), partial=(0.0, 0.0)),
}

# Quality: (baseline ppm, recent ppm, recent window start, main defect types, spike defect type)
QUALITY_PROFILE = {
    "S001": (180, 430, RECENT_START, ["Surface oxidation", "Dross inclusions", "Damaged bundles"], "Dross inclusions"),
    "S002": (120, 120, None, ["Surface oxidation", "Damaged bundles"], None),
    "S003": (210, 210, None, ["Surface oxidation", "Dross inclusions", "Damaged bundles"], None),
    "S004": (160, 160, None, ["Surface oxidation", "Chemical composition out of spec"], None),
    "S005": (2300, 2300, None, ["Moisture / oil contamination", "Foreign material"], None),
    "S006": (2600, 9500, QUALITY_SPIKE_START, ["Moisture / oil contamination", "Foreign material"],
             "Contamination - Fe/Cu out of spec"),
    "S007": (300, 300, None, ["Oxidised ingots"], None),
    "S008": (400, 400, None, ["Granulometry out of spec", "Chemical composition out of spec"], None),
    "S009": (800, 800, None, ["Damaged bags", "Humidity"], None),
    "S010": (500, 500, None, ["Damaged packaging"], None),
    "S011": (250, 250, None, ["Rod diameter out of tolerance"], None),
    "S012": (6000, 6000, None, ["Broken boards"], None),
    "S013": (1500, 1500, None, ["Cracked filters"], None),
    "S014": (150, 150, None, ["Surface oxidation"], None),
}

# Logistics cost: (EUR per unit, fixed EUR per shipment) by supplier lane
LOGISTICS_RATE = {
    "S001": (42.0, 350), "S002": (58.0, 900), "S003": (74.0, 1400), "S004": (51.0, 600),
    "S005": (12.0, 150), "S006": (40.0, 300), "S007": (95.0, 450), "S008": (35.0, 250),
    "S009": (38.0, 300), "S010": (140.0, 550), "S011": (110.0, 400), "S012": (0.9, 280),
    "S013": (0.55, 420), "S014": (48.0, 400),
}
TRANSIT_DAYS = {"Road": 3, "Sea": 14, "Rail": 6, "Road Express": 2}

LINES = [
    dict(line="CL1", family="Extrusion billets 6060/6063", material="RM-P1020", capacity=13.0, yield_=0.955, furnace="F1"),
    dict(line="CL2", family="Rolling slabs 5xxx", material="RM-P1020", capacity=11.5, yield_=0.960, furnace="F1"),
    dict(line="CL3", family="Recycled billets 6063", material="RM-SCRAP", capacity=9.0, yield_=0.900, furnace="F2"),
]


@dataclass
class Slot:
    """One replenishment order (becomes a PO + a shipment + an inspection)."""

    material_id: str
    supplier_id: str
    order_date: pd.Timestamp
    promised_date: pd.Timestamp
    mode: str
    share: float
    cadence: int
    delay: int = 0
    qty: float | None = None
    fixed_qty: bool = False
    recv_factor: float = 1.0
    unit_price: float = 0.0
    eta_override: pd.Timestamp | None = None
    status_note: str = ""

    @property
    def arrival(self) -> pd.Timestamp:
        if self.eta_override is not None:
            return self.eta_override
        return self.promised_date + pd.Timedelta(days=self.delay)

    @property
    def received(self) -> bool:
        return self.eta_override is None and self.arrival <= AS_OF


# --------------------------------------------------------------------------- #
# Price model
# --------------------------------------------------------------------------- #
def _market_indices(rng: np.random.Generator) -> pd.DataFrame:
    idx = pd.date_range(START - pd.Timedelta(days=90), AS_OF + pd.Timedelta(days=60), freq="D")
    n = len(idx)
    t = np.arange(n)
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.93 * noise[i - 1] + rng.normal(0, 4.0)
    p1020 = 2262 + 100 * (t / n) + 20 * np.sin(2 * np.pi * t / 150) + noise
    scrap = 1590 + 45 * (t / n) + 14 * np.sin(2 * np.pi * t / 120 + 1) + 0.6 * noise
    return pd.DataFrame({"RM-P1020": p1020, "RM-SCRAP": scrap}, index=idx)


def _unit_price(slot: Slot, indices: pd.DataFrame, rng: np.random.Generator) -> float:
    d = slot.order_date
    s, m = slot.supplier_id, slot.material_id
    if m in ("RM-P1020", "RM-SCRAP"):
        base = float(indices.at[d, m])
        premium = {
            "S002": 20.0, "S003": -15.0, "S004": 5.0, "S014": 135.0,
            "S005": 0.0, "S006": -45.0,
        }.get(s, 0.0)
        if s == "S001":
            # SCENARIO 2 — MAS Aluminium price drift: premium ramps from -25 to +170 €/t
            if d < MAS_PRICE_RAMP_START:
                premium = -25.0
            else:
                ramp = (d - MAS_PRICE_RAMP_START).days / (AS_OF - MAS_PRICE_RAMP_START).days
                premium = -25.0 + 195.0 * min(ramp, 1.0)
        return round(base + premium + rng.normal(0, 7), 1)

    base = {p[0]: p[7] for p in PRODUCTS}[m]
    if s == "S008" and d >= pd.Timestamp("2026-05-04"):
        # SCENARIO 5 — silicon invoiced above the fixed annual contract price
        return round(2235 + rng.normal(0, 6), 1)
    if m == "PKG-PALLET":
        base = 13.8 if d < pd.Timestamp("2026-01-01") else 14.2
        return round(base + rng.normal(0, 0.1), 2)
    if m == "CON-FILTER":
        return round(base + rng.normal(0, 0.15), 2)
    return round(base * (1 + rng.normal(0, 0.006)), 1)


# --------------------------------------------------------------------------- #
# Production (MES)
# --------------------------------------------------------------------------- #
def generate_production(rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for d in DAYS:
        for i, ln in enumerate(LINES):
            downtime = 0.0
            if pd.Timestamp("2025-12-24") <= d <= pd.Timestamp("2026-01-01"):
                downtime = 24.0  # year-end shutdown
            elif ln["furnace"] == "F1" and pd.Timestamp("2026-08-03") <= d <= pd.Timestamp("2026-08-14"):
                downtime = 24.0  # furnace F1 relining
            else:
                if d.dayofweek == i + 1:
                    downtime += 4.0  # weekly planned maintenance
                if rng.random() < 0.07:
                    downtime += float(rng.uniform(1.5, 9.0))  # breakdowns
            downtime = min(downtime, 24.0)
            util = float(np.clip(rng.normal(0.93, 0.035), 0.78, 1.0))
            output = ln["capacity"] * util * (24.0 - downtime) / 24.0
            consumption = output / ln["yield_"] * float(rng.normal(1.0, 0.008)) if output > 0 else 0.0
            rows.append({
                "date": d,
                "production_line": ln["line"],
                "furnace": ln["furnace"],
                "product_family": ln["family"],
                "material_id": ln["material"],
                "production_quantity": round(output, 2),
                "consumption_quantity": round(consumption, 2),
                "capacity": ln["capacity"],
                "downtime_hours": round(downtime, 1),
            })
    return pd.DataFrame(rows)


def daily_consumption(production: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Daily consumption per material, derived from production."""
    by_mat = production.pivot_table(index="date", columns="material_id",
                                     values="consumption_quantity", aggfunc="sum")
    total_output = production.groupby("date")["production_quantity"].sum()
    cons = pd.DataFrame(index=DAYS)
    cons["RM-P1020"] = by_mat["RM-P1020"]
    cons["RM-SCRAP"] = by_mat["RM-SCRAP"]
    for p in PRODUCTS:
        mid, unit, factor = p[0], p[3], p[11]
        if factor is None:
            continue
        noise = rng.normal(1.0, 0.06, len(DAYS))
        vals = total_output.values * factor * noise
        cons[mid] = np.round(vals) if unit == "pcs" else np.round(vals, 3)
    # furnace relining campaigns consume refractory castable
    for d0, qty, days in ((pd.Timestamp("2026-08-03"), 9.0, 8), (pd.Timestamp("2026-02-09"), 3.0, 3)):
        span = pd.date_range(d0, periods=days)
        cons.loc[span, "CON-REFRACT"] += qty / days
    return cons.round(3)


# --------------------------------------------------------------------------- #
# Purchase slots + inventory simulation
# --------------------------------------------------------------------------- #
def _build_slots(rng: np.random.Generator) -> dict[str, list[Slot]]:
    slots: dict[str, list[Slot]] = {}
    horizon = AS_OF + pd.Timedelta(days=45)
    for mid, lanes in LANES.items():
        mslots = []
        for sid, share, cadence, lt, mode, offset in lanes:
            p = START + pd.Timedelta(days=offset)
            while p <= horizon:
                lead = lt
                if sid == "S001" and p - pd.Timedelta(days=11) >= RECENT_START - pd.Timedelta(days=10):
                    lead = 11  # SCENARIO 3 — MAS quotes longer lead times
                od = p - pd.Timedelta(days=lead)
                if od <= AS_OF:
                    mslots.append(Slot(mid, sid, od, p, mode, share, cadence))
                p += pd.Timedelta(days=cadence)
        slots[mid] = mslots
    # Occasional spot purchases from a trader (short lead time, premium price)
    for off in (80, 185, 262):
        p = START + pd.Timedelta(days=off)
        slots["RM-P1020"].append(Slot("RM-P1020", "S014", p - pd.Timedelta(days=4), p,
                                      "Road Express", 0.0, 0, qty=48.0, fixed_qty=True))
    return slots


def _assign_delivery_behaviour(slots: dict[str, list[Slot]], rng: np.random.Generator) -> None:
    by_sup: dict[str, list[Slot]] = {}
    for mslots in slots.values():
        for s in mslots:
            by_sup.setdefault(s.supplier_id, []).append(s)
    for sid, sl in by_sup.items():
        prof = DELIVERY_PROFILE[sid]
        past = sorted([s for s in sl if s.promised_date <= AS_OF], key=lambda s: s.promised_date)
        for period, subset in (("hist", [s for s in past if s.promised_date < RECENT_START]),
                               ("recent", [s for s in past if s.promised_date >= RECENT_START])):
            k = 0 if period == "hist" else 1
            if not subset:
                continue
            n_late = int(round(len(subset) * prof["late"][k]))
            lo, hi = prof["delay_recent"] if (period == "recent" and "delay_recent" in prof) else prof["delay"]
            late_idx = set(rng.choice(len(subset), size=n_late, replace=False).tolist()) if n_late else set()
            n_part = int(round(len(subset) * prof["partial"][k]))
            part_idx = set(rng.choice(len(subset), size=n_part, replace=False).tolist()) if n_part else set()
            for i, s in enumerate(subset):
                if i in late_idx:
                    s.delay = int(rng.integers(lo, hi + 1))
                elif rng.random() < 0.12:
                    s.delay = -1  # early delivery
                if i in part_idx:
                    s.recv_factor = float(rng.uniform(0.80, 0.95))
        # Express transport on a share of road deliveries (premium freight)
        for s in past:
            if s.mode == "Road" and sid in ("S001", "S005", "S006") and rng.random() < 0.08:
                s.mode = "Road Express"


def _simulate(mid: str, mslots: list[Slot], cons: pd.Series, nominal: float,
              target: float, lot: float, init_stock: float, base_price: float):
    """Day-by-day stock simulation. Returns (stock series, value series)."""
    orders: dict[pd.Timestamp, list[Slot]] = {}
    arrivals: dict[pd.Timestamp, list[Slot]] = {}
    for s in mslots:
        orders.setdefault(max(s.order_date, START), []).append(s)
        if s.received:
            arrivals.setdefault(s.arrival, []).append(s)
    on_hand = init_stock
    mac = base_price
    stock, value = [], []
    for d in DAYS:
        for s in sorted(orders.get(d, []), key=lambda x: x.promised_date):
            if s.fixed_qty:
                continue
            base = s.share * nominal * s.cadence
            if s.order_date < START:
                q = base
            else:
                open_in = sum(o.qty or 0 for o in mslots
                              if o.qty is not None and o.order_date <= d and o.arrival > d
                              and o.promised_date <= s.promised_date and o is not s)
                projected = on_hand + open_in - nominal * (s.promised_date - d).days
                q = base + 0.5 * s.share * (target - projected)
            s.qty = max(lot, round(q / lot) * lot)
        for s in arrivals.get(d, []):
            rq = _received_qty(s, lot)
            mac = (on_hand * mac + rq * s.unit_price) / (on_hand + rq) if on_hand + rq > 0 else mac
            on_hand += rq
        on_hand -= float(cons.at[d])
        stock.append(on_hand)
        value.append(max(on_hand, 0) * mac)
    # orders placed before START still need a quantity
    for s in mslots:
        if s.qty is None:
            s.qty = max(lot, round(s.share * nominal * s.cadence / lot) * lot)
    return pd.Series(stock, index=DAYS), pd.Series(value, index=DAYS)


def _received_qty(s: Slot, lot: float) -> float:
    q = s.qty or 0.0
    if s.recv_factor >= 1.0:
        return q
    if lot >= 10:  # full trucks / full batches only
        return max(lot, math.floor(q * s.recv_factor / lot) * lot) if q > lot else q
    return round(q * s.recv_factor, 3)


def generate(seed: int = SEED) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    indices = _market_indices(rng)
    production = generate_production(rng)
    cons = daily_consumption(production, rng)
    slots = _build_slots(rng)
    _assign_delivery_behaviour(slots, rng)
    for mslots in slots.values():
        for s in mslots:
            s.unit_price = _unit_price(s, indices, rng)

    pinfo = {p[0]: p for p in PRODUCTS}
    nominal = {m: float(cons[m].mean()) for m in cons.columns}
    stock_series, value_series = {}, {}

    for mid, mslots in slots.items():
        p = pinfo[mid]
        target = nominal[mid] * p[9]
        lot = p[10]
        if mid != "RM-P1020":
            st, val = _simulate(mid, mslots, cons[mid], nominal[mid], target, lot, target, p[7])
            stock_series[mid], value_series[mid] = st, val
            continue
        st, val = _p1020_scenario(mslots, cons[mid], nominal[mid], target, lot, p)
        stock_series[mid], value_series[mid] = st, val

    return _assemble(slots, production, cons, stock_series, value_series, rng)


def _p1020_scenario(mslots, cons, nominal, target, lot, p):
    """SCENARIO 1 — stockout risk on primary aluminium caused by MAS Aluminium.

    Stock at the analysis date is above safety stock (nothing alarming in the
    ERP), but the next MAS deliveries are held at origin and no other supplier
    delivers before the stock is exhausted.
    """
    ss = nominal * p[8]
    st, val = _simulate("RM-P1020", mslots, cons, nominal, target, lot, target, p[7])
    # 1) The last MAS deliveries were short-shipped (full trucks missing) so that the
    #    stock at the analysis date sits ~1.6x the safety stock: comfortable in the
    #    ERP, but not enough to bridge the gap until the next inbound delivery.
    desired = 1.6 * ss
    mas_recv = sorted([s for s in mslots if s.supplier_id == "S001" and s.received
                       and s.arrival > AS_OF - pd.Timedelta(days=24)],
                      key=lambda s: s.arrival, reverse=True)
    for s in mas_recv:
        delta = float(st.iloc[-1]) - desired
        s.fixed_qty = True
        if delta < lot / 2:
            if delta < -lot / 2:
                s.qty = (s.qty or 0) + math.ceil(-delta / lot) * lot
                st, val = _simulate("RM-P1020", mslots, cons, nominal, target, lot, target, p[7])
            break
        max_trucks = int(round((s.qty or lot) / lot)) - 1
        trucks = min(int(round(delta / lot)), max_trucks)
        if trucks > 0:
            s.recv_factor = 1 - trucks * lot / (s.qty or lot) + 1e-6
            s.status_note = "short delivery"
        st, val = _simulate("RM-P1020", mslots, cons, nominal, target, lot, target, p[7])

    # 2) Open MAS orders are held at origin -> carrier ETA well beyond coverage
    cons30 = float(cons.iloc[-30:].mean())
    coverage = float(st.iloc[-1]) / cons30
    eta = AS_OF + pd.Timedelta(days=math.ceil(coverage) + 5)
    mas_open = sorted([s for s in mslots if s.supplier_id == "S001" and s.promised_date > AS_OF
                       and s.order_date <= AS_OF], key=lambda s: s.promised_date)
    for i, s in enumerate(mas_open):
        s.eta_override = eta + pd.Timedelta(days=4 * i)
        s.status_note = "held at origin"
    # 3) No other supplier is scheduled to deliver before the MAS ETA
    others = sorted([s for s in mslots if s.supplier_id not in ("S001",) and s.promised_date > AS_OF
                     and s.promised_date <= eta + pd.Timedelta(days=1)], key=lambda s: s.promised_date)
    for i, s in enumerate(others):
        s.promised_date = eta + pd.Timedelta(days=2 + 2 * i)
    st, val = _simulate("RM-P1020", mslots, cons, nominal, target, lot, target, p[7])
    return st, val


# --------------------------------------------------------------------------- #
# Assemble output tables
# --------------------------------------------------------------------------- #
def _assemble(slots, production, cons, stock_series, value_series, rng) -> dict[str, pd.DataFrame]:
    pinfo = {p[0]: p for p in PRODUCTS}
    all_slots = sorted([s for ms in slots.values() for s in ms],
                       key=lambda s: (s.order_date, s.material_id, s.supplier_id))

    po_rows, log_rows, q_rows = [], [], []
    for i, s in enumerate(all_slots, start=1):
        lot = pinfo[s.material_id][10]
        unit = pinfo[s.material_id][3]
        po_id = f"PO-{s.order_date.year}-{i:05d}"
        received = s.received
        rq = _received_qty(s, lot) if received else 0.0
        if received:
            status = "Closed - short delivery" if rq < (s.qty or 0) - 1e-9 else "Received"
        else:
            status = "Open"
        po_rows.append({
            "po_id": po_id,
            "supplier_id": s.supplier_id,
            "order_date": s.order_date,
            "material_id": s.material_id,
            "ordered_quantity": round(s.qty or 0, 3),
            "unit": unit,
            "unit_price": s.unit_price,
            "currency": "EUR",
            "promised_date": s.promised_date,
            "actual_delivery_date": s.arrival if received else pd.NaT,
            "quantity_received": round(rq, 3),
            "status": status,
        })

        # ---- logistics
        mode = s.mode
        transit = TRANSIT_DAYS[mode]
        rate, fixed = LOGISTICS_RATE[s.supplier_id]
        mult = 2.6 if mode == "Road Express" and s.supplier_id != "S014" else 1.0
        cost = ((s.qty or 0) * rate * mult + fixed) * float(rng.normal(1.0, 0.04))
        if received:
            departure = s.arrival - pd.Timedelta(days=transit)
            sh_status, eta, actual = "Delivered", pd.NaT, s.arrival
        elif s.eta_override is not None:
            departure, sh_status, eta, actual = pd.NaT, "Delayed at origin", s.eta_override, pd.NaT
        else:
            planned_dep = s.arrival - pd.Timedelta(days=transit)
            if planned_dep <= AS_OF:
                departure, sh_status = planned_dep, "In transit"
            else:
                departure, sh_status = pd.NaT, "Planned"
            eta, actual = s.arrival, pd.NaT
        ref = actual if received else eta
        log_rows.append({
            "shipment_id": f"SH-{i:05d}",
            "po_id": po_id,
            "supplier_id": s.supplier_id,
            "material_id": s.material_id,
            "departure_date": departure,
            "expected_arrival": s.promised_date,
            "actual_arrival": actual,
            "current_eta": eta,
            "transport_mode": mode,
            "quantity": round(s.qty or 0, 3),
            "logistics_cost": round(cost, 2),
            "delay_days": int((ref - s.promised_date).days),
            "shipment_status": sh_status,
        })

        # ---- goods receipt inspection
        if received:
            base_ppm, recent_ppm, recent_start, types, spike_type = QUALITY_PROFILE[s.supplier_id]
            in_spike = recent_start is not None and s.arrival >= recent_start
            ppm_true = recent_ppm if in_spike else base_ppm
            inspected = rq * 1000 if unit == "t" else rq
            defects = int(rng.poisson(max(inspected * ppm_true / 1e6 * float(rng.lognormal(0, 0.3)), 0)))
            if defects == 0:
                dtype = "None"
            elif in_spike and spike_type and rng.random() < 0.75:
                dtype = spike_type
            else:
                dtype = str(rng.choice(types))
            ppm = defects / inspected * 1e6 if inspected else 0.0
            limit = pinfo[s.material_id][5]
            qstatus = ("Accepted" if ppm <= limit else
                       "Accepted with deviation" if ppm <= 2 * limit else "Quarantined - partial rejection")
            q_rows.append({
                "inspection_id": f"QI-{len(q_rows) + 1:05d}",
                "date": s.arrival,
                "po_id": po_id,
                "supplier_id": s.supplier_id,
                "material_id": s.material_id,
                "quantity_inspected": round(inspected, 1),
                "inspection_unit": "kg" if unit == "t" else unit,
                "defects": defects,
                "defect_type": dtype,
                "ppm": round(ppm, 1),
                "quality_status": qstatus,
            })

    inv_rows = []
    for mid, st in stock_series.items():
        p = pinfo[mid]
        ss = round(float(cons[mid].mean()) * p[8], 1 if p[3] == "t" else 0)
        val = value_series[mid]
        for d in DAYS:
            inv_rows.append({
                "date": d,
                "material_id": mid,
                "material_name": p[1],
                "stock_quantity": round(float(st.at[d]), 3 if p[3] == "t" else 0),
                "unit": p[3],
                "stock_value": round(float(val.at[d]), 2),
                "safety_stock": ss,
                "daily_consumption": float(cons.at[d, mid]),
                "warehouse": p[6],
            })

    products = pd.DataFrame([{
        "material_id": p[0], "material_name": p[1], "category": p[2], "unit": p[3],
        "standard_consumption": round(float(cons[p[0]].mean()), 3),
        "criticality": p[4], "quality_ppm_limit": p[5], "standard_cost": p[7],
        "safety_stock_days": p[8],
    } for p in PRODUCTS])

    po = pd.DataFrame(po_rows)
    # contract volume = planned annual volume at contract signature (rounded)
    vol = po.groupby("supplier_id")["ordered_quantity"].sum()
    first_price = po.sort_values("order_date").groupby("supplier_id")["unit_price"].first()
    lt = {sid: lt for lanes in LANES.values() for sid, _, _, lt, _, _ in lanes}
    lt["S014"] = 4
    suppliers = pd.DataFrame([{
        "supplier_id": s[0], "supplier_name": s[1], "country": s[2], "material": s[3],
        "contract_annual_volume": float(round(vol.get(s[0], 0) / (10 if pinfo[s[3]][3] == "t" else 1000)) *
                                        (10 if pinfo[s[3]][3] == "t" else 1000)),
        "volume_unit": pinfo[s[3]][3],
        "payment_terms": s[4],
        "standard_lead_time_days": lt[s[0]],
        "contracted_price": round(float(first_price.get(s[0], 0)), 1) if s[5] != "Fixed annual"
        else pinfo[s[3]][7],
        "pricing_model": s[5],
        "incoterm": s[7],
        "supplier_status": s[6],
    } for s in SUPPLIERS])
    if (suppliers.supplier_id == "S014").any():
        suppliers.loc[suppliers.supplier_id == "S014", "contract_annual_volume"] = 0.0

    return {
        "products": products,
        "suppliers": suppliers,
        "purchase_orders": po,
        "inventory": pd.DataFrame(inv_rows),
        "production": production,
        "logistics": pd.DataFrame(log_rows),
        "quality": pd.DataFrame(q_rows),
    }


def write_dataset(out_dir: Path, seed: int = SEED) -> dict[str, pd.DataFrame]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = generate(seed)
    for name, df in data.items():
        df.to_csv(out_dir / f"{name}.csv", index=False, date_format="%Y-%m-%d")
    return data


if __name__ == "__main__":
    from src.config import DATA_DIR

    frames = write_dataset(DATA_DIR)
    for name, df in frames.items():
        print(f"{name:16s} {len(df):6d} rows")
