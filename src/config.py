"""Central configuration: paths, analysis windows and business assumptions.

Every business assumption used in a financial calculation lives here, with a
comment. Nothing financial is hard-coded in the UI or in the copilot: the UI
displays values computed from the data using these parameters.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SQLITE_PATH = DATA_DIR / "industrial_ai_os.db"

DATASETS = (
    "products",
    "suppliers",
    "purchase_orders",
    "inventory",
    "production",
    "logistics",
    "quality",
)

# --------------------------------------------------------------------------- #
# Demo company (fictitious)
# --------------------------------------------------------------------------- #
COMPANY_NAME = "Méditerranée Aluminium"
COMPANY_SITE = "Fos-sur-Mer / Marseille"
COMPANY_DISCLAIMER = (
    "Fictitious company and synthetic data generated for demonstration purposes. "
    "No figure refers to a real company."
)

# --------------------------------------------------------------------------- #
# Analysis windows
# --------------------------------------------------------------------------- #
RECENT_WINDOW_DAYS = 90          # "recent" period for supplier performance trends
PRICE_RECENT_DAYS = 60           # "current" price = average over the last N days
PRICE_BASELINE_DAYS = 120        # "historical" price = first N days of the year window
QUALITY_RECENT_DAYS = 60         # recent window for quality spike detection
CONSUMPTION_WINDOW_DAYS = 30     # average daily consumption window
PROJECTION_HORIZON_DAYS = 45     # stock projection horizon
SPEND_WINDOW_DAYS = 365          # purchasing spend window

# --------------------------------------------------------------------------- #
# Supplier performance
# --------------------------------------------------------------------------- #
OTD_TOLERANCE_DAYS = 0           # delivered on/before promised date + tolerance = on time
OTIF_QTY_TOLERANCE = 0.98        # in full = received >= 98% of ordered quantity
OTD_DROP_ALERT_PP = 8.0          # OTD drop (percentage points) triggering an alert
PPM_RATIO_ALERT = 1.8            # recent PPM / baseline PPM triggering an alert
LEAD_TIME_INCREASE_ALERT_PCT = 15.0

# Risk score weights (sum = 1.0) — see src/analytics/suppliers.py
RISK_WEIGHTS = {
    "delivery": 0.30,
    "quality": 0.25,
    "price": 0.20,
    "dependency": 0.15,
    "trend": 0.10,
}
RISK_LEVEL_THRESHOLDS = {"High": 55.0, "Medium": 30.0}   # score >= threshold

# --------------------------------------------------------------------------- #
# Purchasing
# --------------------------------------------------------------------------- #
PRICE_DRIFT_THRESHOLD_PP = 5.0     # supplier price change minus peer change (pp)
BENCHMARK_GAP_THRESHOLD_PCT = 3.0  # current price above peer benchmark (%)
CONTRACT_DEVIATION_THRESHOLD_PCT = 3.0  # fixed-price contracts: paid above contract

# --------------------------------------------------------------------------- #
# Inventory
# --------------------------------------------------------------------------- #
OVERSTOCK_COVERAGE_DAYS = 90       # coverage above which a material is overstocked
TARGET_COVERAGE_DAYS = 45          # target coverage used to compute excess stock
INVENTORY_CARRYING_RATE = 0.20     # annual carrying cost (capital, storage, obsolescence)

# --------------------------------------------------------------------------- #
# Quality
# --------------------------------------------------------------------------- #
QUALITY_SPIKE_RATIO = 2.0          # recent PPM >= 2x baseline -> spike
NON_QUALITY_HANDLING_RATE = 0.35   # handling/sorting/re-melt cost as % of material value

# --------------------------------------------------------------------------- #
# Production & scenario assumptions (demo assumptions, clearly labelled in UI)
# --------------------------------------------------------------------------- #
CONTRIBUTION_MARGIN_EUR_PER_T = 650.0   # contribution margin per tonne of cast product
LINE_RESTART_COST_EUR = 12_000.0        # furnace holding + line restart per stopped line
EXPRESS_TRANSIT_DAYS = 3                # express road transit, EU suppliers
EXPRESS_COST_MULTIPLIER = 2.6           # express road cost vs standard road cost
EXPEDITE_FIXED_FEE_EUR = 3_500.0        # supplier expedite / re-planning fee
TRUCK_LOAD_T = 24.0                     # full truck load (t)

SEVERITY_ORDER = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
