"""Logistics analytics: freight cost, premium freight, delayed shipments."""
from __future__ import annotations

import pandas as pd

from src import config


def logistics_window(logistics: pd.DataFrame, as_of: pd.Timestamp, days: int = config.SPEND_WINDOW_DAYS) -> pd.DataFrame:
    return logistics[(logistics["expected_arrival"] > as_of - pd.Timedelta(days=days))
                     & (logistics["expected_arrival"] <= as_of)]


def cost_by_mode(logistics: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    w = logistics_window(logistics, as_of)
    return (w.groupby("transport_mode")
            .agg(shipments=("shipment_id", "count"), cost=("logistics_cost", "sum"),
                 avg_delay=("delay_days", lambda s: s.clip(lower=0).mean()))
            .reset_index().sort_values("cost", ascending=False))


def express_premium(logistics: pd.DataFrame, suppliers: pd.DataFrame, as_of: pd.Timestamp,
                    multiplier: float = config.EXPRESS_COST_MULTIPLIER) -> pd.DataFrame:
    """Premium paid on express shipments vs the standard mode of the same lane.

    Spot suppliers (whose standard mode *is* express) are excluded.
    premium = express cost x (1 - 1/multiplier)
    """
    w = logistics_window(logistics, as_of)
    spot = suppliers.loc[suppliers["pricing_model"] == "Spot", "supplier_id"]
    ex = w[(w["transport_mode"] == "Road Express") & (~w["supplier_id"].isin(spot))]
    ex = ex.assign(premium=ex["logistics_cost"] * (1 - 1 / multiplier))
    return (ex.groupby("supplier_id")
            .agg(express_shipments=("shipment_id", "count"), express_cost=("logistics_cost", "sum"),
                 premium=("premium", "sum"))
            .reset_index()
            .merge(suppliers[["supplier_id", "supplier_name"]], on="supplier_id", how="left")
            .sort_values("premium", ascending=False))


def delayed_open_shipments(logistics: pd.DataFrame) -> pd.DataFrame:
    """Open shipments whose carrier ETA is later than the ERP promised date."""
    o = logistics[logistics["actual_arrival"].isna() & (logistics["delay_days"] > 0)]
    return o.sort_values("delay_days", ascending=False).reset_index(drop=True)
