"""Supplier performance analytics: OTD, OTIF, delays, lead time, PPM, risk score."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.analytics.purchasing import price_stats, window
from src.analytics.quality import ppm_of

CRITICALITY_FACTOR = {"Critical": 1.0, "High": 0.8, "Medium": 0.5, "Low": 0.3}


def delivery_metrics(po: pd.DataFrame, tolerance_days: int = config.OTD_TOLERANCE_DAYS,
                     qty_tolerance: float = config.OTIF_QTY_TOLERANCE,
                     as_of: pd.Timestamp | None = None) -> dict:
    """Delivery KPIs on purchase order lines due so far.

    OTD  = deliveries on/before promised date (+tolerance) / deliveries due
    OTIF = on time AND quantity received >= qty_tolerance x ordered
    If ``as_of`` is given, open lines already past their promised date count as late.
    """
    rec = po[po["actual_delivery_date"].notna()]
    if as_of is not None:
        overdue = po[po["actual_delivery_date"].isna() & (po["promised_date"] < as_of)]
        rec = pd.concat([rec, overdue.assign(actual_delivery_date=as_of)]) if len(overdue) else rec
    if rec.empty:
        return {"deliveries": 0, "late_deliveries": 0, "otd": float("nan"), "otif": float("nan"),
                "avg_delay": float("nan"), "avg_delay_when_late": float("nan"), "avg_lead_time": float("nan")}
    delay = (rec["actual_delivery_date"] - rec["promised_date"]).dt.days
    on_time = delay <= tolerance_days
    in_full = rec["quantity_received"] >= rec["ordered_quantity"] * qty_tolerance
    late = delay[delay > tolerance_days]
    return {
        "deliveries": int(len(rec)),
        "late_deliveries": int((~on_time).sum()),
        "otd": float(on_time.mean() * 100),
        "otif": float((on_time & in_full).mean() * 100),
        "avg_delay": float(delay.clip(lower=0).mean()),
        "avg_delay_when_late": float(late.mean()) if len(late) else 0.0,
        "avg_lead_time": float((rec["actual_delivery_date"] - rec["order_date"]).dt.days.mean()),
    }


def split_periods(df: pd.DataFrame, date_col: str, as_of: pd.Timestamp,
                  recent_days: int = config.RECENT_WINDOW_DAYS) -> tuple[pd.DataFrame, pd.DataFrame]:
    cut = as_of - pd.Timedelta(days=recent_days)
    year_start = as_of - pd.Timedelta(days=365)
    base = df[(df[date_col] > year_start) & (df[date_col] <= cut)]
    recent = df[(df[date_col] > cut) & (df[date_col] <= as_of)]
    return base, recent


def monthly_delivery(po: pd.DataFrame, supplier_id: str) -> pd.DataFrame:
    rec = po[(po["supplier_id"] == supplier_id) & po["actual_delivery_date"].notna()]
    rec = rec.assign(month=rec["actual_delivery_date"].dt.to_period("M").dt.to_timestamp(),
                     delay=(rec["actual_delivery_date"] - rec["promised_date"]).dt.days,
                     lead_time=(rec["actual_delivery_date"] - rec["order_date"]).dt.days)
    rec = rec.assign(on_time=rec["delay"] <= config.OTD_TOLERANCE_DAYS)
    g = rec.groupby("month").agg(deliveries=("po_id", "count"), otd=("on_time", "mean"),
                                 avg_delay=("delay", lambda s: s.clip(lower=0).mean()),
                                 avg_lead_time=("lead_time", "mean")).reset_index()
    g["otd"] = g["otd"] * 100
    # Rolling 3-month OTD (volume-weighted) — smoother trend than single months
    on_time = (g["otd"] / 100 * g["deliveries"]).rolling(3, min_periods=1).sum()
    g["otd_rolling_3m"] = on_time / g["deliveries"].rolling(3, min_periods=1).sum() * 100
    return g


def compute_risk_score(otd_recent: float, otd_drop_pp: float, ppm_recent: float, ppm_limit: float,
                       ppm_ratio: float, price_gap_pct: float, dependency_share: float,
                       criticality: str, lead_time_increase_pct: float = 0.0) -> dict:
    """Supplier risk score 0-100 (higher = riskier) with its components (0-1 each).

    delivery   : 100% - OTD recent, saturating at 25 pp
    quality    : recent PPM vs material limit, saturating at 2x the limit
    price      : gap vs peer benchmark, saturating at +10%
    dependency : share of the material volume x criticality of the material
    trend      : worst of OTD drop, PPM increase, lead-time increase
    """
    def clip(x):
        return float(np.clip(0.0 if x is None or not np.isfinite(x) else x, 0.0, 1.0))

    comps = {
        "delivery": clip((100.0 - otd_recent) / 25.0) if np.isfinite(otd_recent) else 0.0,
        "quality": clip(ppm_recent / ppm_limit / 2.0) if ppm_limit and np.isfinite(ppm_recent) else 0.0,
        "price": clip(price_gap_pct / 10.0),
        "dependency": clip(dependency_share / 0.5) * CRITICALITY_FACTOR.get(criticality, 0.5),
        "trend": max(clip(otd_drop_pp / 15.0), clip((ppm_ratio - 1.0) / 2.0),
                     clip(lead_time_increase_pct / 30.0)),
    }
    score = 100.0 * sum(config.RISK_WEIGHTS[k] * v for k, v in comps.items())
    return {"score": round(score, 1), "level": risk_level(score), "components": comps}


def risk_level(score: float) -> str:
    if score >= config.RISK_LEVEL_THRESHOLDS["High"]:
        return "High"
    if score >= config.RISK_LEVEL_THRESHOLDS["Medium"]:
        return "Medium"
    return "Low"


def supplier_scorecard(po: pd.DataFrame, quality: pd.DataFrame, suppliers: pd.DataFrame,
                       products: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """One row per supplier with delivery, quality, price and risk KPIs (12 months)."""
    year_po = po[(po["promised_date"] > as_of - pd.Timedelta(days=365)) & (po["promised_date"] <= as_of)]
    spend_po = window(po, as_of, config.SPEND_WINDOW_DAYS)
    pstats = price_stats(po, suppliers, as_of)
    mat_volume = spend_po.groupby("material_id")["ordered_quantity"].sum()
    crit = products.set_index("material_id")["criticality"]
    limits = products.set_index("material_id")["quality_ppm_limit"]

    rows = []
    for _, s in suppliers.iterrows():
        sid, mid = s["supplier_id"], s["material"]
        spo = year_po[year_po["supplier_id"] == sid]
        base_po, recent_po = split_periods(spo, "promised_date", as_of)
        sq = quality[quality["supplier_id"] == sid]
        base_q, recent_q = split_periods(sq, "date", as_of)
        d_all = delivery_metrics(spo, as_of=as_of)
        d_base, d_rec = delivery_metrics(base_po, as_of=as_of), delivery_metrics(recent_po, as_of=as_of)
        sp = spend_po[spend_po["supplier_id"] == sid]
        ps = pstats[(pstats["supplier_id"] == sid) & (pstats["material_id"] == mid)] if len(pstats) else pstats
        ps = ps.iloc[0] if len(ps) else None
        vol = float(sp.loc[sp["material_id"] == mid, "ordered_quantity"].sum())
        share = vol / float(mat_volume.get(mid, np.nan)) if mat_volume.get(mid, 0) else 0.0

        ppm_rec, ppm_base = ppm_of(recent_q), ppm_of(base_q)
        ppm_ratio = ppm_rec / ppm_base if ppm_base and np.isfinite(ppm_base) and ppm_base > 0 else 1.0
        otd_drop = d_base["otd"] - d_rec["otd"] if np.isfinite(d_base["otd"]) and np.isfinite(d_rec["otd"]) else 0.0
        lt_inc = ((d_rec["avg_lead_time"] - d_base["avg_lead_time"]) / d_base["avg_lead_time"] * 100
                  if np.isfinite(d_base["avg_lead_time"]) and np.isfinite(d_rec["avg_lead_time"]) else 0.0)
        gap_pct = float(ps["gap_vs_benchmark_pct"]) if ps is not None and pd.notna(ps["gap_vs_benchmark_pct"]) else 0.0
        if s["pricing_model"] == "Spot":
            gap_pct = 0.0  # spot premium is expected, not a drift
        if s["pricing_model"] == "Fixed annual" and ps is not None and pd.notna(ps["current_price"]):
            gap_pct = (float(ps["current_price"]) - float(s["contracted_price"])) / float(s["contracted_price"]) * 100
        risk = compute_risk_score(d_rec["otd"] if d_rec["deliveries"] else d_all["otd"], otd_drop,
                                  ppm_rec if np.isfinite(ppm_rec) else ppm_of(sq), float(limits.get(mid, np.nan)),
                                  ppm_ratio, gap_pct, share, str(crit.get(mid, "Medium")), lt_inc)
        rows.append({
            "supplier_id": sid,
            "supplier_name": s["supplier_name"],
            "country": s["country"],
            "material_id": mid,
            "supplier_status": s["supplier_status"],
            "pricing_model": s["pricing_model"],
            "spend": float((sp["ordered_quantity"] * sp["unit_price"]).sum()),
            "volume": vol,
            "volume_share_pct": share * 100,
            "deliveries": d_all["deliveries"],
            "otd": d_all["otd"],
            "otd_baseline": d_base["otd"],
            "otd_recent": d_rec["otd"],
            "otif": d_all["otif"],
            "otif_recent": d_rec["otif"],
            "avg_delay": d_all["avg_delay"],
            "avg_delay_baseline": d_base["avg_delay"],
            "avg_delay_recent": d_rec["avg_delay"],
            "delay_when_late_baseline": d_base["avg_delay_when_late"],
            "delay_when_late_recent": d_rec["avg_delay_when_late"],
            "late_deliveries_recent": d_rec["late_deliveries"],
            "deliveries_recent": d_rec["deliveries"],
            "avg_lead_time_baseline": d_base["avg_lead_time"],
            "avg_lead_time_recent": d_rec["avg_lead_time"],
            "lead_time_increase_pct": lt_inc,
            "ppm": ppm_of(sq),
            "ppm_baseline": ppm_base,
            "ppm_recent": ppm_rec,
            "ppm_limit": float(limits.get(mid, np.nan)),
            "avg_price": float(ps["avg_price_12m"]) if ps is not None else float("nan"),
            "current_price": float(ps["current_price"]) if ps is not None else float("nan"),
            "baseline_price": float(ps["baseline_price"]) if ps is not None else float("nan"),
            "price_evolution_pct": float(ps["price_change_pct"]) if ps is not None else float("nan"),
            "price_gap_pct": gap_pct,
            "otd_drop_pp": otd_drop,
            "risk_score": risk["score"],
            "risk_level": risk["level"],
            **{f"risk_{k}": v for k, v in risk["components"].items()},
        })
    return pd.DataFrame(rows).sort_values("risk_score", ascending=False).reset_index(drop=True)


def detect_deteriorating_suppliers(scorecard: pd.DataFrame) -> pd.DataFrame:
    """Suppliers whose recent performance degraded significantly vs their baseline."""
    sc = scorecard
    ppm_ratio = sc["ppm_recent"] / sc["ppm_baseline"].replace(0, np.nan)
    signals = pd.DataFrame({
        "otd": sc["otd_drop_pp"] >= config.OTD_DROP_ALERT_PP,
        "ppm": (ppm_ratio >= config.PPM_RATIO_ALERT) & (sc["ppm_recent"] > sc["ppm_limit"]),
        "lead_time": sc["lead_time_increase_pct"] >= config.LEAD_TIME_INCREASE_ALERT_PCT,
    })
    out = sc.assign(ppm_ratio=ppm_ratio, n_signals=signals.sum(axis=1),
                    signal_otd=signals["otd"], signal_ppm=signals["ppm"], signal_lead_time=signals["lead_time"])
    # A deterioration requires the delivery signal plus at least one corroborating signal
    return out[out["signal_otd"] & (out["n_signals"] >= 2)].reset_index(drop=True)
