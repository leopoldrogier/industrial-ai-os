"""Data access layer.

V1 sources: CSV extracts (default) or a local SQLite database built from them.
V2: add connectors (SAP / SQL / Excel upload / API) that return the same
``DataBundle`` — nothing downstream needs to change.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import DATA_DIR, DATASETS, SQLITE_PATH

DATE_COLUMNS = {
    "purchase_orders": ["order_date", "promised_date", "actual_delivery_date"],
    "inventory": ["date"],
    "production": ["date"],
    "logistics": ["departure_date", "expected_arrival", "actual_arrival", "current_eta"],
    "quality": ["date"],
}

# Which operational system each dataset stands for (shown in the UI)
SOURCE_SYSTEMS = {
    "purchase_orders": "ERP — purchasing",
    "suppliers": "ERP — vendor master",
    "products": "ERP — material master",
    "inventory": "WMS — stock levels",
    "production": "MES — production reporting",
    "logistics": "TMS / carrier tracking",
    "quality": "QMS — goods receipt inspection",
}


@dataclass(frozen=True)
class DataBundle:
    products: pd.DataFrame
    suppliers: pd.DataFrame
    purchase_orders: pd.DataFrame
    inventory: pd.DataFrame
    production: pd.DataFrame
    logistics: pd.DataFrame
    quality: pd.DataFrame

    @property
    def as_of(self) -> pd.Timestamp:
        """Analysis date = last inventory snapshot (never the wall clock)."""
        return pd.Timestamp(self.inventory["date"].max())

    def supplier_name(self, supplier_id: str) -> str:
        m = self.suppliers.loc[self.suppliers.supplier_id == supplier_id, "supplier_name"]
        return str(m.iloc[0]) if len(m) else supplier_id

    def material_name(self, material_id: str) -> str:
        m = self.products.loc[self.products.material_id == material_id, "material_name"]
        return str(m.iloc[0]) if len(m) else material_id

    def material_unit(self, material_id: str) -> str:
        m = self.products.loc[self.products.material_id == material_id, "unit"]
        return str(m.iloc[0]) if len(m) else ""

    def row_counts(self) -> dict[str, int]:
        return {name: len(getattr(self, name)) for name in DATASETS}


def _parse(name: str, df: pd.DataFrame) -> pd.DataFrame:
    for col in DATE_COLUMNS.get(name, []):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def ensure_data(data_dir: Path = DATA_DIR) -> None:
    """Generate the demo dataset if CSV files are missing (first launch)."""
    if all((Path(data_dir) / f"{n}.csv").exists() for n in DATASETS):
        return
    from src.data_generation.generator import write_dataset

    write_dataset(Path(data_dir))


def load_csv(data_dir: Path = DATA_DIR) -> DataBundle:
    ensure_data(data_dir)
    frames = {n: _parse(n, pd.read_csv(Path(data_dir) / f"{n}.csv")) for n in DATASETS}
    return DataBundle(**frames)


def export_to_sqlite(bundle: DataBundle, db_path: Path = SQLITE_PATH) -> Path:
    with sqlite3.connect(db_path) as con:
        for name in DATASETS:
            getattr(bundle, name).to_sql(name, con, if_exists="replace", index=False)
    return db_path


def load_sqlite(db_path: Path = SQLITE_PATH) -> DataBundle:
    if not Path(db_path).exists():
        export_to_sqlite(load_csv(), db_path)
    with sqlite3.connect(db_path) as con:
        frames = {n: _parse(n, pd.read_sql(f"SELECT * FROM {n}", con)) for n in DATASETS}
    return DataBundle(**frames)


def load_data(source: str = "csv") -> DataBundle:
    if source == "sqlite":
        return load_sqlite()
    return load_csv()
