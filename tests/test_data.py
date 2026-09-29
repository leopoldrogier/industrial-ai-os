"""Dataset generation and data access tests."""
import pandas as pd
import pytest

from src.config import DATA_DIR
from src.data_generation.generator import generate
from src.data_loader import export_to_sqlite, load_sqlite


@pytest.fixture(scope="module")
def generated():
    return generate()


def test_generator_is_deterministic(generated):
    again = generate()
    for name, df in generated.items():
        pd.testing.assert_frame_equal(df, again[name], check_dtype=False)


def test_committed_csv_matches_generator(generated, data):
    for name, df in generated.items():
        assert len(getattr(data, name)) == len(df), name
    po_csv = data.purchase_orders
    assert po_csv["ordered_quantity"].sum() == pytest.approx(generated["purchase_orders"]["ordered_quantity"].sum())


def test_dataset_volume_and_period(data):
    assert len(data.purchase_orders) >= 300
    assert 10 <= len(data.suppliers) <= 15
    days = data.inventory["date"].nunique()
    assert days >= 365
    assert (data.inventory["date"].max() - data.inventory["date"].min()).days >= 364


def test_dataset_consistency(data):
    po = data.purchase_orders
    assert (po["promised_date"] > po["order_date"]).all()
    assert (po["quantity_received"] <= po["ordered_quantity"] + 1e-9).all()
    received = po[po["actual_delivery_date"].notna()]
    assert (received["actual_delivery_date"] <= data.as_of).all()
    assert (data.inventory["stock_quantity"] >= 0).all()
    assert set(po["supplier_id"]) <= set(data.suppliers["supplier_id"])
    assert set(po["material_id"]) <= set(data.products["material_id"])
    # one inspection per received PO, one shipment per PO
    assert len(data.quality) == len(received)
    assert set(data.logistics["po_id"]) == set(po["po_id"])


def test_inventory_balance_matches_receipts_and_consumption(data):
    """Stock(t) = Stock(t-1) + receipts(t) - consumption(t) for every material."""
    po = data.purchase_orders
    for mid, inv in data.inventory.groupby("material_id"):
        inv = inv.sort_values("date").set_index("date")
        rec = po[(po["material_id"] == mid) & po["actual_delivery_date"].notna()]
        rec = rec.groupby("actual_delivery_date")["quantity_received"].sum()
        delta = inv["stock_quantity"].diff().iloc[1:]
        expected = rec.reindex(delta.index, fill_value=0) - inv["daily_consumption"].iloc[1:]
        assert (delta - expected).abs().max() < 0.05, mid


def test_sqlite_roundtrip(tmp_path, data):
    db = export_to_sqlite(data, tmp_path / "t.db")
    back = load_sqlite(db)
    assert len(back.purchase_orders) == len(data.purchase_orders)
    assert back.as_of == data.as_of
    assert DATA_DIR.exists()
