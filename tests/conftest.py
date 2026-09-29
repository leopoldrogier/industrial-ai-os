import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_loader import load_data  # noqa: E402
from src.engine.analysis import run_analysis  # noqa: E402


@pytest.fixture(scope="session")
def data():
    return load_data("csv")


@pytest.fixture(scope="session")
def result(data):
    return run_analysis(data)


def make_po(rows):
    """Build a small purchase-order frame: (order, promised, actual, ordered, received)."""
    df = pd.DataFrame(rows, columns=["order_date", "promised_date", "actual_delivery_date",
                                     "ordered_quantity", "quantity_received"])
    for c in ["order_date", "promised_date", "actual_delivery_date"]:
        df[c] = pd.to_datetime(df[c])
    df["po_id"] = [f"PO-{i}" for i in range(len(df))]
    return df
