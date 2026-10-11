import numpy as np
import pandas as pd
import pytest


def test_numpy_available():
    values = np.array([10, 20, 30])

    assert values.mean() == 20


def test_pandas_available():
    df = pd.DataFrame({
        "vendor_id": ["V001", "V002"],
        "delivery_score": [90, 75],
    })

    assert len(df) == 2
    assert df["delivery_score"].mean() == 82.5


def test_python_environment():
    assert True