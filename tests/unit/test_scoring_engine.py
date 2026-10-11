import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from supplier_performance.scoring.scoring_engine import (
    load_scoring_config,
    score_suppliers,
    validate_scoring_config,
)


def config():
    return load_scoring_config(Path("config/scoring_config.json"))


def feature_rows():
    return pd.DataFrame([
        {
            "vendor_id": "VEN0001", "vendor_name": "Reliable Ltd",
            "on_time_delivery_rate": 0.96, "defect_rate": 0.005,
            "invoice_price_variance_pct": 1.0, "lead_time_cv": 0.15,
            "evaluable_delivery_count": 30, "inspected_qty": 2000,
            "invoiced_po_count": 25, "po_count": 30, "overdue_open_count": 0,
            "recent_on_time_delivery_rate": 0.95, "recent_defect_rate": 0.006,
        },
        {
            "vendor_id": "VEN0002", "vendor_name": "Poor Ltd",
            "on_time_delivery_rate": 0.60, "defect_rate": 0.08,
            "invoice_price_variance_pct": 8.0, "lead_time_cv": 0.65,
            "evaluable_delivery_count": 25, "inspected_qty": 1000,
            "invoiced_po_count": 20, "po_count": 25, "overdue_open_count": 2,
            "recent_on_time_delivery_rate": 0.45, "recent_defect_rate": 0.12,
        },
        {
            "vendor_id": "VEN0003", "vendor_name": "New Ltd",
            "on_time_delivery_rate": 1.0, "defect_rate": 0.0,
            "invoice_price_variance_pct": 0.0, "lead_time_cv": 0.0,
            "evaluable_delivery_count": 1, "inspected_qty": 10,
            "invoiced_po_count": 1, "po_count": 1, "overdue_open_count": 0,
            "recent_on_time_delivery_rate": 1.0, "recent_defect_rate": 0.0,
        },
    ])


def test_weights_must_sum_to_one():
    cfg = config()
    cfg["weights"]["delivery"] = 0.5
    with pytest.raises(ValueError, match="sum to 1.0"):
        validate_scoring_config(cfg)


def test_scorecards_are_ordered_and_explain_alerts():
    scored = score_suppliers(feature_rows(), config()).set_index("vendor_id")
    assert scored.loc["VEN0001", "performance_score"] > scored.loc["VEN0002", "performance_score"]
    assert scored.loc["VEN0001", "risk_classification"] == "Strong"
    assert scored.loc["VEN0002", "risk_classification"] in {"At risk", "Critical"}
    assert "LOW_DELIVERY_RELIABILITY" in scored.loc["VEN0002", "risk_alerts"]
    assert "HIGH_DEFECT_RATE" in scored.loc["VEN0002", "risk_alerts"]
    assert "OVERDUE_OPEN_PURCHASE_ORDERS" in scored.loc["VEN0002", "risk_alerts"]


def test_low_sample_supplier_is_not_classified_as_strong():
    scored = score_suppliers(feature_rows(), config()).set_index("vendor_id")
    assert scored.loc["VEN0003", "risk_classification"] == "Insufficient data"
    assert not bool(scored.loc["VEN0003", "all_dimensions_sample_sufficient"])


def test_cost_savings_do_not_exceed_score_cap():
    rows = feature_rows().iloc[[0]].copy()
    rows["invoice_price_variance_pct"] = -20
    scored = score_suppliers(rows, config())
    assert scored.iloc[0]["cost_score"] == 100
