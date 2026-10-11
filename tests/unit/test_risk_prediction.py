import numpy as np
import pandas as pd

from supplier_performance.ml.risk_prediction import NUMERIC_FEATURES, _model_pipeline, _snapshot_features


def test_model_pipeline_excludes_identifiers_and_profile_labels():
    model = _model_pipeline({
        "n_estimators": 10, "max_depth": 3, "min_samples_leaf": 1, "random_state": 7
    })
    frame = pd.DataFrame({name: [1.0, np.nan, 2.0] for name in NUMERIC_FEATURES})
    model.fit(frame, [0, 1, 0])
    assert model.predict_proba(frame).shape == (3, 2)
    assert "supplier_profile" not in NUMERIC_FEATURES
    assert "vendor_id" not in NUMERIC_FEATURES


def test_snapshot_features_only_count_events_observed_by_snapshot():
    snapshot = pd.Timestamp("2026-06-30")
    tables = {
        "vendors": pd.DataFrame([{"vendor_id": "V1", "supplier_profile": "quality_risk"}]),
        "purchase_orders": pd.DataFrame([
            {"po_id": "P1", "vendor_id": "V1", "order_date": "2026-05-01", "promised_delivery_date": "2026-05-20", "ordered_qty": 10, "unit_price": 2.0, "po_status": "Closed"}
        ]),
        "delivery_logs": pd.DataFrame([
            {"po_id": "P1", "delivery_date": "2026-05-18", "delivered_qty": 10}
        ]),
        "quality_inspections": pd.DataFrame([
            {"po_id": "P1", "inspection_date": "2026-05-18", "inspected_qty": 10, "defective_qty": 1}
        ]),
        "invoices": pd.DataFrame([
            {"po_id": "P1", "invoice_date": "2026-05-22", "invoiced_qty": 10, "invoice_total": 20.0}
        ]),
    }
    from supplier_performance.ml.risk_prediction import _dates
    dated = _dates(tables)
    features = _snapshot_features(dated, "V1", snapshot, 180)
    assert features["history_po_count"] == 1
    assert features["history_on_time_delivery_rate"] == 1.0
    assert features["history_defect_rate"] == 0.1
