"""Point-in-time supplier risk dataset, temporal model evaluation, and predictions.

The target is underperformance during the 90 days after a supplier snapshot.
Supplier profile/scenario labels are deliberately excluded from model features.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATE_COLUMNS = {
    "purchase_orders": ["order_date", "promised_delivery_date"],
    "delivery_logs": ["delivery_date"],
    "quality_inspections": ["inspection_date"],
    "invoices": ["invoice_date"],
}


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _dates(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    out = {name: frame.copy() for name, frame in tables.items()}
    for table, columns in DATE_COLUMNS.items():
        for col in columns:
            if col in out[table]:
                out[table][col] = pd.to_datetime(out[table][col], errors="coerce").dt.normalize()
    return out


def _rate(num: float, den: float) -> float:
    return float(num / den) if den and den > 0 else np.nan


def _snapshot_features(
    tables: dict[str, pd.DataFrame], vendor_id: str, snapshot: pd.Timestamp, history_days: int
) -> dict[str, Any]:
    """Compute only features observable at snapshot time from a trailing window."""
    start = snapshot - pd.Timedelta(days=history_days)
    pos_all = tables["purchase_orders"]
    pos = pos_all.loc[
        (pos_all["vendor_id"] == vendor_id)
        & (pos_all["order_date"] <= snapshot)
        & (pos_all["order_date"] > start)
        & (pos_all["po_status"].astype(str).str.lower() != "cancelled")
    ].copy()
    po_ids = set(pos["po_id"])
    deliveries = tables["delivery_logs"]
    deliveries = deliveries.loc[
        deliveries["po_id"].isin(po_ids) & (deliveries["delivery_date"] <= snapshot)
    ]
    inspections = tables["quality_inspections"]
    inspections = inspections.loc[
        inspections["po_id"].isin(po_ids) & (inspections["inspection_date"] <= snapshot)
    ]
    invoices = tables["invoices"]
    invoices = invoices.loc[invoices["po_id"].isin(po_ids) & (invoices["invoice_date"] <= snapshot)]

    delivered_by_po = deliveries.groupby("po_id").agg(
        delivered_qty=("delivered_qty", "sum"), final_delivery_date=("delivery_date", "max")
    ) if len(deliveries) else pd.DataFrame(columns=["delivered_qty", "final_delivery_date"])
    delivery_view = pos.set_index("po_id").join(delivered_by_po)
    delivery_view["delivered_qty"] = delivery_view["delivered_qty"].fillna(0)
    delivery_view["is_fully_delivered"] = delivery_view["delivered_qty"] >= pd.to_numeric(
        delivery_view["ordered_qty"], errors="coerce"
    ).fillna(0)
    delivery_view["is_evaluable"] = (
        (delivery_view["promised_delivery_date"] < snapshot) | delivery_view["is_fully_delivered"]
    )
    delivery_view["is_on_time"] = (
        delivery_view["is_fully_delivered"]
        & delivery_view["final_delivery_date"].notna()
        & (delivery_view["final_delivery_date"] <= delivery_view["promised_delivery_date"])
    )
    evaluable = delivery_view.loc[delivery_view["is_evaluable"]]

    inspected = pd.to_numeric(inspections.get("inspected_qty", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
    defective = pd.to_numeric(inspections.get("defective_qty", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
    if len(invoices):
        invoice_view = invoices.merge(pos[["po_id", "unit_price"]], on="po_id", how="left", validate="many_to_one")
        inv_qty = pd.to_numeric(invoice_view["invoiced_qty"], errors="coerce").fillna(0)
        inv_total = pd.to_numeric(invoice_view["invoice_total"], errors="coerce").fillna(0)
        base_price = pd.to_numeric(invoice_view["unit_price"], errors="coerce")
        row_invoice_unit_price = inv_total / inv_qty.replace(0, np.nan)
        row_variance = (row_invoice_unit_price - base_price) / base_price.replace(0, np.nan) * 100
        valid = row_variance.notna() & inv_qty.gt(0)
        cost_var = float(np.average(row_variance[valid], weights=inv_qty[valid])) if valid.any() else np.nan
    else:
        cost_var = np.nan
    lead = (delivery_view.loc[
        delivery_view["is_fully_delivered"] & delivery_view["final_delivery_date"].notna()
    ]["final_delivery_date"] - delivery_view.loc[
        delivery_view["is_fully_delivered"] & delivery_view["final_delivery_date"].notna()
    ]["order_date"]).dt.days
    lead_mean = lead.mean() if len(lead) else np.nan
    lead_cv = (lead.std(ddof=1) / abs(lead_mean)) if len(lead) > 1 and pd.notna(lead_mean) and lead_mean else np.nan

    overdue = (
        (delivery_view["promised_delivery_date"] < snapshot)
        & ~delivery_view["is_fully_delivered"]
    ).sum()
    return {
        "snapshot_date": snapshot,
        "vendor_id": vendor_id,
        "history_po_count": int(len(pos)),
        "history_evaluable_delivery_count": int(len(evaluable)),
        "history_on_time_delivery_rate": _rate(float(evaluable["is_on_time"].sum()), float(len(evaluable))),
        "history_inspected_units": float(inspected),
        "history_defective_units": float(defective),
        "history_defect_rate": _rate(float(defective), float(inspected)),
        "history_invoice_count": int(len(invoices)),
        "history_cost_variance_pct": float(cost_var) if pd.notna(cost_var) else np.nan,
        "history_mean_lead_days": float(lead_mean) if pd.notna(lead_mean) else np.nan,
        "history_lead_time_cv": float(lead_cv) if pd.notna(lead_cv) else np.nan,
        "history_overdue_open_count": int(overdue),
    }


def _future_label(
    tables: dict[str, pd.DataFrame], vendor_id: str, snapshot: pd.Timestamp,
    horizon_days: int, cutoff: pd.Timestamp, config: dict[str, Any]
) -> tuple[float, int | None]:
    """Label future window only when enough matured delivery and inspection evidence exists."""
    end = snapshot + pd.Timedelta(days=horizon_days)
    if end > cutoff:
        return np.nan, None
    pos = tables["purchase_orders"]
    future = pos.loc[
        (pos["vendor_id"] == vendor_id)
        & (pos["order_date"] > snapshot)
        & (pos["order_date"] <= end)
        & (pos["promised_delivery_date"] <= cutoff)
        & (pos["po_status"].astype(str).str.lower() != "cancelled")
    ].copy()
    if len(future) == 0:
        return np.nan, None
    ids = set(future["po_id"])
    deliveries = tables["delivery_logs"]
    deliveries = deliveries.loc[deliveries["po_id"].isin(ids) & (deliveries["delivery_date"] <= cutoff)]
    delivery_agg = deliveries.groupby("po_id").agg(
        delivered_qty=("delivered_qty", "sum"), final_delivery_date=("delivery_date", "max")
    ) if len(deliveries) else pd.DataFrame(columns=["delivered_qty", "final_delivery_date"])
    future = future.set_index("po_id").join(delivery_agg)
    future["delivered_qty"] = future["delivered_qty"].fillna(0)
    future["is_fully_delivered"] = future["delivered_qty"] >= pd.to_numeric(future["ordered_qty"], errors="coerce").fillna(0)
    # Promised date is past the global cutoff only for excluded POs. All included POs are evaluable.
    future["is_on_time"] = (
        future["is_fully_delivered"] & future["final_delivery_date"].notna()
        & (future["final_delivery_date"] <= future["promised_delivery_date"])
    )
    delivery_rate = _rate(float(future["is_on_time"].sum()), float(len(future)))

    inspections = tables["quality_inspections"]
    inspections = inspections.loc[inspections["po_id"].isin(ids) & (inspections["inspection_date"] <= cutoff)]
    inspected = pd.to_numeric(inspections.get("inspected_qty", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
    defective = pd.to_numeric(inspections.get("defective_qty", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
    min_pos = int(config["minimum_future_evaluable_pos"])
    min_units = int(config["minimum_future_inspected_units"])
    if len(future) < min_pos or inspected < min_units:
        return np.nan, None
    thresholds = config["underperformance_thresholds"]
    poor_delivery = pd.notna(delivery_rate) and delivery_rate < thresholds["on_time_delivery_rate_below"]
    defect_rate = _rate(float(defective), float(inspected))
    poor_quality = pd.notna(defect_rate) and defect_rate > thresholds["defect_rate_above"]
    return float(bool(poor_delivery or poor_quality)), int(len(future))


def build_training_panel(
    tables: dict[str, pd.DataFrame], config: dict[str, Any], cutoff_date: str | pd.Timestamp
) -> pd.DataFrame:
    tables = _dates(tables)
    cutoff = pd.Timestamp(cutoff_date).normalize()
    vendor_ids = sorted(tables["vendors"]["vendor_id"].astype(str).unique())
    first_order = tables["purchase_orders"]["order_date"].min()
    first_snapshot = (first_order + pd.Timedelta(days=int(config["history_window_days"]))).to_period("M").to_timestamp("M")
    snapshots: list[pd.Timestamp] = []
    current = first_snapshot
    while current <= cutoff:
        snapshots.append(current)
        current = current + pd.offsets.MonthEnd(1)
    if cutoff not in snapshots:
        snapshots.append(cutoff)
    rows = []
    for snapshot in sorted(set(snapshots)):
        for vendor_id in vendor_ids:
            features = _snapshot_features(tables, vendor_id, snapshot, int(config["history_window_days"]))
            label, future_po_count = _future_label(
                tables, vendor_id, snapshot, int(config["prediction_horizon_days"]), cutoff, config
            )
            features["future_underperformance"] = label
            features["future_evaluable_po_count"] = future_po_count
            rows.append(features)
    return pd.DataFrame(rows).sort_values(["snapshot_date", "vendor_id"]).reset_index(drop=True)


NUMERIC_FEATURES = [
    "history_po_count", "history_evaluable_delivery_count", "history_on_time_delivery_rate",
    "history_inspected_units", "history_defective_units", "history_defect_rate",
    "history_invoice_count", "history_cost_variance_pct", "history_mean_lead_days",
    "history_lead_time_cv", "history_overdue_open_count",
]


def _model_pipeline(config: dict[str, Any]) -> Pipeline:
    preprocess = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                              ("scaler", StandardScaler())]), NUMERIC_FEATURES),
    ], remainder="drop")
    classifier = RandomForestClassifier(
        n_estimators=int(config["n_estimators"]),
        max_depth=int(config["max_depth"]),
        min_samples_leaf=int(config["min_samples_leaf"]),
        class_weight="balanced_subsample",
        random_state=int(config["random_state"]),
        n_jobs=-1,
    )
    return Pipeline([("preprocess", preprocess), ("classifier", classifier)])


def train_evaluate_predict(panel: pd.DataFrame, config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    labelled = panel.loc[panel["future_underperformance"].notna()].copy()
    if len(labelled) < int(config["minimum_training_rows"]):
        raise ValueError(
            f"Only {len(labelled)} labelled supplier snapshots; need at least "
            f"{config['minimum_training_rows']}. Check the history window, label thresholds, and generated data."
        )
    labelled["snapshot_date"] = pd.to_datetime(labelled["snapshot_date"])
    unique_dates = sorted(labelled["snapshot_date"].unique())
    if len(unique_dates) < 4:
        raise ValueError("At least four distinct labelled snapshot dates are required for temporal evaluation.")
    split_idx = max(1, int(len(unique_dates) * (1 - float(config["temporal_test_fraction"]))))
    split_idx = min(split_idx, len(unique_dates) - 1)
    test_start = pd.Timestamp(unique_dates[split_idx])
    train = labelled.loc[labelled["snapshot_date"] < test_start]
    test = labelled.loc[labelled["snapshot_date"] >= test_start]
    if train["future_underperformance"].nunique() < 2 or test["future_underperformance"].nunique() < 2:
        raise ValueError(
            "Temporal train/test split does not contain both classes. Review synthetic data volume or label thresholds; "
            "do not switch to a random split because it can leak time patterns."
        )
    model = _model_pipeline(config)
    model.fit(train[NUMERIC_FEATURES], train["future_underperformance"].astype(int))
    probabilities = model.predict_proba(test[NUMERIC_FEATURES])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    y_true = test["future_underperformance"].astype(int)
    metrics: dict[str, Any] = {
        "target": "future_underperformance",
        "prediction_horizon_days": int(config["prediction_horizon_days"]),
        "history_window_days": int(config["history_window_days"]),
        "temporal_test_start": test_start.date().isoformat(),
        "training_rows": int(len(train)),
        "test_rows": int(len(test)),
        "training_positive_rate": float(train["future_underperformance"].mean()),
        "test_positive_rate": float(y_true.mean()),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "pr_auc_average_precision": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "classification_report": classification_report(y_true, predictions, output_dict=True, zero_division=0),
        "features": NUMERIC_FEATURES,
        "model": "RandomForestClassifier",
        "note": "Metrics are from a chronological holdout. Review them before using predictions for operational decisions.",
    }
    with (output_dir / "model_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Retrain on all mature labelled snapshots after reporting the temporal holdout metrics.
    final_model = _model_pipeline(config)
    final_model.fit(labelled[NUMERIC_FEATURES], labelled["future_underperformance"].astype(int))
    joblib.dump(final_model, output_dir / "supplier_risk_model.joblib")

    latest_date = pd.to_datetime(panel["snapshot_date"]).max()
    latest = panel.loc[pd.to_datetime(panel["snapshot_date"]) == latest_date].copy()
    latest_probs = final_model.predict_proba(latest[NUMERIC_FEATURES])[:, 1]
    latest["predicted_underperformance_probability"] = latest_probs.round(4)
    latest["predicted_risk_band"] = pd.cut(
        latest["predicted_underperformance_probability"],
        bins=[-0.001, 0.25, 0.50, 0.75, 1.001],
        labels=["Low", "Moderate", "High", "Very High"],
    ).astype(str)
    latest["prediction_horizon_days"] = int(config["prediction_horizon_days"])
    latest["model_snapshot_date"] = latest_date.date().isoformat()
    latest.drop(columns=["future_underperformance", "future_evaluable_po_count"], errors="ignore").to_csv(
        output_dir / "supplier_risk_predictions.csv", index=False
    )

    importance = final_model.named_steps["classifier"].feature_importances_
    # Numeric preprocessing preserves the original columns and adds missingness indicators.
    transformed_names = final_model.named_steps["preprocess"].get_feature_names_out()
    pd.DataFrame({"feature": transformed_names, "importance": importance}).sort_values(
        "importance", ascending=False
    ).to_csv(output_dir / "feature_importance.csv", index=False)

    summary = {
        "panel_rows": int(len(panel)),
        "labelled_rows": int(len(labelled)),
        "latest_snapshot_date": latest_date.date().isoformat(),
        "predictions_written": int(len(latest)),
        "positive_label_rate": float(labelled["future_underperformance"].mean()),
        "files": ["model_metrics.json", "supplier_risk_model.joblib", "supplier_risk_predictions.csv", "feature_importance.csv"],
    }
    with (output_dir / "ml_pipeline_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and run supplier future-risk prediction.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"), help="Directory containing raw synthetic tables")
    parser.add_argument("--config", type=Path, default=Path("config/ml_config.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/ml"))
    parser.add_argument("--as-of-date", default=None, help="Optional cutoff date; otherwise read data/manifest.json")
    args = parser.parse_args()

    config = load_config(args.config)
    manifest_path = args.data_dir / "manifest.json"
    if not manifest_path.exists():
        manifest_path = args.data_dir.parent / "manifest.json"
    cutoff = args.as_of_date
    if cutoff is None:
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        cutoff = manifest.get("as_of_date") or manifest.get("metadata", {}).get("as_of_date")
        if cutoff is None:
            raise ValueError(f"Could not find as_of_date in {manifest_path}; pass --as-of-date explicitly.")
    tables = {}
    for name in ["vendors", "purchase_orders", "delivery_logs", "quality_inspections", "invoices"]:
        parquet_path = args.data_dir / f"{name}.parquet"
        csv_path = args.data_dir / f"{name}.csv"
        if parquet_path.exists():
            tables[name] = pd.read_parquet(parquet_path)
        elif csv_path.exists():
            tables[name] = pd.read_csv(csv_path)
        else:
            raise FileNotFoundError(f"Missing raw table: expected {parquet_path} or {csv_path}")
    panel = build_training_panel(tables, config, cutoff)
    panel.to_csv(args.output_dir / "supplier_risk_training_panel.csv", index=False) if args.output_dir.exists() else None
    summary = train_evaluate_predict(panel, config, args.output_dir)
    # Save panel after output folder creation as well.
    panel.to_csv(args.output_dir / "supplier_risk_training_panel.csv", index=False)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
