"""Build PO-level and supplier-period features without fan-out double counting."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _date(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_datetime(df[column], errors="coerce").dt.normalize()


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace({0: np.nan})
    return numerator / denominator


def build_po_features(tables: dict[str, pd.DataFrame], as_of_date: str | pd.Timestamp) -> pd.DataFrame:
    """Return one row per PO; aggregate event tables independently before joins."""
    as_of = pd.Timestamp(as_of_date).normalize()
    pos = tables["purchase_orders"].copy()
    pos["order_date"] = _date(pos, "order_date")
    pos["promised_delivery_date"] = _date(pos, "promised_delivery_date")
    pos["ordered_qty"] = pd.to_numeric(pos["ordered_qty"], errors="coerce").fillna(0)
    pos["unit_price"] = pd.to_numeric(pos["unit_price"], errors="coerce").fillna(0)
    pos["po_status"] = pos["po_status"].astype(str).str.strip().str.title()

    deliveries = tables["delivery_logs"].copy()
    if len(deliveries):
        deliveries["delivery_date"] = _date(deliveries, "delivery_date")
        deliveries["delivered_qty"] = pd.to_numeric(deliveries["delivered_qty"], errors="coerce").fillna(0)
        deliveries["actual_lead_days"] = pd.to_numeric(deliveries["actual_lead_days"], errors="coerce")
        delivery_agg = deliveries.groupby("po_id", as_index=False).agg(
            delivery_event_count=("delivery_id", "count"),
            delivered_qty=("delivered_qty", "sum"),
            first_delivery_date=("delivery_date", "min"),
            final_delivery_date=("delivery_date", "max"),
            mean_event_lead_days=("actual_lead_days", "mean"),
        )
    else:
        delivery_agg = pd.DataFrame(columns=[
            "po_id", "delivery_event_count", "delivered_qty", "first_delivery_date",
            "final_delivery_date", "mean_event_lead_days",
        ])

    inspections = tables["quality_inspections"].copy()
    if len(inspections):
        for col in ["inspected_qty", "defective_qty", "accepted_qty"]:
            inspections[col] = pd.to_numeric(inspections[col], errors="coerce").fillna(0)
        inspection_agg = inspections.groupby("po_id", as_index=False).agg(
            inspected_qty=("inspected_qty", "sum"),
            defective_qty=("defective_qty", "sum"),
            accepted_qty=("accepted_qty", "sum"),
            inspection_count=("inspection_id", "count"),
        )
    else:
        inspection_agg = pd.DataFrame(columns=[
            "po_id", "inspected_qty", "defective_qty", "accepted_qty", "inspection_count",
        ])

    invoices = tables["invoices"].copy()
    if len(invoices):
        for col in ["invoiced_qty", "invoice_unit_price", "invoice_total"]:
            invoices[col] = pd.to_numeric(invoices[col], errors="coerce").fillna(0)
        invoice_agg = invoices.groupby("po_id", as_index=False).agg(
            invoice_count=("invoice_id", "count"),
            invoiced_qty=("invoiced_qty", "sum"),
            invoice_total=("invoice_total", "sum"),
        )
        invoice_agg["effective_invoice_unit_price"] = _safe_divide(
            invoice_agg["invoice_total"], invoice_agg["invoiced_qty"]
        )
    else:
        invoice_agg = pd.DataFrame(columns=[
            "po_id", "invoice_count", "invoiced_qty", "invoice_total",
            "effective_invoice_unit_price",
        ])

    # Each child table has one row per PO after aggregation, preventing join fan-out.
    po = pos.merge(delivery_agg, on="po_id", how="left", validate="one_to_one")
    po = po.merge(inspection_agg, on="po_id", how="left", validate="one_to_one")
    po = po.merge(invoice_agg, on="po_id", how="left", validate="one_to_one")

    count_cols = ["delivery_event_count", "inspection_count", "invoice_count"]
    quantity_cols = ["delivered_qty", "inspected_qty", "defective_qty", "accepted_qty", "invoiced_qty", "invoice_total"]
    for col in count_cols + quantity_cols:
        po[col] = pd.to_numeric(po[col], errors="coerce").fillna(0)

    po["delivery_completion_ratio"] = _safe_divide(po["delivered_qty"], po["ordered_qty"]).clip(0, 1.5)
    po["is_cancelled"] = po["po_status"].eq("Cancelled")
    po["is_open"] = po["po_status"].eq("Open")
    po["is_fully_delivered"] = po["delivered_qty"] >= po["ordered_qty"]
    po["is_overdue_open"] = (
        po["is_open"]
        & (po["promised_delivery_date"] < as_of)
        & ~po["is_fully_delivered"]
        & ~po["is_cancelled"]
    )
    po["is_delivery_evaluable"] = (
        ~po["is_cancelled"]
        & (
            po["is_fully_delivered"]
            | (po["promised_delivery_date"] < as_of)
        )
    )
    # A PO is on time only when the final delivery completed by the promised date.
    po["is_on_time"] = (
        po["is_fully_delivered"]
        & po["final_delivery_date"].notna()
        & (po["final_delivery_date"] <= po["promised_delivery_date"])
    )
    po["delivery_timeliness"] = np.where(
        po["is_delivery_evaluable"], po["is_on_time"].astype(float), np.nan
    )
    po["actual_lead_days"] = (
        po["final_delivery_date"] - po["order_date"]
    ).dt.days
    po["lead_time_delta_days"] = (
        po["final_delivery_date"] - po["promised_delivery_date"]
    ).dt.days
    po["defect_rate"] = _safe_divide(po["defective_qty"], po["inspected_qty"])
    po["quality_coverage"] = _safe_divide(po["inspected_qty"], po["delivered_qty"]).clip(0, 1.5)
    po["cost_variance_pct"] = (
        _safe_divide(po["effective_invoice_unit_price"] - po["unit_price"], po["unit_price"]) * 100
    )
    po["has_invoice"] = po["invoice_count"] > 0
    po["invoice_quantity_variance"] = po["invoiced_qty"] - po["ordered_qty"]
    po["reporting_as_of_date"] = as_of

    # Future promised POs are not delivery failures yet. Preserve them for pipeline monitoring.
    return po.sort_values(["order_date", "po_id"]).reset_index(drop=True)


def build_supplier_features(
    po_features: pd.DataFrame,
    vendors: pd.DataFrame,
    as_of_date: str | pd.Timestamp,
    recent_days: int = 90,
) -> pd.DataFrame:
    """Aggregate PO features to one row per vendor, with recent-window indicators."""
    as_of = pd.Timestamp(as_of_date).normalize()
    po = po_features.copy()
    po["order_date"] = pd.to_datetime(po["order_date"], errors="coerce")
    po["reporting_as_of_date"] = as_of
    active_pos = po.loc[~po["is_cancelled"]].copy()

    grouped = active_pos.groupby("vendor_id", as_index=False).agg(
        po_count=("po_id", "nunique"),
        ordered_qty=("ordered_qty", "sum"),
        delivered_qty=("delivered_qty", "sum"),
        evaluable_delivery_count=("is_delivery_evaluable", "sum"),
        on_time_delivery_count=("is_on_time", "sum"),
        overdue_open_count=("is_overdue_open", "sum"),
        inspected_qty=("inspected_qty", "sum"),
        defective_qty=("defective_qty", "sum"),
        accepted_qty=("accepted_qty", "sum"),
        invoice_total=("invoice_total", "sum"),
        invoiced_qty=("invoiced_qty", "sum"),
        invoiced_po_count=("has_invoice", "sum"),
        mean_cost_variance_pct=("cost_variance_pct", "mean"),
        median_cost_variance_pct=("cost_variance_pct", "median"),
        mean_actual_lead_days=("actual_lead_days", "mean"),
        std_actual_lead_days=("actual_lead_days", "std"),
        mean_lead_time_delta_days=("lead_time_delta_days", "mean"),
        quality_po_count=("inspection_count", lambda s: int((s > 0).sum())),
    )

    grouped["on_time_delivery_rate"] = _safe_divide(
        grouped["on_time_delivery_count"], grouped["evaluable_delivery_count"]
    )
    grouped["defect_rate"] = _safe_divide(grouped["defective_qty"], grouped["inspected_qty"])
    grouped["quality_acceptance_rate"] = _safe_divide(
        grouped["accepted_qty"], grouped["inspected_qty"]
    )
    grouped["invoice_price_variance_pct"] = grouped["mean_cost_variance_pct"]
    grouped["lead_time_cv"] = _safe_divide(
        grouped["std_actual_lead_days"], grouped["mean_actual_lead_days"].abs()
    )
    grouped["delivery_sample_sufficient"] = grouped["evaluable_delivery_count"] >= 5
    grouped["quality_sample_sufficient"] = grouped["inspected_qty"] >= 100
    grouped["cost_sample_sufficient"] = grouped["invoiced_po_count"] >= 3

    recent_start = as_of - pd.Timedelta(days=recent_days)
    recent = active_pos.loc[
        (active_pos["order_date"] >= recent_start)
        & (active_pos["order_date"] <= as_of)
    ]
    recent_group = recent.groupby("vendor_id", as_index=False).agg(
        recent_po_count=("po_id", "nunique"),
        recent_evaluable_delivery_count=("is_delivery_evaluable", "sum"),
        recent_on_time_delivery_count=("is_on_time", "sum"),
        recent_inspected_qty=("inspected_qty", "sum"),
        recent_defective_qty=("defective_qty", "sum"),
        recent_mean_cost_variance_pct=("cost_variance_pct", "mean"),
        recent_mean_lead_days=("actual_lead_days", "mean"),
    )
    recent_group["recent_on_time_delivery_rate"] = _safe_divide(
        recent_group["recent_on_time_delivery_count"],
        recent_group["recent_evaluable_delivery_count"],
    )
    recent_group["recent_defect_rate"] = _safe_divide(
        recent_group["recent_defective_qty"], recent_group["recent_inspected_qty"]
    )
    recent_group = recent_group.drop(columns=[
        "recent_on_time_delivery_count", "recent_defective_qty",
    ])

    result = vendors.copy().merge(grouped, on="vendor_id", how="left", validate="one_to_one")
    result = result.merge(recent_group, on="vendor_id", how="left", validate="one_to_one")
    count_and_sum_cols = [
        "po_count", "ordered_qty", "delivered_qty", "evaluable_delivery_count",
        "on_time_delivery_count", "overdue_open_count", "inspected_qty",
        "defective_qty", "accepted_qty", "invoice_total", "invoiced_qty",
        "invoiced_po_count", "quality_po_count", "recent_po_count",
        "recent_evaluable_delivery_count", "recent_inspected_qty",
    ]
    for col in count_and_sum_cols:
        if col in result.columns:
            result[col] = result[col].fillna(0)
    for col in ["delivery_sample_sufficient", "quality_sample_sufficient", "cost_sample_sufficient"]:
        result[col] = result[col].fillna(False).astype(bool)

    result["reporting_as_of_date"] = as_of
    result["recent_window_days"] = recent_days
    result["data_sufficiency_status"] = np.select(
        [
            result["po_count"].eq(0),
            ~result["delivery_sample_sufficient"] & ~result["quality_sample_sufficient"],
            ~result["delivery_sample_sufficient"] | ~result["quality_sample_sufficient"] | ~result["cost_sample_sufficient"],
        ],
        ["No history", "Limited history", "Partial coverage"],
        default="Sufficient history",
    )
    return result.sort_values("vendor_id").reset_index(drop=True)


def build_monthly_supplier_features(po_features: pd.DataFrame) -> pd.DataFrame:
    """Build vendor-month metrics for trend charts and future time-aware ML features."""
    po = po_features.loc[~po_features["is_cancelled"]].copy()
    po["order_date"] = pd.to_datetime(po["order_date"], errors="coerce")
    po["month"] = po["order_date"].dt.to_period("M").dt.to_timestamp()
    monthly = po.groupby(["vendor_id", "month"], as_index=False).agg(
        po_count=("po_id", "nunique"),
        evaluable_delivery_count=("is_delivery_evaluable", "sum"),
        on_time_delivery_count=("is_on_time", "sum"),
        inspected_qty=("inspected_qty", "sum"),
        defective_qty=("defective_qty", "sum"),
        mean_cost_variance_pct=("cost_variance_pct", "mean"),
        mean_actual_lead_days=("actual_lead_days", "mean"),
        overdue_open_count=("is_overdue_open", "sum"),
    )
    monthly["on_time_delivery_rate"] = _safe_divide(
        monthly["on_time_delivery_count"], monthly["evaluable_delivery_count"]
    )
    monthly["defect_rate"] = _safe_divide(monthly["defective_qty"], monthly["inspected_qty"])
    return monthly.sort_values(["vendor_id", "month"]).reset_index(drop=True)


def save_dataframe(df: pd.DataFrame, output_dir: Path, stem: str, formats: list[str]) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for fmt in formats:
        path = output_dir / f"{stem}.{fmt}"
        if fmt == "csv":
            df.to_csv(path, index=False, date_format="%Y-%m-%d")
        elif fmt == "parquet":
            df.to_parquet(path, index=False)
        else:
            raise ValueError(f"Unsupported output format: {fmt}")
        written.append(path.as_posix())
    return written
