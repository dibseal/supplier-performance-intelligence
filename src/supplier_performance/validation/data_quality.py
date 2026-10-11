"""Validate synthetic procurement tables and emit a machine-readable quality report."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


REQUIRED_COLUMNS: dict[str, set[str]] = {
    "categories": {"category_id", "category_name", "typical_lead_days"},
    "vendors": {"vendor_id", "vendor_name", "status", "monthly_capacity_units"},
    "vendor_category_contracts": {
        "contract_id", "vendor_id", "category_id", "contract_unit_price",
        "valid_from", "valid_to", "is_active",
    },
    "purchase_orders": {
        "po_id", "vendor_id", "category_id", "contract_id", "order_date",
        "promised_delivery_date", "ordered_qty", "unit_price", "po_status",
    },
    "delivery_logs": {
        "delivery_id", "po_id", "delivery_date", "delivered_qty", "actual_lead_days",
    },
    "quality_inspections": {
        "inspection_id", "po_id", "delivery_id", "inspection_date",
        "inspected_qty", "defective_qty", "accepted_qty",
    },
    "invoices": {
        "invoice_id", "po_id", "invoice_date", "invoiced_qty",
        "invoice_unit_price", "invoice_total",
    },
}

FOREIGN_KEYS = [
    ("purchase_orders", "vendor_id", "vendors", "vendor_id"),
    ("purchase_orders", "category_id", "categories", "category_id"),
    ("purchase_orders", "contract_id", "vendor_category_contracts", "contract_id"),
    ("delivery_logs", "po_id", "purchase_orders", "po_id"),
    ("quality_inspections", "po_id", "purchase_orders", "po_id"),
    ("quality_inspections", "delivery_id", "delivery_logs", "delivery_id"),
    ("invoices", "po_id", "purchase_orders", "po_id"),
    ("vendor_category_contracts", "vendor_id", "vendors", "vendor_id"),
    ("vendor_category_contracts", "category_id", "categories", "category_id"),
]

UNIQUE_KEYS = {
    "categories": ["category_id"],
    "vendors": ["vendor_id"],
    "vendor_category_contracts": ["contract_id"],
    "purchase_orders": ["po_id"],
    "delivery_logs": ["delivery_id"],
    "quality_inspections": ["inspection_id"],
    "invoices": ["invoice_id"],
}


def _check(name: str, passed: bool, severity: str = "error", details: str = "") -> dict[str, Any]:
    return {
        "check": name,
        "passed": bool(passed),
        "severity": severity,
        "details": details,
    }


def validate_tables(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Validate required tables, schemas, keys, date logic and numeric constraints.

    Raises ValueError for missing tables or failed error-severity checks. Warning
    checks are included in the report but do not stop the pipeline.
    """
    missing_tables = sorted(set(REQUIRED_COLUMNS) - set(tables))
    if missing_tables:
        raise ValueError(f"Missing required tables: {missing_tables}")

    results: list[dict[str, Any]] = []

    for table_name, required in REQUIRED_COLUMNS.items():
        missing_columns = sorted(required - set(tables[table_name].columns))
        results.append(_check(
            f"{table_name}.required_columns",
            not missing_columns,
            details=f"Missing columns: {missing_columns}" if missing_columns else "All required columns present",
        ))

    # Do not continue with checks that would fail noisily when columns are absent.
    failed_schema = [r for r in results if not r["passed"]]
    if failed_schema:
        missing = [r["check"] for r in failed_schema]
        raise ValueError(f"Data quality schema validation failed: {missing}")

    for table_name, key_columns in UNIQUE_KEYS.items():
        df = tables[table_name]
        has_null = df[key_columns].isna().any().any()
        has_duplicates = df.duplicated(key_columns).any()
        results.append(_check(
            f"{table_name}.primary_key",
            not has_null and not has_duplicates,
            details=f"null_keys={bool(has_null)}, duplicate_keys={int(df.duplicated(key_columns).sum())}",
        ))

    for child_name, child_col, parent_name, parent_col in FOREIGN_KEYS:
        child = tables[child_name]
        parent = tables[parent_name]
        invalid = child[child_col].isna() | ~child[child_col].isin(parent[parent_col])
        results.append(_check(
            f"{child_name}.{child_col}_references_{parent_name}",
            not invalid.any(),
            details=f"Invalid references: {int(invalid.sum())}",
        ))

    positive_rules = [
        ("purchase_orders.ordered_qty_positive", "purchase_orders", "ordered_qty", lambda s: s > 0),
        ("purchase_orders.unit_price_nonnegative", "purchase_orders", "unit_price", lambda s: s >= 0),
        ("delivery_logs.delivered_qty_nonnegative", "delivery_logs", "delivered_qty", lambda s: s >= 0),
        ("quality_inspections.inspected_qty_nonnegative", "quality_inspections", "inspected_qty", lambda s: s >= 0),
        ("quality_inspections.defective_qty_nonnegative", "quality_inspections", "defective_qty", lambda s: s >= 0),
        ("quality_inspections.accepted_qty_nonnegative", "quality_inspections", "accepted_qty", lambda s: s >= 0),
        ("invoices.invoiced_qty_nonnegative", "invoices", "invoiced_qty", lambda s: s >= 0),
        ("invoices.invoice_total_nonnegative", "invoices", "invoice_total", lambda s: s >= 0),
    ]
    for name, table_name, column, predicate in positive_rules:
        values = pd.to_numeric(tables[table_name][column], errors="coerce")
        valid = values.notna() & predicate(values)
        results.append(_check(name, bool(valid.all()), details=f"Invalid rows: {int((~valid).sum())}"))

    # Date columns must parse; PO promises should not predate their order dates.
    date_columns = {
        "purchase_orders": ["order_date", "promised_delivery_date"],
        "delivery_logs": ["delivery_date"],
        "quality_inspections": ["inspection_date"],
        "invoices": ["invoice_date"],
        "vendor_category_contracts": ["valid_from", "valid_to"],
    }
    parsed: dict[tuple[str, str], pd.Series] = {}
    for table_name, columns in date_columns.items():
        for column in columns:
            values = pd.to_datetime(tables[table_name][column], errors="coerce")
            parsed[(table_name, column)] = values
            results.append(_check(
                f"{table_name}.{column}_parseable",
                bool(values.notna().all()),
                details=f"Unparseable/null dates: {int(values.isna().sum())}",
            ))

    po_order = parsed[("purchase_orders", "order_date")]
    po_promised = parsed[("purchase_orders", "promised_delivery_date")]
    results.append(_check(
        "purchase_orders.promised_date_not_before_order",
        bool((po_promised >= po_order).all()),
        details=f"Invalid rows: {int((po_promised < po_order).sum())}",
    ))

    contract_start = parsed[("vendor_category_contracts", "valid_from")]
    contract_end = parsed[("vendor_category_contracts", "valid_to")]
    results.append(_check(
        "vendor_category_contracts.validity_range",
        bool((contract_end >= contract_start).all()),
        details=f"Invalid rows: {int((contract_end < contract_start).sum())}",
    ))

    inspection = tables["quality_inspections"]
    quantity_consistent = (
        pd.to_numeric(inspection["defective_qty"], errors="coerce")
        + pd.to_numeric(inspection["accepted_qty"], errors="coerce")
        == pd.to_numeric(inspection["inspected_qty"], errors="coerce")
    )
    results.append(_check(
        "quality_inspections.quantity_balance",
        bool(quantity_consistent.all()),
        details=f"Invalid rows: {int((~quantity_consistent).sum())}",
    ))

    # These are warnings: multiple invoices and split deliveries are legitimate.
    po = tables["purchase_orders"]
    delivery = tables["delivery_logs"]
    results.append(_check(
        "purchase_orders.has_records",
        len(po) > 0,
        details=f"PO rows: {len(po)}",
    ))
    results.append(_check(
        "vendors.has_active_suppliers",
        bool(tables["vendors"]["status"].eq("Active").any()),
        severity="warning",
        details="No active suppliers found" if not tables["vendors"]["status"].eq("Active").any() else "Active suppliers present",
    ))
    results.append(_check(
        "delivery_logs.nonempty",
        len(delivery) > 0,
        severity="warning",
        details=f"Delivery rows: {len(delivery)}",
    ))

    report = _summarize(results)
    if report["failed_error_checks"]:
        failed_names = [r["check"] for r in results if not r["passed"] and r["severity"] == "error"]
        raise ValueError(f"Data quality validation failed: {failed_names}")
    return report


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    errors = [r for r in results if not r["passed"] and r["severity"] == "error"]
    warnings = [r for r in results if not r["passed"] and r["severity"] == "warning"]
    return {
        "total_checks": len(results),
        "passed_checks": sum(1 for r in results if r["passed"]),
        "failed_error_checks": len(errors),
        "failed_warning_checks": len(warnings),
        "checks": results,
    }
