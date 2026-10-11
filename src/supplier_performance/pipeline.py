"""Run data quality validation and feature engineering over generated source tables."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from supplier_performance.features.feature_engineering import (
    build_monthly_supplier_features,
    build_po_features,
    build_supplier_features,
    save_dataframe,
)
from supplier_performance.validation.data_quality import validate_tables

LOGGER = logging.getLogger("supplier_performance.pipeline")

TABLES = [
    "categories",
    "vendors",
    "vendor_category_contracts",
    "purchase_orders",
    "delivery_logs",
    "quality_inspections",
    "invoices",
]


def _read_table(source_dir: Path, name: str) -> pd.DataFrame:
    parquet_path = source_dir / f"{name}.parquet"
    csv_path = source_dir / f"{name}.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"Could not find {name}.parquet or {name}.csv in {source_dir}")


def run_pipeline(
    source_dir: Path,
    output_dir: Path,
    as_of_date: str | None = None,
    formats: list[str] | None = None,
) -> dict[str, Any]:
    formats = formats or ["csv", "parquet"]
    tables = {name: _read_table(source_dir, name) for name in TABLES}

    manifest_path = source_dir.parent / "manifest.json"
    if as_of_date is None and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        as_of_date = manifest.get("as_of_date")
    if as_of_date is None:
        raise ValueError("Provide --as-of-date or ensure the source manifest contains as_of_date.")

    quality_report = validate_tables(tables)
    quality_report_path = output_dir / "data_quality_report.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    quality_report_path.write_text(json.dumps(quality_report, indent=2), encoding="utf-8")

    po_features = build_po_features(tables, as_of_date)
    supplier_features = build_supplier_features(po_features, tables["vendors"], as_of_date)
    monthly_features = build_monthly_supplier_features(po_features)

    outputs = {
        "po_features": save_dataframe(po_features, output_dir, "po_features", formats),
        "supplier_features": save_dataframe(supplier_features, output_dir, "supplier_features", formats),
        "supplier_monthly_features": save_dataframe(monthly_features, output_dir, "supplier_monthly_features", formats),
    }
    summary = {
        "as_of_date": str(pd.Timestamp(as_of_date).date()),
        "quality_report": quality_report_path.as_posix(),
        "quality_checks_total": quality_report["total_checks"],
        "quality_checks_passed": quality_report["passed_checks"],
        "quality_checks_failed": quality_report["failed_error_checks"],
        "outputs": outputs,
        "row_counts": {
            "po_features": int(len(po_features)),
            "supplier_features": int(len(supplier_features)),
            "supplier_monthly_features": int(len(monthly_features)),
        },
    }
    (output_dir / "feature_pipeline_manifest.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate source data and build supplier features.")
    parser.add_argument("--source-dir", default="data/raw")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--as-of-date", default=None)
    parser.add_argument("--formats", nargs="+", choices=["csv", "parquet"], default=["csv", "parquet"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    summary = run_pipeline(
        Path(args.source_dir),
        Path(args.output_dir),
        as_of_date=args.as_of_date,
        formats=args.formats,
    )
    print("Step 3 pipeline completed successfully")
    print(f"As of date: {summary['as_of_date']}")
    print(
        f"Data quality checks: {summary['quality_checks_passed']}/"
        f"{summary['quality_checks_total']} passed"
    )
    print("Output row counts:")
    for name, count in summary["row_counts"].items():
        print(f"  {name:30s} {count:>10,}")
    print(f"Manifest: {args.output_dir}/feature_pipeline_manifest.json")


if __name__ == "__main__":
    main()
