# Step 3 — Data validation and feature engineering

## Run

```powershell
$env:PYTHONPATH = "src"
python -m supplier_performance.pipeline
python -m pytest tests/unit -v
```

## Outputs

- `data/processed/po_features.csv` and `.parquet`: one row per PO.
- `data/processed/supplier_features.csv` and `.parquet`: one row per vendor.
- `data/processed/supplier_monthly_features.csv` and `.parquet`: one row per vendor/month.
- `data/processed/data_quality_report.json`: detailed checks.
- `data/processed/feature_pipeline_manifest.json`: row counts and output paths.

## Aggregation safety

Delivery events, inspections, and invoices are aggregated independently to PO grain before they are joined. This prevents split deliveries and multiple invoices from multiplying each other's quantities or values.

## Metric semantics

- A PO is on time only if it is fully delivered and its final delivery date is on or before the promised date.
- An open PO is evaluable as a failure only if it is overdue at the reporting as-of date; future open POs are not treated as late.
- Defect rate is total defective units divided by inspected units.
- Effective invoice unit price is total invoice amount divided by total invoiced quantity.
- Cost variance is effective invoice unit price relative to the PO unit price.
- Lead time is measured from PO order date to final delivery date.
- Cancelled POs are excluded from supplier performance aggregates.
- Missing metrics remain null where no denominator exists; they are not automatically converted to zero.
- `scenario_truth` is not read by the feature pipeline and must not be used as an ML feature.

## Known simplifications

This MVP uses PO-level promised dates and the final delivery event to determine on-time completion. Production systems may require line-level promises, approved schedule changes, goods-receipt status, returns, credit notes, taxes, freight, and currency conversion.
