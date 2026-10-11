# Step 2 — Synthetic Procurement Data Dictionary

All records are fictional and reproducible from the configured seed.

| Table | Grain | Purpose |
|---|---|---|
| `categories` | One row per category | Product category and typical lead-time/cost ranges |
| `vendors` | One row per vendor | Supplier master, tier, region, capacity and status |
| `vendor_category_contracts` | One row per vendor/category agreement | Eligibility, price, capacity, contract validity and certification |
| `purchase_orders` | One row per PO | Ordered quantity, price, promised delivery date and status |
| `delivery_logs` | One row per shipment/delivery event | Delivered quantity, date, lead time and lateness |
| `quality_inspections` | One row per inspection event | Inspected, defective and accepted quantities |
| `invoices` | One row per invoice | Invoiced quantity, unit price, amount and status |
| `scenario_truth` | One row per vendor | Synthetic scenario label and deterioration start date; evaluation-only metadata |

## Important interpretation notes

- `scenario_truth` is stored separately from operational source tables. Do not use its profile or deterioration date as model features.
- Delivery and quality tables are event-level. Aggregate them before comparing vendors.
- A PO may have multiple delivery events and multiple invoices.
- Invoice totals are synthetic amounts in GBP.
- `is_late_vs_promised` compares a delivery event's date to the PO promised date. For split shipments, a later shipment may be late even when the first shipment was on time.
- The generator creates synthetic scenarios to test pipeline behavior; the scenario labels are not real-world supplier labels.
