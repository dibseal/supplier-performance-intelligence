"""Generate linked, reproducible synthetic procurement datasets.

The generated data is fictional. Scenario-truth metadata is written separately
from operational tables so it can support later model evaluation without being
mistaken for a model feature.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from faker import Faker

LOGGER = logging.getLogger("supplier_performance.data_generation")

CATEGORY_DEFINITIONS = [
    ("CAT001", "Packaging", "units", 14, 5.0, 35.0),
    ("CAT002", "Raw Materials", "kg", 21, 8.0, 120.0),
    ("CAT003", "Electronics", "units", 28, 25.0, 900.0),
    ("CAT004", "Textiles", "meters", 18, 3.0, 45.0),
    ("CAT005", "Office Supplies", "units", 7, 0.5, 80.0),
    ("CAT006", "Industrial Components", "units", 30, 12.0, 500.0),
    ("CAT007", "Safety Equipment", "units", 10, 4.0, 150.0),
    ("CAT008", "Chemicals", "liters", 16, 2.0, 75.0),
    ("CAT009", "Food Ingredients", "kg", 9, 1.5, 40.0),
    ("CAT010", "Maintenance Parts", "units", 12, 5.0, 350.0),
    ("CAT011", "Labels and Printing", "units", 8, 0.2, 15.0),
    ("CAT012", "Logistics Materials", "units", 11, 1.0, 60.0),
]

PROFILE_NAMES = [
    "reliable", "late_delivery", "quality_risk", "cost_risk",
    "inconsistent_lead_time", "deteriorating", "mixed"
]

def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        cfg = json.load(handle)
    required = {"seed", "history_months", "as_of_date", "output_dir", "formats", "counts", "behavior"}
    missing = required - set(cfg)
    if missing:
        raise ValueError(f"Configuration missing keys: {sorted(missing)}")
    if cfg["history_months"] < 6:
        raise ValueError("history_months must be at least 6 to support temporal scenarios.")
    if cfg["counts"]["vendors"] < cfg["counts"]["categories"]:
        raise ValueError("vendors must be at least the number of categories.")
    if not set(cfg["formats"]).issubset({"csv", "parquet"}):
        raise ValueError("formats may contain only 'csv' and/or 'parquet'.")
    return cfg

def _unique_names(fake: Faker, count: int) -> list[str]:
    names: set[str] = set()
    while len(names) < count:
        names.add(f"{fake.company()} {fake.random_element(elements=('Supply', 'Industries', 'Trading', 'Manufacturing', 'Solutions'))}")
    return sorted(names)

def _pick_profile(rng: np.random.Generator, behavior: dict[str, Any], count: int) -> list[str]:
    fractions = {
        "reliable": behavior["reliable_vendor_fraction"],
        "late_delivery": behavior["late_vendor_fraction"],
        "quality_risk": behavior["quality_risk_vendor_fraction"],
        "cost_risk": behavior["cost_risk_vendor_fraction"],
        "inconsistent_lead_time": behavior["inconsistent_vendor_fraction"],
        "deteriorating": behavior["deteriorating_vendor_fraction"],
    }
    # Normalize configured fractions if their sum exceeds one; remaining vendors are mixed.
    total = sum(fractions.values())
    if total > 1:
        fractions = {k: v / total for k, v in fractions.items()}
    profiles: list[str] = []
    for profile, fraction in fractions.items():
        profiles.extend([profile] * int(round(count * fraction)))
    profiles = profiles[:count]
    profiles.extend(["mixed"] * (count - len(profiles)))
    rng.shuffle(profiles)
    return profiles

def _write_table(df: pd.DataFrame, stem: str, output_dir: Path, formats: list[str]) -> dict[str, Any]:
    outputs = []
    for fmt in formats:
        path = output_dir / f"{stem}.{fmt}"
        if fmt == "csv":
            df.to_csv(path, index=False, date_format="%Y-%m-%dT%H:%M:%S")
        elif fmt == "parquet":
            df.to_parquet(path, index=False)
        outputs.append(str(path.as_posix()))
    return {"rows": int(len(df)), "columns": list(df.columns), "files": outputs}

def validate_datasets(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Run structural and referential-integrity checks; raise on invalid data."""
    vendors = tables["vendors"]
    categories = tables["categories"]
    contracts = tables["vendor_category_contracts"]
    pos = tables["purchase_orders"]
    deliveries = tables["delivery_logs"]
    inspections = tables["quality_inspections"]
    invoices = tables["invoices"]

    checks: dict[str, bool] = {
        "vendor_id_unique": vendors["vendor_id"].is_unique,
        "category_id_unique": categories["category_id"].is_unique,
        "po_id_unique": pos["po_id"].is_unique,
        "delivery_id_unique": deliveries["delivery_id"].is_unique,
        "inspection_id_unique": inspections["inspection_id"].is_unique,
        "invoice_id_unique": invoices["invoice_id"].is_unique,
        "po_vendor_fk_valid": pos["vendor_id"].isin(vendors["vendor_id"]).all(),
        "po_category_fk_valid": pos["category_id"].isin(categories["category_id"]).all(),
        "delivery_po_fk_valid": deliveries["po_id"].isin(pos["po_id"]).all(),
        "inspection_po_fk_valid": inspections["po_id"].isin(pos["po_id"]).all(),
        "invoice_po_fk_valid": invoices["po_id"].isin(pos["po_id"]).all(),
        "contract_vendor_fk_valid": contracts["vendor_id"].isin(vendors["vendor_id"]).all(),
        "contract_category_fk_valid": contracts["category_id"].isin(categories["category_id"]).all(),
        "positive_order_quantities": (pos["ordered_qty"] > 0).all(),
        "nonnegative_delivered_quantities": (deliveries["delivered_qty"] >= 0).all(),
        "nonnegative_defect_counts": (inspections["defective_qty"] >= 0).all(),
        "invoice_totals_nonnegative": (invoices["invoice_total"] >= 0).all(),
        "no_duplicate_vendor_category_contracts": not contracts.duplicated(["vendor_id", "category_id"]).any(),
    }
    failed = [name for name, passed in checks.items() if not passed]
    
    # Convert NumPy and pandas boolean results to native Python bool.
    checks = {
        name: bool(passed)
        for name, passed in checks.items()
    }

    failed = [name for name, passed in checks.items() if not passed]

    if failed:
        raise ValueError(f"Synthetic data validation failed: {failed}")

    return {
        "passed": len(checks),
        "failed": 0,
        "checks": checks,
    }


def generate_datasets(cfg: dict[str, Any], output_dir: Path | None = None) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    seed = int(cfg["seed"])
    rng = np.random.default_rng(seed)
    fake = Faker("en_GB")
    Faker.seed(seed)
    fake.seed_instance(seed)

    as_of = pd.Timestamp(cfg["as_of_date"])
    history_start = as_of - pd.DateOffset(months=int(cfg["history_months"]))
    behavior = cfg["behavior"]
    n_vendors = int(cfg["counts"]["vendors"])
    n_categories = int(cfg["counts"]["categories"])
    n_pos = int(cfg["counts"]["purchase_orders"])
    if n_categories > len(CATEGORY_DEFINITIONS):
        raise ValueError(f"At most {len(CATEGORY_DEFINITIONS)} category definitions are currently available.")

    category_rows = CATEGORY_DEFINITIONS[:n_categories]
    categories = pd.DataFrame([
        {
            "category_id": row[0], "category_name": row[1], "unit_of_measure": row[2],
            "typical_lead_days": row[3], "typical_unit_cost_min": row[4],
            "typical_unit_cost_max": row[5],
        } for row in category_rows
    ])

    profiles = _pick_profile(rng, behavior, n_vendors)
    names = _unique_names(fake, n_vendors)
    regions = ["United Kingdom", "India", "Germany", "Netherlands", "Poland", "China", "Vietnam", "Turkey"]
    vendor_rows = []
    scenario_rows = []
    for i in range(n_vendors):
        vendor_id = f"VEN{i+1:04d}"
        profile = profiles[i]
        onboard_date = history_start - pd.Timedelta(days=int(rng.integers(30, 1500)))
        active = bool(rng.random() > 0.06)
        capacity = int(rng.integers(500, 15000))
        vendor_rows.append({
            "vendor_id": vendor_id,
            "vendor_name": names[i],
            "vendor_tier": rng.choice(["Strategic", "Preferred", "Approved", "Transactional"], p=[0.12, 0.23, 0.45, 0.20]),
            "region": rng.choice(regions),
            "onboard_date": onboard_date.date().isoformat(),
            "status": "Active" if active else "Inactive",
            "monthly_capacity_units": capacity,
            "payment_terms_days": int(rng.choice([15, 30, 45, 60], p=[0.1, 0.55, 0.25, 0.1])),
            "certifications": "|".join(rng.choice(["ISO9001", "ISO14001", "ISO45001", "REACH", "None"], size=int(rng.integers(1, 3)), replace=False)),
            "supplier_profile": profile,
        })
        deterioration_start = None
        if profile == "deteriorating":
            deterioration_start = history_start + pd.Timedelta(days=int((as_of - history_start).days * rng.uniform(0.55, 0.82)))
        scenario_rows.append({
            "vendor_id": vendor_id,
            "scenario_profile": profile,
            "deterioration_start_date": deterioration_start.date().isoformat() if deterioration_start is not None else None,
            "scenario_seed": seed,
            "is_synthetic_ground_truth": True,
        })
    vendors = pd.DataFrame(vendor_rows)
    scenario_truth = pd.DataFrame(scenario_rows)

    # Each supplier is eligible for 1-4 categories. Ensure every category has eligible vendors.
    contract_rows = []
    contract_id = 1
    vendor_categories: dict[str, list[str]] = {}
    category_ids = categories["category_id"].tolist()
    for i, vendor in vendors.iterrows():
        chosen = set(rng.choice(category_ids, size=int(rng.integers(1, min(4, len(category_ids)) + 1)), replace=False).tolist())
        if i < len(category_ids):
            chosen.add(category_ids[i])
        vendor_categories[vendor["vendor_id"]] = sorted(chosen)
        for category_id in sorted(chosen):
            cat = categories.loc[categories["category_id"] == category_id].iloc[0]
            contract_rows.append({
                "contract_id": f"CON{contract_id:06d}",
                "vendor_id": vendor["vendor_id"],
                "category_id": category_id,
                "contract_unit_price": round(float(rng.uniform(cat["typical_unit_cost_min"], cat["typical_unit_cost_max"])), 2),
                "valid_from": (history_start - pd.Timedelta(days=int(rng.integers(0, 365)))).date().isoformat(),
                "valid_to": (as_of + pd.Timedelta(days=int(rng.integers(30, 365)))).date().isoformat(),
                "minimum_order_qty": int(rng.choice([10, 25, 50, 100, 250])),
                "max_monthly_capacity_units": int(vendors.loc[vendors["vendor_id"] == vendor["vendor_id"], "monthly_capacity_units"].iloc[0]),
                "certification_required": rng.choice(["None", "ISO9001", "ISO14001"], p=[0.6, 0.3, 0.1]),
                "is_active": bool(vendor["status"] == "Active"),
            })
            contract_id += 1
    contracts = pd.DataFrame(contract_rows)

    # Select PO vendors and categories from eligible contracts.
    contracts_by_category = {cat: contracts[contracts["category_id"] == cat] for cat in category_ids}
    po_rows = []
    po_vendor_profile: dict[str, str] = {}
    po_meta: dict[str, dict[str, Any]] = {}
    for i in range(n_pos):
        cat_id = str(rng.choice(category_ids))
        candidates = contracts_by_category[cat_id]
        if candidates.empty:
            candidates = contracts
        chosen = candidates.iloc[int(rng.integers(0, len(candidates)))]
        vendor_id = chosen["vendor_id"]
        vendor = vendors.loc[vendors["vendor_id"] == vendor_id].iloc[0]
        cat = categories.loc[categories["category_id"] == cat_id].iloc[0]
        order_ts = history_start + pd.Timedelta(seconds=int(rng.integers(0, max(1, int((as_of - history_start).total_seconds())))))
        promised_lead = max(2, int(round(rng.normal(float(cat["typical_lead_days"]), max(1.5, float(cat["typical_lead_days"]) * 0.12)))))
        promised_date = order_ts.normalize() + pd.Timedelta(days=promised_lead)
        qty = int(np.clip(rng.lognormal(mean=4.1, sigma=0.85), 5, 3000))
        base_price = float(chosen["contract_unit_price"])
        # Contract price can move modestly over time; risk profile affects invoice variance, not the agreed price.
        po_id = f"PO{i+1:07d}"
        po_rows.append({
            "po_id": po_id,
            "vendor_id": vendor_id,
            "category_id": cat_id,
            "contract_id": chosen["contract_id"],
            "order_date": order_ts.date().isoformat(),
            "promised_delivery_date": promised_date.date().isoformat(),
            "ordered_qty": qty,
            "unit_price": round(base_price, 2),
            "currency": "GBP",
            "po_status": rng.choice(["Open", "Closed", "Cancelled"], p=[0.08, 0.88, 0.04]),
            "shipping_method": rng.choice(["Road", "Air", "Sea", "Rail"], p=[0.55, 0.12, 0.23, 0.10]),
        })
        po_vendor_profile[po_id] = vendor["supplier_profile"]
        po_meta[po_id] = {
            "vendor_id": vendor_id, "category_id": cat_id, "order_date": order_ts,
            "promised_date": promised_date, "qty": qty, "unit_price": base_price,
            "profile": vendor["supplier_profile"],
            "deterioration_start": scenario_truth.loc[scenario_truth["vendor_id"] == vendor_id, "deterioration_start_date"].iloc[0],
            "po_status": po_rows[-1]["po_status"],
        }
    purchase_orders = pd.DataFrame(po_rows)

    delivery_rows = []
    inspection_rows = []
    invoice_rows = []
    delivery_counter = inspection_counter = invoice_counter = 1
    profile_lateness = {
        "reliable": -1.0, "late_delivery": 8.0, "quality_risk": 1.5,
        "cost_risk": 1.0, "inconsistent_lead_time": 2.0, "deteriorating": 0.5, "mixed": 1.5,
    }
    profile_defect_rate = {
        "reliable": 0.006, "late_delivery": 0.025, "quality_risk": 0.085,
        "cost_risk": 0.025, "inconsistent_lead_time": 0.025, "deteriorating": 0.02, "mixed": 0.025,
    }
    profile_cost_overrun = {
        "reliable": 0.002, "late_delivery": 0.008, "quality_risk": 0.008,
        "cost_risk": 0.075, "inconsistent_lead_time": 0.008, "deteriorating": 0.01, "mixed": 0.012,
    }
    for po in po_rows:
        meta = po_meta[po["po_id"]]
        if meta["po_status"] == "Cancelled":
            continue
        order_date = meta["order_date"]
        promised_date = meta["promised_date"]
        qty = meta["qty"]
        profile = meta["profile"]
        lead_typical = int(categories.loc[categories["category_id"] == meta["category_id"], "typical_lead_days"].iloc[0])
        deterioration_start = pd.Timestamp(meta["deterioration_start"]) if meta["deterioration_start"] else None
        deteriorated = profile == "deteriorating" and deterioration_start is not None and order_date >= deterioration_start
        delay_mean = profile_lateness[profile] + (7.0 if deteriorated else 0.0)
        if profile == "inconsistent_lead_time":
            delay_sd = max(5.0, lead_typical * 0.55)
        else:
            delay_sd = max(2.0, lead_typical * 0.18)
        delay_days = int(round(rng.normal(delay_mean, delay_sd)))
        actual_delivery_date = max(order_date.normalize(), promised_date + pd.Timedelta(days=delay_days))
        # Occasional split shipments, each with a portion of ordered quantity.
        split = rng.random() < behavior["split_delivery_probability"]
        if split and qty >= 2:
            first_qty = int(rng.integers(1, qty))
            quantities = [first_qty, qty - first_qty]
            if rng.random() < behavior["second_split_probability"] and qty >= 3:
                # Split the second part once more, preserving exact total quantity.
                second_part = quantities.pop()
                piece = int(rng.integers(1, second_part)) if second_part > 1 else second_part
                quantities.extend([piece, second_part - piece] if piece < second_part else [piece])
        else:
            quantities = [qty]
        for j, delivered_qty in enumerate(quantities):
            delivery_date = actual_delivery_date + pd.Timedelta(days=j * int(rng.integers(1, 5)))
            delivery_rows.append({
                "delivery_id": f"DEL{delivery_counter:08d}",
                "po_id": po["po_id"],
                "delivery_date": delivery_date.date().isoformat(),
                "delivered_qty": int(delivered_qty),
                "delivery_sequence": j + 1,
                "delivery_status": "Received",
                "actual_lead_days": max(0, int((delivery_date.normalize() - order_date.normalize()).days)),
                "is_late_vs_promised": bool(delivery_date.normalize() > promised_date.normalize()),
            })
            delivery_counter += 1

            if rng.random() < behavior["inspection_coverage"]:
                defect_rate = profile_defect_rate[profile] * (2.2 if deteriorated else 1.0)
                defect_rate = float(np.clip(rng.normal(defect_rate, defect_rate * 0.22 + 0.001), 0, 0.35))
                defective = int(rng.binomial(int(delivered_qty), defect_rate))
                inspected = int(delivered_qty)
                inspection_rows.append({
                    "inspection_id": f"INS{inspection_counter:08d}",
                    "po_id": po["po_id"],
                    "delivery_id": f"DEL{delivery_counter-1:08d}",
                    "inspection_date": delivery_date.date().isoformat(),
                    "inspected_qty": inspected,
                    "defective_qty": defective,
                    "accepted_qty": inspected - defective,
                    "defect_type": rng.choice(["None", "Cosmetic", "Dimensional", "Functional", "Packaging"],
                                              p=[0.55, 0.15, 0.12, 0.10, 0.08]) if defective > 0 else "None",
                })
                inspection_counter += 1

        # Most POs have one invoice; a small fraction have two invoices/adjustments.
        n_invoice = 2 if rng.random() < behavior["invoice_multiple_probability"] else 1
        invoice_qty_remaining = qty
        for inv_idx in range(n_invoice):
            inv_qty = invoice_qty_remaining if inv_idx == n_invoice - 1 else max(1, int(qty * rng.uniform(0.25, 0.65)))
            invoice_qty_remaining = max(0, invoice_qty_remaining - inv_qty)
            overrun = profile_cost_overrun[profile] + (0.025 if deteriorated else 0.0)
            unit_price = meta["unit_price"] * (1 + max(-0.02, rng.normal(overrun, 0.025)))
            invoice_date = max(order_date, actual_delivery_date) + pd.Timedelta(days=int(rng.integers(0, 31)))
            invoice_rows.append({
                "invoice_id": f"INV{invoice_counter:08d}",
                "po_id": po["po_id"],
                "invoice_date": invoice_date.date().isoformat(),
                "invoiced_qty": int(inv_qty),
                "invoice_unit_price": round(float(unit_price), 2),
                "invoice_total": round(float(inv_qty * unit_price), 2),
                "currency": "GBP",
                "invoice_status": rng.choice(["Paid", "Pending", "Disputed"], p=[0.78, 0.17, 0.05]),
            })
            invoice_counter += 1

    delivery_logs = pd.DataFrame(delivery_rows, columns=[
        "delivery_id", "po_id", "delivery_date", "delivered_qty", "delivery_sequence",
        "delivery_status", "actual_lead_days", "is_late_vs_promised"
    ])
    quality_inspections = pd.DataFrame(inspection_rows, columns=[
        "inspection_id", "po_id", "delivery_id", "inspection_date", "inspected_qty",
        "defective_qty", "accepted_qty", "defect_type"
    ])
    invoices = pd.DataFrame(invoice_rows, columns=[
        "invoice_id", "po_id", "invoice_date", "invoiced_qty", "invoice_unit_price",
        "invoice_total", "currency", "invoice_status"
    ])

    tables = {
        "categories": categories,
        "vendors": vendors,
        "vendor_category_contracts": contracts,
        "purchase_orders": purchase_orders,
        "delivery_logs": delivery_logs,
        "quality_inspections": quality_inspections,
        "invoices": invoices,
    }
    validation = validate_datasets(tables)
    metadata = {
        "seed": seed,
        "history_start": history_start.date().isoformat(),
        "as_of_date": as_of.date().isoformat(),
        "history_months": int(cfg["history_months"]),
        "validation": validation,
        "scenario_truth_rows": int(len(scenario_truth)),
        "tables": {},
    }
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, df in tables.items():
            metadata["tables"][name] = _write_table(df, name, output_dir, cfg["formats"])
        truth_dir = output_dir.parent / "synthetic"
        truth_dir.mkdir(parents=True, exist_ok=True)
        metadata["tables"]["scenario_truth"] = _write_table(scenario_truth, "scenario_truth", truth_dir, cfg["formats"])
        manifest_path = output_dir.parent / "manifest.json"
        manifest_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        LOGGER.info("Wrote manifest: %s", manifest_path)
    return tables | {"scenario_truth": scenario_truth}, metadata

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic supplier procurement datasets.")
    parser.add_argument("--config", default="config/synthetic_data.json", help="Path to JSON configuration.")
    parser.add_argument("--output-dir", default=None, help="Override configured output directory.")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = load_config(Path(args.config))
    if args.seed is not None:
        cfg["seed"] = args.seed
    out = Path(args.output_dir or cfg["output_dir"])
    tables, metadata = generate_datasets(cfg, out)
    print("\nSynthetic procurement data generated successfully")
    print(f"Seed: {metadata['seed']}")
    print(f"Period: {metadata['history_start']} to {metadata['as_of_date']}")
    print(f"Validation: {metadata['validation']['passed']} passed, {metadata['validation']['failed']} failed")
    print("\nTable row counts:")
    for name, df in tables.items():
        print(f"  {name:28s} {len(df):>9,}")

if __name__ == "__main__":
    main()
