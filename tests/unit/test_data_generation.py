import json
from pathlib import Path

from supplier_performance.data.data_generation import generate_datasets, load_config, validate_datasets


def small_config():
    config_path = Path("config/synthetic_data.json")
    cfg = load_config(config_path)
    cfg["seed"] = 1234
    cfg["counts"] = {"vendors": 20, "categories": 6, "purchase_orders": 250}
    cfg["history_months"] = 12
    cfg["formats"] = ["csv"]
    return cfg


def test_generation_is_reproducible():
    cfg = small_config()
    first, _ = generate_datasets(cfg)
    second, _ = generate_datasets(cfg)
    assert first["vendors"].equals(second["vendors"])
    assert first["purchase_orders"].equals(second["purchase_orders"])
    assert first["delivery_logs"].equals(second["delivery_logs"])


def test_data_relationships_and_validation():
    tables, metadata = generate_datasets(small_config())
    assert metadata["validation"]["failed"] == 0
    assert tables["purchase_orders"]["vendor_id"].isin(tables["vendors"]["vendor_id"]).all()
    assert tables["delivery_logs"]["po_id"].isin(tables["purchase_orders"]["po_id"]).all()
    assert tables["quality_inspections"]["delivery_id"].isin(tables["delivery_logs"]["delivery_id"]).all()


def test_generated_profiles_include_scenarios():
    tables, _ = generate_datasets(small_config())
    profiles = set(tables["scenario_truth"]["scenario_profile"])
    assert "mixed" in profiles or len(profiles) > 1
    assert tables["scenario_truth"]["is_synthetic_ground_truth"].all()
