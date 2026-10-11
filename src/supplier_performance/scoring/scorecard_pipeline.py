"""Command-line entry point for deterministic supplier scorecards."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from supplier_performance.scoring.scoring_engine import (
    load_scoring_config,
    save_scorecards,
    score_suppliers,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score supplier performance features.")
    parser.add_argument("--input", default="data/processed/supplier_features.parquet")
    parser.add_argument("--config", default="config/scoring_config.json")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--formats", nargs="+", choices=["csv", "parquet"], default=["csv", "parquet"])
    args = parser.parse_args()

    input_path = Path(args.input)
    if input_path.suffix.lower() == ".parquet":
        features = pd.read_parquet(input_path)
    elif input_path.suffix.lower() == ".csv":
        features = pd.read_csv(input_path)
    else:
        raise ValueError("Input must be a .csv or .parquet file.")

    config = load_scoring_config(Path(args.config))
    scorecards = score_suppliers(features, config)
    paths = save_scorecards(scorecards, Path(args.output_dir), args.formats)

    summary = {
        "supplier_count": int(len(scorecards)),
        "risk_classification_counts": {
            str(key): int(value)
            for key, value in scorecards["risk_classification"].value_counts(dropna=False).items()
        },
        "mean_performance_score": (
            round(float(scorecards["performance_score"].mean()), 2)
            if scorecards["performance_score"].notna().any() else None
        ),
        "scorecard_paths": paths,
    }
    summary_path = Path(args.output_dir) / "scorecard_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Supplier scoring completed.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
