# Step 4 — Supplier scoring engine

## Run

```powershell
$env:PYTHONPATH = "src"
python -m supplier_performance.scoring.scorecard_pipeline
python -m pytest tests/unit -v
```

## Outputs

- `data/processed/supplier_scorecards.csv`
- `data/processed/supplier_scorecards.parquet`
- `data/processed/scorecard_summary.json`

## Design

- Weighted score: delivery 35%, quality 30%, cost 20%, lead-time consistency 15%.
- Each dimension is scored on a 0–100 scale.
- Missing dimension metrics are excluded and available weights are renormalized.
- The final risk classification is `Insufficient data` unless all configured minimum sample requirements are met.
- Risk alerts are deterministic rule outputs and can coexist with an acceptable overall score when a recent deterioration signal exists.
- Weights, thresholds and alert triggers are configured in `config/scoring_config.json`.

## Score mappings

- Delivery: on-time delivery rate multiplied by 100.
- Quality: 100 at/below the target defect rate; 0 at/above the unacceptable defect rate; linear between.
- Cost: 100 when invoice price variance is at or below the no-penalty threshold; 0 at/above the configured zero-score overrun; linear between.
- Lead-time consistency: 100 at/below the ideal coefficient of variation; 0 at/above the zero-score coefficient; linear between.

## Important caveat

The thresholds and weights are initial synthetic-MVP defaults. Validate them with procurement stakeholders before using the scores for real sourcing decisions.
