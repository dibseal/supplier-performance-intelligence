# Step 5 — Future Supplier Risk Prediction

## Objective
Predict whether a supplier is likely to underperform during the next 90 days, using only information that was observable at a historical snapshot date.

## Design decisions
- The label is `future_underperformance`: 1 if the future 90-day window has either an on-time delivery rate below 85% or a defect rate above 3%; otherwise 0.
- A label is created only when the future window has matured by the configured data cutoff and has at least five eligible POs and 100 inspected units. Insufficiently observed future windows are left unlabeled.
- Predictors are computed from the preceding 180 days and only include events dated on or before the snapshot.
- `vendor_id`, `supplier_profile`, scenario truth, names, and future outcomes are not model inputs. This prevents direct label leakage from the synthetic generator's behavior profiles.
- Evaluation uses a chronological holdout by snapshot date, not a random row split.
- Random Forest is a baseline model, not assumed to be production-ready. Check PR-AUC, precision, recall, F1 and Brier score before operational use.
- The predicted probability is not the Step 4 performance score. It estimates future underperformance risk, not current performance quality.

## Files
- `config/ml_config.json`: horizon, thresholds, sample minimums, and model settings.
- `src/supplier_performance/ml/risk_prediction.py`: point-in-time features, target creation, temporal evaluation, final training, predictions.
- `tests/unit/test_risk_prediction.py`: tests for leakage exclusion and point-in-time event filtering.

## Run
From the project root in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "src"
python -m pytest tests/unit -v
python -m supplier_performance.ml.risk_prediction
```

The script reads raw tables from `data/raw` and `as_of_date` from `data/manifest.json`. Override paths if required:

```powershell
python -m supplier_performance.ml.risk_prediction --data-dir data/raw --config config/ml_config.json --output-dir data/processed/ml --as-of-date 2026-10-01
```

## Outputs
- `data/processed/ml/supplier_risk_training_panel.csv`
- `data/processed/ml/supplier_risk_predictions.csv`
- `data/processed/ml/model_metrics.json`
- `data/processed/ml/feature_importance.csv`
- `data/processed/ml/supplier_risk_model.joblib`
- `data/processed/ml/ml_pipeline_summary.json`

## Interpretation
`predicted_underperformance_probability` ranges from 0 to 1. The displayed Low/Moderate/High/Very High bands are initial configurable communication bands, not validated business risk appetite. Keep them separate from Step 4's deterministic `performance_score` and `risk_classification`.

## Limitations
Synthetic data is useful for testing the pipeline but cannot establish real-world model performance. The label thresholds, minimum sample counts, feature windows, calibration, and decision thresholds should be reviewed with procurement stakeholders. If the chronological test set is too small or contains only one class, the pipeline intentionally stops instead of silently falling back to a random split.
