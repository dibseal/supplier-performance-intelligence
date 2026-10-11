# Step 6 — Supplier Performance Dashboard

## What this adds

A Streamlit dashboard that combines the deterministic Step 4 scorecards and Step 5 ML predictions. It provides supplier search, filters, current and predicted risk distributions, a prioritisation table, supplier details, and CSV export.

## Files

- `dashboard.py` — copy to the root of `D:\Hackathon\supplier-performance-intelligence`
- `requirements-step6.txt` — adds Streamlit without replacing the project's existing requirements

## Install and run (PowerShell)

From the project root:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-step6.txt
$env:PYTHONPATH = "src"
streamlit run dashboard.py
```

Streamlit will print a local URL, usually `http://localhost:8501`. Open it in your browser.

## Expected input files

- `data/processed/supplier_scorecards.csv` — Step 4
- `data/processed/supplier_features.csv` — Step 3 (optional enrichment)
- `data/processed/supplier_monthly_features.csv` — reserved for future trend charts
- `data/processed/ml/supplier_risk_predictions.csv` — Step 5
- `data/processed/ml/model_metrics.json` — Step 5 evaluation summary (optional)

Run Steps 3, 4, and 5 before opening the dashboard. If scorecards and predictions are both missing, the dashboard explains which commands to run.

## Design notes

- Uses `vendor_id` as the join key, never supplier display name.
- Treats Step 4 scores and Step 5 predicted probabilities as separate signals.
- Does not show `supplier_profile` or other synthetic scenario labels as model explanations.
- Reads CSVs only; no pipeline outputs are changed.
- Model metrics are visible under **Data and model notes**.
- The training data is synthetic. Do not interpret model metrics as evidence of real-world performance.
