"""Supplier Performance Intelligence dashboard (Streamlit).

Run from the project root with: streamlit run dashboard.py
The dashboard reads existing CSV outputs and never modifies pipeline data.
"""
from __future__ import annotations

from pathlib import Path
import json
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
ML_DIR = PROCESSED / "ml"

st.set_page_config(page_title="Supplier Performance Intelligence", page_icon="📊", layout="wide")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.warning(f"Could not read {path.relative_to(ROOT)}: {exc}")
        return pd.DataFrame()


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    return next((name for name in candidates if name in df.columns), None)


def numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(float("nan"), index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def pct(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    # Rates are stored as fractions in this project; percentages are for display only.
    return values * 100


st.title("Supplier Performance Intelligence")
st.caption("Current supplier performance, emerging risk, and procurement priorities")

scorecards = read_csv(PROCESSED / "supplier_scorecards.csv")
predictions = read_csv(ML_DIR / "supplier_risk_predictions.csv")
features = read_csv(PROCESSED / "supplier_features.csv")
monthly = read_csv(PROCESSED / "supplier_monthly_features.csv")

if scorecards.empty and predictions.empty:
    st.error("No scorecards or ML predictions were found. Run Steps 3, 4, and 5 first.")
    st.code("python -m supplier_performance.scoring.scorecard_pipeline\npython -m supplier_performance.ml.risk_prediction")
    st.stop()

# Join on the stable synthetic vendor identifier; avoid joining on display names.
if "vendor_id" in scorecards.columns and "vendor_id" in predictions.columns:
    risk_cols = [c for c in ["vendor_id", "predicted_underperformance_probability", "predicted_risk_band"] if c in predictions.columns]
    view = scorecards.merge(predictions[risk_cols].drop_duplicates("vendor_id"), on="vendor_id", how="outer", validate="one_to_one")
elif scorecards.empty:
    view = predictions.copy()
else:
    view = scorecards.copy()

if not features.empty and "vendor_id" in features.columns and "vendor_id" in view.columns:
    useful = [c for c in ["vendor_id", "vendor_name", "category_name", "po_count", "on_time_delivery_rate", "defect_rate", "cost_variance_pct", "overdue_open_count"] if c in features.columns]
    add = features[useful].drop_duplicates("vendor_id")
    # Scorecards are the authoritative source when a feature column already exists.
    add = add[[c for c in add.columns if c == "vendor_id" or c not in view.columns]]
    if len(add.columns) > 1:
        view = view.merge(add, on="vendor_id", how="left", validate="one_to_one")

name_col = first_existing_column(view, ["vendor_name", "supplier_name", "vendor_id"])
score_col = first_existing_column(view, ["performance_score"])
class_col = first_existing_column(view, ["risk_classification"])
prob_col = first_existing_column(view, ["predicted_underperformance_probability"])
band_col = first_existing_column(view, ["predicted_risk_band"])
category_col = first_existing_column(view, ["category_name", "category", "product_category"])

with st.sidebar:
    st.header("Filters")
    search_text = st.text_input("Search supplier", placeholder="Supplier name or ID")
    if class_col:
        classes = sorted(view[class_col].dropna().astype(str).unique().tolist())
        selected_classes = st.multiselect("Current performance classification", classes, default=classes)
    else:
        selected_classes = []
    if band_col:
        bands = sorted(view[band_col].dropna().astype(str).unique().tolist())
        selected_bands = st.multiselect("Predicted risk band", bands, default=bands)
    else:
        selected_bands = []
    if category_col:
        categories = sorted(view[category_col].dropna().astype(str).unique().tolist())
        selected_categories = st.multiselect("Category", categories, default=categories)
    else:
        selected_categories = []

filtered = view.copy()
if search_text and name_col:
    filtered = filtered[filtered[name_col].astype(str).str.contains(search_text, case=False, na=False)]
if class_col and selected_classes:
    filtered = filtered[filtered[class_col].astype(str).isin(selected_classes)]
if band_col and selected_bands:
    filtered = filtered[filtered[band_col].astype(str).isin(selected_bands)]
if category_col and selected_categories:
    filtered = filtered[filtered[category_col].astype(str).isin(selected_categories)]

# KPI cards
k1, k2, k3, k4 = st.columns(4)
k1.metric("Suppliers in view", f"{len(filtered):,}")
if score_col:
    k2.metric("Average performance score", f"{numeric(filtered, score_col).mean():.1f}" if numeric(filtered, score_col).notna().any() else "N/A")
else:
    k2.metric("Average performance score", "N/A")
if class_col:
    at_risk_count = filtered[class_col].astype(str).isin(["At risk", "Critical"]).sum()
    k3.metric("At risk / Critical", f"{int(at_risk_count):,}")
else:
    k3.metric("At risk / Critical", "N/A")
if prob_col:
    p = numeric(filtered, prob_col)
    k4.metric("Average predicted risk", f"{p.mean() * 100:.1f}%" if p.notna().any() else "N/A")
else:
    k4.metric("Average predicted risk", "N/A")

st.divider()
left, right = st.columns(2)
with left:
    st.subheader("Current performance distribution")
    if class_col and not filtered.empty:
        counts = filtered[class_col].fillna("Unknown").astype(str).value_counts().rename_axis("Classification").to_frame("Suppliers")
        st.bar_chart(counts)
    else:
        st.info("Current classification is not available in the scorecard file.")
with right:
    st.subheader("Predicted future risk distribution")
    if band_col and not filtered.empty:
        counts = filtered[band_col].fillna("Unknown").astype(str).value_counts().rename_axis("Predicted risk band").to_frame("Suppliers")
        st.bar_chart(counts)
    else:
        st.info("ML predictions are not available. Run Step 5 to generate them.")

st.subheader("Supplier prioritisation")
priority = filtered.copy()
if prob_col:
    priority[prob_col] = numeric(priority, prob_col)
    priority = priority.sort_values(prob_col, ascending=False, na_position="last")
elif score_col:
    priority[score_col] = numeric(priority, score_col)
    priority = priority.sort_values(score_col, ascending=True, na_position="last")

preferred = [
    name_col, "vendor_id", category_col, score_col, class_col,
    "delivery_score", "quality_score", "cost_score", "lead_time_consistency_score",
    band_col, prob_col, "on_time_delivery_rate", "defect_rate", "cost_variance_pct",
    "overdue_open_count", "risk_alert_count", "risk_alerts_text",
]
columns = []
for col in preferred:
    if col and col in priority.columns and col not in columns:
        columns.append(col)
if not columns:
    columns = list(priority.columns[:12])
st.dataframe(priority[columns], use_container_width=True, hide_index=True)

st.download_button(
    "Download filtered supplier list (CSV)",
    data=priority[columns].to_csv(index=False).encode("utf-8"),
    file_name="supplier_performance_filtered.csv",
    mime="text/csv",
)

st.subheader("Supplier detail")
if name_col and not filtered.empty:
    labels = filtered[name_col].astype(str).tolist()
    chosen = st.selectbox("Select a supplier", labels)
    detail = filtered.loc[filtered[name_col].astype(str) == chosen].head(1)
    if not detail.empty:
        row = detail.iloc[0]
        d1, d2, d3 = st.columns(3)
        d1.metric("Performance score", f"{row[score_col]:.1f}" if score_col and pd.notna(row.get(score_col)) else "N/A")
        d2.metric("Current classification", str(row.get(class_col, "N/A")) if class_col else "N/A")
        risk_value = row.get(prob_col) if prob_col else None
        d3.metric("Predicted underperformance", f"{float(risk_value)*100:.1f}%" if risk_value is not None and pd.notna(risk_value) else "N/A")
        alert_col = first_existing_column(detail, ["risk_alerts_text", "risk_alerts"])
        if alert_col and pd.notna(row.get(alert_col)) and str(row.get(alert_col)).strip():
            st.markdown("**Risk alerts**")
            st.write(str(row[alert_col]).replace("|", " · "))
        with st.expander("View all available supplier fields"):
            st.dataframe(detail.T.rename(columns={detail.index[0]: "Value"}), use_container_width=True)
else:
    st.info("Supplier details are unavailable until supplier identifiers or names are present in the data.")

with st.expander("Data and model notes"):
    st.markdown(
        "- Current classification comes from the deterministic Step 4 scorecard.\n"
        "- Predicted risk comes from the Step 5 ML model and is an estimate, not a guarantee.\n"
        "- The model was trained on synthetic data; validate against real historical outcomes before operational use.\n"
        "- This dashboard reads CSV outputs only and does not modify source data."
    )
    metrics_path = ML_DIR / "model_metrics.json"
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            st.json(metrics)
        except (OSError, json.JSONDecodeError) as exc:
            st.warning(f"Could not load model metrics: {exc}")
