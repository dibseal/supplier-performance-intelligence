"""Configurable, deterministic supplier scoring and explainable risk alerts."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_scoring_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    validate_scoring_config(config)
    return config


def validate_scoring_config(config: dict[str, Any]) -> None:
    required_weights = {"delivery", "quality", "cost", "lead_time_consistency"}
    weights = config.get("weights", {})
    if set(weights) != required_weights:
        raise ValueError(f"weights must contain exactly: {sorted(required_weights)}")
    if any(not isinstance(v, (int, float)) or v < 0 for v in weights.values()):
        raise ValueError("Scoring weights must be non-negative numbers.")
    if not np.isclose(sum(weights.values()), 1.0, atol=1e-8):
        raise ValueError("Scoring weights must sum to 1.0.")
    thresholds = config.get("risk_thresholds", {})
    if not (0 <= thresholds.get("at_risk_min", -1) < thresholds.get("acceptable_min", -1)
            < thresholds.get("strong_min", -1) <= 100):
        raise ValueError("Risk thresholds must be ordered and within 0..100.")
    quality = config.get("quality", {})
    if not (0 <= quality["target_defect_rate"] < quality["unacceptable_defect_rate"] <= 1):
        raise ValueError("Quality defect-rate thresholds must be ordered in 0..1.")
    consistency = config.get("lead_time_consistency", {})
    if not (0 <= consistency["ideal_cv"] < consistency["zero_score_cv"]):
        raise ValueError("Lead-time CV thresholds are invalid.")


def _numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def _linear_higher_is_better(value: float, bad: float, good: float) -> float:
    """Map value <= bad to 0 and value >= good to 100."""
    if pd.isna(value):
        return np.nan
    if good <= bad:
        raise ValueError("good must be greater than bad.")
    return float(np.clip((value - bad) / (good - bad) * 100.0, 0, 100))


def _linear_lower_is_better(value: float, good: float, bad: float) -> float:
    """Map value <= good to 100 and value >= bad to 0."""
    if pd.isna(value):
        return np.nan
    if bad <= good:
        raise ValueError("bad must be greater than good.")
    return float(np.clip((bad - value) / (bad - good) * 100.0, 0, 100))


def _classify(score: float, thresholds: dict[str, float]) -> str:
    if pd.isna(score):
        return "Insufficient data"
    if score >= thresholds["strong_min"]:
        return "Strong"
    if score >= thresholds["acceptable_min"]:
        return "Acceptable"
    if score >= thresholds["at_risk_min"]:
        return "At risk"
    return "Critical"


def _alerts(row: pd.Series, config: dict[str, Any]) -> list[str]:
    thresholds = config["alerts"]
    alerts: list[str] = []

    if pd.notna(row.get("on_time_delivery_rate")):
        if row["on_time_delivery_rate"] < thresholds["low_delivery_rate"]:
            alerts.append("LOW_DELIVERY_RELIABILITY")
    if pd.notna(row.get("defect_rate")):
        if row["defect_rate"] > thresholds["high_defect_rate"]:
            alerts.append("HIGH_DEFECT_RATE")
    if pd.notna(row.get("invoice_price_variance_pct")):
        if row["invoice_price_variance_pct"] > thresholds["high_cost_variance_pct"]:
            alerts.append("HIGH_COST_VARIANCE")
    if pd.notna(row.get("lead_time_cv")):
        if row["lead_time_cv"] > thresholds["high_lead_time_cv"]:
            alerts.append("INCONSISTENT_LEAD_TIME")
    if (
        pd.notna(row.get("recent_on_time_delivery_rate"))
        and pd.notna(row.get("on_time_delivery_rate"))
        and row["recent_on_time_delivery_rate"]
        < row["on_time_delivery_rate"] - thresholds["recent_delivery_drop"]
    ):
        alerts.append("RECENT_DELIVERY_DETERIORATION")
    if (
        pd.notna(row.get("recent_defect_rate"))
        and pd.notna(row.get("defect_rate"))
        and row["recent_defect_rate"]
        > row["defect_rate"] + thresholds["recent_defect_increase"]
    ):
        alerts.append("RECENT_QUALITY_DETERIORATION")
    if int(row.get("overdue_open_count", 0) or 0) > 0:
        alerts.append("OVERDUE_OPEN_PURCHASE_ORDERS")

    return alerts


def score_suppliers(
    supplier_features: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Return supplier scorecards; dimension scores are nullable if data is absent.

    Overall score is the weighted sum of available dimensions, normalized by the
    sum of their weights. Missing dimensions are not treated as zero. A supplier
    receives a final classification only if all required minimum sample criteria
    are met; otherwise the classification is Insufficient data.
    """
    validate_scoring_config(config)
    df = supplier_features.copy()

    delivery = _numeric(df, "on_time_delivery_rate")
    defect_rate = _numeric(df, "defect_rate")
    cost_variance = _numeric(df, "invoice_price_variance_pct")
    lead_cv = _numeric(df, "lead_time_cv")

    quality_cfg = config["quality"]
    cost_cfg = config["cost"]
    consistency_cfg = config["lead_time_consistency"]

    df["delivery_score"] = delivery.clip(0, 1) * 100
    df["quality_score"] = defect_rate.apply(
        lambda x: _linear_lower_is_better(
            x, quality_cfg["target_defect_rate"], quality_cfg["unacceptable_defect_rate"]
        )
    )
    df["cost_score"] = cost_variance.apply(
        lambda x: (
            cost_cfg["savings_score_cap"]
            if pd.notna(x) and x <= cost_cfg["no_penalty_variance_pct"]
            else _linear_lower_is_better(
                x,
                cost_cfg["no_penalty_variance_pct"],
                cost_cfg["zero_score_variance_pct"],
            )
        )
    )
    df["lead_time_consistency_score"] = lead_cv.apply(
        lambda x: _linear_lower_is_better(
            x, consistency_cfg["ideal_cv"], consistency_cfg["zero_score_cv"]
        )
    )

    score_columns = {
        "delivery": "delivery_score",
        "quality": "quality_score",
        "cost": "cost_score",
        "lead_time_consistency": "lead_time_consistency_score",
    }
    weighted_numerator = pd.Series(0.0, index=df.index)
    available_weight = pd.Series(0.0, index=df.index)
    for dimension, column in score_columns.items():
        weight = float(config["weights"][dimension])
        present = df[column].notna()
        weighted_numerator = weighted_numerator + df[column].fillna(0) * weight
        available_weight = available_weight + present.astype(float) * weight
    df["performance_score"] = (weighted_numerator / available_weight.replace(0, np.nan)).round(2)

    min_samples = config["minimum_samples"]
    checks = {
        "delivery": _numeric(df, "evaluable_delivery_count") >= min_samples["delivery_evaluable_pos"],
        "quality": _numeric(df, "inspected_qty") >= min_samples["quality_inspected_units"],
        "cost": _numeric(df, "invoiced_po_count") >= min_samples["cost_invoiced_pos"],
        "lead_time_consistency": _numeric(df, "po_count") >= min_samples["lead_time_observations"],
    }
    for dimension, check in checks.items():
        df[f"{dimension}_sample_sufficient"] = check.fillna(False).astype(bool)
    df["all_dimensions_sample_sufficient"] = pd.concat(
        [df[f"{dimension}_sample_sufficient"] for dimension in checks], axis=1
    ).all(axis=1)

    df["risk_classification"] = df["performance_score"].apply(
        lambda score: _classify(score, config["risk_thresholds"])
    )
    df.loc[~df["all_dimensions_sample_sufficient"], "risk_classification"] = "Insufficient data"

    df["risk_alerts"] = df.apply(lambda row: _alerts(row, config), axis=1)
    df["risk_alert_count"] = df["risk_alerts"].apply(len)
    df["risk_alerts_text"] = df["risk_alerts"].apply(lambda alerts: "|".join(alerts))
    df["scoring_config_version"] = "1.0"
    return df


def save_scorecards(
    scorecards: pd.DataFrame,
    output_dir: Path,
    formats: list[str] | None = None,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    formats = formats or ["csv", "parquet"]
    written = []
    for fmt in formats:
        path = output_dir / f"supplier_scorecards.{fmt}"
        if fmt == "csv":
            export = scorecards.copy()
            for col in export.columns:
                if export[col].apply(lambda value: isinstance(value, list)).any():
                    export[col] = export[col].apply(
                        lambda value: "|".join(value) if isinstance(value, list) else value
                    )
            export.to_csv(path, index=False)
        elif fmt == "parquet":
            export = scorecards.copy()
            export["risk_alerts"] = export["risk_alerts"].apply(lambda alerts: "|".join(alerts))
            export.to_parquet(path, index=False)
        else:
            raise ValueError(f"Unsupported format: {fmt}")
        written.append(path.as_posix())
    return written
