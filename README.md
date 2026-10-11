
# Supplier Performance Intelligence Platform

## Business Objective

Evaluate supplier performance, identify underperforming vendors,
predict future supplier risk, and recommend suitable alternatives.

## Core Capabilities

- Synthetic enterprise procurement data generation
- Data quality validation and feature engineering
- Explainable weighted supplier performance scoring
- Rule-based early-warning alerts
- ML-based prediction of future supplier underperformance
- Constraint-based alternative supplier recommendations
- Interactive supplier performance dashboard
- Optional LLM-generated explanations and corrective-action drafts

## Technology Stack

- Python, NumPy, pandas, Faker
- Parquet and CSV
- scikit-learn
- Pandera and pytest
- FastAPI
- React, TypeScript and Material UI
- Optional LLM provider integration

## Architecture Principles

1. Business calculations remain deterministic and auditable.
2. ML is used only for evaluated predictive tasks.
3. LLMs are used only for language-oriented tasks.
4. Synthetic data is reproducible and referentially consistent.
5. Recommendations respect supplier eligibility constraints.

## Development Status

- [x] Project structure and Python environment
- [ ] Synthetic procurement data generator
- [ ] Data validation and feature engineering
- [ ] Supplier scoring engine
- [ ] Predictive risk model
- [ ] Alternative supplier recommendations
- [ ] FastAPI backend
- [ ] React dashboard
- [ ] Optional LLM integration
- [ ] End-to-end testing
