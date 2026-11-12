# Renter default-risk model

A Python and SQLite example of a basic underwriting analysis. It makes up a portfolio of 5,000 renters, queries it with SQL, and fits a calibrated logistic regression to estimate a made-up 12-month default outcome. No real renter records are used.

## How to run

Python 3.10 or later.

```bash
python -m pip install -r requirements.txt
python risk_model.py --db renters.sqlite3 --out output
python -m unittest discover -s tests -v
```

The script creates a SQLite database and writes four files to `output/`: `sql_summary.csv`, `calibration.csv`, `approval_tradeoff.csv`, and `metrics.txt`. Open the database with `sqlite3 renters.sqlite3` and run `.read queries.sql` for extra SQL breakdowns.

## The features

- `rent_to_income`: monthly rent divided by monthly income. A larger value means a heavier rent burden.
- `credit_band`: a made-up credit group from 1 (weakest) to 4 (strongest). Not a real credit score.
- `cash_flow_stability`: a made-up score from 0 to 1. Higher means steadier cash flow.
- `employment_tenure_months`: months in current employment.
- `default_12m`: a simulated yes or no for default within 12 months.

The outcome is sampled from a known formula using these same features, so the model is expected to find the pattern that was planted. Good results here say nothing about real-world accuracy.

## Calibration

A stratified 75/25 train/test split holds back a quarter of the rows for evaluation. The training rows fit a standardized logistic regression with five-fold sigmoid calibration. The test rows are only used for reporting. `metrics.txt` has ROC AUC (ranking quality) and Brier score (squared probability error; lower is better). On the example data: ROC AUC 0.6525, Brier score 0.0484. `calibration.csv` groups held-out predictions into ten roughly equal groups and compares average predicted risk with the actual default rate. With small bins the observed rates can be noisy.

## Approval and loss tradeoff

`approval_tradeoff.csv` varies the highest predicted default probability allowed for approval. A higher threshold approves more applications, including more higher-risk ones. The file reports the approval share, the mean predicted default probability among approved rows, and the observed default rate among approved rows. The expected default rate is a proxy, not expected dollar loss. To estimate dollar loss you would also need exposure and loss-given-default assumptions. The demo does not model pricing, selection effects, protected-class fairness, or policy constraints.

The SQL queries help check the shape of the portfolio before fitting the model. This is a teaching example, not a production underwriting or approval system.
