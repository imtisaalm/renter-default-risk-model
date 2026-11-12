import argparse
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ['rent_to_income', 'credit_band', 'cash_flow_stability', 'employment_tenure_months']


def generate(n=5000, seed=42):
    rng = np.random.default_rng(seed)
    income = rng.lognormal(np.log(5500), 0.42, n)
    rent = rng.uniform(800, 3100, n)
    ratio = rent / income
    credit = rng.choice([1, 2, 3, 4], size=n, p=[0.14, 0.28, 0.37, 0.21])
    stability = rng.beta(5, 2, n)
    tenure = np.minimum(rng.exponential(32, n), 180)
    logit = -2.4 + 2.1 * (ratio - 0.35) - 0.48 * (credit - 2) - 2.0 * (stability - 0.65) - 0.012 * tenure
    prob = 1 / (1 + np.exp(-logit))
    default = rng.binomial(1, prob)
    return pd.DataFrame({'application_id': np.arange(1, n+1), 'monthly_income': income.round(2),
        'monthly_rent': rent.round(2), 'rent_to_income': ratio.round(4), 'credit_band': credit,
        'cash_flow_stability': stability.round(4), 'employment_tenure_months': tenure.round(1), 'default_12m': default})


def store(frame, db):
    with sqlite3.connect(db) as conn:
        frame.to_sql('applications', conn, if_exists='replace', index=False)
        conn.execute('CREATE INDEX IF NOT EXISTS idx_credit_band ON applications(credit_band)')


def sql_summary(db):
    query = """SELECT credit_band, COUNT(*) AS applications,
        ROUND(AVG(rent_to_income), 3) AS avg_rent_to_income,
        ROUND(AVG(default_12m), 4) AS default_rate
        FROM applications GROUP BY credit_band ORDER BY credit_band"""
    with sqlite3.connect(db) as conn:
        return pd.read_sql_query(query, conn)


def train_and_evaluate(frame, seed=42):
    X = frame[FEATURES]
    y = frame['default_12m']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
    base = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model = CalibratedClassifierCV(base, method='sigmoid', cv=5)
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = {'auc': float(roc_auc_score(y_test, probabilities)),
               'brier': float(brier_score_loss(y_test, probabilities)),
               'test_rows': len(y_test), 'observed_default_rate': float(y_test.mean())}
    calibration = pd.DataFrame({'predicted': probabilities, 'actual': y_test.to_numpy()})
    calibration['bin'] = pd.qcut(calibration['predicted'], 10, labels=False, duplicates='drop')
    calibration = calibration.groupby('bin').agg(count=('actual','size'), predicted=('predicted','mean'), observed=('actual','mean')).reset_index()
    tradeoff = []
    for cutoff in np.arange(0.05, 0.51, 0.025):
        approved = probabilities <= cutoff
        tradeoff.append({'max_predicted_risk': round(float(cutoff), 3),
                         'approval_rate': float(approved.mean()),
                         'expected_default_rate_approved': float(probabilities[approved].mean()) if approved.any() else np.nan,
                         'observed_default_rate_approved': float(y_test.to_numpy()[approved].mean()) if approved.any() else np.nan,
                         'approved_count': int(approved.sum())})
    return metrics, calibration, pd.DataFrame(tradeoff)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default='renters.sqlite3')
    parser.add_argument('--out', default='output')
    parser.add_argument('--rows', type=int, default=5000)
    args = parser.parse_args()
    if args.rows < 200:
        parser.error('--rows must be at least 200')
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    frame = generate(args.rows)
    store(frame, args.db)
    summary = sql_summary(args.db)
    metrics, calibration, tradeoff = train_and_evaluate(frame)
    summary.to_csv(out / 'sql_summary.csv', index=False)
    calibration.to_csv(out / 'calibration.csv', index=False)
    tradeoff.to_csv(out / 'approval_tradeoff.csv', index=False)
    (out / 'metrics.txt').write_text('\n'.join(f'{k}: {v:.4f}' if isinstance(v,float) else f'{k}: {v}' for k,v in metrics.items()) + '\n')
    print(summary.to_string(index=False))
    print('\n' + (out / 'metrics.txt').read_text())
    print('Wrote CSV tables to', out)

if __name__ == '__main__': main()
