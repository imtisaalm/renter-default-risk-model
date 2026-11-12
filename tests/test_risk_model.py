import tempfile
import unittest
from pathlib import Path
from risk_model import generate, store, sql_summary, train_and_evaluate

class RiskTests(unittest.TestCase):
    def test_reproducible_data(self):
        a, b = generate(300), generate(300)
        self.assertTrue(a.equals(b))
        self.assertTrue(a['default_12m'].isin([0,1]).all())
    def test_sql_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test.sqlite3'
            store(generate(300), path)
            self.assertEqual(int(sql_summary(path)['applications'].sum()), 300)
    def test_tradeoff(self):
        metrics, calibration, tradeoff = train_and_evaluate(generate(1200))
        self.assertTrue(0 <= metrics['auc'] <= 1)
        self.assertTrue(tradeoff['approval_rate'].is_monotonic_increasing)
        self.assertTrue((calibration['predicted'].between(0,1)).all())

if __name__ == '__main__': unittest.main()
