"""核对领域原语的业务数值。"""
import unittest
from analysis_data.io import load
from analysis_data.domain import summarize
class DomainTests(unittest.TestCase):
    def test_observed_returns(self):
        rows=summarize(load('data/normal.json')[0]);r={x['episode_id']:x for x in rows}
        self.assertAlmostEqual(r['S1-left-right']['observed_return'],2.8)
        self.assertAlmostEqual(r['S1-right-right']['observed_return'],1.8)
        self.assertEqual(r['S1-right-left']['observed_return'],0)
