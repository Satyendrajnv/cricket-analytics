"""
Unit Tests for Cricket Advanced Performance Metrics
"""

import sys
import os
import unittest
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from cricket_analytics.metrics import BattingMetricsCalculator, BowlingMetricsCalculator
from cricket_analytics.data_loader import CricketDataLoader, MatchPhase


class TestCricketMetrics(unittest.TestCase):
    def test_batting_metrics_formulas(self):
        sr = BattingMetricsCalculator.calculate_strike_rate(runs=50, balls=25)
        self.assertEqual(sr, 200.0)

        bpct = BattingMetricsCalculator.calculate_boundary_percentage(boundaries=5, balls=20)
        self.assertEqual(bpct, 25.0)

        dpct = BattingMetricsCalculator.calculate_dot_percentage(dots=8, balls=20)
        self.assertEqual(dpct, 40.0)

        pi = BattingMetricsCalculator.calculate_pressure_index(dot_pct=40.0, strike_rate=150.0)
        self.assertEqual(pi, 10.0)

    def test_bowling_metrics_formulas(self):
        econ = BowlingMetricsCalculator.calculate_economy_rate(runs_conceded=24, balls_bowled=24)
        self.assertEqual(econ, 6.0)

        bii = BowlingMetricsCalculator.calculate_bowling_impact_index(wickets=2, balls=24, economy=6.0, dot_pct=50.0)
        # (2 * 20 / 24 = 1.67) + (10 - 6 = 4) + (50 / 10 = 5) = 10.67
        self.assertEqual(bii, 10.67)

    def test_phase_classification(self):
        self.assertEqual(CricketDataLoader.classify_phase(3), MatchPhase.POWERPLAY.value)
        self.assertEqual(CricketDataLoader.classify_phase(10), MatchPhase.MIDDLE.value)
        self.assertEqual(CricketDataLoader.classify_phase(18), MatchPhase.DEATH.value)


if __name__ == "__main__":
    unittest.main()
