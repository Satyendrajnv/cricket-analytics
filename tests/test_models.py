"""
Unit Tests for Win Probability Machine Learning Pipeline
"""

import sys
import os
import unittest
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from cricket_analytics.data_loader import CricketDataLoader
from cricket_analytics.win_probability import WinProbabilityModel
from cricket_analytics.player_eval import PlayerEvaluator


class TestWinProbabilityModel(unittest.TestCase):
    def setUp(self):
        loader = CricketDataLoader()
        self.matches_df, self.deliveries_df = loader.load_data()

    def test_feature_construction(self):
        model = WinProbabilityModel()
        feature_df = model.build_features(self.matches_df, self.deliveries_df)
        self.assertFalse(feature_df.empty)
        self.assertIn("runs_required", feature_df.columns)
        self.assertIn("wickets_in_hand", feature_df.columns)

    def test_model_training_and_prediction(self):
        model = WinProbabilityModel(model_type="logistic")
        feature_df = model.build_features(self.matches_df, self.deliveries_df)
        train_res = model.train(feature_df)

        self.assertEqual(train_res["status"], "trained")
        self.assertGreaterEqual(train_res["accuracy"], 0.0)

        prob = model.predict_win_probability(
            runs_required=30,
            balls_remaining=24,
            wickets_in_hand=7,
            current_rr=8.5,
            required_rr=7.5,
        )
        self.assertTrue(0.0 <= prob <= 1.0)

    def test_player_matchup_matrix(self):
        evaluator = PlayerEvaluator(self.deliveries_df)
        batter_name = self.deliveries_df["batter"].iloc[0]
        matchups = evaluator.get_matchup_matrix(batter_name, min_balls=1)
        self.assertIsInstance(matchups, pd.DataFrame)


if __name__ == "__main__":
    unittest.main()
