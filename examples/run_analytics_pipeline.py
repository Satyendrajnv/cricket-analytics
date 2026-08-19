"""
Executable End-to-End Cricket Analytics Pipeline Demonstration
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from cricket_analytics.data_loader import CricketDataLoader
from cricket_analytics.metrics import BattingMetricsCalculator, BowlingMetricsCalculator
from cricket_analytics.player_eval import PlayerEvaluator
from cricket_analytics.win_probability import WinProbabilityModel


def main():
    print("=" * 70)
    print("🏏 SCOUTEDGE BRIDGING PORTFOLIO: CRICKET ANALYTICS PIPELINE")
    print("=" * 70)

    # 1. Ingestion & Phase Classification
    print("\n[1/4] Ingesting IPL Match Data & Segmenting Phases...")
    loader = CricketDataLoader(data_dir="data")
    matches_df, deliveries_df = loader.load_data()
    print(f"      ✓ Loaded {len(matches_df)} matches and {len(deliveries_df):,} delivery records.")

    # 2. Player Metrics & Profile Breakdown
    sample_batter = deliveries_df["batter"].iloc[0]
    sample_bowler = deliveries_df["bowler"].iloc[0]

    print(f"\n[2/4] Summarizing Batter Profile: '{sample_batter}'...")
    b_summary = BattingMetricsCalculator.summarize_batter(deliveries_df, sample_batter)
    print(f"      Runs: {b_summary['runs']} | SR: {b_summary['strike_rate']} | Boundary %: {b_summary['boundary_pct']}% | Pressure Index: {b_summary['pressure_index']}")

    print(f"\n[3/4] Summarizing Bowler Profile: '{sample_bowler}'...")
    w_summary = BowlingMetricsCalculator.summarize_bowler(deliveries_df, sample_bowler)
    print(f"      Wickets: {w_summary['wickets']} | Econ: {w_summary['economy']} | Death Econ: {w_summary['death_economy']} | Impact Index: {w_summary['impact_index']}")

    # 3. Head-to-Head Matchup Matrix
    evaluator = PlayerEvaluator(deliveries_df)
    matchup_df = evaluator.get_matchup_matrix(sample_batter, min_balls=1)
    print(f"\n[3.5] Matchup Matrix for '{sample_batter}':")
    if not matchup_df.empty:
        print(matchup_df.head(5).to_string(index=False))

    # 4. Machine Learning Win Probability Predictor
    print("\n[4/4] Training Second-Innings Win Probability Model...")
    model = WinProbabilityModel(model_type="logistic")
    features_df = model.build_features(matches_df, deliveries_df)
    train_res = model.train(features_df)
    print(f"      ✓ Model Trained ({train_res['model_type']}) | Test Accuracy: {train_res['accuracy'] * 100:.2f}%")

    # Simulate chase context: 40 runs needed off 24 balls with 6 wickets in hand
    sim_prob = model.predict_win_probability(
        runs_required=40,
        balls_remaining=24,
        wickets_in_hand=6,
        current_rr=8.0,
        required_rr=10.0,
    )
    print(f"\n🎯 Chase Context Simulation:")
    print(f"   Runs Required: 40 | Balls Remaining: 24 | Wickets Left: 6")
    print(f"   Predicted Win Probability: {sim_prob * 100:.1f}%\n")
    print("=" * 70)


if __name__ == "__main__":
    main()
