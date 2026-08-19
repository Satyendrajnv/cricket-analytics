"""
Player Profiling & Head-to-Head Matchup Matrix Evaluator
"""

from typing import Dict, Any, List, Optional
import pandas as pd
from .metrics import BattingMetricsCalculator, BowlingMetricsCalculator


class PlayerEvaluator:
    """
    Evaluates individual player phase breakdowns and Batter vs Bowler head-to-head matchups.
    """

    def __init__(self, deliveries_df: pd.DataFrame):
        self.df = deliveries_df

    def get_phase_batting_breakdown(self, batter_name: str) -> pd.DataFrame:
        """Returns phase-wise batting breakdown (Powerplay, Middle, Death)."""
        b_df = self.df[self.df["batter"] == batter_name]
        if b_df.empty:
            return pd.DataFrame()

        records = []
        for phase in ["Powerplay", "Middle", "Death"]:
            phase_df = b_df[b_df["phase"] == phase]
            if phase_df.empty:
                continue
            runs = int(phase_df["batsman_runs"].sum())
            balls = len(phase_df)
            boundaries = int((phase_df["batsman_runs"] >= 4).sum())
            dots = int((phase_df["batsman_runs"] == 0).sum())
            sr = BattingMetricsCalculator.calculate_strike_rate(runs, balls)

            records.append({
                "phase": phase,
                "runs": runs,
                "balls": balls,
                "strike_rate": sr,
                "boundaries": boundaries,
                "dots": dots,
            })

        return pd.DataFrame(records)

    def get_matchup_matrix(self, batter_name: str, min_balls: int = 5) -> pd.DataFrame:
        """
        Computes Batter vs Bowler head-to-head matchup metrics with sample-size safeguards.
        """
        b_df = self.df[self.df["batter"] == batter_name]
        if b_df.empty:
            return pd.DataFrame()

        matchups = []
        for bowler, group in b_df.groupby("bowler"):
            balls = len(group)
            if balls < min_balls:
                continue

            runs = int(group["batsman_runs"].sum())
            dismissals = int(group["is_wicket"].sum())
            dots = int((group["batsman_runs"] == 0).sum())
            sr = BattingMetricsCalculator.calculate_strike_rate(runs, balls)

            matchups.append({
                "bowler": bowler,
                "balls_faced": balls,
                "runs_scored": runs,
                "dismissals": dismissals,
                "dot_balls": dots,
                "strike_rate": sr,
                "average": round(runs / max(1, dismissals), 2),
            })

        result_df = pd.DataFrame(matchups)
        if not result_df.empty:
            result_df = result_df.sort_values(by="balls_faced", ascending=False).reset_index(drop=True)
        return result_df
