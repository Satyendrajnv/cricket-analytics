"""
Cricket Advanced Metrics Engineering (Batting & Bowling)
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


class BattingMetricsCalculator:
    """Calculates batting performance metrics and pressure indices."""

    @staticmethod
    def calculate_strike_rate(runs: int, balls: int) -> float:
        """Strike Rate: (Runs / Balls) * 100"""
        if balls == 0:
            return 0.0
        return round((runs / balls) * 100.0, 2)

    @staticmethod
    def calculate_boundary_percentage(boundaries: int, balls: int) -> float:
        """Boundary %: (Boundaries / Balls) * 100"""
        if balls == 0:
            return 0.0
        return round((boundaries / balls) * 100.0, 2)

    @staticmethod
    def calculate_dot_percentage(dots: int, balls: int) -> float:
        """Dot Ball %: (Dots / Balls) * 100"""
        if balls == 0:
            return 0.0
        return round((dots / balls) * 100.0, 2)

    @staticmethod
    def calculate_pressure_index(dot_pct: float, strike_rate: float) -> float:
        """
        Pressure Index (PI) = Dot% * (1 - SR / 200)
        Higher PI indicates batter is under scoring pressure.
        """
        return round(dot_pct * (1.0 - (strike_rate / 200.0)), 2)

    @classmethod
    def calculate_expected_runs(cls, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates Expected Runs (xR) per delivery conditioned on match phase.
        """
        if "phase" not in df.columns:
            return df

        phase_means = df.groupby("phase")["total_runs"].transform("mean")
        df_out = df.copy()
        df_out["expected_runs"] = phase_means.round(2)
        return df_out

    @classmethod
    def summarize_batter(cls, df: pd.DataFrame, batter_name: str) -> Dict[str, Any]:
        """Summarizes complete batting profile for a target batter."""
        b_df = df[df["batter"] == batter_name]
        if b_df.empty:
            return {
                "batter": batter_name,
                "runs": 0,
                "balls": 0,
                "strike_rate": 0.0,
                "boundary_pct": 0.0,
                "dot_pct": 0.0,
                "pressure_index": 0.0,
            }

        runs = int(b_df["batsman_runs"].sum())
        balls = len(b_df)
        boundaries = int((b_df["batsman_runs"] >= 4).sum())
        dots = int((b_df["batsman_runs"] == 0).sum())

        sr = cls.calculate_strike_rate(runs, balls)
        bpct = cls.calculate_boundary_percentage(boundaries, balls)
        dpct = cls.calculate_dot_percentage(dots, balls)
        pi = cls.calculate_pressure_index(dpct, sr)

        return {
            "batter": batter_name,
            "runs": runs,
            "balls": balls,
            "strike_rate": sr,
            "boundary_pct": bpct,
            "dot_pct": dpct,
            "pressure_index": pi,
        }


class BowlingMetricsCalculator:
    """Calculates bowling efficiency, economy, and impact indices."""

    @staticmethod
    def calculate_economy_rate(runs_conceded: int, balls_bowled: int) -> float:
        """Economy Rate: Runs / (Balls / 6)"""
        if balls_bowled == 0:
            return 0.0
        overs = balls_bowled / 6.0
        return round(runs_conceded / overs, 2)

    @staticmethod
    def calculate_bowling_impact_index(wickets: int, balls: int, economy: float, dot_pct: float) -> float:
        """
        Bowling Impact Index (BII) = (Wickets * 20 / Balls) + (10 - Economy) + (Dot% / 10)
        """
        if balls == 0:
            return 0.0
        wicket_component = (wickets * 20.0) / balls
        econ_component = 10.0 - economy
        dot_component = dot_pct / 10.0
        return round(wicket_component + econ_component + dot_component, 2)

    @classmethod
    def summarize_bowler(cls, df: pd.DataFrame, bowler_name: str) -> Dict[str, Any]:
        """Summarizes complete bowling profile for a target bowler."""
        w_df = df[df["bowler"] == bowler_name]
        if w_df.empty:
            return {
                "bowler": bowler_name,
                "wickets": 0,
                "balls": 0,
                "economy": 0.0,
                "dot_pct": 0.0,
                "impact_index": 0.0,
            }

        runs = int(w_df["total_runs"].sum())
        balls = len(w_df)
        wickets = int(w_df["is_wicket"].sum())
        dots = int((w_df["total_runs"] == 0).sum())

        econ = cls.calculate_economy_rate(runs, balls)
        dpct = round((dots / balls) * 100.0, 2) if balls > 0 else 0.0
        bii = cls.calculate_bowling_impact_index(wickets, balls, econ, dpct)

        # Death overs economy
        death_df = w_df[w_df["over"] >= 16]
        death_econ = cls.calculate_economy_rate(
            int(death_df["total_runs"].sum()), len(death_df)
        ) if not death_df.empty else econ

        return {
            "bowler": bowler_name,
            "wickets": wickets,
            "balls": balls,
            "economy": econ,
            "death_economy": death_econ,
            "dot_pct": dpct,
            "impact_index": bii,
        }
