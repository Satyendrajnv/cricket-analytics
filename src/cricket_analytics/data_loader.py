"""
Data Loader & Match Phase Segmentation Engine
"""

import os
from enum import Enum
from typing import Optional, Tuple, Dict, Any
import pandas as pd
import numpy as np


class MatchPhase(Enum):
    POWERPLAY = "Powerplay"  # Overs 1-6
    MIDDLE = "Middle"        # Overs 7-15
    DEATH = "Death"          # Overs 16-20


class CricketDataLoader:
    """
    Ingests, cleans, and segments ball-by-ball T20 match data.
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.matches_path = os.path.join(data_dir, "matches.csv")
        self.deliveries_path = os.path.join(data_dir, "deliveries.csv")

    @staticmethod
    def classify_phase(over: int) -> str:
        """Categorizes over number into T20 match phase."""
        if over <= 6:
            return MatchPhase.POWERPLAY.value
        elif over <= 15:
            return MatchPhase.MIDDLE.value
        else:
            return MatchPhase.DEATH.value

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads matches and deliveries datasets. Falls back to synthetic match signals if CSVs missing.
        """
        if os.path.exists(self.matches_path) and os.path.exists(self.deliveries_path):
            matches_df = pd.read_csv(self.matches_path)
            deliveries_df = pd.read_csv(self.deliveries_path)
        else:
            matches_df, deliveries_df = self._generate_synthetic_dataset()

        # Add match phase column
        if "over" in deliveries_df.columns:
            deliveries_df["phase"] = deliveries_df["over"].apply(self.classify_phase)

        return matches_df, deliveries_df

    def _generate_synthetic_dataset(self, num_matches: int = 50) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generates realistic synthetic IPL ball-by-ball dataset for testing."""
        np.random.seed(42)
        batters = ["Julian Vance", "Kylian Vance", "Arda Vance", "Florian Wirtz", "Rohit Sharma", "Virat Kohli"]
        bowlers = ["Jasprit Bumrah", "Rashid Khan", "Mitchell Starc", "Yuzvendra Chahal", "Sunil Narine"]
        teams = ["Apex XI", "Capital Elite", "Iberia Sports", "Rhine Valley", "Mumbai Indians", "Royal Challengers"]

        matches_list = []
        deliveries_list = []

        for match_id in range(1, num_matches + 1):
            team1, team2 = np.random.choice(teams, size=2, replace=False)
            toss_winner = np.random.choice([team1, team2])
            winner = np.random.choice([team1, team2], p=[0.55, 0.45])

            matches_list.append({
                "id": match_id,
                "season": 2024,
                "city": "Mumbai",
                "date": "2024-04-15",
                "team1": team1,
                "team2": team2,
                "toss_winner": toss_winner,
                "toss_decision": "bat",
                "winner": winner,
                "result": "runs",
                "result_margin": 15,
                "venue": "Wankhede Stadium",
            })

            # Generate 20 overs per inning
            for inning in [1, 2]:
                batting_team = team1 if inning == 1 else team2
                bowling_team = team2 if inning == 1 else team1

                for over in range(1, 21):
                    bowler = np.random.choice(bowlers)
                    for ball in range(1, 7):
                        batter = np.random.choice(batters)
                        runs = int(np.random.choice([0, 1, 2, 4, 6], p=[0.40, 0.35, 0.10, 0.10, 0.05]))
                        is_wicket = int(np.random.choice([0, 1], p=[0.95, 0.05]))

                        deliveries_list.append({
                            "match_id": match_id,
                            "inning": inning,
                            "batting_team": batting_team,
                            "bowling_team": bowling_team,
                            "over": over,
                            "ball": ball,
                            "batter": batter,
                            "bowler": bowler,
                            "non_striker": "Non Striker",
                            "batsman_runs": runs,
                            "extra_runs": 0,
                            "total_runs": runs,
                            "is_wicket": is_wicket,
                            "dismissal_kind": "caught" if is_wicket else np.nan,
                            "player_dismissed": batter if is_wicket else np.nan,
                        })

        return pd.DataFrame(matches_list), pd.DataFrame(deliveries_list)
