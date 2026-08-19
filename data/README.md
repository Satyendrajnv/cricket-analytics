# Cricket Analytics Dataset Provenance & Schema

This directory documents the data structure and schema expected by the **Cricket Analytics Engine**. Raw data files (`deliveries.csv` and `matches.csv`) are excluded from Git repository tracking via `.gitignore` to maintain repository lightweight integrity.

---

## Dataset Provenance & Acquisition

- **Source**: Indian Premier League (IPL) Ball-by-Ball Dataset (2008–2024).
- **Format**: CSV (Comma Separated Values).
- **Expected Files**:
  1. `data/matches.csv` (~225 KB, 1,000+ match records).
  2. `data/deliveries.csv` (~27 MB, 250,000+ ball-by-ball delivery records).

---

## Data Schema Reference

### 1. `matches.csv` Schema

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `id` | Integer | Unique match identifier |
| `season` | String / Integer | Season year (e.g. `2024`) |
| `city` | String | Match host city |
| `date` | String (YYYY-MM-DD) | Match date |
| `match_type` | String | Match format classification (T20) |
| `team1` | String | First team |
| `team2` | String | Second team |
| `toss_winner` | String | Team winning the toss |
| `toss_decision` | String | Decision (`bat` or `field`) |
| `winner` | String | Match winning team |
| `result` | String | Win type (`runs`, `wickets`, `tie`, `no result`) |
| `result_margin` | Float | Win margin quantity |
| `target_runs` | Float / Integer | Innings 2 run target |
| `venue` | String | Stadium venue name |

### 2. `deliveries.csv` Schema

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `match_id` | Integer | Foreign key matching `matches.csv(id)` |
| `inning` | Integer | Innings number (1 or 2) |
| `batting_team` | String | Team currently batting |
| `bowling_team` | String | Team currently bowling |
| `over` | Integer | Over number (1 to 20) |
| `ball` | Integer | Ball number within over (1 to 6) |
| `batter` | String | Striker batter name |
| `bowler` | String | Bowler name |
| `non_striker` | String | Non-striker batter name |
| `batsman_runs` | Integer | Runs scored directly off bat |
| `extra_runs` | Integer | Extra runs conceded |
| `total_runs` | Integer | Total runs on ball (`batsman_runs` + `extra_runs`) |
| `is_wicket` | Integer | Binary flag (1 if wicket fell, 0 otherwise) |
| `dismissal_kind` | String | Type of wicket (`caught`, `bowled`, `lbw`, `run out`, etc.) |
| `player_dismissed` | String | Name of dismissed batter |

---

## How to Load Data Locally

Place `matches.csv` and `deliveries.csv` directly inside the `data/` directory:

```bash
cricket-analytics-dashboard/
└── data/
    ├── README.md
    ├── matches.csv
    └── deliveries.csv
```

The pipeline automatically falls back to synthetic match signal generators if CSV data files are absent.
