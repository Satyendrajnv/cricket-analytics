# 🏏 Cricket Analytics & Performance Modeling Engine

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Active Development](https://img.shields.io/badge/Status-Active%20Development-orange.svg)](#)

A modular Python framework for ball-by-ball T20 cricket data processing, phase-wise performance metrics engineering, batter-bowler matchup matrix extraction, and machine learning match outcome win probability modeling.

---

## 🎯 Executive Summary & Strategic Narrative

The **Cricket Analytics Engine** bridges raw sports event data with actionable performance intelligence. Built on 16+ seasons of IPL ball-by-ball match data (2008–2024), this codebase demonstrates how domain-specific feature engineering turns low-level delivery records into tactical decision support.

```text
Cricket Ball-by-Ball Data
          ↓
Phase Segmentation & Feature Engineering
          ↓
Performance & Matchup Analytics
          ↓
Calibrated Win Probability ML Model
          ↓
Interactive Analytics Dashboard
```

---

## 🏗️ Architecture & Component Roadmap

```text
cricket-analytics-dashboard/
├── README.md                          # Production overview & methodology summary
├── LICENSE                            # MIT License
├── .gitignore                         # Exclusions for binary datasets & caches
├── requirements.txt                   # Dependency manifest
├── data/
│   └── README.md                      # Dataset lineage, schema & local path setup
├── docs/
│   └── methodology.md                 # Mathematical specs for metrics & ML models
├── notebooks/
│   └── ipl_analytics_eda.ipynb        # Exploratory analysis & model validation
├── src/
│   └── cricket_analytics/
│       ├── __init__.py                # Package exports
│       ├── data_loader.py             # Ingestion & match phase segmentation
│       ├── metrics.py                 # Batting, Bowling & Matchup metric formulas
│       ├── player_eval.py             # Batter/Bowler profiling & matchup matrix
│       └── win_probability.py         # Win probability ML prediction model
├── dashboard/
│   └── app.py                         # Interactive Streamlit analytics dashboard
├── tests/
│   ├── test_metrics.py                # Unit tests for metrics
│   └── test_models.py                 # Unit tests for ML predictions
└── examples/
    └── run_analytics_pipeline.py      # Executable demonstration script
```

---

## 📊 Analytical Methodology

### 1. Match Phase Segmentation
- **Powerplay ($P_1$)**: Overs 1 to 6 (Field restrictions active).
- **Middle Overs ($P_2$)**: Overs 7 to 15 (Spin & rotation phase).
- **Death Overs ($P_3$)**: Overs 16 to 20 (High-risk boundary hitting & yorker execution).

### 2. Core Metrics
- **Batting**: Strike Rate ($\text{SR}$), Boundary $\%$, Dot Ball $\%$, Phase Average, Pressure Index ($\text{PI}$), Expected Runs ($xR$).
- **Bowling**: Economy Rate ($\text{Econ}$), Dot Ball $\%$, Boundary Conceded $\%$, Death Overs Economy, Bowling Impact Index ($\text{BII}$).
- **Matchups**: Batter $\times$ Bowler head-to-head metrics with minimum sample-size safeguards ($|D_{b,w}| \ge 10$).

---

## 🚀 Quickstart & Setup

### 1. Installation
```bash
git clone https://github.com/Satyendrajnv/cricket-analytics-dashboard.git
cd cricket-analytics-dashboard
pip install -r requirements.txt
```

### 2. Dataset Setup
Download IPL dataset (`matches.csv` and `deliveries.csv`) and place inside `data/`. Refer to [data/README.md](data/README.md) for data schema.

---

## 🛡️ License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.
