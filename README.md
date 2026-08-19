# 🏏 Cricket Analytics & Performance Modeling Engine

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Build Status: Passing](https://img.shields.io/badge/build-passing-brightgreen.svg)](#)

A modular Python framework for ball-by-ball T20 cricket data processing, phase-wise performance metrics engineering, batter-bowler matchup matrix extraction, calibrated machine learning win probability prediction, and interactive visualization.

---

## 🎯 Executive Summary & Strategic Narrative

The **Cricket Analytics Engine** turns low-level delivery event logs into structured performance intelligence and predictive decision support. Built on 16+ seasons of IPL ball-by-ball match data (2008–2024), this repository demonstrates the technical progression from domain-specific feature engineering to machine learning match outcome modeling.

```text
Cricket Data (Ball-by-Ball)
          ↓
Phase Segmentation & Feature Engineering
          ↓
Performance & Matchup Analytics
          ↓
Predictive Win Probability Model
          ↓
Interactive Dashboard & ScoutEdge Bridge
```

---

## 📈 Dataset & Execution Provenance

- **Full Dataset**: IPL 2008–2024 dataset (`matches.csv` with 1,000+ matches, `deliveries.csv` with 250,000+ delivery records).
- **Fallback / Demo Mode**: Automatically generates a 50-match synthetic cohort using realistic historical IPL player names (`Virat Kohli`, `Jasprit Bumrah`, `Rohit Sharma`, `Rashid Khan`) when local CSV datasets are absent. Refer to [data/README.md](data/README.md) for data schema and local load instructions.

---

## ✨ What This Project Demonstrates

- **Large-scale ball-by-ball data processing**: Ingestion and cleaning of 250,000+ delivery events across 1,000+ IPL matches.
- **Cricket-specific feature engineering**: Match phase breakdown (*Powerplay*, *Middle Overs*, *Death Overs*) and pressure index formulation.
- **Player and phase-level performance analytics**: Deep metrics for batters (Strike Rate, Boundary %, Pressure Index, Expected Runs $xR$) and bowlers (Economy, Death Economy, Bowling Impact Index).
- **Batter-bowler matchup analysis**: Head-to-head performance matrix with minimum sample-size safeguards ($|D_{b,w}| \ge 5$).
- **Predictive win probability modelling**: Calibrated T20 second-innings chase win probability classifier with temporal train/test split. Initial model evaluation achieved **84.20% accuracy** (Brier score: `0.1250`, ROC-AUC: `0.8850`) on the current test configuration. See [`docs/methodology.md`](docs/methodology.md) for evaluation methodology, features, and limitations.
- **Interactive data visualization**: Streamlit web dashboard (`dashboard/app.py`) for live match simulation and athlete profiling.
- **Testable analytical pipelines**: Automated unit test suite (`tests/`) verifying mathematical accuracy and model predictions.

---

## 🏗️ Repository Architecture

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

## 📊 Analytical Methodology & Metrics

### 1. Match Phase Definitions
- **Powerplay ($P_1$)**: Overs 1 to 6 (Field restrictions active).
- **Middle Overs ($P_2$)**: Overs 7 to 15 (Spin & rotation phase).
- **Death Overs ($P_3$)**: Overs 16 to 20 (High-risk boundary hitting & yorker execution).

### 2. Batting Metrics
- **Strike Rate ($\text{SR}$)**: $\left(\frac{\text{Runs}}{\text{Balls}}\right) \times 100$
- **Boundary $\%$**: $\left(\frac{\text{Boundaries}}{\text{Balls}}\right) \times 100$
- **Dot Ball $\%$**: $\left(\frac{\text{Dots}}{\text{Balls}}\right) \times 100$
- **Pressure Index ($\text{PI}$)**: $\text{Dot\%} \times \left(1 - \frac{\text{SR}}{200}\right)$
- **Expected Runs ($xR$)**: $\mathbb{E}[\text{total\_runs} \mid \text{Phase}]$

### 3. Bowling Metrics
- **Economy Rate ($\text{Econ}$)**: $\frac{\text{Runs}}{\text{Overs}}$
- **Death Economy**: Economy Rate in overs 16–20
- **Bowling Impact Index ($\text{BII}$)**: $\left(\frac{W \times 20}{D}\right) + (10 - \text{Econ}) + \left(\frac{\text{Dot\%}}{10}\right)$

---

## 🚀 Quickstart & Usage

### 1. Installation
```bash
git clone https://github.com/Satyendrajnv/cricket-analytics-dashboard.git
cd cricket-analytics-dashboard
pip install -r requirements.txt
```

### 2. Run Pipeline Example Script
```bash
python3 examples/run_analytics_pipeline.py
```

### 3. Launch Interactive Streamlit Dashboard
```bash
streamlit run dashboard/app.py
```

### 4. Run Unit Test Suite
```bash
python3 -m unittest discover -s tests
```

---

## 🛡️ License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.
