"""
Interactive Cricket Analytics & Performance Streamlit Dashboard
"""

import sys
import os
import streamlit as st
import pandas as pd
import numpy as np

# Ensure src/ package is on PATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from cricket_analytics.data_loader import CricketDataLoader
from cricket_analytics.metrics import BattingMetricsCalculator, BowlingMetricsCalculator
from cricket_analytics.player_eval import PlayerEvaluator
from cricket_analytics.win_probability import WinProbabilityModel

st.set_page_config(
    page_title="Cricket Analytics Dashboard",
    page_icon="🏏",
    layout="wide",
)

st.title("🏏 Cricket Analytics & Win Probability Engine")
st.caption("Ball-by-ball T20 match data processing, phase-wise performance metrics, and ML win probability modeling.")

@st.cache_data
def load_dataset():
    loader = CricketDataLoader(data_dir="data")
    return loader.load_data()

matches_df, deliveries_df = load_dataset()

# Sidebar Navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Select Module",
    ["Overview & Dataset", "Batter Profiling", "Bowler Profiling", "Matchup Matrix", "Win Probability Model"],
)

if page == "Overview & Dataset":
    st.header("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Matches", len(matches_df))
    col2.metric("Total Deliveries", f"{len(deliveries_df):,}")
    col3.metric("Unique Batters", deliveries_df["batter"].nunique())

    st.subheader("Sample Deliveries Signal Stream")
    st.dataframe(deliveries_df.head(10), use_container_width=True)

elif page == "Batter Profiling":
    st.header("Batter Performance & Phase Breakdown")
    all_batters = sorted(deliveries_df["batter"].unique().tolist())
    selected_batter = st.selectbox("Select Batter", all_batters)

    summary = BattingMetricsCalculator.summarize_batter(deliveries_df, selected_batter)
    evaluator = PlayerEvaluator(deliveries_df)
    phase_df = evaluator.get_phase_batting_breakdown(selected_batter)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Runs", summary["runs"])
    col2.metric("Strike Rate", summary["strike_rate"])
    col3.metric("Boundary %", f"{summary['boundary_pct']}%")
    col4.metric("Pressure Index", summary["pressure_index"])

    st.subheader("Phase Breakdown (Powerplay vs Middle vs Death)")
    if not phase_df.empty:
        st.dataframe(phase_df, use_container_width=True)
        st.bar_chart(phase_df.set_index("phase")["strike_rate"])

elif page == "Bowler Profiling":
    st.header("Bowler Performance & Economy Analysis")
    all_bowlers = sorted(deliveries_df["bowler"].unique().tolist())
    selected_bowler = st.selectbox("Select Bowler", all_bowlers)

    summary = BowlingMetricsCalculator.summarize_bowler(deliveries_df, selected_bowler)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Wickets", summary["wickets"])
    col2.metric("Economy Rate", summary["economy"])
    col3.metric("Death Economy", summary["death_economy"])
    col4.metric("Impact Index", summary["impact_index"])

elif page == "Matchup Matrix":
    st.header("Batter vs Bowler Head-to-Head Matrix")
    all_batters = sorted(deliveries_df["batter"].unique().tolist())
    selected_batter = st.selectbox("Select Batter for Matchup Analysis", all_batters)

    evaluator = PlayerEvaluator(deliveries_df)
    matchup_df = evaluator.get_matchup_matrix(selected_batter, min_balls=3)

    if not matchup_df.empty:
        st.dataframe(matchup_df, use_container_width=True)
    else:
        st.info("No matchup data meeting minimum ball threshold.")

elif page == "Win Probability Model":
    st.header("Live Second-Innings Win Probability Predictor")

    st.subheader("Simulate Chase Context")
    col1, col2, col3 = st.columns(3)
    runs_req = col1.number_input("Runs Required", min_value=1, max_value=250, value=45)
    balls_rem = col2.number_input("Balls Remaining", min_value=1, max_value=120, value=30)
    wickets_left = col3.number_input("Wickets in Hand", min_value=1, max_value=10, value=6)

    crr = round((160 - runs_req) / max(1, (120 - balls_rem) / 6), 2)
    rrr = round(runs_req / max(1, balls_rem / 6), 2)

    st.caption(f"Current Run Rate: **{crr}** | Required Run Rate: **{rrr}**")

    wp_model = WinProbabilityModel(model_type="logistic")
    feature_df = wp_model.build_features(matches_df, deliveries_df)
    train_res = wp_model.train(feature_df)

    if train_res["status"] == "trained":
        st.success(f"Model Trained on {train_res['train_samples']} deliveries | Test Accuracy: {train_res['accuracy'] * 100:.1f}%")

    prob = wp_model.predict_win_probability(runs_req, balls_rem, wickets_left, crr, rrr)

    st.subheader("Predicted Win Probability")
    st.progress(prob)
    st.metric("Batting Team Win Probability", f"{prob * 100:.1f}%")
