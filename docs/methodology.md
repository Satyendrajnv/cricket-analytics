# Cricket Analytics Methodology & Mathematical Specifications

This document outlines the analytical formulas, phase definitions, and machine learning methodology implemented in the **Cricket Analytics Infrastructure**.

---

## 1. Match Phase Definitions

In modern T20 cricket, match dynamics shift significantly across overs. The pipeline segments ball-by-ball deliveries into three distinct phases:

- **Powerplay** ($P_1$): Overs $1.1$ to $6.0$ (Overs index $1\text{--}6$). Field restrictions in place.
- **Middle Overs** ($P_2$): Overs $6.1$ to $15.0$ (Overs index $7\text{--}15$). Spin and control-oriented bowling strategies.
- **Death Overs** ($P_3$): Overs $15.1$ to $20.0$ (Overs index $16\text{--}20$). High-risk boundary hitting and yorker execution.

---

## 2. Batting Metrics Engineering

Let $D_b$ be the set of legal deliveries faced by batter $b$, $R_b$ be the total runs scored off the bat, $W_b$ be total dismissals, $B_b$ be boundary deliveries ($\text{runs} \ge 4$), and $Z_b$ be dot ball deliveries ($\text{runs} = 0$).

### 2.1 Strike Rate ($\text{SR}$)
$$\text{SR}_b = \left( \frac{\sum R_b}{|D_b|} \right) \times 100$$

### 2.2 Boundary Percentage ($\text{BP}$)
$$\text{BP}_b = \left( \frac{|B_b|}{|D_b|} \right) \times 100$$

### 2.3 Dot Ball Percentage ($\text{DP}$)
$$\text{DP}_b = \left( \frac{|Z_b|}{|D_b|} \right) \times 100$$

### 2.4 Pressure Index ($\text{PI}$)
Quantifies pressure built on the batter due to dot ball density relative to recent scoring rates:
$$\text{PI}_b = \text{DP}_b \times \left(1 - \frac{\text{SR}_b}{200}\right)$$

### 2.5 Phase-Context Expected Runs ($xR$)
Expected runs per delivery conditioned on match phase $p \in \{P_1, P_2, P_3\}$ and wicket context $w$:
$$xR(p, w) = \mathbb{E}[\text{runs} \mid \text{Phase} = p, \text{Wickets Lost} = w]$$

---

## 3. Bowling Metrics Engineering

Let $D_w$ be legal deliveries bowled by bowler $w$, $R_w$ be total runs conceded (excluding leg-byes/byes), $W_w$ be bowler wickets taken, $Z_w$ be dot balls bowled, and $B_w$ be boundaries conceded.

### 3.1 Economy Rate ($\text{Econ}$)
$$\text{Econ}_w = \frac{\sum R_w}{|D_w| / 6}$$

### 3.2 Dot Ball Percentage ($\text{DP}_w$)
$$\text{DP}_w = \left( \frac{|Z_w|}{|D_w|} \right) \times 100$$

### 3.3 Bowling Impact Index ($\text{BII}$)
$$\text{BII}_w = \left(\frac{W_w \times 20}{|D_w|}\right) + \left(10 - \text{Econ}_w\right) + \left(\frac{|Z_w|}{|D_w|} \times 10\right)$$

---

## 4. Head-to-Head Matchup Matrix

For any Batter $b$ and Bowler $w$ with sample size $|D_{b,w}| \ge 10$ legal deliveries:
$$\text{Matchup SR}_{b,w} = \left( \frac{\sum R_{b,w}}{|D_{b,w}|} \right) \times 100$$
$$\text{Dismissal Frequency}_{b,w} = \frac{|W_{b,w}|}{|D_{b,w}|}$$

---

## 5. Win Probability Machine Learning Pipeline

### 5.1 Problem Framing
Predict the probability $P(\text{Team}_1 \text{ Wins} \mid S_t)$ at any ball state $S_t = (\text{Runs Required}, \text{Balls Remaining}, \text{Wickets in Hand}, \text{Current Run Rate}, \text{Required Run Rate}, \text{Venue Score Factor})$.

### 5.2 Model Architecture & Calibration
1. **Temporal Train/Test Split**: Training on historical seasons (2008–2021); testing on held-out recent seasons (2022–2024) to eliminate temporal data leakage.
2. **Classifiers**: Logistic Regression (L2 regularized baseline) and Random Forest Classifier.
3. **Probability Calibration**: Calibrated via Isotonic Regression / Sigmoid scaling to ensure predicted probabilities match empirical win rates.
