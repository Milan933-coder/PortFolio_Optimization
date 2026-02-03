# 🚀 GARCH–LSTM vs SVJ  
### Advanced Portfolio Simulation & CVaR Optimization Framework

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Finance](https://img.shields.io/badge/Domain-Quant%20Finance-green)
![ML](https://img.shields.io/badge/ML-LSTM%20%7C%20GARCH-orange)
![Risk](https://img.shields.io/badge/Risk-CVaR-red)
![Status](https://img.shields.io/badge/Status-Research--Grade-purple)

---

## 📌 Overview

This repository implements a **robust, research-grade comparison** between two state-of-the-art **financial scenario generation pipelines** for **portfolio risk optimization**:

- **📉 Stochastic Volatility with Jumps (SVJ)**
- **📉 Rgime Update with SVJ**
- **📈 GARCH–LSTM Hybrid Model with EVT Tail Smoothing**
- **📈 GARCH–SSM Hybrid Model with EVT Tail Smoothing Needs Futher Improvement and  Ablation Studies on the Block Size **

The goal is to **generate realistic multi-asset return scenarios**, optimize portfolios under **CVaR constraints**, and evaluate **out-of-sample performance**.

> ⚠️ Designed for **quant research, risk modeling, and agentic AI financial systems**

---

## 🧠 Key Concepts Covered

- Stochastic Volatility (Heston-style)
- Jump Diffusion Models
- GARCH(1,1) Volatility Modeling
- Extreme Value Theory (EVT / POT)
- LSTM Mean Dynamics
- Copula-based Dependence
- Monte Carlo Simulation
- CVaR Portfolio Optimization
- Backtesting & Risk Attribution

---

## 📂 Repository Structure

```text
.
├── GARCH_LSTM_SVJ.py   # Main experiment pipeline
├── svj_engine.py       # SVJ calibration & simulation engine
├── metric.py           # contains different objective(CVaR,Sharpe,C-Sharpe)
└── README.md
