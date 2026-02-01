"""
regime_vs_noregime_svj.py

Drop this script in the same folder as svj_engine.py and run:
    python regime_vs_noregime_svj.py

What it does:
  - Calibrates SVJ per-asset on training window (START_DATE..END_DATE)
  - Detects regimes (Gaussian HMM on abs(mean returns) proxy)
  - Builds:
       * Baseline (pooled) SVJ params
       * Regime-conditioned SVJ params (per-asset per-regime; fallback to pooled if too few obs)
  - Simulates forward from the first trading day after END_DATE up to BACKTEST_DATE
  - Optimizes CVaR (alpha) for both sets of scenarios
  - Computes CVaR of optimized portfolios, shows bar chart and weight comparison, and plots realized backtest cumulative returns
"""

import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM

# Import functions from your svj engine
from svj_engine import (
    load_price_data,
    compute_log_returns,
    calibrate_svj,
    simulate_multi_svj,
    optimize_cvar,
    compute_cvar
)

# ----------------------------
# USER CONFIG
# ----------------------------
TICKERS = [
    "AAPL",   # Tech - Apple (1980)
    "MSFT",   # Tech - Microsoft (1986)
    "JNJ",    # Healthcare - Johnson & Johnson (1944)
    "JPM",    # Finance - JP Morgan (1980)
    "XOM",    # Energy - Exxon (1970)
    "PG",     # Consumer - Procter & Gamble (1890)
    "KO",     # Consumer - Coca-Cola (1919)
    "DIS",    # Entertainment - Disney (1957)
    "BA",     # Industrial - Boeing (1962)
    "WMT"     # Retail - Walmart (1972)
]   # change as required, must match in yfinance
START_DATE = "2012-01-01"          # training start
END_DATE = "2024-01-31"            # training end (calibration uses data up to this date)
BACKTEST_DATE = "2026-01-31"       # backtest end (simulate from next trading day after END_DATE up to this date)
N_PATHS = 3000                     # total Monte Carlo paths (split by regime probabilities)
ALPHA = 0.95                       # CVaR level
N_REGIMES = 2              # calm / crisis
MIN_POINTS_PER_REGIME = 60         # fallback threshold for per-regime calibration
SEED = 42

np.random.seed(SEED)
warnings.filterwarnings("ignore")

# ----------------------------
# Helper: Regime detection + labeling
# ----------------------------
def detect_regimes(series, n_states=2):
    """
    Fit a Gaussian HMM on absolute values of the input series (vol proxy).
    series: pd.Series indexed by date
    Returns: states (array), last_probabilities (array), hmm_model
    """
    vol_proxy = np.abs(series).values.reshape(-1, 1)
    model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=500)
    model.fit(vol_proxy)
    states = model.predict(vol_proxy)
    probs = model.predict_proba(vol_proxy)
    last_probs = probs[-1]  # probability distribution of most recent day
    return states, last_probs, model

def label_regimes_by_volatility(series, states):
    """
    Label numeric HMM states -> 'calm' and 'crisis' by realized vol on the series.
    Returns mapping dict: {'calm': state_id, 'crisis': state_id}
    """
    vols = {}
    for s in np.unique(states):
        vols[s] = series[states == s].std()
    sorted_states = sorted(vols, key=vols.get)
    return {"calm": sorted_states[0], "crisis": sorted_states[-1]}

# ----------------------------
# Core experiment functions
# ----------------------------
def calibrate_pooled_params(returns_df):
    """Calibrate SVJ per-asset on pooled training returns (returns_df: DataFrame columns=tickers)"""
    params_list = []
    for t in returns_df.columns:
        s = returns_df[t].dropna()
        if len(s) < 30:
            raise ValueError(f"Not enough data to calibrate asset {t}; need >30 points")
        params_list.append(calibrate_svj(s))
    return params_list

def calibrate_regime_params(returns_df, states, regime_id_map):
    """
    Calibrate SVJ parameters per asset per regime.
    returns_df: DataFrame (T x n_assets)
    states: array of length T mapping each date to HMM state
    regime_id_map: {'calm': id, 'crisis': id}
    Returns a dict: { regime_name: params_list_for_assets }
    """
    results = {}
    for name, sid in regime_id_map.items():
        params_list = []
        print(f"Calibrating params for regime '{name}' (state id={sid})")
        for t in returns_df.columns:
            series_r = returns_df[t][states == sid].dropna()
            if len(series_r) < MIN_POINTS_PER_REGIME:
                # fallback to pooled (we'll calibrate pooled separately and reuse)
                params_list.append(None)
                print(f"  - asset {t}: insufficient obs in regime ({len(series_r)}), will use pooled fallback")
            else:
                params_list.append(calibrate_svj(series_r))
        results[name] = params_list
    return results

def assemble_regime_params_or_fallback(regime_params_dict, pooled_params):
    """
    Replace None entries in regime params with pooled_params for that asset.
    """
    fixed = {}
    for rname, plist in regime_params_dict.items():
        fixed_list = []
        for i, p in enumerate(plist):
            if p is None:
                fixed_list.append(pooled_params[i])
            else:
                fixed_list.append(p)
        fixed[rname] = fixed_list
    return fixed

def simulate_baseline(S0_vec, pooled_params, corr_matrix, T, n_paths):
    """Simulate using pooled params"""
    _, _, terminal_returns = simulate_multi_svj(
        S0_vec=S0_vec,
        params_list=pooled_params,
        corr_matrix=corr_matrix,
        T=T,
        n_paths=n_paths,
        seed=SEED
    )
    return terminal_returns  # shape (n_paths, n_assets)

def simulate_regime_mixture(S0_vec, regime_params_fixed, corr_matrix, T, n_paths, regime_probs):
    """
    For each regime, simulate #paths = round(n_paths * regime_prob).
    Stack terminal returns from all regimes and return combined array.
    regime_params_fixed: dict { 'calm': [params...], 'crisis': [params...] }
    regime_probs: array aligned with HMM state ids (we will map using regime_id_map)
    """
    parts = []
    # Map regime names to their probability using order of regime_params_fixed
    # regime_params_fixed keys correspond to names; we need the HMM state's id ordering to map prob -> name.
    # Instead the caller will pass regime_probs_by_name (dict name->prob). We'll implement that.
    raise NotImplementedError("This function is replaced by simulate_regime_mixture_by_name below.")


def simulate_regime_mixture_by_name(S0_vec, regime_params_fixed_by_name, corr_matrix, T, n_paths, regime_probs_by_name):
    """
    regime_params_fixed_by_name: dict { 'calm': params_list, 'crisis': params_list }
    regime_probs_by_name: dict { 'calm': p_cal, 'crisis': p_cri }
    """
    parts = []
    total_assigned = 0
    for i, (rname, params_list) in enumerate(regime_params_fixed_by_name.items()):
        p = float(regime_probs_by_name.get(rname, 0.0))
        n = int(round(n_paths * p))
        # Ensure at least a few paths
        if n < 10:
            # allow small regimes but ensure sum == n_paths at the end
            n = min(max(n, 10), n_paths - total_assigned) if (n_paths - total_assigned) > 0 else 0
        total_assigned += n
        if n <= 0:
            continue
        _, _, terminal_returns = simulate_multi_svj(
            S0_vec=S0_vec,
            params_list=params_list,
            corr_matrix=corr_matrix,
            T=T,
            n_paths=n,
            seed=SEED + i  # different seed per regime block
        )
        parts.append(terminal_returns)
        print(f"Simulated {n} paths for regime '{rname}'")
    # If rounding left some unassigned, simulate them using pooled of last regime
    remaining = n_paths - sum([p.shape[0] for p in parts]) if parts else n_paths
    if remaining > 0:
        # Use 'calm' if exists else first regime
        fallback_name = list(regime_params_fixed_by_name.keys())[0]
        print(f"Rounding leftover paths: simulating {remaining} fallback paths using regime '{fallback_name}' params")
        _, _, terminal_returns = simulate_multi_svj(
            S0_vec=S0_vec,
            params_list=regime_params_fixed_by_name[fallback_name],
            corr_matrix=corr_matrix,
            T=T,
            n_paths=remaining,
            seed=SEED + 999
        )
        parts.append(terminal_returns)
    combined = np.vstack(parts)
    return combined  # shape (M_combined, n_assets)

# ----------------------------
# Main experiment orchestration
# ----------------------------
def main():
    print("Loading training prices...")
    prices_train = load_price_data(TICKERS, START_DATE, END_DATE)
    returns_train = compute_log_returns(prices_train)  # DataFrame (T x n_assets)
    print(f"Training data rows: {returns_train.shape[0]}, assets: {returns_train.shape[1]}")

    # ---------- Regime detection ----------
    print("Detecting regimes (HMM on abs(mean returns) ) ...")
    vol_series = returns_train.mean(axis=1)  # aggregate proxy
    states, last_state_probs, hmm_model = detect_regimes(vol_series, n_states=N_REGIMES)
    regime_id_map = label_regimes_by_volatility(vol_series, states)
    # convert last_state_probs (which is array indexed by numeric state id) to dict name->prob
    # e.g. if regime_id_map={'calm':0,'crisis':1} then regime_probs_by_name['calm']=last_state_probs[0]
    regime_probs_by_name = { name: float(last_state_probs[sid]) for name, sid in regime_id_map.items() }

    print("Detected Regimes:", regime_id_map)
    print("Latest regime probabilities (by name):", regime_probs_by_name)

    # ---------- Calibrate pooled params (baseline) ----------
    print("Calibrating pooled (baseline) SVJ per asset ...")
    pooled_params = calibrate_pooled_params(returns_train)

    # ---------- Calibrate regime params (per asset per regime) ----------
    regime_params_raw = calibrate_regime_params(returns_train, states, regime_id_map)
    regime_params_fixed = assemble_regime_params_or_fallback(regime_params_raw, pooled_params)

    # ---------- Prepare forward/backtest period and S0 ---------
    # Next trading day after END_DATE -> use as sim start; we will simulate until BACKTEST_DATE inclusive
    start_for_forward = pd.to_datetime(END_DATE) + pd.Timedelta(days=1)
    end_for_forward = pd.to_datetime(BACKTEST_DATE)
    if end_for_forward < start_for_forward:
        raise ValueError("BACKTEST_DATE must be after END_DATE + 1 day")

    # Download actual market data between start_for_forward and BACKTEST_DATE (inclusive)
    df_forward = yf.download(TICKERS, start=start_for_forward, end=end_for_forward + pd.Timedelta(days=1), auto_adjust=True, progress=False)
    # yfinance returns multiindex (Open/Close/...), we want Close
    if isinstance(df_forward.columns, pd.MultiIndex):
        prices_forward = df_forward["Close"].dropna()
    else:
        prices_forward = df_forward.to_frame(name=TICKERS[0]).dropna()  # single ticker case fallback

    if prices_forward.empty:
        raise ValueError("No market data found between the next trading day after END_DATE and BACKTEST_DATE. Check dates or connectivity.")

    # S0 vector = first available row in prices_forward
    S0_vec = prices_forward.iloc[0].values
    # Horizon in days for simulation = number of trading steps from S0 to backtest date
    T = prices_forward.shape[0] - 1
    if T <= 0:
        raise ValueError("Forward/backtest window too short (no trading days).")

    print(f"Simulating forward from {prices_forward.index[0].date()} to {prices_forward.index[-1].date()} -> T = {T} trading days")
    print("S0_vec (first forward prices):", S0_vec)

    # Correlation matrix from training returns (pooled)
    corr_matrix = returns_train.corr().values

    # ---------- Baseline simulation & optimization ----------
    print("Simulating baseline pooled SVJ ...")
    terminal_returns_baseline = simulate_baseline(S0_vec, pooled_params, corr_matrix, T=T, n_paths=N_PATHS)
    w_baseline = optimize_cvar(terminal_returns_baseline, alpha=ALPHA)
    portfolio_returns_baseline = terminal_returns_baseline @ w_baseline  # terminal portfolio log returns across simulated paths
    cvar_baseline = compute_cvar(portfolio_returns_baseline, alpha=ALPHA)
    print(f"Baseline: CVaR (alpha={ALPHA}) = {cvar_baseline:.6f}")

    # ---------- Regime-aware simulation & optimization ----------
    print("Simulating regime-aware SVJ (mixture) ...")
    # regime_params_fixed is name->params_list; ensure regime_probs_by_name contains same keys
    terminal_returns_regime = simulate_regime_mixture_by_name(
        S0_vec=S0_vec,
        regime_params_fixed_by_name=regime_params_fixed,
        corr_matrix=corr_matrix,
        T=T,
        n_paths=N_PATHS,
        regime_probs_by_name=regime_probs_by_name
    )
    w_regime = optimize_cvar(terminal_returns_regime, alpha=ALPHA)
    portfolio_returns_regime = terminal_returns_regime @ w_regime
    cvar_regime = compute_cvar(portfolio_returns_regime, alpha=ALPHA)
    print(f"Regime-aware: CVaR (alpha={ALPHA}) = {cvar_regime:.6f}")

    # ---------- Print & Plot CVaR comparison ----------
    print("\n--- CVaR comparison ---")
    print(f"Baseline CVaR: {cvar_baseline:.6f}")
    print(f"Regime CVaR:   {cvar_regime:.6f}")

    # Bar plot for CVaR
    plt.figure(figsize=(6,4))
    plt.bar(["Baseline", "Regime"], [cvar_baseline, cvar_regime], color=["#1f77b4", "#ff7f0e"])
    plt.title(f"Portfolio CVaR (alpha={ALPHA})")
    plt.ylabel("CVaR (log-return)")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ---------- Weight comparison bar chart (side-by-side) ----------
    idx = np.arange(len(TICKERS))
    width = 0.35
    plt.figure(figsize=(10,5))
    plt.bar(idx - width/2, w_baseline, width, label="Baseline")
    plt.bar(idx + width/2, w_regime, width, label="Regime-aware")
    plt.xticks(idx, TICKERS)
    plt.ylabel("Weights")
    plt.title("Optimized Portfolio Weights: Baseline vs Regime-aware")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()

    # ---------- Backtest realized performance (apply weights to actual forward returns) ----------
    actual_log_rets = compute_log_returns(prices_forward)  # shape (T, n_assets); first row corresponds to return from S0 to next day
    # Align columns/tickers if needed
    actual_log_rets = actual_log_rets[TICKERS]  # ensure column order
    # daily portfolio returns
    daily_port_baseline = actual_log_rets.dot(w_baseline)
    daily_port_regime = actual_log_rets.dot(w_regime)
    cum_baseline = np.exp(np.cumsum(daily_port_baseline)) - 1.0
    cum_regime = np.exp(np.cumsum(daily_port_regime)) - 1.0

    plt.figure(figsize=(10,5))
    plt.plot(actual_log_rets.index[1:], np.exp(np.cumsum(daily_port_baseline))[1:], label="Baseline")
    plt.plot(actual_log_rets.index[1:], np.exp(np.cumsum(daily_port_regime))[1:], label="Regime-aware")
    plt.title("Backtest: Cumulative Portfolio Growth (starting at 1.0)")
    plt.ylabel("Portfolio value (relative)")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()

    # Final realized cumulative returns (log)
    realized_log_ret_baseline = np.log(prices_forward.iloc[-1].values / prices_forward.iloc[0].values) @ w_baseline
    realized_log_ret_regime = np.log(prices_forward.iloc[-1].values / prices_forward.iloc[0].values) @ w_regime
    print("\n--- Realized backtest results ---")
    print(f"Baseline realized cumulative return (log) : {realized_log_ret_baseline:.6f}   => pct {(np.exp(realized_log_ret_baseline)-1)*100:.2f}%")
    print(f"Regime realized cumulative return (log)   : {realized_log_ret_regime:.6f}   => pct {(np.exp(realized_log_ret_regime)-1)*100:.2f}%")

    # Print weights
    print("\nPortfolio weights (Baseline):")
    for t, w in zip(TICKERS, w_baseline):
        print(f"  {t}: {w:.4f}")
    print("\nPortfolio weights (Regime-aware):")
    for t, w in zip(TICKERS, w_regime):
        print(f"  {t}: {w:.4f}")

if __name__ == "__main__":
    main()
