import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import cvxpy as cp
from scipy.optimize import minimize

# ============================================================
# DATA LOADING
# ============================================================
N_steps=252
def load_price_data(tickers, start_date: str, end_date: str):
    """Load historical price data from Yahoo Finance."""
    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False
    )
    
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data.to_frame(name=tickers[0])
    
    return prices.dropna()

def get_current_prices(tickers, end_date: str):
    """
    Get initial prices for simulation:
    first trading day strictly AFTER end_date.
    """
    end_date = pd.to_datetime(end_date)

    prices = yf.download(
        tickers,
        start=end_date,
        auto_adjust=True,
        progress=False
    )["Close"]

    # Ensure strictly after END_DATE
    prices = prices.loc[prices.index > end_date]

    if prices.empty:
        raise ValueError("No trading day found after END_DATE")

    if isinstance(prices, pd.DataFrame):
        return prices.iloc[0].values
    else:
        return np.array([prices.iloc[0]])

# ============================================================
# RETURNS & STATS
# ============================================================

def compute_log_returns(prices):
    """Compute log returns from price series."""
    return np.log(prices / prices.shift(1)).dropna()

def compute_basic_stats(log_returns, trading_days=252):
    """Compute basic statistics from log returns."""
    return {
        "mean_annual": log_returns.mean() * trading_days,
        "cov_annual": log_returns.cov() * trading_days,
        "corr": log_returns.corr()
    }

# ============================================================
# JUMP DETECTION & CALIBRATION
# ============================================================

def detect_jumps(log_returns, threshold_std=2.5):
    """
    Detect significant jumps in returns (outliers > 2.5 standard deviations).
    Returns: (jump_dates, jump_sizes, jump_count, jump_percentage)
    """
    returns = log_returns.values.flatten()
    std = np.std(returns)
    jump_threshold = threshold_std * std
    
    # Identify jumps
    jump_mask = np.abs(returns) > jump_threshold
    jump_dates = log_returns.index[jump_mask]
    jump_sizes = returns[jump_mask]
    
    jump_count = len(jump_sizes)
    jump_percentage = (jump_count / len(returns)) * 100
    
    return jump_dates, jump_sizes, jump_count, jump_percentage

def calibrate_svj(log_returns, trading_days=252, jump_threshold_std=2.5,variance_scale=0.3,jump_scale=0.6):
    """
    Calibrate SVJ (Stochastic Volatility with Jumps) parameters.
    
    Parameters
    ----------
    log_returns : pd.Series or pd.DataFrame
        Log returns of asset(s)
    trading_days : int
        Number of trading days per year
    jump_threshold_std : float
        Number of standard deviations for jump detection
    
    Returns
    -------
    dict : SVJ parameters {'mu', 'kappa', 'theta', 'sigma', 'rho', 'v0',
                           'lambda_j', 'mu_j', 'sigma_j'}
    """
    
    # Step 1: Detect jumps
    returns = log_returns.values.flatten()
    std = np.std(returns)
    jump_threshold = jump_threshold_std * std
    
    jump_mask = np.abs(returns) > jump_threshold
    jump_sizes = returns[jump_mask]
    
    # Jump parameters
    n_jumps = len(jump_sizes)
    lambda_j = n_jumps / (len(returns) / trading_days)*jump_scale # Jump intensity per year
    mu_j = (np.mean(jump_sizes) if n_jumps > 0 else -0.03)*jump_scale
    sigma_j =(np.std(jump_sizes) if n_jumps > 0 else 0.08)*jump_scale
    
    print(f"    Jump Statistics:")
    print(f"    Detected jumps: {n_jumps}")
    print(f"    Jump frequency (per year): {lambda_j:.2f}")
    print(f"    Mean jump size: {mu_j:.4f}")
    print(f"    Jump volatility: {sigma_j:.4f}")
    
    # Step 2: Remove jumps and calibrate Heston on diffusion component
    filtered_returns = returns[~jump_mask]
    filtered_series = pd.Series(filtered_returns, 
                                index=log_returns.index[~jump_mask])
    
    # Compute variance of non-jump returns
    mu = filtered_series.mean() * trading_days
    
    # Realized variance using rolling window
    window = 21  # 21-day window
    rv = filtered_series.rolling(window).var().dropna()
    
    v = rv.values
    v_t, v_tp1 = v[:-1], v[1:]
    
    dt = 1.0 / trading_days
    y = (v_tp1 - v_t) / dt
    
    X = np.column_stack([np.ones(len(v_t)), -v_t])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    
    kappa = beta[1]
    theta = beta[0] / (kappa + 1e-8)
    
    residuals = y - X @ beta
    sigma = np.std(residuals) * np.sqrt(dt)
    
    # Correlation between returns and variance changes
    aligned_returns = filtered_returns[:len(v) - 1]
    vol_changes = np.diff(v)
    
    rho = np.corrcoef(aligned_returns, vol_changes)[0, 1]
    if np.isnan(rho):
        rho = 0.0
    
    v0 = max(rv.iloc[-1], 0.0001) * variance_scale  # Scale down
    theta = max(theta, 0.0001) * variance_scale     # Scale down
    sigma = sigma * np.sqrt(variance_scale)      # Initial variance floor
    
    # Ensure Feller condition: 2*kappa*theta > sigma^2
    if 2 * kappa * theta <= sigma**2:
        sigma = np.sqrt(2 * kappa * theta) * 0.9
    
    return {
        "mu": float(mu.iloc[0]) if isinstance(mu, pd.Series) else float(mu),
        "kappa": float(max(kappa, 0.1)),
        "theta": float(max(theta, 0.0001)),
        "sigma": float(sigma),
        "rho": float(np.clip(rho, -0.99, 0.99)),
        "v0": float(v0),
        "lambda_j": float(lambda_j),
        "mu_j": float(mu_j),
        "sigma_j": float(max(sigma_j, 0.01))
    }

def return_svj_params(tickers):
    """Calibrate SVJ parameters for multiple tickers."""
    params = {}
    for ticker in tickers:
        prices = load_price_data([ticker], "2020-01-01", "2025-12-31")
        log_returns = compute_log_returns(prices[ticker])
        params[ticker] = calibrate_svj(log_returns)
    return params

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def _make_pos_def_corr(corr, eps=1e-8):
    """Ensure correlation matrix is positive definite."""
    corr = np.array(corr, dtype=float)
    try:
        np.linalg.cholesky(corr)
        return corr
    except np.linalg.LinAlgError:
        jitter = eps
        for _ in range(50):
            corr_j = corr.copy()
            np.fill_diagonal(corr_j, np.diag(corr_j) + jitter)
            try:
                np.linalg.cholesky(corr_j)
                return corr_j
            except np.linalg.LinAlgError:
                jitter *= 10
        raise

# ============================================================
# SVJ MULTI-ASSET SIMULATION
# ============================================================

def simulate_multi_svj(
    S0_vec,
    params_list,
    corr_matrix,
    T=60,
    N=N_steps,
    n_paths=10000,
    trading_days=252,
    seed=42,
    risk_neutral=True
):
    """
    Simulate multi-asset SVJ (Stochastic Volatility with Jumps) paths.
    
    Parameters
    ----------
    S0_vec : array-like
        Initial stock prices
    params_list : list of dict
        SVJ parameters for each asset
    corr_matrix : ndarray
        Correlation matrix
    T : int
        Time horizon in days
    n_paths : int
        Number of Monte Carlo paths
    trading_days : int
        Trading days per year (default 252)
    seed : int
        Random seed
    risk_neutral : bool
        If True, use zero drift (risk-neutral pricing)
    
    Returns
    -------
    prices : ndarray (n_paths, T+1, n_assets)
        Simulated price paths
    variances : ndarray (n_paths, T+1, n_assets)
        Simulated variance paths
    terminal_returns : ndarray (n_paths, n_assets)
        Terminal log returns
    """
    
    np.random.seed(seed)
    S0_vec = np.asarray(S0_vec)
    
    
    # Extract parameters
    if risk_neutral:
        mus = np.zeros(N)  # Zero drift for risk-neutral
    else:
        mus = np.array([p["mu"] for p in params_list])
    
    kappas = np.array([p["kappa"] for p in params_list])
    thetas = np.array([p["theta"] for p in params_list])
    sigmas = np.array([p["sigma"] for p in params_list])
    rhos = np.array([p["rho"] for p in params_list])
    v0s = np.array([p["v0"] for p in params_list])
    
    # Jump parameters
    lambdas = np.array([p["lambda_j"] for p in params_list])
    mu_js = np.array([p["mu_j"] for p in params_list])
    sigma_js = np.array([p["sigma_j"] for p in params_list])
    
    # Ensure positive definite correlation
    corr_matrix = _make_pos_def_corr(corr_matrix)
    L = np.linalg.cholesky(corr_matrix)
    
    dt = 1.0 / trading_days
    
    # Initialize arrays
    prices = np.zeros((n_paths, T + 1, N))
    variances = np.zeros((n_paths, T + 1, N))
    
    prices[:, 0, :] = S0_vec
    variances[:, 0, :] = v0s
    
    # Simulation loop
    for t in range(T):
        # Generate correlated shocks for price process
        U = np.random.randn(n_paths, N)
        Z1 = U @ L.T  # Correlated price shocks
        
        # Independent shocks for variance and jumps
        eps = np.random.randn(n_paths, N)
        Z2 = rhos * Z1 + np.sqrt(1 - rhos**2) * eps  # Correlated variance shocks
        
        # Jump arrivals (Poisson process)
        jump_arrivals = np.random.poisson(lambdas * dt, (n_paths, N))
        
        # Jump sizes
        jump_sizes = np.random.randn(n_paths, N)
        
        # Current variance (ensure positive)
        vt = np.maximum(variances[:, t, :], 1e-6)
        sqrt_vt_dt = np.sqrt(vt * dt)
        
        # VARIANCE PROCESS (Heston part)
        variances[:, t + 1, :] = np.maximum(
            vt + kappas * (thetas - vt) * dt +
            sigmas * sqrt_vt_dt * Z2,
            0
        )
        variances[:, t + 1, :] = np.minimum(variances[:, t + 1, :], 0.05)
        
        diffusion = (mus - 0.5 * vt) * dt + sqrt_vt_dt * Z1
        
        # Jump component
        jump_contribution = np.zeros((n_paths, N))
        for i in range(N):
            # Add jump for each arrival
            has_jump = jump_arrivals[:, i] > 0
            jump_contribution[has_jump,i]= (
                mu_js[i] + sigma_js[i] * jump_sizes[has_jump, i]
            )
        
        prices[:, t + 1, :] = prices[:, t, :] * np.exp(diffusion + jump_contribution)
    
    # Compute terminal returns (log returns)
    terminal_prices = prices[:, -1, :]
    terminal_returns = np.log(terminal_prices / S0_vec)
    
    return prices, variances, terminal_returns

# ============================================================
# SINGLE ASSET SVJ SIMULATION
# ============================================================

def simulate_svj(
    S0,
    params,
    T=1.0,
    N=252,
    M=5000,
    seed=None,
    risk_neutral=True
):
    """
    Simulate single-asset SVJ price paths.
    
    Parameters
    ----------
    S0 : float
        Current price
    params : dict
        SVJ parameters
    T : float
        Time horizon in years
    N : int
        Number of time steps
    M : int
        Number of Monte Carlo paths
    seed : int or None
        Random seed
    risk_neutral : bool
        If True, use zero drift
    
    Returns
    -------
    S : ndarray (N+1, M)
        Simulated price paths
    """
    
    if seed is not None:
        np.random.seed(seed)
    
    dt = T / N
    
    # Extract parameters
    mu = 0.0 if risk_neutral else params['mu']
    kappa = params['kappa']
    theta = params['theta']
    sigma = params['sigma']
    rho = params['rho']
    v0 = params['v0']
    
    # Jump parameters
    lambda_j = params['lambda_j']
    mu_j = params['mu_j']
    sigma_j = params['sigma_j']
    
    # Initialize arrays
    S = np.zeros((N + 1, M))
    v = np.zeros((N + 1, M))
    
    S[0, :] = S0
    v[0, :] = v0
    
    # Random numbers
    Z1 = np.random.randn(N, M)  # Price shocks
    Z2 = np.random.randn(N, M)  # Variance shocks
    
    # Jump process
    jump_arrivals = np.random.poisson(lambda_j * dt, (N, M))
    jump_sizes = np.random.randn(N, M)
    
    # Simulation
    for t in range(N):
        v_t = np.maximum(v[t], 0)
        
        # Variance process
        v[t + 1] = np.maximum(
            v_t + kappa * (theta - v_t) * dt +
            sigma * np.sqrt(v_t * dt) * Z2[t],
            0
        )
        
        # Correlated shocks
        W = rho * Z2[t] + np.sqrt(1 - rho**2) * Z1[t]
        
        # Jump contribution
        jump_contribution = (
            mu_j * jump_arrivals[t] +
            sigma_j * np.sqrt(np.maximum(jump_arrivals[t], 0)) * jump_sizes[t]
        )
        
        # Price process
        S[t + 1] = S[t] * np.exp(
            (mu - 0.5 * v_t) * dt +
            np.sqrt(v_t * dt) * W +
            jump_contribution
        )
    
    return S

# ============================================================
# PORTFOLIO SCENARIOS
# ============================================================

def build_portfolio_scenarios(simulated_paths_dict):
    """
    Build scenario returns from simulated paths.
    
    Parameters
    ----------
    simulated_paths_dict : dict
        {ticker: price_paths (T+1, M)}
    
    Returns
    -------
    scenario_returns : ndarray (M, n_assets)
        Returns for each scenario
    """
    scenario_returns = []
    
    for ticker, paths in simulated_paths_dict.items():
        log_rets = np.diff(np.log(paths), axis=0)
        cumulative_rets = log_rets.sum(axis=0)  # Sum over time
        scenario_returns.append(cumulative_rets)
    
    return np.column_stack(scenario_returns)

# ============================================================
# CVaR OPTIMIZATION (CORRECTED)
# ============================================================

def optimize_cvar(returns, alpha=0.95, max_weight=None):
    """
    Optimize portfolio to minimize CVaR (Conditional Value at Risk).
    
    Parameters
    ----------
    returns : ndarray (M, n)
        Scenario returns (M scenarios, n assets)
    alpha : float
        Confidence level (default 0.95)
    max_weight : float or None
        Maximum weight per asset (default None = no limit)
    
    Returns
    -------
    weights : ndarray (n,)
        Optimal portfolio weights
    """
    M, n = returns.shape
    
    # Decision variables
    w = cp.Variable(n)      # Portfolio weights
    z = cp.Variable()       # VaR threshold
    u = cp.Variable(M)      # Auxiliary variables for tail losses
    
    # Convert returns to losses (negative returns)
    losses = -returns
    
    # Constraints
    constraints = [
        cp.sum(w) == 1,                    # Fully invested
        w >= 0,                            # Long-only
        u >= 0,                            # Non-negative auxiliary
        u >= losses @ w - z                # CVaR constraint
    ]
    
    # Add maximum weight constraint if specified
    if max_weight is not None:
        constraints.append(w <= max_weight)
    
    # Objective: Minimize CVaR
    objective = cp.Minimize(z + (1 / ((1 - alpha) * M)) * cp.sum(u))
    
    problem = cp.Problem(objective, constraints)
    
    # Solve
    try:
        problem.solve(solver=cp.ECOS, verbose=False)
    except:
        try:
            problem.solve(solver=cp.SCS, verbose=False)
        except:
            print(f"⚠️ Solver failed. Reverting to equal weights.")
            return np.ones(n) / n
    
    # Check solution status
    if w.value is None or problem.status in ["unbounded", "infeasible"]:
        print(f"⚠️ Solver Status: {problem.status}. Reverting to equal weights.")
        return np.ones(n) / n
    
    # Clean up numerical errors
    weights = np.array(w.value).flatten()
    weights = np.maximum(weights, 0)  # Remove tiny negatives
    weights = weights / weights.sum()  # Renormalize
    
    return weights

def compute_cvar(portfolio_returns, alpha=0.95):
    """
    Compute Conditional Value at Risk.
    
    Parameters
    ----------
    portfolio_returns : ndarray
        Portfolio returns across scenarios
    alpha : float
        Confidence level
    
    Returns
    -------
    cvar : float
        CVaR value (expected loss in worst alpha% of cases)
    """
    var = np.percentile(portfolio_returns, (1 - alpha) * 100)
    return portfolio_returns[portfolio_returns <= var].mean()

# ============================================================
# VISUALIZATION
# ============================================================

def plot_sample_paths(prices, tickers, asset_idx=0, n_paths=40):
    """Plot sample price paths."""
    plt.figure(figsize=(10, 6))
    for i in range(n_paths):
        plt.plot(prices[i, :, asset_idx], alpha=0.4, linewidth=0.8)
    plt.title(f"SVJ Paths — {tickers[asset_idx]}")
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_fan_chart(prices, tickers, asset_idx=0):
    """Plot fan chart with percentile bands."""
    qs = np.percentile(prices[:, :, asset_idx], [5, 25, 50, 75, 95], axis=0)
    
    plt.figure(figsize=(10, 6))
    plt.fill_between(range(len(qs[0])), qs[0], qs[4], alpha=0.2, label='5-95%')
    plt.fill_between(range(len(qs[0])), qs[1], qs[3], alpha=0.3, label='25-75%')
    plt.plot(qs[2], color='black', lw=2, label='Median')
    
    plt.title(f"Fan Chart — {tickers[asset_idx]}")
    plt.xlabel("Days")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_return_distribution(returns, title="Return Distribution"):
    """Plot histogram of returns."""
    plt.figure(figsize=(10, 6))
    plt.hist(returns * 100, bins=100, alpha=0.7, edgecolor='black')
    plt.xlabel("Return (%)")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
