"""
Portfolio Optimization Comparison: SVJ vs GARCH-LSTM
- Train on historical data (START_DATE to END_DATE)
- Simulate from next trading day after END_DATE
- Track actual performance to current date
- Compare which method performs better
"""
import cvxpy as cp
import numpy as np


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.preprocessing import StandardScaler
from arch import arch_model
from datetime import datetime, timedelta
import yfinance as yf
import warnings
from scipy import stats
from metric import optimize_c_sharpe,optimize_portfolio_multi_objective,optimize_robust_cvar015f9d5ae9588f6211ea3a2e
warnings.filterwarnings('ignore')

from svj_engine import (
    load_price_data, compute_log_returns, calibrate_svj,
    optimize_cvar, compute_cvar
)

# ============================================================
# CONFIGURATION
# ============================================================

# Diversified stocks (all IPO'd before 2020)
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
]

START_DATE = "2016-01-01"
END_DATE = "2025-01-01"
HORIZON_DAYS = 60
N_PATHS = 5000
LOOKBACK = 60

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

print("=" * 80)
print("PORTFOLIO OPTIMIZATION: SVJ vs GARCH-LSTM COMPARISON")
print("=" * 80)
print(f"\nTickers: {', '.join(TICKERS)}")
print(f"Training Period: {START_DATE} to {END_DATE}")
print(f"Simulation Horizon: {HORIZON_DAYS} days")
print(f"Monte Carlo Paths: {N_PATHS}")
print(f"\n" + "=" * 80)

# ============================================================
# STEP 1: LOAD TRAINING DATA
# ============================================================

print("\nStep 1: Loading historical data...")
print("-" * 80)

prices_train = load_price_data(TICKERS, START_DATE, END_DATE)
returns_train = pd.DataFrame({t: compute_log_returns(prices_train[t]) for t in TICKERS})

# Get initial prices (first trading day AFTER END_DATE)
end_date_dt = pd.to_datetime(END_DATE)
prices_after = yf.download(TICKERS, start=END_DATE, progress=False)["Close"]
prices_after = prices_after[prices_after.index > end_date_dt]

if prices_after.empty:
    raise ValueError(f"No trading day found after {END_DATE}")

S0_vec = prices_after.iloc[0].values  # Initial prices for simulation
simulation_start_date = prices_after.index[0].strftime("%Y-%m-%d")

print(f"✓ Training data: {len(returns_train)} days")
print(f"✓ Simulation starts: {simulation_start_date}")
print(f"✓ Initial prices (${simulation_start_date}):")
for ticker, price in zip(TICKERS, S0_vec):
    print(f"    {ticker}: ${price:.2f}")

# Correlation matrix
corr_matrix = returns_train.corr().values

# ============================================================
# STEP 2: METHOD 1 - SVJ (Stochastic Volatility + Jumps)
# ============================================================

print("\n" + "=" * 80)
print("METHOD 1: SVJ (Stochastic Volatility with Jumps)")
print("=" * 80)

svj_params = {}
for ticker in TICKERS:
    log_rets = compute_log_returns(prices_train[ticker])
    svj_params[ticker] = calibrate_svj(
        log_rets, 
        jump_threshold_std=3.0,
        variance_scale=0.5, 
        jump_scale=0.6
    )
    print(f"  ✓ {ticker} calibrated")

params_list_svj = [svj_params[t] for t in TICKERS]

# Fix the simulate_multi_svj call
print("\nSimulating SVJ paths...")

# Manual SVJ simulation with correct dimensions
def simulate_svj_fixed(S0_vec, params_list, corr_matrix, T, n_paths, seed=42,risk_neutral=False):
    """Fixed SVJ simulation"""
    np.random.seed(seed)
    
    n_assets = len(S0_vec)
    S0_vec = np.array(S0_vec)
    
    # Extract parameters
    if risk_neutral:
        # In a risk-neutral world, the expected log return drift is often set to 0
        # or (r - 0.5*sigma^2 - lambda*k). Here we simplify to 0 to measure pure risk.
        mus = np.zeros(n_assets)
    else:
        mus = np.array([p["mu"] for p in params_list])
    kappas = np.array([p["kappa"] for p in params_list])
    thetas = np.array([p["theta"] for p in params_list])
    sigmas = np.array([p["sigma"] for p in params_list])
    rhos = np.array([p["rho"] for p in params_list])
    v0s = np.array([p["v0"] for p in params_list])
    lambdas = np.array([p["lambda_j"] for p in params_list])
    mu_js = np.array([p["mu_j"] for p in params_list])
    sigma_js = np.array([p["sigma_j"] for p in params_list])
    
    # Cholesky decomposition
    L = np.linalg.cholesky(corr_matrix)
    
    dt = 1.0 / 252
    
    # Initialize
    prices = np.zeros((n_paths, T + 1, n_assets))
    variances = np.zeros((n_paths, T + 1, n_assets))
    prices[:, 0, :] = S0_vec
    variances[:, 0, :] = v0s
    
    # Simulate
    for t in range(T):
        # Correlated shocks
        U = np.random.randn(n_paths, n_assets)
        Z1 = U @ L.T
        
        eps = np.random.randn(n_paths, n_assets)
        Z2 = rhos * Z1 + np.sqrt(1 - rhos**2) * eps
        
        # Jump arrivals
        jump_arrivals = np.random.poisson(lambdas * dt, (n_paths, n_assets))
        jump_sizes = np.random.randn(n_paths, n_assets)
        
        # Current variance
        vt = np.maximum(variances[:, t, :], 1e-6)
        sqrt_vt_dt = np.sqrt(vt * dt)
        
        # Variance process
        variances[:, t + 1, :] = np.maximum(
            vt + kappas * (thetas - vt) * dt + sigmas * sqrt_vt_dt * Z2,
            0
        )
        variances[:, t + 1, :] = np.minimum(variances[:, t + 1, :], 0.05)
        
        # Price process
        diffusion = (mus - 0.5 * vt) * dt + sqrt_vt_dt * Z1
        
        # Jump contribution
        jump_contribution = np.zeros((n_paths, n_assets))
        for i in range(n_assets):
            has_jump = jump_arrivals[:, i] > 0
            jump_contribution[has_jump, i] = (
                mu_js[i] + sigma_js[i] * jump_sizes[has_jump, i]
            )
        
        prices[:, t + 1, :] = prices[:, t, :] * np.exp(diffusion + jump_contribution)
    
    # Terminal returns
    terminal_returns = np.log(prices[:, -1, :] / S0_vec)
    
    return prices, terminal_returns

_, svj_terminal_rets = simulate_svj_fixed(
    S0_vec, params_list_svj, corr_matrix, HORIZON_DAYS, N_PATHS, seed=42,risk_neutral=True
)

svj_weights = optimize_robust_cvar(svj_terminal_rets, svj_terminal_rets.mean(axis=0), epsilon=0.015, alpha=0.95, max_weight=0.40)
svj_portfolio_rets = svj_terminal_rets @ svj_weights

print(f"✓ SVJ Portfolio optimized")
print(f"  Expected Return: {svj_portfolio_rets.mean():.2%}")
print(f"  CVaR (95%): {compute_cvar(svj_portfolio_rets):.2%}")

# ============================================================
# STEP 3: METHOD 2 - GARCH-LSTM
# ============================================================

print("\n" + "=" * 80)
print("METHOD 2: GARCH-LSTM (Hybrid Deep Learning)")
print("=" * 80)
def apply_evt_smoothing(residuals, threshold_q=0.95):
    """
    Stabilizes residuals by replacing extreme outliers with GPD samples.
    Add this to your GARCH_LSTM_SVJ.py script.
    """
    from scipy import stats
    threshold = np.quantile(np.abs(residuals), threshold_q)
    extreme_mask = np.abs(residuals) > threshold
    
    if np.any(extreme_mask):
        exceedances = np.abs(residuals[extreme_mask]) - threshold
        # Fit GPD to the exceedances
        shape, loc, scale = stats.genpareto.fit(exceedances)
        # Generate 'realistic' extreme samples
        gpd_samples = stats.genpareto.rvs(shape, loc=loc, scale=scale, size=np.sum(extreme_mask))
        
        smoothed = residuals.copy()
        smoothed[extreme_mask] = np.sign(residuals[extreme_mask]) * (threshold + gpd_samples)
        return smoothed
    return residuals
class LSTMMean(nn.Module):
    def __init__(self, input_size=1, hidden_size=50):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def simulate_garch_lstm_joint(returns, corr_matrix, S0_vec, lookback, horizon, n_paths):
    """
    Stabilized GARCH-EVT-LSTM Simulator with Integrated Training.
    """
    n_assets = returns.shape[1]
    garch_results = []
    lstm_models = [] 
    scalers = []
    all_stable_residuals = []
    
    # --- PHASE 1: ASSET-BY-ASSET TRAINING & SMOOTHING ---
    for i in range(n_assets):
        series = returns.iloc[:, i]
        print(f"Processing {TICKERS[i]}:", end=" ")
        
        # 1. GARCH & EVT Smoothing
        res = arch_model(series * 100, vol="Garch", p=1, q=1).fit(disp="off", show_warning=False)
        garch_results.append(res)
        
        # Apply POT/GPD to tame COVID residuals
        smoothed_z = apply_evt_smoothing(res.std_resid)
        all_stable_residuals.append(np.sort(smoothed_z))
        
        # 2. LSTM Training Logic
        scaler = StandardScaler().fit(series.values.reshape(-1, 1))
        scalers.append(scaler)
        scaled_series = scaler.transform(series.values.reshape(-1, 1))
        
        # Prepare Tensors
        X_train, y_train = [], []
        for j in range(lookback, len(scaled_series)):
            X_train.append(scaled_series[j-lookback:j, 0])
            y_train.append(scaled_series[j, 0])
            
        X_train = torch.tensor(np.array(X_train), dtype=torch.float32).unsqueeze(-1).to(DEVICE)
        y_train = torch.tensor(np.array(y_train), dtype=torch.float32).unsqueeze(-1).to(DEVICE)
        
        model = LSTMMean().to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = torch.nn.MSELoss()
        
        # Training Loop
        for epoch in range(25):
            model.train()
            optimizer.zero_grad()
            loss = loss_fn(model(X_train), y_train)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
        model.eval()
        lstm_models.append(model)
        print(f"GARCH+EVT✓ LSTM Trained (Loss: {loss.item():.5f})✓")

    # --- PHASE 2: VECTORIZED INITIALIZATION ---
    # Use Copula-based tail dependence via Cholesky
    L = np.linalg.cholesky(corr_matrix)
    shocks = (np.random.normal(size=(horizon, n_paths, n_assets)) @ L.T)
    u_shocks = stats.norm.cdf(shocks) 
    
    current_sigmas = np.tile(
        np.array([np.sqrt(res.conditional_volatility[-1] / 10000) for res in garch_results]),
        (n_paths, 1)
    )
    
    all_sequences = []
    for i in range(n_assets):
        seq = scalers[i].transform(returns.iloc[-lookback:, i].values.reshape(-1, 1))
        seq_batch = torch.tensor(seq, dtype=torch.float32).repeat(n_paths, 1, 1).to(DEVICE)
        all_sequences.append(seq_batch)

    sim_returns = np.zeros((n_paths, horizon, n_assets))
    eps_prev = np.zeros((n_paths, n_assets))

    # --- PHASE 3: VECTORIZED SIMULATION ---
    with torch.no_grad():
        for t in range(horizon):
            for i in range(n_assets):
                # Batch Prediction for speed
                mu_scaled = lstm_models[i](all_sequences[i]).cpu().numpy()
                mu = scalers[i].inverse_transform(mu_scaled).flatten()
                
                # GARCH Update
                p = garch_results[i].params
                var_t = (p["omega"]/10000) + (p["alpha[1]"] * eps_prev[:, i]**2) + (p["beta[1]"] * current_sigmas[:, i]**2)
                current_sigmas[:, i] = np.sqrt(np.clip(var_t, 1e-6, 0.01))
                
                # EVT Sampling
                u = u_shocks[t, :, i]
                indices = (u * (len(all_stable_residuals[i]) - 1)).astype(int)
                z_stable = all_stable_residuals[i][indices]
                
                # Combine & Step
                r = mu + current_sigmas[:, i] * z_stable
                sim_returns[:, t, i] = r
                eps_prev[:, i] = current_sigmas[:, i] * z_stable 
                
                # Update window for next step
                next_scaled = (r - scalers[i].mean_[0]) / scalers[i].scale_[0]
                next_scaled_tensor = torch.tensor(next_scaled, dtype=torch.float32).view(-1, 1, 1).to(DEVICE)
                all_sequences[i] = torch.cat([all_sequences[i][:, 1:, :], next_scaled_tensor], dim=1)

    return sim_returns.sum(axis=1)

gl_terminal_rets = simulate_garch_lstm_joint(
    returns_train, corr_matrix, S0_vec, LOOKBACK, HORIZON_DAYS, N_PATHS
)

gl_weights=optimize_robust_cvar(gl_terminal_rets, gl_terminal_rets.mean(axis=0), epsilon=0.015, alpha=0.95, max_weight=0.40)
gl_portfolio_rets = gl_terminal_rets @ gl_weights

print(f"✓ GARCH-LSTM Portfolio optimized")
print(f"  Expected Return: {gl_portfolio_rets.mean():.2%}")
print(f"  CVaR (95%): {compute_cvar(gl_portfolio_rets):.2%}")

# ============================================================
# STEP 4: BACKTEST - ACTUAL PERFORMANCE
# ============================================================

print("\n" + "=" * 80)
print("BACKTEST: Actual Performance from Simulation Start to Today")
print("=" * 80)

# Load actual prices from simulation start to today
today = datetime.now().strftime("%Y-%m-%d")
prices_test = yf.download(TICKERS, start=simulation_start_date, end=today, progress=False)["Close"]

if len(prices_test) > 1:
    # Compute actual returns
    returns_test = pd.DataFrame({t: compute_log_returns(prices_test[t]) for t in TICKERS})
    
    # Portfolio performance
    svj_daily_rets = (returns_test * svj_weights).sum(axis=1)
    gl_daily_rets = (returns_test * gl_weights).sum(axis=1)
    
    # Cumulative returns
    svj_total_return = (1 + svj_daily_rets).prod() - 1
    gl_total_return = (1 + gl_daily_rets).prod() - 1
    
    # Sharpe ratio
    svj_sharpe = (svj_daily_rets.mean() / svj_daily_rets.std()) * np.sqrt(252) if svj_daily_rets.std() > 0 else 0
    gl_sharpe = (gl_daily_rets.mean() / gl_daily_rets.std()) * np.sqrt(252) if gl_daily_rets.std() > 0 else 0
    
    # Max drawdown
    svj_cum = (1 + svj_daily_rets).cumprod()
    gl_cum = (1 + gl_daily_rets).cumprod()
    svj_dd = (svj_cum / svj_cum.cummax() - 1).min()
    gl_dd = (gl_cum / gl_cum.cummax() - 1).min()
    
    print(f"\nTest Period: {simulation_start_date} to {today}")
    print(f"Trading Days: {len(returns_test)}")
    print(f"\nSVJ Portfolio:")
    print(f"  Total Return:     {svj_total_return:+.2%}")
    print(f"  Annualized:       {((1 + svj_total_return)**(252/len(returns_test)) - 1):+.2%}")
    print(f"  Sharpe Ratio:     {svj_sharpe:.4f}")
    print(f"  Max Drawdown:     {svj_dd:.2%}")
    
    print(f"\nGARCH-LSTM Portfolio:")
    print(f"  Total Return:     {gl_total_return:+.2%}")
    print(f"  Annualized:       {((1 + gl_total_return)**(252/len(returns_test)) - 1):+.2%}")
    print(f"  Sharpe Ratio:     {gl_sharpe:.4f}")
    print(f"  Max Drawdown:     {gl_dd:.2%}")
    
    # Winner
    if svj_total_return > gl_total_return:
        winner = "SVJ"
        diff = svj_total_return - gl_total_return
    else:
        winner = "GARCH-LSTM"
        diff = gl_total_return - svj_total_return
    
    print(f"\n🏆 WINNER: {winner} (outperformed by {diff:+.2%})")
    
else:
    print("⚠️ Insufficient test data (simulation started too recently)")
    svj_total_return = gl_total_return = 0
    svj_sharpe = gl_sharpe = 0
    svj_cum = gl_cum = None

# ============================================================
# STEP 5: COMPARISON SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("COMPLETE COMPARISON SUMMARY")
print("=" * 80)

summary = pd.DataFrame({
    'Metric': [
        'Expected Return (60d)',
        'Volatility',
        'Sharpe (expected)',
        'CVaR (95%)',
        'Actual Return',
        'Actual Sharpe'
    ],
    'SVJ': [
        f"{svj_portfolio_rets.mean():.2%}",
        f"{svj_portfolio_rets.std():.2%}",
        f"{(svj_portfolio_rets.mean() / svj_portfolio_rets.std()):.2f}",
        f"{compute_cvar(svj_portfolio_rets):.2%}",
        f"{svj_total_return:+.2%}" if svj_total_return else "N/A",
        f"{svj_sharpe:.2f}" if svj_sharpe else "N/A"
    ],
    'GARCH-LSTM': [
        f"{gl_portfolio_rets.mean():.2%}",
        f"{gl_portfolio_rets.std():.2%}",
        f"{(gl_portfolio_rets.mean() / gl_portfolio_rets.std()):.2f}",
        f"{compute_cvar(gl_portfolio_rets):.2%}",
        f"{gl_total_return:+.2%}" if gl_total_return else "N/A",
        f"{gl_sharpe:.2f}" if gl_sharpe else "N/A"
    ]
})

print("\n" + summary.to_string(index=False))

# ============================================================
# STEP 6: VISUALIZATIONS
# ============================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Plot 1: Return distributions
ax = axes[0, 0]
ax.hist(svj_portfolio_rets * 100, bins=100, alpha=0.6, label='SVJ', color='blue', density=True)
ax.hist(gl_portfolio_rets * 100, bins=100, alpha=0.6, label='GARCH-LSTM', color='green', density=True)
ax.axvline(svj_portfolio_rets.mean() * 100, color='blue', linestyle='--', lw=2)
ax.axvline(gl_portfolio_rets.mean() * 100, color='green', linestyle='--', lw=2)
ax.set_xlabel('Return (%)')
ax.set_ylabel('Density')
ax.set_title('Simulated Return Distribution (60-day)')
ax.legend()
ax.grid(alpha=0.3)

# Plot 2: Portfolio weights
ax = axes[0, 1]
x = np.arange(len(TICKERS))
width = 0.35
ax.bar(x - width/2, svj_weights * 100, width, label='SVJ', alpha=0.7, color='blue')
ax.bar(x + width/2, gl_weights * 100, width, label='GARCH-LSTM', alpha=0.7, color='green')
ax.set_ylabel('Weight (%)')
ax.set_title('Portfolio Allocation Comparison')
ax.set_xticks(x)
ax.set_xticklabels(TICKERS, rotation=45, ha='right')
ax.legend()
ax.grid(alpha=0.3, axis='y')

# Plot 3: CVaR comparison
ax = axes[0, 2]
cvar_data = [compute_cvar(svj_portfolio_rets), compute_cvar(gl_portfolio_rets)]
bars = ax.bar(['SVJ', 'GARCH-LSTM'], np.array(cvar_data) * 100, color=['blue', 'green'], alpha=0.7)
ax.set_ylabel('CVaR 95% (%)')
ax.set_title('Tail Risk Comparison')
ax.axhline(0, color='black', linestyle=':', lw=1)
ax.grid(alpha=0.3, axis='y')
for bar, val in zip(bars, cvar_data):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, height, f'{val:.2%}',
            ha='center', va='bottom' if val > 0 else 'top')

# Plot 4: Actual cumulative returns
if svj_cum is not None:
    ax = axes[1, 0]
    ax.plot((svj_cum - 1) * 100, label='SVJ', lw=2, color='blue')
    ax.plot((gl_cum - 1) * 100, label='GARCH-LSTM', lw=2, color='green')
    ax.axhline(0, color='black', linestyle=':', lw=1)
    ax.set_xlabel('Trading Days')
    ax.set_ylabel('Cumulative Return (%)')
    ax.set_title(f'Actual Performance ({simulation_start_date} to {today})')
    ax.legend()
    ax.grid(alpha=0.3)
else:
    axes[1, 0].text(0.5, 0.5, 'Insufficient\nbacktest data', 
                    ha='center', va='center', transform=axes[1, 0].transAxes)
    axes[1, 0].axis('off')

# Plot 5: Daily returns scatter
if svj_cum is not None:
    ax = axes[1, 1]
    ax.scatter(svj_daily_rets * 100, gl_daily_rets * 100, alpha=0.5, s=20)
    ax.plot([-10, 10], [-10, 10], 'r--', lw=2, label='45° line')
    ax.set_xlabel('SVJ Daily Return (%)')
    ax.set_ylabel('GARCH-LSTM Daily Return (%)')
    ax.set_title('Daily Return Comparison')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.axis('equal')
else:
    axes[1, 1].axis('off')

# Plot 6: Performance metrics summary
ax = axes[1, 2]
ax.axis('off')

if svj_cum is not None:
    metrics_text = f"""
    BACKTEST SUMMARY
    ────────────────────────────
    
    Period: {len(returns_test)} days
    
    SVJ Portfolio:
      Return: {svj_total_return:+.2%}
      Sharpe: {svj_sharpe:.2f}
      MaxDD: {svj_dd:.2%}
    
    GARCH-LSTM Portfolio:
      Return: {gl_total_return:+.2%}
      Sharpe: {gl_sharpe:.2f}
      MaxDD: {gl_dd:.2%}
    
    Winner: {winner}
    Outperformance: {diff:+.2%}
    """
else:
    metrics_text = "Insufficient test data\nfor backtesting"

ax.text(0.1, 0.9, metrics_text, transform=ax.transAxes, 
        fontfamily='monospace', fontsize=10, verticalalignment='top')

plt.tight_layout()
plt.show()

print("\n" + "=" * 80)
print("Analysis Complete!")
print("=" * 80)
