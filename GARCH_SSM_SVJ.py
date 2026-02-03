
import cvxpy as cp
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

# Assuming these are available in the working directory
from svj_engine import (
    load_price_data, compute_log_returns, calibrate_svj,
    compute_cvar,optimize_cvar
)
# Re-importing or defining metric functions if not present
# Assuming optimize_robust_cvar is defined or imported
from metric import optimize_robust_cvar

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================

TICKERS = [
    "AAPL", "MSFT", "JNJ", "JPM", "XOM",
    "PG", "KO", "DIS", "BA", "WMT"
]

START_DATE = "2016-01-01"
END_DATE = "2025-01-01"
HORIZON_DAYS = 60
N_PATHS = 5000
LOOKBACK = 60

DEVICE = torch.device("cpu")
torch.manual_seed(42)
np.random.seed(42)
import math
import torch.nn.functional as F

class NonCudaMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_inner = int(expand * d_model)
        self.d_state = d_state
        self.dt_rank = math.ceil(d_model / 16)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1
        )

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)

        A = torch.arange(1, d_state + 1).float()
        self.A_log = nn.Parameter(torch.log(A).repeat(self.d_inner, 1))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        B, L, _ = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = self.conv1d(x.transpose(1, 2))[:, :, :L]
        x = F.silu(x.transpose(1, 2))

        params = self.x_proj(x)
        dt, Bp, Cp = torch.split(params, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt)).clamp(1e-4, 0.1)


        y = self.selective_scan(x, dt, Bp, Cp)
        y = y * F.silu(z)
        return self.out_proj(y)

    def selective_scan(self, u, dt, Bp, Cp):
        B, L, D = u.shape
        N = self.d_state

        A = -torch.exp(self.A_log)

        h = torch.zeros(B, D, N, device=u.device)
        ys = []

        for t in range(L):
            dA = torch.exp(dt[:, t].unsqueeze(-1) * A)
            dB = dt[:, t].unsqueeze(-1) * Bp[:, t].unsqueeze(1)

            h = dA * h + dB * u[:, t].unsqueeze(-1)
            y = torch.einsum("bdn,bn->bd", h, Cp[:, t])
            ys.append(y)

        y = torch.stack(ys, dim=1)
        return y + u * self.D

# ============================================================
# SSM ARCHITECTURE
# ============================================================

class SSMMean(nn.Module):
    def __init__(self, d_model=1, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.mamba = NonCudaMambaBlock(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        out = self.mamba(x)
        return self.fc(out[:, -1, :])


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def apply_evt_smoothing(residuals, threshold_q=0.95):
    threshold = np.quantile(np.abs(residuals), threshold_q)
    extreme_mask = np.abs(residuals) > threshold
    
    if np.any(extreme_mask):
        exceedances = np.abs(residuals[extreme_mask]) - threshold
        shape, loc, scale = stats.genpareto.fit(exceedances)
        gpd_samples = stats.genpareto.rvs(shape, loc=loc, scale=scale, size=np.sum(extreme_mask))
        
        smoothed = residuals.copy()
        smoothed[extreme_mask] = np.sign(residuals[extreme_mask]) * (threshold + gpd_samples)
        return smoothed
    return residuals

def simulate_svj_fixed(S0_vec, params_list, corr_matrix, T, n_paths, seed=42, risk_neutral=False):
    np.random.seed(seed)
    n_assets = len(S0_vec)
    L = np.linalg.cholesky(corr_matrix)
    dt = 1.0 / 252
    
    mus = np.zeros(n_assets) if risk_neutral else np.array([p['mu'] for p in params_list])
    kappas = np.array([p["kappa"] for p in params_list])
    thetas = np.array([p["theta"] for p in params_list])
    sigmas = np.array([p["sigma"] for p in params_list])
    rhos = np.array([p["rho"] for p in params_list])
    v0s = np.array([p["v0"] for p in params_list])
    lambdas = np.array([p["lambda_j"] for p in params_list])
    mu_js = np.array([p["mu_j"] for p in params_list])
    sigma_js = np.array([p["sigma_j"] for p in params_list])
    
    prices = np.zeros((n_paths, T + 1, n_assets))
    variances = np.zeros((n_paths, T + 1, n_assets))
    prices[:, 0, :] = S0_vec
    variances[:, 0, :] = v0s
    
    for t in range(T):
        U = np.random.randn(n_paths, n_assets)
        Z1 = U @ L.T
        eps = np.random.randn(n_paths, n_assets)
        Z2 = rhos * Z1 + np.sqrt(1 - rhos**2) * eps
        jump_arrivals = np.random.poisson(lambdas * dt, (n_paths, n_assets))
        jump_sizes = np.random.randn(n_paths, n_assets)
        vt = np.maximum(variances[:, t, :], 1e-6)
        sqrt_vt_dt = np.sqrt(vt * dt)
        
        variances[:, t + 1, :] = np.maximum(
            vt + kappas * (thetas - vt) * dt + sigmas * sqrt_vt_dt * Z2, 0
        )
        variances[:, t + 1, :] = np.minimum(variances[:, t + 1, :], 0.05)
        
        diffusion = (mus - 0.5 * vt) * dt + sqrt_vt_dt * Z1
        jump_contribution = np.zeros((n_paths, n_assets))
        for i in range(n_assets):
            has_jump = jump_arrivals[:, i] > 0
            jump_contribution[has_jump, i] = mu_js[i] + sigma_js[i] * jump_sizes[has_jump, i]
        
        prices[:, t + 1, :] = prices[:, t, :] * np.exp(diffusion + jump_contribution)
    
    return prices, np.log(prices[:, -1, :] / S0_vec)

def simulate_garch_ssm_joint(returns, corr_matrix, S0_vec, lookback, horizon, n_paths, apply_evt=False):
    n_assets = returns.shape[1]
    garch_results = []
    ssm_models = [] 
    scalers = []
    all_stable_residuals = []
    
    for i in range(n_assets):
        series = returns.iloc[:, i]
        print(f"Processing {TICKERS[i]}:", end=" ")
        
        res = arch_model(series * 100, vol="Garch", p=1, q=1).fit(disp="off", show_warning=False)
        garch_results.append(res)
        
        resid = res.std_resid
        if apply_evt:
            resid = apply_evt_smoothing(resid)
        all_stable_residuals.append(np.sort(resid))
        
        scaler = StandardScaler().fit(series.values.reshape(-1, 1))
        scalers.append(scaler)
        scaled_series = scaler.transform(series.values.reshape(-1, 1))
        
        X_train, y_train = [], []
        for j in range(lookback, len(scaled_series)):
            X_train.append(scaled_series[j-lookback:j, 0])
            y_train.append(scaled_series[j, 0])
            
        X_train = torch.tensor(np.array(X_train), dtype=torch.float32).unsqueeze(-1).to(DEVICE)
        y_train = torch.tensor(np.array(y_train), dtype=torch.float32).unsqueeze(-1).to(DEVICE)
        
        model = SSMMean(d_model=1).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = torch.nn.MSELoss()
        
        for epoch in range(50):
            model.train()
            optimizer.zero_grad()
            output = model(X_train)
            loss = loss_fn(output, y_train)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
        model.eval()
        ssm_models.append(model)
        print(f"GARCH+SSM✓ Trained (Loss: {loss.item():.5f})✓")

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

    with torch.no_grad():
        for t in range(horizon):
            for i in range(n_assets):
                mu_scaled = ssm_models[i](all_sequences[i]).cpu().numpy()
                mu_scaled = np.clip(mu_scaled, -5, 5)
                mu = scalers[i].inverse_transform(mu_scaled).flatten()
                
                p = garch_results[i].params
                var_t = (p["omega"]/10000) + (p["alpha[1]"] * eps_prev[:, i]**2) + (p["beta[1]"] * current_sigmas[:, i]**2)
                current_sigmas[:, i] = np.sqrt(np.clip(var_t, 1e-6, 0.01))
                
                u = u_shocks[t, :, i]
                indices = (u * (len(all_stable_residuals[i]) - 1)).astype(int)
                z_stable = all_stable_residuals[i][indices]
                
                r = mu + current_sigmas[:, i] * z_stable
                sim_returns[:, t, i] = r
                eps_prev[:, i] = current_sigmas[:, i] * z_stable 
                
                next_scaled = (r - scalers[i].mean_[0]) / scalers[i].scale_[0]
                next_scaled_tensor = torch.tensor(next_scaled, dtype=torch.float32).view(-1, 1, 1).to(DEVICE)
                all_sequences[i] = torch.cat([all_sequences[i][:, 1:, :], next_scaled_tensor], dim=1)

    return sim_returns.sum(axis=1)

# ============================================================
# MAIN EXECUTION
# ============================================================

def run_analysis(apply_evt=False):
    print("=" * 80)
    print(f"PORTFOLIO OPTIMIZATION: SVJ vs GARCH-SSM (EVT={apply_evt})")
    print("=" * 80)

    # Step 1: Data Loading
    prices_train = load_price_data(TICKERS, START_DATE, END_DATE)
    returns_train = pd.DataFrame({t: compute_log_returns(prices_train[t]) for t in TICKERS})
    
    end_date_dt = pd.to_datetime(END_DATE)
    prices_after = yf.download(TICKERS, start=END_DATE, progress=False)["Close"]
    prices_after = prices_after[prices_after.index > end_date_dt]
    if prices_after.empty: raise ValueError(f"No trading day found after {END_DATE}")
    S0_vec = prices_after.iloc[0].values
    simulation_start_date = prices_after.index[0].strftime("%Y-%m-%d")
    corr_matrix = returns_train.corr().values

    # Step 2: Method 1 - SVJ
    print("\nMETHOD 1: SVJ Calibration...")
    svj_params = {}
    for ticker in TICKERS:
        log_rets = compute_log_returns(prices_train[ticker])
        svj_params[ticker] = calibrate_svj(log_rets, 3.0, 0.5, 0.6)
    params_list_svj = [svj_params[t] for t in TICKERS]
    _, svj_terminal_rets = simulate_svj_fixed(S0_vec, params_list_svj, corr_matrix, HORIZON_DAYS, N_PATHS, seed=42, risk_neutral=True)
    svj_weights = optimize_cvar(svj_terminal_rets , alpha=0.95, max_weight=0.40)
    svj_portfolio_rets = svj_terminal_rets @ svj_weights

    # Step 3: Method 2 - GARCH-SSM
    print("\nMETHOD 2: GARCH-SSM Training...")
    ssm_terminal_rets = simulate_garch_ssm_joint(returns_train, corr_matrix, S0_vec, LOOKBACK, HORIZON_DAYS, N_PATHS, apply_evt=apply_evt)
    ssm_weights = optimize_cvar(ssm_terminal_rets , alpha=0.95, max_weight=0.40)
    ssm_portfolio_rets = ssm_terminal_rets @ ssm_weights

    # Step 4: Backtest
    today = datetime.now().strftime("%Y-%m-%d")
    prices_test = yf.download(TICKERS, start=simulation_start_date, end=today, progress=False)["Close"]
    
    if len(prices_test) > 1:
        returns_test = pd.DataFrame({t: compute_log_returns(prices_test[t]) for t in TICKERS})
        svj_daily_rets = (returns_test * svj_weights).sum(axis=1)
        ssm_daily_rets = (returns_test * ssm_weights).sum(axis=1)
        svj_total_return = (1 + svj_daily_rets).prod() - 1
        ssm_total_return = (1 + ssm_daily_rets).prod() - 1
        svj_sharpe = (svj_daily_rets.mean() / svj_daily_rets.std()) * np.sqrt(252)
        ssm_sharpe = (ssm_daily_rets.mean() / ssm_daily_rets.std()) * np.sqrt(252)
        svj_cum = (1 + svj_daily_rets).cumprod()
        ssm_cum = (1 + ssm_daily_rets).cumprod()
        svj_dd = (svj_cum / svj_cum.cummax() - 1).min()
        ssm_dd = (ssm_cum / ssm_cum.cummax() - 1).min()
        winner = "SVJ" if svj_total_return > ssm_total_return else "GARCH-SSM"
        diff = abs(svj_total_return - ssm_total_return)
    else:
        svj_cum = ssm_cum = None

    # Step 5: Visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    # Return distributions
    ax = axes[0, 0]
    ax.hist(svj_portfolio_rets * 100, bins=100, alpha=0.6, label='SVJ', color='blue', density=True)
    ax.hist(ssm_portfolio_rets * 100, bins=100, alpha=0.6, label='GARCH-SSM', color='orange', density=True)
    ax.set_title('Simulated Return Distribution')
    ax.legend()
    # Weights
    ax = axes[0, 1]
    x = np.arange(len(TICKERS))
    width = 0.35
    ax.bar(x - width/2, svj_weights * 100, width, label='SVJ', color='blue')
    ax.bar(x + width/2, ssm_weights * 100, width, label='GARCH-SSM', color='orange')
    ax.set_xticks(x); ax.set_xticklabels(TICKERS, rotation=45); ax.legend()
    # CVaR
    ax = axes[0, 2]
    cvar_data = [compute_cvar(svj_portfolio_rets), compute_cvar(ssm_portfolio_rets)]
    ax.bar(['SVJ', 'GARCH-SSM'], np.array(cvar_data) * 100, color=['blue', 'orange'])
    ax.set_title('Tail Risk (CVaR)')
    # Performance
    if svj_cum is not None:
        ax = axes[1, 0]
        ax.plot((svj_cum - 1) * 100, label='SVJ', color='blue')
        ax.plot((ssm_cum - 1) * 100, label='GARCH-SSM', color='orange')
        ax.set_title('Cumulative Return (%)'); ax.legend()
        ax = axes[1, 1]
        ax.scatter(svj_daily_rets * 100, ssm_daily_rets * 100, alpha=0.5)
        ax.plot([-5, 5], [-5, 5], 'r--'); ax.set_title('Daily Return Comparison')
    
    plt.tight_layout()
    plt.savefig('GARCH_SSM_Comparison_evt.png')
    print("Analysis Complete! Saved plot as GARCH_SSM_Comparison.png")

if __name__ == "__main__":
    run_analysis(apply_evt=True)
