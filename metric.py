
import cvxpy as cp
import numpy as np

def optimize_c_sharpe(terminal_returns, mu_pred, alpha=0.95, max_weight=0.40):
    """
    Optimizes the Conditional Sharpe Ratio (C-Sharpe) using CVXPY.
    Unified Metric: (E[R] - rf) / CVaR
    """
    M, n = terminal_returns.shape
    risk_free_daily = 0.02 / 252
    excess_mu = mu_pred - risk_free_daily
    
    # Transformation variables
    y = cp.Variable(n)      # Scaled weights
    t = cp.Variable()       # Scaling factor (t > 0)
    zeta_bar = cp.Variable()# Scaled VaR threshold
    u = cp.Variable(M)      # Scaled auxiliary variables for tail loss
    
    # Objective: Minimize the scaled CVaR 
    # (equivalent to maximizing E[R]-rf / CVaR)
    scaled_cvar = zeta_bar + (1 / ((1 - alpha) * M)) * cp.sum(u)
    objective = cp.Minimize(scaled_cvar)
    
    # Constraints
    constraints = [
        excess_mu @ y == 1,                # Set excess return to 1 (normalization)
        cp.sum(y) == t,                    # Sum of scaled weights equals t
        y >= 0,                            # Long-only
        y <= max_weight * t,               # Diversification cap
        t >= 0,
        u >= 0,
        # Scaled CVaR constraint: u >= -y*R - zeta_bar
        u >= (-terminal_returns @ y) - (zeta_bar)
    ]
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    
    if y.value is None:
        return np.ones(n) / n
        
    # Recover original weights: w = y / t
    optimal_weights = y.value / t.value
    return optimal_weights
def optimize_portfolio_multi_objective(terminal_returns, lambda_ret=0.5, lambda_risk=0.5, 
                                       alpha=0.95, max_weight=0.40):
    """
    Multi-Objective Portfolio Optimization using CVXPY.
    
    Objective: Minimize [ -lambda_ret * Mean_Return + lambda_risk * CVaR_alpha ]
    
    Parameters:
    -----------
    terminal_returns : ndarray (M scenarios, N assets)
    lambda_ret : float, weight for return maximization
    lambda_risk : float, weight for tail-risk minimization
    alpha : float, confidence level for CVaR
    max_weight : float, individual asset weight cap
    """
    M, n = terminal_returns.shape
    mu = np.mean(terminal_returns, axis=0)
    
    # Decision Variables
    w = cp.Variable(n)      # Portfolio weights
    zeta = cp.Variable()    # VaR threshold (auxiliary variable for CVaR)
    u = cp.Variable(M)      # Auxiliary variables for tail losses (exceedances)
    
    # 1. Expected Return component
    expected_return = mu @ w
    
    # 2. CVaR component (Rockafellar-Uryasev formulation)
    # CVaR = zeta + (1 / ((1 - alpha) * M)) * sum(u)
    # subject to u >= -R*w - zeta and u >= 0
    cvar = zeta + (1 / ((1 - alpha) * M)) * cp.sum(u)
    
    # Objective: Maximize Return (minimize -Return) and Minimize Risk
    # Combined: Minimize [ -lambda_ret * Return + lambda_risk * CVaR ]
    objective = cp.Minimize(-lambda_ret * expected_return + lambda_risk * cvar)
    
    # Constraints
    constraints = [
        cp.sum(w) == 1,                      # Fully invested
        w >= 0,                              # Long-only
        w <= max_weight,                     # Diversification cap
        u >= 0,                              # Non-negativity of exceedances
        u >= (-terminal_returns @ w) - zeta  # Tail loss definition
    ]
    
    # Solve using ECOS (fast for linear/cone problems)
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    
    if w.value is None:
        return np.ones(n) / n # Fallback to equal weights
        
    return np.array(w.value).flatten()
def optimize_robust_cvar(terminal_returns, mu_pred, epsilon=0.01, alpha=0.95, max_weight=0.40):
    """
    Distributionally Robust CVaR Optimization.
    Minimizes the worst-case CVaR within a Wasserstein ambiguity set.
    """
    M, n = terminal_returns.shape
    w = cp.Variable(n)
    zeta = cp.Variable()
    u = cp.Variable(M)
    
    # Standard CVaR components
    expected_return = mu_pred @ w
    cvar_base = zeta + (1 / ((1 - alpha) * M)) * cp.sum(u)
    
    # Robustness Term (Wasserstein DRO)
    # This term adds a penalty based on the norm of the weights
    # It accounts for uncertainty in the scenario probabilities
    robust_penalty = epsilon * cp.norm(w, 2)
    
    # Objective: Maximize Return - lambda * (Robust CVaR)
    # Here we simplify to a risk-minimization focus
    objective = cp.Minimize(cvar_base + robust_penalty - expected_return)
    
    constraints = [
        cp.sum(w) == 1,
        w >= 0.05,  # Increased min-weight to solve the "Zero Allocation" issue
        w <= max_weight,
        u >= 0,
        u >= (-terminal_returns @ w) - zeta
    ]
    
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)
    
    return w.value