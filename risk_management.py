import numpy as np
import pandas as pd
from scipy.stats import norm

class RiskManager:
    def __init__(self, returns, cov_matrix, confidence_level=0.95):
        self.returns = returns
        self.cov_matrix = cov_matrix
        self.confidence_level = confidence_level

    def calculate_var(self, weights, portfolio_value=1_000_000):
        """Calculates Value-at-Risk (VaR) for the portfolio."""
        portfolio_return = np.sum(self.returns * weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        z_score = norm.ppf(self.confidence_level)
        var = portfolio_value * (portfolio_return - z_score * portfolio_volatility)
        return var

    def calculate_cvar(self, weights, portfolio_value=1_000_000):
        """Calculates Conditional Value-at-Risk (CVaR) for the portfolio."""
        portfolio_return = np.sum(self.returns * weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        z_score = norm.ppf(self.confidence_level)
        cvar = portfolio_value * (portfolio_return - (norm.pdf(z_score) / (1 - self.confidence_level)) * portfolio_volatility)
        return cvar

    def var_constraint(self, weights, max_var, portfolio_value=1_000_000):
        """Returns penalty if VaR exceeds max_var."""
        var = self.calculate_var(weights, portfolio_value)
        return max_var - var if var > max_var else 0

    def cvar_constraint(self, weights, max_cvar, portfolio_value=1_000_000):
        """Returns penalty if CVaR exceeds max_cvar."""
        cvar = self.calculate_cvar(weights, portfolio_value)
        return max_cvar - cvar if cvar > max_cvar else 0