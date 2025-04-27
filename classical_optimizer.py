import numpy as np
from scipy.optimize import minimize
from src.risk_management import RiskManager

class ClassicalPortfolioOptimizer:
    def __init__(self, returns, cov_matrix, risk_free_rate=0.01, max_var=None):
        self.returns = returns
        self.cov_matrix = cov_matrix
        self.risk_free_rate = risk_free_rate
        self.num_assets = len(returns)
        self.max_var = max_var
        self.risk_manager = RiskManager(returns, cov_matrix)

    def _objective(self, weights):
        portfolio_return = np.sum(self.returns * weights)
        portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        penalty = 0
        if self.max_var:
            penalty = self.risk_manager.var_constraint(weights, self.max_var) * 1000
        return - (portfolio_return - self.risk_free_rate) / portfolio_risk + penalty

    def optimize(self):
        initial_weights = np.ones(self.num_assets) / self.num_assets
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = [(0, 1) for _ in range(self.num_assets)]
        result = minimize(
            self._objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        return result.x