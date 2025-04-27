import unittest
import numpy as np
from src.risk_management import RiskManager

class TestRiskManager(unittest.TestCase):
    def setUp(self):
        self.returns = np.array([0.01, 0.02, 0.015])
        self.cov_matrix = np.array([[0.0001, 0.00002, 0.00001],
                                   [0.00002, 0.00015, 0.00003],
                                   [0.00001, 0.00003, 0.00012]])
        self.risk_manager = RiskManager(self.returns, self.cov_matrix, confidence_level=0.95)

    def test_var_calculation(self):
        weights = np.array([0.4, 0.4, 0.2])
        var = self.risk_manager.calculate_var(weights)
        self.assertGreater(var, 0, "VaR should be positive")
        self.assertLess(var, 1_000_000, "VaR should be reasonable")

    def test_cvar_calculation(self):
        weights = np.array([0.4, 0.4, 0.2])
        cvar = self.risk_manager.calculate_cvar(weights)
        self.assertGreater(cvar, 0, "CVaR should be positive")
        self.assertLess(cvar, 1_000_000, "CVaR should be reasonable")

    def test_var_constraint(self):
        weights = np.array([0.4, 0.4, 0.2])
        penalty = self.risk_manager.var_constraint(weights, max_var=100_000)
        self.assertGreaterEqual(penalty, 0, "Penalty should be non-negative")

if __name__ == '__main__':
    unittest.main()