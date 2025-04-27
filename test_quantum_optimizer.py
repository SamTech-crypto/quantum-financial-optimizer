import unittest
import numpy as np
from src.quantum_optimizer import QuantumPortfolioOptimizer

class TestQuantumOptimizer(unittest.TestCase):
    def setUp(self):
        self.returns = np.array([0.01, 0.02, 0.015])
        self.cov_matrix = np.array([[0.0001, 0.00002, 0.00001],
                                   [0.00002, 0.00015, 0.00003],
                                   [0.00001, 0.00003, 0.00012]])
        self.optimizer = QuantumPortfolioOptimizer(self.returns, self.cov_matrix, risk_free_rate=0.01, max_var=100_000)

    def test_qaoa_optimization(self):
        result = self.optimizer.optimize_with_qaoa()
        weights = self.optimizer.get_optimal_weights(result)
        self.assertAlmostEqual(sum(weights), 1.0, places=5)
        self.assertTrue(all(w >= 0 for w in weights))

    def test_quantum_annealing(self):
        result = self.optimizer.optimize_with_quantum_annealing()
        weights = self.optimizer.get_optimal_weights(result)
        self.assertAlmostEqual(sum(weights), 1.0, places=5)
        self.assertTrue(all(w >= 0 for w in weights))

    def test_zero_returns(self):
        optimizer = QuantumPortfolioOptimizer(np.zeros(3), self.cov_matrix, risk_free_rate=0.01)
        result = optimizer.optimize_with_qaoa()
        weights = optimizer.get_optimal_weights(result)
        self.assertAlmostEqual(sum(weights), 1.0, places=5)

    def test_single_asset(self):
        optimizer = QuantumPortfolioOptimizer(np.array([0.01]), np.array([[0.0001]]), risk_free_rate=0.01)
        result = optimizer.optimize_with_qaoa()
        weights = optimizer.get_optimal_weights(result)
        self.assertAlmostEqual(weights[0], 1.0, places=5)

if __name__ == '__main__':
    unittest.main()