import numpy as np
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler
from braket.aws import AwsDevice
from braket.annealing import Problem, ProblemType
import os
from src.risk_management import RiskManager

class QuantumPortfolioOptimizer:
    def __init__(self, returns, cov_matrix, risk_free_rate=0.01, num_qubits=None, max_var=None):
        self.returns = returns
        self.cov_matrix = cov_matrix
        self.risk_free_rate = risk_free_rate
        self.num_assets = len(returns)
        self.num_qubits = num_qubits if num_qubits else self.num_assets
        self.max_var = max_var
        self.risk_manager = RiskManager(returns, cov_matrix)
        self.qp = self._build_quadratic_program()

    def _build_quadratic_program(self):
        """Constructs the quadratic program for portfolio optimization."""
        qp = QuadraticProgram()
        for i in range(self.num_assets):
            qp.binary_var(f'x_{i}')

        linear = {f'x_{i}': self.returns[i] for i in range(self.num_assets)}
        quadratic = {
            (f'x_{i}', f'x_{j}'): self.cov_matrix[i, j]
            for i in range(self.num_assets)
            for j in range(self.num_assets)
        }
        qp.maximize(linear=linear, quadratic=quadratic)

        qp.linear_constraint(
            linear={f'x_{i}': 1 for i in range(self.num_assets)},
            sense='==',
            rhs=1,
            name='budget'
        )
        return qp

    def optimize_with_qaoa(self):
        """Optimizes portfolio using QAOA on Qiskit."""
        qaoa = QAOA(sampler=Sampler(), optimizer=COBYLA(), reps=1)
        optimizer = MinimumEigenOptimizer(qaoa)
        result = optimizer.solve(self.qp)
        return result

    def optimize_with_quantum_annealing(self):
        """Optimizes portfolio using quantum annealing on AWS Braket."""
        problem = Problem(ProblemType.QUBO)
        for i in range(self.num_assets):
            problem.add_linear_term(i, -self.returns[i])
        for i in range(self.num_assets):
            for j in range(i, self.num_assets):
                problem.add_quadratic_term(i, j, self.cov_matrix[i, j])

        penalty = 1000
        for i in range(self.num_assets):
            problem.add_linear_term(i, penalty)
            for j in range(i + 1, self.num_assets):
                problem.add_quadratic_term(i, j, 2 * penalty)
        problem.add_linear_term(self.num_assets, -penalty * self.num_assets)

        if self.max_var:
            weights = np.ones(self.num_assets) / self.num_assets  # Initial guess
            var_penalty = self.risk_manager.var_constraint(weights, self.max_var)
            if var_penalty > 0:
                for i in range(self.num_assets):
                    problem.add_linear_term(i, var_penalty * 100)

        device_arn = os.getenv("AWS_BRAKET_DEVICE_ARN", "arn:aws:braket:::device/qpu/d-wave/Advantage_system4")
        device = AwsDevice(device_arn)
        task = device.run(problem, shots=1000)
        result = task.result()

        best_solution = result.record.sample[np.argmin(result.record.energy)]
        return type('Result', (), {'x': best_solution})()

    def get_optimal_weights(self, result):
        """Extracts optimal portfolio weights from optimization result."""
        weights = np.zeros(self.num_assets)
        for i, val in enumerate(result.x):
            weights[i] = val
        return weights / np.sum(weights)