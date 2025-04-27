import pandas as pd
import numpy as np

class DataPreprocessor:
    def __init__(self, data):
        self.data = data

    def clean_data(self):
        """Removes missing values and ensures numeric data."""
        self.data = self.data.dropna()
        self.data = self.data.select_dtypes(include=[np.number])
        return self.data

    def calculate_returns(self, period='daily'):
        """Calculates returns from price data."""
        if period == 'daily':
            returns = self.data.pct_change().dropna()
        else:
            returns = self.data.resample(period).ffill().pct_change().dropna()
        return returns

    def get_cov_matrix(self, returns):
        """Calculates covariance matrix from returns."""
        return returns.cov().values

    def get_expected_returns(self, returns):
        """Calculates expected returns (mean)."""
        return returns.mean().values