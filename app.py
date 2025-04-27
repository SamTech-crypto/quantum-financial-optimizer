import streamlit as st
import pandas as pd
import numpy as np
from src.quantum_optimizer import QuantumPortfolioOptimizer
from src.classical_optimizer import ClassicalPortfolioOptimizer
from src.risk_management import RiskManager
from src.data_preprocessing import DataPreprocessor
import plotly.express as px

st.set_page_config(page_title="Quantum Financial Optimizer", layout="wide")

st.title("Quantum Financial Optimizer")
st.markdown("Optimize your portfolio using quantum computing with QAOA, quantum annealing, or classical methods.")

# Sidebar for file upload and settings
st.sidebar.header("Portfolio Settings")
uploaded_file = st.sidebar.file_uploader("Upload portfolio data (CSV)", type="csv")
risk_free_rate = st.sidebar.slider("Risk-Free Rate (%)", 0.0, 5.0, 1.0) / 100
confidence_level = st.sidebar.slider("Confidence Level for VaR/CVaR (%)", 90.0, 99.0, 95.0) / 100
max_var = st.sidebar.number_input("Max VaR ($)", min_value=0.0, value=50000.0, step=1000.0)
optimization_method = st.sidebar.selectbox("Optimization Method", ["QAOA (Quantum)", "Quantum Annealing", "Classical"])

# Main content
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    preprocessor = DataPreprocessor(data)
    data_clean = preprocessor.clean_data()
    returns_data = preprocessor.calculate_returns()
    
    st.write("### Portfolio Data Preview")
    st.dataframe(data_clean.head())

    returns = preprocessor.get_expected_returns(returns_data)
    cov_matrix = preprocessor.get_cov_matrix(returns_data)

    if st.button("Optimize Portfolio"):
        with st.spinner("Optimizing portfolio..."):
            # Optimization
            if optimization_method == "QAOA (Quantum)":
                optimizer = QuantumPortfolioOptimizer(returns, cov_matrix, risk_free_rate, max_var=max_var)
                result = optimizer.optimize_with_qaoa()
                weights = optimizer.get_optimal_weights(result)
            elif optimization_method == "Quantum Annealing":
                optimizer = QuantumPortfolioOptimizer(returns, cov_matrix, risk_free_rate, max_var=max_var)
                result = optimizer.optimize_with_quantum_annealing()
                weights = optimizer.get_optimal_weights(result)
            else:
                optimizer = ClassicalPortfolioOptimizer(returns, cov_matrix, risk_free_rate, max_var=max_var)
                weights = optimizer.optimize()

            # Risk Metrics
            risk_manager = RiskManager(returns, cov_matrix, confidence_level)
            var = risk_manager.calculate_var(weights)
            cvar = risk_manager.calculate_cvar(weights)

            # Display results
            st.write("### Optimal Portfolio Weights")
            weights_df = pd.DataFrame({"Asset": data_clean.columns, "Weight": weights})
            st.dataframe(weights_df)

            st.write("### Risk Metrics")
            st.write(f"**Value-at-Risk (VaR)**: ${var:,.2f}")
            st.write(f"**Conditional Value-at-Risk (CVaR)**: ${cvar:,.2f}")

            # Visualize weights
            fig = px.pie(weights_df, values="Weight", names="Asset", title="Portfolio Allocation")
            st.plotly_chart(fig)
else:
    st.info("Please upload a CSV file with portfolio data to begin.")