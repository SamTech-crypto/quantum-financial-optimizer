import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from src.quantum_optimizer import QuantumPortfolioOptimizer
from src.classical_optimizer import ClassicalPortfolioOptimizer
from src.risk_management import RiskManager
from src.data_preprocessing import DataPreprocessor

# --- PAGE CONFIG ---
st.set_page_config(page_title="Quantum Financial Optimizer", layout="wide")

# --- CUSTOM BACKGROUND STYLE ---
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f3f6fd;
        font-family: 'Segoe UI', sans-serif;
    }
    .css-1d391kg, .css-ffhzg2 {
        background-color: white !important;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0px 2px 10px rgba(0,0,0,0.1);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- HEADER ---
st.title("🧠 Quantum Financial Optimizer")
st.markdown("Optimize your portfolio using **quantum computing** with QAOA, quantum annealing (via AWS Braket), or classical methods.")

# --- SIDEBAR SETTINGS ---
st.sidebar.header("⚙️ Portfolio Settings")
uploaded_file = st.sidebar.file_uploader("📁 Upload portfolio data (CSV)", type="csv")
risk_free_rate = st.sidebar.slider("Risk-Free Rate (%)", 0.0, 5.0, 1.0) / 100
confidence_level = st.sidebar.slider("Confidence Level for VaR/CVaR (%)", 90.0, 99.0, 95.0) / 100
max_var = st.sidebar.number_input("Max VaR ($)", min_value=0.0, value=50000.0, step=1000.0)
optimization_method = st.sidebar.selectbox("Optimization Method", ["QAOA (Quantum)", "Quantum Annealing", "Classical"])

# --- MAIN LOGIC ---
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    preprocessor = DataPreprocessor(data)
    data_clean = preprocessor.clean_data()
    returns_data = preprocessor.calculate_returns()

    st.subheader("📊 Portfolio Data Preview")
    st.dataframe(data_clean.head())

    returns = preprocessor.get_expected_returns(returns_data)
    cov_matrix = preprocessor.get_cov_matrix(returns_data)

    if st.button("🚀 Optimize Portfolio"):
        with st.spinner("Running optimization..."):
            try:
                if optimization_method == "QAOA (Quantum)":
                    optimizer = QuantumPortfolioOptimizer(returns, cov_matrix, risk_free_rate, max_var=max_var)
                    result = optimizer.optimize_with_qaoa()
                    weights = optimizer.get_optimal_weights(result)

                elif optimization_method == "Quantum Annealing":
                    if not os.getenv("AWS_ACCESS_KEY_ID"):
                        st.error("Missing AWS credentials! Please add them in your .env or Streamlit secrets.")
                        st.stop()
                    optimizer = QuantumPortfolioOptimizer(returns, cov_matrix, risk_free_rate, max_var=max_var)
                    result = optimizer.optimize_with_quantum_annealing()
                    weights = optimizer.get_optimal_weights(result)

                else:  # Classical
                    optimizer = ClassicalPortfolioOptimizer(returns, cov_matrix, risk_free_rate, max_var=max_var)
                    weights = optimizer.optimize()

                # RISK METRICS
                risk_manager = RiskManager(returns, cov_matrix, confidence_level)
                var = risk_manager.calculate_var(weights)
                cvar = risk_manager.calculate_cvar(weights)

                # DISPLAY RESULTS
                st.subheader("📈 Optimal Portfolio Weights")
                weights_df = pd.DataFrame({"Asset": data_clean.columns, "Weight": weights})
                st.dataframe(weights_df)

                st.subheader("🛡️ Risk Metrics")
                st.markdown(f"**Value-at-Risk (VaR)**: `${var:,.2f}`")
                st.markdown(f"**Conditional Value-at-Risk (CVaR)**: `${cvar:,.2f}`")

                fig = px.pie(weights_df, values="Weight", names="Asset", title="Portfolio Allocation")
                st.plotly_chart(fig)

            except Exception as e:
                st.error(f"Optimization failed: {e}")
else:
    st.info("📎 Please upload a CSV file with portfolio data to begin.")
