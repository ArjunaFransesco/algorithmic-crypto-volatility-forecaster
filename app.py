import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="High-Frequency Crypto Volatility & Regime Switching Engine",
    page_icon="🤖",
    layout="wide"
)

st.title("🎯 High-Frequency Crypto Volatility & Regime Switching Engine")
st.markdown("**Domain**: `Machine Learning / Quantitative Trading` | **Tech Stack**: `LightGBM, Time Series, Streamlit`")
st.markdown("**Author**: [Arjuna Fransesco](https://github.com/ArjunaFransesco) | **GitHub**: [Portfolio Repositories](https://github.com/ArjunaFransesco?tab=repositories)")
st.markdown("---")

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("⚙️ Domain Input Telemetry")
    realized_volatility_5m = st.slider("Realized Volatility 5M", float(0.005), float(0.15), float(0.035))
    orderbook_imbalance_ratio = st.slider("Orderbook Imbalance Ratio", float(-0.8), float(0.8), float(0.02))
    relative_strength_index_rsi = st.slider("Relative Strength Index Rsi", float(15.0), float(85.0), float(52.0))
    exponential_moving_avg_spread = st.slider("Exponential Moving Avg Spread", float(-0.05), float(0.05), float(0.008))
    bid_ask_spread_bps = st.slider("Bid Ask Spread Bps", float(0.5), float(20.0), float(4.5))
    trading_volume_usd_k = st.slider("Trading Volume Usd K", float(50.0), float(3500.0), float(850.0))

with col2:
    st.subheader("🔮 Predictive Model Inference")
    model_path = os.path.join(os.path.dirname(__file__), "models/model_pipeline.joblib")
    scaler_path = os.path.join(os.path.dirname(__file__), "models/scaler.joblib")
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        input_df = pd.DataFrame([{"realized_volatility_5m": realized_volatility_5m, "orderbook_imbalance_ratio": orderbook_imbalance_ratio, "relative_strength_index_rsi": relative_strength_index_rsi, "exponential_moving_avg_spread": exponential_moving_avg_spread, "bid_ask_spread_bps": bid_ask_spread_bps, "trading_volume_usd_k": trading_volume_usd_k}])
        input_scaled = scaler.transform(input_df)
        pred = model.predict(input_scaled)[0]
        
        st.markdown("#### Real-Time Prediction Output")
        st.info(f"Predicted `target_future_volatility`: **{pred}**")
        
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_scaled)[0]
            st.progress(float(probs[1]) if len(probs) > 1 else float(probs[0]))
            st.caption(f"Confidence Probability Score: **{np.max(probs):.2%}**")
    else:
        st.warning("Model or Scaler artifact not found in models/ directory.")

st.markdown("---")
st.markdown("### 📊 Benchmark Metrics")
metrics_path = os.path.join(os.path.dirname(__file__), "reports/metrics.json")
if os.path.exists(metrics_path):
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics_data = json.load(f)
    st.json(metrics_data)
