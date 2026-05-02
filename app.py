import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import random
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}
.stButton>button {
    background-color: #00c6ff;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL (FIXED) ----------------
model_path = os.path.join(os.path.dirname(__file__), 'fraud_model.pkl')
model = joblib.load(model_path)

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Controls")

st.sidebar.markdown("### 🔧 Model Info")
st.sidebar.write("Model: Random Forest")
st.sidebar.write("Features: V1 - V28 + Amount")

mode = st.sidebar.selectbox("🎯 Mode", ["Manual", "Demo"])

# ---------------- MAIN ----------------
st.title("💳 Fraud Detection Dashboard")
st.markdown("### AI-powered transaction risk analysis")

st.divider()

# ---------------- INPUT ----------------
st.subheader("📊 Transaction Input")

col1, col2 = st.columns(2)

with col1:
    amount = st.number_input("💰 Transaction Amount", min_value=0.0)

with col2:
    if mode == "Demo":
        st.info("Demo mode uses random fraud-like patterns")

st.divider()

# ---------------- PREDICT ----------------
if st.button("🔍 Analyze Transaction"):

    columns = ['V1','V2','V3','V4','V5','V6','V7','V8','V9','V10',
               'V11','V12','V13','V14','V15','V16','V17','V18','V19','V20',
               'V21','V22','V23','V24','V25','V26','V27','V28','Amount']

    if mode == "Demo":
        values = list(np.random.uniform(-3, 3, 28)) + [amount]
    else:
        values = [0]*28 + [amount]

    input_df = pd.DataFrame([values], columns=columns)

    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0][1]

    st.divider()

    colA, colB = st.columns(2)

    # ---------------- RESULT ----------------
    with colA:
        st.subheader("🔎 Prediction Result")

        if prediction[0] == 1:
            st.error("🚨 Fraud Detected")
        else:
            st.success("✅ Normal Transaction")

        st.write(f"Risk Score: **{probability:.2%}**")

    # ---------------- GAUGE CHART ----------------
    with colB:
        st.subheader("📊 Fraud Risk Meter")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={'text': "Fraud Risk %"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "red"},
                'steps': [
                    {'range': [0, 30], 'color': "green"},
                    {'range': [30, 70], 'color': "orange"},
                    {'range': [70, 100], 'color': "red"}
                ],
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

    # ---------------- TRANSACTION HISTORY ----------------
    st.subheader("📈 Transaction History")

    history = [random.uniform(0, 100) for _ in range(20)]
    st.line_chart(history)