import streamlit as st
import numpy as np
import pickle
import pandas as pd

# -----------------------------
# FAST MODEL + SCALER LOADING
# -----------------------------
@st.cache_resource
def load_model():
    with open("models/xgb_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_scaler():
    with open("models/amount_scaler.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
scaler = load_scaler()

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="💳 Credit Card Fraud Detection",
    page_icon="💳",
    layout="centered"
)

# -----------------------------
# DARK UI + ANIMATIONS
# -----------------------------
st.markdown("""
<style>
.stApp {background: linear-gradient(135deg, #0a0f1f, #141a2e); color: white;}
.app-title {font-size: 44px; text-align:center; font-weight:900; color:#4DA3FF; text-shadow:0px 0px 12px rgba(77,163,255,0.6);}
.sub-title {text-align:center; font-size:18px; color:#b8c6ff; margin-bottom:30px;}
label, .css-1d391kg {color:#c8d3ff !important;}
.stNumberInput input {background-color:#111726 !important; color:white !important; border:1px solid #3a72ff !important; border-radius:6px !important;}
.streamlit-expanderHeader:hover {color:#4DA3FF !important; transition:0.3s ease;}
.stButton > button {width:100%; background:#255DFF !important; color:white !important; padding:12px !important; border-radius:10px !important; font-size:18px !important; font-weight:700 !important; border:none !important; box-shadow:0px 0px 12px #255DFF; transition:0.25s ease-in-out;}
.stButton > button:hover {background:#1e4ad1 !important; box-shadow:0px 0px 18px #3a73ff; transform:scale(1.02);}
.result-box {text-align:center; padding:16px; border-radius:12px; margin-top:20px; font-size:20px; font-weight:700;}
.legit {background: rgba(0,255,120,0.15); border:1px solid #00ff88; color:#00ff88; box-shadow:0px 0px 10px rgba(0,255,120,0.6);}
.fraud {background: rgba(255,60,60,0.20); border:1px solid #ff4d4d; color:#ff4d4d; box-shadow:0px 0px 10px rgba(255,60,60,0.6);}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# TITLES
# -----------------------------
st.markdown("<div class='app-title'>💳 Credit Card Fraud Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Real-time ML prediction with confidence</div>", unsafe_allow_html=True)

# -----------------------------
# LOAD SAMPLE FRAUD DATA
# -----------------------------
fraud_samples = {
    "Fraud Sample 1": [472.0, 1.191857111, 0.266150712, 0.166480113, 0.448154078,
                       0.060017649, -0.082360808, -0.078802983, 0.085101654, -0.255425128,
                       -1.106822788, 0.634736198, 0.463232402, -0.114804663, -0.183361270,
                       -0.145783041, -0.069083135, -0.225775248, -0.638671953, 0.101288021,
                       -0.339846477, 0.167170404, 0.125894532, -0.008983099, 0.014724169,
                       -0.567437284, 0.045909881, 0.166974414, 0.070794979, 0.0],
    "Fraud Sample 2": [0.0]*30  # You can replace with another known fraud sample
}

sample_choice = st.selectbox("Select a sample for testing", ["Manual Entry"] + list(fraud_samples.keys()))

# -----------------------------
# MAIN INPUTS
# -----------------------------
if sample_choice == "Manual Entry":
    colA, colB = st.columns(2)
    with colA:
        time = st.number_input("Transaction Time", min_value=0.0, value=0.0)
    with colB:
        amount = st.number_input("Amount (₹)", min_value=0.0, value=0.0)
    
    # PCA Features
    with st.expander("🔧 Advanced PCA Features (V1–V28)"):
        st.write("Modify V1–V28 if needed (default 0).")
        pca_values = []
        cols = st.columns(4)
        for idx in range(28):
            col = cols[idx % 4]
            pca_values.append(col.number_input(f"V{idx+1}", value=0.0, format="%.5f"))
else:
    # Use selected sample
    full_sample = fraud_samples[sample_choice]
    time = full_sample[0]
    amount = full_sample[-1]
    pca_values = full_sample[1:-1]
    st.info(f"Loaded {sample_choice}. Edit values in the PCA expander if needed.")
    with st.expander("🔧 Advanced PCA Features (V1–V28)"):
        st.write("Modify V1–V28 if needed (default is loaded sample).")
        cols = st.columns(4)
        for idx in range(28):
            col = cols[idx % 4]
            pca_values[idx] = col.number_input(f"V{idx+1}", value=pca_values[idx], format="%.5f")

# -----------------------------
# PREDICT BUTTON
# -----------------------------
if st.button("Predict Fraud"):
    # SCALE AMOUNT BEFORE PREDICTION
    amount_scaled = scaler.transform(np.array([[amount]])).flatten()[0]
    features = np.array([time] + pca_values + [amount_scaled]).reshape(1, -1)
    
    pred = model.predict(features)[0]
    pred_prob = model.predict_proba(features)[0][1]

    if pred == 0:
        st.markdown(f"<div class='result-box legit'>✅ Legit Transaction ({(1-pred_prob)*100:.2f}% confidence)</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='result-box fraud'>🚨 Fraud Detected! ({pred_prob*100:.2f}% confidence)</div>", unsafe_allow_html=True)
