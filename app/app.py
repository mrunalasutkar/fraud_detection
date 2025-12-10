import streamlit as st
import numpy as np
import pickle
from pathlib import Path

# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # assumes your app is in 'app' folder
MODEL_PATH = BASE_DIR / "models" / "xgb_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "amount_scaler.pkl"

# -----------------------------
# LOAD MODEL + SCALER
# -----------------------------
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_scaler():
    with open(SCALER_PATH, "rb") as f:
        return pickle.load(f)

model = load_model()
scaler = load_scaler()

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="centered"
)

# -----------------------------
# DARK UI + ANIMATIONS
# -----------------------------
st.markdown("""
<style>
.stApp {background: linear-gradient(135deg, #0a0f1f 0%, #141a2e 100%); color: white; animation: fadeIn 0.7s ease-in-out;}
.app-title {font-size: 44px; text-align:center; font-weight:900; color:#4DA3FF; text-shadow:0px 0px 12px rgba(77,163,255,0.6);}
.sub-title {text-align:center; font-size:18px; color:#b8c6ff; margin-bottom:30px;}
.stNumberInput input {background-color:#111726 !important; color:white !important; border:1px solid #3a72ff !important; border-radius:6px !important;}
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
st.markdown("<div class='sub-title'>Detect Anamalous Transaction</div>", unsafe_allow_html=True)

# -----------------------------
# INPUTS
# -----------------------------
colA, colB = st.columns(2)
with colA:
    time = st.number_input("Transaction Time", min_value=0.0, value=0.0)
with colB:
    amount = st.number_input("Amount (₹)", min_value=0.0, value=0.0)

# PCA FEATURES (V1–V28)
with st.expander("🔧 Advanced PCA Features (V1–V28)"):
    st.write("Modify V1–V28 if needed (default 0).")
    pca_values = []
    cols = st.columns(4)
    for idx in range(28):
        col = cols[idx % 4]
        pca_values.append(col.number_input(f"V{idx+1}", value=0.0, format="%.5f"))

# -----------------------------
# PREDICTION
# -----------------------------
if st.button("Predict Fraud"):
    amount_scaled = scaler.transform(np.array([[amount]])).flatten()[0]
    features = np.array([time] + pca_values + [amount_scaled]).reshape(1, -1)
    prediction = model.predict(features)[0]

    if prediction == 0:
        st.markdown("<div class='result-box legit'>✅ Legit Transaction</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='result-box fraud'>🚨 Fraud Detected!</div>", unsafe_allow_html=True)
