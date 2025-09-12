import streamlit as st
import pickle
import numpy as np

# ---------------------------
# Load the model + scaler
# ---------------------------
try:
    with open('model.pkl', 'rb') as model_file:
        clf = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
except FileNotFoundError:
    st.error("❌ Model or scaler file missing. Please upload 'model.pkl' & 'scaler.pkl'.")
    st.stop()

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Placement Predictor",
    page_icon="🎓",
    layout="centered",
)

# Custom CSS for Dark Theme
st.markdown(
    """
    <style>
    body {
        background-color: #000000;
        color: #ffffff;
        font-family: "Helvetica Neue", sans-serif;
    }
    .stApp {
        background-color: #000000;
        color: #ffffff;
        padding: 2rem;
    }
    h1, h2, h3, label, .css-16huue1, .css-1d391kg {
        color: #ffffff !important;
    }
    .result {
        font-size: 1.3rem;
        font-weight: bold;
        margin-top: 1rem;
        padding: 1rem;
        border-radius: 10px;
    }
    .success {
        background-color: #111111;
        color: #00ff88;
        border: 1px solid #00ff88;
    }
    .error {
        background-color: #111111;
        color: #ff4d4d;
        border: 1px solid #ff4d4d;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# App Title
# ---------------------------
st.title("🎓 Placement Predictor (Hinglish)")
st.caption("Enter your **CGPA and IQ** to know your placement chances!")

# ---------------------------
# Inputs
# ---------------------------
cgpa = st.number_input(
    '📊 Enter your CGPA',
    min_value=0.0,
    max_value=10.0,
    step=0.1,
    value=7.0
)

iq = st.number_input(
    '🧠 Enter your IQ',
    min_value=50,
    max_value=200,
    step=1,
    value=100
)

# ---------------------------
# Prediction
# ---------------------------
if st.button("🔮 Predict Placement"):
    # Convert input into numpy array
    query_point = np.array([cgpa, iq]).reshape(1, -1)

    # Scale input
    query_point_scaled = scaler.transform(query_point)

    # Predict
    prediction = clf.predict(query_point_scaled)

    # Display Hinglish Result
    if prediction[0] == 1:
        st.markdown(
            "<div class='result success'>🎉 Waah! Lagta hai ki aapki <b>placement pakki hai!</b> 🚀</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div class='result error'>😔 Abhi placement thoda mushkil lag raha hai... Thoda aur mehnat karo! 💪</div>",
            unsafe_allow_html=True
        )
