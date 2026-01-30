import streamlit as st
import pickle
import numpy as np
from PIL import Image
from datetime import datetime

# ---------------- CONFIG & GLASSY THEME ----------------
st.set_page_config(page_title="UPI Fraud Detection System", layout="wide", page_icon="🛡️")

# CSS for Glassmorphism Design
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #ffffff;
    }

    /* The Glass Card Effect */
    div[data-testid="stForm"] {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border-radius: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 40px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }

    /* Teal Gradient Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #26b99a 0%, #1da1f2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        height: 3.5em !important;
        font-weight: 700 !important;
    }

    /* Reset Button Styling */
    .reset-box button {
        background: rgba(255, 255, 255, 0.1) !important;
        color: #cbd5e1 !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
    }

    input, select, .stSelectbox div {
        background-color: rgba(255, 255, 255, 0.05) !important;
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)


# ---------------- LOAD ASSETS ----------------
@st.cache_resource
def load_assets():
    model = pickle.load(open("model_rfc.pkl", "rb"))
    scaler = pickle.load(open("sc_.pkl", "rb"))
    le_day = pickle.load(open("le_day.pkl", "rb"))
    le_month = pickle.load(open("le_month.pkl", "rb"))
    le_cat = pickle.load(open("le_cat.pkl", "rb"))
    return model, scaler, le_day, le_month, le_cat


model, scaler, le_day, le_month, le_cat = load_assets()


# ---------------- RESET CALLBACK ----------------
def hard_reset():
    for key in ["r_day", "r_month", "r_year", "r_cat", "r_upi", "r_age", "r_amt", "r_state", "r_zip"]:
        if key in st.session_state:
            del st.session_state[key]


# ---------------- HEADER ----------------
with st.container():
    col_img, col_text = st.columns([1, 2])
    with col_img:
        try:
            # Responsive image implementation
            st.image("Cybercrime Contrast Scene.png", use_container_width=True)
        except:
            st.info("Image Asset Missing")
    with col_text:
        st.title("UPI Fraud Detection System")
        st.markdown("Real-time Fraud Detection using Machine Learning")

st.divider()

# ---------------- DATA LISTS ----------------
# Custom ordered lists for the dropdowns
days_list = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
months_list = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]
years_list = list(range(2020, 2026)) # Creates [2020, 2021, 2022, 2023, 2024, 2025]

# ---------------- MAIN UI ----------------
with st.form("glass_form"):
    st.subheader("📋 Transaction Parameters")
    c1, c2, c3 = st.columns(3)
    with c1:
        # Day dropdown starts from Monday to Sunday
        trans_day = st.selectbox("Day", days_list, key="r_day")
        upi_number = st.number_input("UPI ID", min_value=0, key="r_upi")
    with c2:
        # Month dropdown starts from January to December
        trans_month = st.selectbox("Month", months_list, key="r_month")
        age = st.number_input("User Age", min_value=18, key="r_age")
    with c3:
        # Year dropdown from 2020 to 2025
        trans_year = st.selectbox("Year", years_list, key="r_year")
        zip_code = st.number_input("ZIP Code", key="r_zip")

    st.subheader("💰 Financial Context")
    c4, c5, c6 = st.columns(3)
    with c4:
        category = st.selectbox("Merchant Type", le_cat.classes_, key="r_cat")
    with c5:
        trans_amount = st.number_input("Amount (₹)", min_value=0.0, key="r_amt")
    with c6:
        state = st.number_input("State Code", key="r_state")

    _, center_btn, _ = st.columns([1, 1, 1])
    with center_btn:
        submit = st.form_submit_button("DETECT FRAUD")

# ---------------- RESET SECTION ----------------
_, reset_col, _ = st.columns([1.2, 0.6, 1.2])
with reset_col:
    st.markdown('<div class="reset-box">', unsafe_allow_html=True)
    st.button("CLEAR DASHBOARD", on_click=hard_reset)
    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- PREDICTION ----------------
if submit:
    try:
        # Transform the ordered labels using the pre-loaded encoders
        features = np.array([[
            le_day.transform([st.session_state.r_day])[0],
            le_month.transform([st.session_state.r_month])[0],
            int(st.session_state.r_year),
            le_cat.transform([st.session_state.r_cat])[0],
            int(st.session_state.r_upi),
            float(st.session_state.r_age),
            float(st.session_state.r_amt),
            int(st.session_state.r_state),
            int(st.session_state.r_zip)
        ]])

        scaled = scaler.transform(features)
        prediction = model.predict(scaled)

        if prediction[0] == 1:
            st.error("### 🚨 HIGH RISK: Fraud Detected")
        else:
            st.success("### ✅ SECURE: Verified")
            st.balloons()
    except Exception as e:
        st.error(f"Error: {e}")
        st.info("Check if the Day or Month names in your encoder match the custom lists provided.")

st.markdown("---")
