import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import pickle
from datetime import date, timedelta

@st.cache_resource
def load_model_keras():
    model = load_model("./Models/Pol_Bmkg_Hybrid_Model_40.keras")
    return model

# Set the layout
st.set_page_config(page_title="PM2.5 Prediction", layout="wide")

# --- Custom CSS ---
st.markdown("""
    <style>
    .sidebar-button button {
        width: 100% !important;
        text-align: left;
        padding: 0.5rem 1rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 10px;
        margin-bottom: 0.5rem;
        background-color: #262730;
        color: white;
    }
    .sidebar-button button:hover {
        background-color: #444;
    }
    .sidebar-section {
        padding-bottom: 1rem;
        margin-bottom: 1rem;
        border-bottom: 1px solid gray;
    }
    </style>
""", unsafe_allow_html=True)

# --- Session state for page navigation ---
if "page" not in st.session_state:
    st.session_state.page = "Main"

# --- Sidebar ---
with st.sidebar:
    st.markdown("### 📌 Menu")
    with st.container():
        st.markdown('<div class="sidebar-button">', unsafe_allow_html=True)
        if st.button("🏠 Main"):
            st.session_state.page = "Main"
        if st.button("📊 Prediction"):
            st.session_state.page = "Prediction"
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Conditional Instruction section ---
    if st.session_state.page == "Prediction":
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### 🧾 Page Instruction")
        st.markdown(
            """
            <div style="border: 1px solid gray; padding: 10px; border-radius: 5px;">
            This page is used to predict PM2.5 index in Central Jakarta
            <br>
            <br>

            Follow these steps to use this application:
            <ol>
            <li>Insert PM2.5 index for the past 7 days (not including today's index)</li>
            <li>Input no. 1 = 7th past day’s PM2.5, Input no. 2 = 6th past day’s PM2.5, etc.</li>
            <li>Click “Predict” after filling in all the inputs</li>
            </ol>
            </div>
            """,
            unsafe_allow_html=True
        )


# --- Page content ---
if st.session_state.page == "Main":
    st.markdown(
        """
        <div style="display: flex; justify-content: center; align-items: center; height: 90vh; text-align: center;">
            <div>
                <h1>Predict PM2.5 In Central Jakarta</h1>
                <p>A website that helps predict Central's Jakarta PM2.5 Index for 7 days ahead</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

elif st.session_state.page == "Prediction":
    st.markdown("<h2 style='text-align: center;'>Predict PM2.5 In Central Jakarta</h2>", unsafe_allow_html=True)
    st.write("")  # spacing

    # --- Input form for 7 PM2.5 values ---
    pm_values = []
    for i in range(7, 0, -1):
        val = st.number_input(f"PM2.5 input from {i} days ago", min_value=0, step=1, key=f"pm_{i}")
        pm_values.append(val)

    pm_values = np.array(pm_values, dtype=np.float64)

    if st.button("Predict"):
        data = pd.read_csv('./data_for_app.csv', sep=',')

        data['tanggal'] = pd.to_datetime(data['tanggal'], format='%Y-%m-%d').dt.date
        data.set_index('tanggal', inplace=True)

        ### Start of original code for data_window
        # # Get yesterday's date
        # yesterday = date.today() - timedelta(days=1)

        # # Get data from 7 days before yesterday up to yesterday (inclusive)
        # start_date = yesterday - timedelta(days=6)
        # data_window = data.loc[start_date:yesterday]
        ### End of original code for data_window

        ### Start of hardcoded date range
        start = pd.to_datetime('2025-06-09').date()
        end = pd.to_datetime('2025-06-15').date()
        data_window = data.loc[start:end] 
        ### End of hardcoded date range

        data_window['pm25'] = pm_values
        scaler_x = pickle.load(open("./Scaler/scaler_x.pkl", "rb"))
        scaler_y = pickle.load(open("./Scaler/scaler_y.pkl", "rb"))

        # Scale the input data
        data_window_scaled = scaler_x.transform(data_window)

        data_window_scaled = data_window_scaled.reshape(1, 7, 13)

        # Load the model
        model = load_model_keras()

        y_pred = model.predict(data_window_scaled)
        pm_values_pred = scaler_y.inverse_transform(y_pred)
        pm_values_pred = pm_values_pred.round()

        st.success(f"Predicted PM2.5 for the next 7 days: {pm_values_pred.flatten()}")
