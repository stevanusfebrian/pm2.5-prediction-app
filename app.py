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

def get_pm25_level(value):
    if value <= 50:
        return "Baik"
    elif value <= 100:
        return "Sedang"
    elif value <= 199:
        return "Tidak Sehat"
    elif value <= 299:
        return "Sangat Tidak Sehat"
    elif value <= 500:
        return "Berbahaya"
    else:
        return "Di luar jangkauan"

# Set the layout
st.set_page_config(page_title="Prediksi PM2.5", layout="wide")

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
    st.session_state.page = "Utama"

# --- Sidebar ---
with st.sidebar:
    st.markdown("### 📌 Menu")
    with st.container():
        st.markdown('<div class="sidebar-button">', unsafe_allow_html=True)
        if st.button("🏠 Utama"):
            st.session_state.page = "Utama"
        if st.button("📊 Prediksi"):
            st.session_state.page = "Prediksi"
        if st.button("📘 Tentang ISPU"):
            st.session_state.page = "Tentang ISPU"
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- Conditional Instruction section ---
    if st.session_state.page == "Prediksi":
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### 🧾 Instruksi Halaman")
        st.markdown(
            """
            <div style="border: 1px solid gray; padding: 10px; border-radius: 5px;">
            Halaman ini digunakan untuk memprediksi indeks PM2.5 di Jakarta Pusat
            <br>
            <br>

            Ikuti langkah-langkah berikut untuk menggunakan aplikasi ini:
            <ol>
            <li>Masukkan indeks PM2.5 selama 7 hari terakhir (tidak termasuk hari ini)</li>
            <li>Input no. 1 = PM2.5 pada hari ke-7 sebelumnya, Input no. 2 = PM2.5 pada hari ke-6 sebelumnya, dan seterusnya</li>
            <li>Klik “Predict” setelah semua input terisi</li>
            </ol>
            </div>
            """,
            unsafe_allow_html=True
        )



# --- Page content ---
if st.session_state.page == "Utama":
    st.markdown(
        """
        <div style="display: flex; justify-content: center; align-items: center; height: 90vh; text-align: center;">
            <div>
                <h1>Prediksi Indeks PM2.5 Jakarta Pusat untuk 7 Hari Ke Depan</h1>
                <p>Sebuah web yang membantu memprediksi indeks PM2.5 Jakarta Pusat untuk 7 hari ke depan</p>
        </div>
        """,
        unsafe_allow_html=True
    )

elif st.session_state.page == "Prediksi":
    st.markdown("<h2 style='text-align: center;'>Prediksi Indeks PM2.5 di Jakarta Pusat untuk 7 Hari Ke Depan</h2>", unsafe_allow_html=True)
    st.write("")  # spacing

    # --- Input form for 7 PM2.5 values ---
    pm_values = []
    for i in range(7, 0, -1):
        val = st.number_input(f"Input PM2.5 {i} hari lalu", min_value=0, step=1, key=f"pm_{i}")
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

        #  Display the prediction result
        week_dates = [date.today() + timedelta(days=i) for i in range(0, 7)]

        prediction_output = "Prediksi PM2.5 7 hari ke depan:\n\n"
        for d, v in zip(week_dates, pm_values_pred.flatten()):
            level = get_pm25_level(v)
            prediction_output += f"{d.strftime('%A, %d %b %Y')} = {int(v)} ({level})\n\n"

        st.success(prediction_output)
        
elif st.session_state.page == "Tentang ISPU":
        
        st.markdown("<h2 style='text-align: center;'>Tentang ISPU (Indeks Standar Pencemar Udara)</h2>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])  # You can adjust 1-2-1 ratio if needed

        with col2:
            st.image("ISPU.png", caption="Kategori ISPU dan Dampaknya terhadap Kesehatan", use_container_width =True)
