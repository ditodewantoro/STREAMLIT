import pandas as pd
import pickle
import streamlit as st
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load pipeline model (menggunakan pickle)
with open("obesity_pipeline_model.pkl", "rb") as f:
    model = pickle.load(f)

# Pilihan kategori sesuai data latih
gender_options = ['Male', 'Female', 'Other']
calc_options = ['no', 'Sometimes', 'Frequently', 'Always']
favc_options = ['yes', 'no']
scc_options = ['yes', 'no']
smoke_options = ['yes', 'no']
family_history_options = ['yes', 'no']
caec_options = ['no', 'Sometimes', 'Frequently', 'Always']
mtrans_options = ['Automobile', 'Motorbike', 'Bike', 'Public_Transportation', 'Walking']

# Fungsi prediksi
def make_prediction(input_data):
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)[0]
    categories = ['Berat_Badan_Normal', 'Kelebihan_Berat_Badan_Level_I', 'Kelebihan_Berat_Badan_Level_II',
                  'Obesitas_Tipe_I', 'Obesitas_Tipe_II', 'Obesitas_Tipe_III', 'Berat_Badan_Kurang']
    return f"Prediksi tingkat obesitas Anda adalah: {categories[prediction]}"

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Obesitas", page_icon=":guardsman:", layout="wide")
st.title("Prediksi Tingkat Obesitas Berdasarkan Data Pengguna")
st.markdown("### Silakan lengkapi data berikut:")

# Form input
with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input('Usia', 1, 100, 25)
        height = st.number_input('Tinggi Badan (cm)', 100, 250, 170)
        weight = st.number_input('Berat Badan (kg)', 30, 200, 70)
        fcvc = st.slider('Frekuensi Konsumsi Sayur (FCVC)', 0.0, 3.0, 2.0)
        ncp = st.slider('Jumlah Makanan Pokok per Hari (NCP)', 1.0, 4.0, 3.0)
        ch2o = st.slider('Konsumsi Air Harian (liter)', 0.0, 3.0, 2.0)
        faf = st.slider('Aktivitas Fisik (jam/minggu)', 0.0, 5.0, 1.0)
        tue = st.slider('Durasi Layar (jam/hari)', 0.0, 5.0, 2.0)
    with col2:
        gender = st.selectbox('Jenis Kelamin', gender_options)
        calc = st.selectbox('Konsumsi Alkohol (CALC)', calc_options)
        favc = st.selectbox('Konsumsi Makanan Tinggi Kalori (FAVC)', favc_options)
        scc = st.selectbox('Pantau Konsumsi Kalori (SCC)', scc_options)
        smoke = st.selectbox('Merokok (SMOKE)', smoke_options)
        family = st.selectbox('Riwayat Kegemukan dalam Keluarga', family_history_options)
        caec = st.selectbox('Kebiasaan Ngemil (CAEC)', caec_options)
        mtrans = st.selectbox('Transportasi Utama (MTRANS)', mtrans_options)

    submitted = st.form_submit_button("Prediksi")
    if submitted:
        user_input = {
            'Age': age,
            'Height': height,
            'Weight': weight,
            'FCVC': fcvc,
            'NCP': ncp,
            'CH2O': ch2o,
            'FAF': faf,
            'TUE': tue,
            'Gender': gender,
            'CALC': calc,
            'FAVC': favc,
            'SCC': scc,
            'SMOKE': smoke,
            'family_history_with_overweight': family,
            'CAEC': caec,
            'MTRANS': mtrans
        }
        with st.spinner("Melakukan prediksi..."):
            result = make_prediction(user_input)
            st.success(result)

# Footer
st.markdown("---")
st.markdown("#### Aplikasi ini menggunakan model pipeline klasifikasi obesitas berdasarkan data WHO.")
