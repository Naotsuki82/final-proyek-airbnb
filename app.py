import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json

# ==============================================================================
# Konfigurasi Halaman dan Judul
# ==============================================================================
st.set_page_config(page_title="Prediksi Harga Airbnb di New York", layout="wide")

st.title("Prediksi Harga Airbnb di New York")
st.write("Aplikasi ini menggunakan input lokasi yang dinamis untuk prediksi yang lebih realistis.")
st.write("Dirancang oleh: Joseph H. A. dan Cliff Jordan J. A.")

# ==============================================================================
# Memuat Semua Aset yang Dibutuhkan (dengan Caching)
# ==============================================================================
@st.cache_resource
def load_resources():
    """Fungsi untuk memuat model, scaler, dan semua file data."""
    try:
        model = joblib.load('final_model.joblib')
        scaler = joblib.load('scaler.joblib')
        model_features = joblib.load('final_model_features_136.joblib')
        all_scaler_features = joblib.load('all_scaler_features.joblib')
        with open('geo_data.json', 'r') as f:
            geo_data = json.load(f)
        return model, scaler, model_features, all_scaler_features, geo_data
    except FileNotFoundError as e:
        st.error(f"File tidak ditemukan: {e.filename}. Pastikan semua file (.joblib, .json) ada di folder yang sama.")
        return None, None, None, None, None

model, scaler, model_features_136, all_scaler_features, geo_data = load_resources()

if not all([model, scaler, model_features_136, all_scaler_features, geo_data]):
    st.stop()

# ==============================================================================
# Helper Function (tidak berubah)
# ==============================================================================
def get_options_from_features(prefix, features):
    options = [f.replace(prefix, "").replace("_", " ") for f in features if f.startswith(prefix)]
    return ["Lainnya"] + sorted(list(set(options)))

# ==============================================================================
# BAGIAN 1: INPUT LOKASI (DI LUAR FORM)
# ==============================================================================
st.header("1: Pilih Lokasi Properti")
col1, col2 = st.columns(2)

with col1:
    borough_options = list(geo_data.keys())
    selected_borough = st.selectbox("Pilih Wilayah Utama (Borough)", borough_options, key="borough_selector")

with col2:
    neighbourhoods_in_borough = list(geo_data[selected_borough].keys())
    selected_neighbourhood = st.selectbox("Pilih Lingkungan", neighbourhoods_in_borough, key="neighbourhood_selector")

st.header(f"2: Sesuaikan lokasi presisi di dalam **{selected_neighbourhood}**:")
bounds = geo_data[selected_borough][selected_neighbourhood]

lat_min, lat_max = float(bounds['lat_min']), float(bounds['lat_max'])
lon_min, lon_max = float(bounds['lon_min']), float(bounds['lon_max'])

if lat_min >= lat_max: lat_max = lat_min + 0.0001
if lon_min >= lon_max: lon_max = lon_min + 0.0001

lat_col, lon_col = st.columns(2)
with lat_col:
    # --- PERUBAHAN: Menambahkan parameter 'step' ---
    latitude = st.slider(
        "Latitude",
        min_value=lat_min,
        max_value=lat_max,
        value=float(np.mean([lat_min, lat_max])),
        format="%.6f",  # Menambah presisi format
        step=0.000001,     # Menentukan langkah pergerakan slider
        key="lat_slider"
    )
with lon_col:
    # --- PERUBAHAN: Menambahkan parameter 'step' ---
    longitude = st.slider(
        "Longitude",
        min_value=lon_min,
        max_value=lon_max,
        value=float(np.mean([lon_min, lon_max])),
        format="%.6f",  # Menambah presisi format
        step=0.000001,     # Menentukan langkah pergerakan slider
        key="lon_slider"
    )

st.divider()

# ==============================================================================
# BAGIAN 2: INPUT DETAIL LAINNYA (DI DALAM FORM)
# ==============================================================================
room_type_options = ["Entire home/apt", "Private room", "Shared room"]
property_type_options = get_options_from_features("property_type_", all_scaler_features)
host_neighbourhood_options = get_options_from_features("host_neighbourhood_", all_scaler_features)

with st.form("property_details_form"):
    st.header("3: Masukkan Detail Properti Lainnya")
    
    st.subheader("Detail Utama Properti")
    d_col1, d_col2 = st.columns(2)
    with d_col1:
        accommodates = st.slider("Jumlah Tamu (Accommodates)", 1, 16, 2)
        bedrooms = st.slider("Jumlah Kamar Tidur (Bedrooms)", 0, 10, 1)
        beds = st.slider("Jumlah Tempat Tidur (Beds)", 0, 20, 1)
    with d_col2:
        minimum_nights = st.number_input("Minimum Malam Inap", min_value=1, value=1, step=1)
        minimum_minimum_nights = st.number_input("Minimum dari Minimum Malam", min_value=1, value=1, step=1)

    with st.expander("Klik untuk Detail Tambahan"):
        col3, col4, col5 = st.columns(3)
        with col3:
            st.markdown("**Tipe Properti & Ketersediaan**")
            room_type = st.selectbox("Tipe Ruangan", room_type_options)
            property_type = st.selectbox("Tipe Properti Spesifik", property_type_options)
            availability_30 = st.slider("Ketersediaan (30 hari)", 0, 30, 15)
            availability_60 = st.slider("Ketersediaan (60 hari)", 0, 60, 45)
            availability_90 = st.slider("Ketersediaan (90 hari)", 0, 90, 75)
            availability_365 = st.slider("Ketersediaan (365 hari)", 0, 365, 300)
            availability_eoy = st.slider("Ketersediaan Akhir Tahun (eoy)", 0, 1, 0)
        with col4:
            st.markdown("**Informasi Host**")
            host_total_listings_count = st.number_input("Total Listing Host", min_value=1, value=1)
            host_is_superhost = st.radio("Apakah Host adalah Superhost?", ["Ya", "Tidak"], index=1)
            host_neighbourhood = st.selectbox("Lingkungan Host", host_neighbourhood_options)
            host_response_rate = st.slider("Tingkat Respons Host (%)", 0, 100, 95)
            host_acceptance_rate = st.slider("Tingkat Penerimaan Host (%)", 0, 100, 90)
            host_response_time = st.selectbox("Waktu Respons Host", ["within an hour", "Lainnya"])
        with col5:
            st.markdown("**Ulasan & Estimasi**")
            number_of_reviews_l30d = st.slider("Jumlah Ulasan (30 hari terakhir)", 0, 10, 1)
            review_scores_rating = st.slider("Skor Ulasan (Rating Keseluruhan)", 1.0, 5.0, 4.5, 0.1)
            review_scores_cleanliness = st.slider("Skor Kebersihan", 1.0, 5.0, 4.5, 0.1)
            review_scores_location = st.slider("Skor Lokasi", 1.0, 5.0, 4.5, 0.1)
            estimated_occupancy_l365d = st.number_input("Estimasi Hari Terisi (1 thn)", value=150)
            estimated_revenue_l365d = st.number_input("Estimasi Pendapatan (1 thn)", value=5000)

    submitted = st.form_submit_button("Prediksi Harga per Malam")

# ==============================================================================
# Logika Prediksi
# ==============================================================================
if submitted:
    if model and scaler and model_features_136 and all_scaler_features:
        input_data = pd.DataFrame(columns=all_scaler_features)
        input_data.loc[0] = 0

        # Isi semua nilai numerik
        numerics = {
            'host_total_listings_count': host_total_listings_count, 'latitude': latitude, 'longitude': longitude,
            'accommodates': accommodates, 'bedrooms': bedrooms, 'beds': beds, 'minimum_nights': minimum_nights,
            'minimum_minimum_nights': minimum_minimum_nights, 'availability_30': availability_30,
            'availability_60': availability_60, 'availability_90': availability_90,
            'availability_365': availability_365, 'availability_eoy': availability_eoy,
            'calculated_host_listings_count': 1, # Menambahkan nilai default untuk kolom yang mungkin hilang
            'number_of_reviews_l30d': number_of_reviews_l30d,
            'estimated_occupancy_l365d': estimated_occupancy_l365d, 'estimated_revenue_l365d': estimated_revenue_l365d,
            'review_scores_rating': review_scores_rating, 'review_scores_cleanliness': review_scores_cleanliness,
            'review_scores_location': review_scores_location
        }
        for key, value in numerics.items():
            if key in input_data.columns: input_data[key] = value

        # Handle One-Hot Encoded Features
        def set_one_hot(prefix, value):
            if value != "Lainnya":
                col_name = f"{prefix}{value.replace(' ', '_')}"
                if col_name in input_data.columns: input_data[col_name] = 1
        
        set_one_hot("neighbourhood_group_cleansed_", selected_borough)
        set_one_hot("neighbourhood_cleansed_", selected_neighbourhood)
        set_one_hot("room_type_", room_type if room_type != "Entire home/apt" else "Lainnya")
        set_one_hot("property_type_", property_type)
        set_one_hot("host_neighbourhood_", host_neighbourhood)
        if host_response_time == 'within an hour': set_one_hot("host_response_time_", "within an hour")
        if host_is_superhost == 'Ya':
            if 'host_is_superhost_t' in input_data.columns:
                input_data['host_is_superhost_t'] = 1
        
        def set_closest_rate(prefix, value, all_features):
            rate_cols = [f for f in all_features if f.startswith(prefix)]
            if rate_cols:
                available_rates = [int(''.join(filter(str.isdigit, f))) for f in rate_cols if ''.join(filter(str.isdigit, f))]
                if available_rates:
                    closest_rate = min(available_rates, key=lambda x: abs(x - value))
                    col_name = [f for f in rate_cols if str(closest_rate) in f][0]
                    if col_name in input_data.columns: input_data[col_name] = 1
        set_closest_rate("host_response_rate_", host_response_rate, all_scaler_features)
        set_closest_rate("host_acceptance_rate_", host_acceptance_rate, all_scaler_features)
        
        input_data = input_data[all_scaler_features]
        input_df_scaled = scaler.transform(input_data)
        scaled_df = pd.DataFrame(input_df_scaled, columns=all_scaler_features)
        final_input_for_model = scaled_df[model_features_136]
        
        prediction_log = model.predict(final_input_for_model)
        prediction_dollar = np.expm1(prediction_log)[0]
        
        st.success(f"## Prediksi Harga per Malam: ${prediction_dollar:,.2f}")
