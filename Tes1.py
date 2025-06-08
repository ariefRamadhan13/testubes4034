import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def downhole_app():
    # Fungsi untuk memuat dan memproses data
    def load_and_process_data(uploaded_file):
        # Memuat data dari file Excel
        data = pd.read_excel(uploaded_file, sheet_name='Master Data')
        
        # Menampilkan lima baris pertama untuk memeriksa data
        st.write("Data pertama (head):", data.head())
        
        # Memeriksa kolom dan tipe data setiap kolom
        st.write("Tipe data kolom:\n", data.dtypes)
        
        # Memeriksa jumlah data yang hilang di setiap kolom
        missing_data = data.isnull().sum()
        st.write("Data yang hilang per kolom:\n", missing_data)

        # Memisahkan data yang tidak hilang untuk pelatihan model
        data_non_missing = data[data['AVG_DOWNHOLE_PRESSURE'] != 0]
        
        # Memisahkan data yang hilang pada AVG_DOWNHOLE_PRESSURE dan AVG_DOWNHOLE_TEMPERATURE
        data_missing = data[data['AVG_DOWNHOLE_PRESSURE'] == 0]
        
        # Menghapus baris dengan target yang memiliki nilai NaN (hilang)
        data_non_missing = data_non_missing.dropna(subset=['AVG_DOWNHOLE_PRESSURE', 'AVG_DOWNHOLE_TEMPERATURE'])
        
        # Memisahkan fitur dan target
        features = ['AVG_DP_TUBING', 'AVG_ANNULUS_PRESS', 'AVG_CHOKE_SIZE_P', 'AVG_WHP_P', 'BORE_OIL_VOL', 'BORE_GAS_VOL']
        X_train = data_non_missing[features]
        y_train_pressure = data_non_missing['AVG_DOWNHOLE_PRESSURE']
        y_train_temperature = data_non_missing['AVG_DOWNHOLE_TEMPERATURE']

        return data, X_train, y_train_pressure, y_train_temperature, features

    # Fungsi untuk membuat model prediksi dan evaluasi
    def train_and_predict(X_train, y_train_pressure, y_train_temperature, data, features):
        # Membuat model Random Forest untuk memprediksi AVG_DOWNHOLE_PRESSURE
        model_pressure = RandomForestRegressor(n_estimators=100, random_state=42)
        model_pressure.fit(X_train, y_train_pressure)

        # Membuat model Random Forest untuk memprediksi AVG_DOWNHOLE_TEMPERATURE
        model_temperature = RandomForestRegressor(n_estimators=100, random_state=42)
        model_temperature.fit(X_train, y_train_temperature)

        # Memprediksi nilai untuk seluruh data (termasuk data yang hilang)
        predicted_pressure_complete = model_pressure.predict(data[features])
        predicted_temperature_complete = model_temperature.predict(data[features])

        # Menyimpan hasil prediksi pada seluruh data (termasuk data yang hilang)
        data['Predicted_AVG_DOWNHOLE_PRESSURE'] = predicted_pressure_complete
        data['Predicted_AVG_DOWNHOLE_TEMPERATURE'] = predicted_temperature_complete

        return model_pressure, model_temperature, data

    # Fungsi untuk plotting data
    def plot_data(data):
        # Memisahkan data untuk yang telah diprediksi (nilai yang hilang) untuk plotting
        data_predicted = data[data['Predicted_AVG_DOWNHOLE_PRESSURE'].notnull()]

        # Memisahkan data lengkap untuk plotting
        data_true = data[data['AVG_DOWNHOLE_PRESSURE'] != 0]

        # Membuat plot dengan semua data dalam satu plot
        plt.figure(figsize=(12, 6))

        # Plot untuk data asli AVG_DOWNHOLE_PRESSURE (warna biru)
        plt.plot(data_true['DATEPRD'], data_true['AVG_DOWNHOLE_PRESSURE'], label='True AVG_DOWNHOLE_PRESSURE', color='blue')

        # Plot untuk data prediksi AVG_DOWNHOLE_PRESSURE (warna merah)
        plt.plot(data_predicted['DATEPRD'], data_predicted['Predicted_AVG_DOWNHOLE_PRESSURE'], label='Predicted AVG_DOWNHOLE_PRESSURE', color='red')

        # Plot untuk data asli AVG_DOWNHOLE_TEMPERATURE (warna hijau)
        plt.plot(data_true['DATEPRD'], data_true['AVG_DOWNHOLE_TEMPERATURE'], label='True AVG_DOWNHOLE_TEMPERATURE', color='green')

        # Plot untuk data prediksi AVG_DOWNHOLE_TEMPERATURE (warna orange)
        plt.plot(data_predicted['DATEPRD'], data_predicted['Predicted_AVG_DOWNHOLE_TEMPERATURE'], label='Predicted AVG_DOWNHOLE_TEMPERATURE', color='orange')

        # Menambahkan judul dan label
        plt.title('Data Asli dan Prediksi AVG_DOWNHOLE_PRESSURE dan AVG_DOWNHOLE_TEMPERATURE vs Waktu')
        plt.xlabel('Tanggal')
        plt.ylabel('Nilai')
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Menampilkan plot
        st.pyplot(plt)

    # Fungsi untuk evaluasi model
    def evaluate_model(model_pressure, model_temperature, X_train, y_train_pressure, y_train_temperature):
        # Evaluasi model untuk AVG_DOWNHOLE_PRESSURE (menggunakan data yang tidak hilang)
        y_pred_pressure = model_pressure.predict(X_train)
        st.write("Evaluasi untuk AVG_DOWNHOLE_PRESSURE:")
        st.write("R²:", r2_score(y_train_pressure, y_pred_pressure))
        st.write("MAE:", mean_absolute_error(y_train_pressure, y_pred_pressure))

        # Evaluasi model untuk AVG_DOWNHOLE_TEMPERATURE (menggunakan data yang tidak hilang)
        y_pred_temperature = model_temperature.predict(X_train)
        st.write("Evaluasi untuk AVG_DOWNHOLE_TEMPERATURE:")
        st.write("R²:", r2_score(y_train_temperature, y_pred_temperature))
        st.write("MAE:", mean_absolute_error(y_train_temperature, y_pred_temperature))

    # Upload file Excel
    uploaded_file = st.file_uploader("Pilih file Excel", type="xlsx")
    
    if uploaded_file is not None:
        # Memuat dan memproses data
        data, X_train, y_train_pressure, y_train_temperature, features = load_and_process_data(uploaded_file)
        
        # Melatih model dan prediksi
        model_pressure, model_temperature, data = train_and_predict(X_train, y_train_pressure, y_train_temperature, data, features)
        
        # Menampilkan plot
        plot_data(data)
        
        # Menampilkan evaluasi model
        evaluate_model(model_pressure, model_temperature, X_train, y_train_pressure, y_train_temperature)

