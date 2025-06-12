import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Fungsi utama aplikasi
def downhole_app():
    st.title("Prediksi Tekanan dan Temperatur Downhole dari Surface Data")

    # Memuat dan memproses data
    def load_and_process_data(uploaded_file):
        st.write("Memuat data...")
        data = pd.read_excel(uploaded_file, sheet_name='Master Data')
        
        # Menampilkan data pertama (head)
        st.write("Data pertama (head):", data.head())
        
        # Menampilkan tipe data kolom
        st.write("Tipe data kolom:", data.dtypes)
        
        # Menampilkan data yang hilang
        missing_data = data.isnull().sum()
        st.write("Data yang hilang per kolom:\n", missing_data)
        
        # Menyaring data yang tidak hilang
        data_non_missing = data[data['AVG_DOWNHOLE_PRESSURE'] != 0]
        data_non_missing = data_non_missing.dropna(subset=['AVG_DOWNHOLE_PRESSURE', 'AVG_DOWNHOLE_TEMPERATURE'])
        
        # Mengambil fitur yang akan digunakan untuk prediksi
        features = ['AVG_DP_TUBING', 'AVG_ANNULUS_PRESS', 'AVG_CHOKE_SIZE_P', 'AVG_WHP_P', 'BORE_OIL_VOL', 'BORE_GAS_VOL']
        
        # Menentukan data fitur (X) dan target (y) untuk tekanan dan temperatur
        X_train = data_non_missing[features]
        y_train_pressure = data_non_missing['AVG_DOWNHOLE_PRESSURE']
        y_train_temperature = data_non_missing['AVG_DOWNHOLE_TEMPERATURE']
        
        return data, X_train, y_train_pressure, y_train_temperature, features

    # Menampilkan distribusi data sebelum pelatihan model
    def plot_distribution_before_model(data):
        st.write("Distribusi Tekanan dan Temperatur Downhole Sebelum Pelatihan Model")

        # Memisahkan data untuk yang asli
        data_true = data[data['AVG_DOWNHOLE_PRESSURE'] != 0]

        # Plot distribusi tekanan (sebelum pelatihan model)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.hist(data_true['AVG_DOWNHOLE_PRESSURE'], bins=20, alpha=0.7, label='True AVG_DOWNHOLE_PRESSURE', color='blue')
        ax.set_title('Distribusi Tekanan Downhole Sebelum Pelatihan Model')
        ax.set_xlabel('Tekanan (PSI)')
        ax.set_ylabel('Frekuensi')
        ax.legend()
        st.pyplot(fig)

        # Plot distribusi temperatur (sebelum pelatihan model)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.hist(data_true['AVG_DOWNHOLE_TEMPERATURE'], bins=20, alpha=0.7, label='True AVG_DOWNHOLE_TEMPERATURE', color='green')
        ax.set_title('Distribusi Temperatur Downhole Sebelum Pelatihan Model')
        ax.set_xlabel('Temperatur (°C)')
        ax.set_ylabel('Frekuensi')
        ax.legend()
        st.pyplot(fig)

    # Melatih model dan prediksi
    def train_and_predict(X_train, y_train_pressure, y_train_temperature, data, features):
        st.write("Melatih model dan membuat prediksi...")
        
        # Membuat model Random Forest untuk tekanan downhole
        model_pressure = RandomForestRegressor(n_estimators=100, random_state=42)
        model_pressure.fit(X_train, y_train_pressure)
        
        # Membuat model Random Forest untuk temperatur downhole
        model_temperature = RandomForestRegressor(n_estimators=100, random_state=42)
        model_temperature.fit(X_train, y_train_temperature)
        
        # Memprediksi seluruh data
        predicted_pressure_complete = model_pressure.predict(data[features])
        predicted_temperature_complete = model_temperature.predict(data[features])
        
        # Menyimpan hasil prediksi pada data
        data['Predicted_AVG_DOWNHOLE_PRESSURE'] = predicted_pressure_complete
        data['Predicted_AVG_DOWNHOLE_TEMPERATURE'] = predicted_temperature_complete
        
        return model_pressure, model_temperature, data

    # Menampilkan visualisasi data setelah pelatihan model
    def plot_data_after_model(data):
        st.write("Menampilkan visualisasi data setelah pelatihan model...")

        # Memisahkan data untuk yang telah diprediksi dan yang asli
        data_predicted = data[data['Predicted_AVG_DOWNHOLE_PRESSURE'].notnull()]
        data_true = data[data['AVG_DOWNHOLE_PRESSURE'] != 0]
        
        # Membuat plot dengan Matplotlib untuk tekanan dan temperatur over waktu
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(data_true['DATEPRD'], data_true['AVG_DOWNHOLE_PRESSURE'], label='True AVG_DOWNHOLE_PRESSURE', color='blue')
        ax.plot(data_predicted['DATEPRD'], data_predicted['Predicted_AVG_DOWNHOLE_PRESSURE'], label='Predicted AVG_DOWNHOLE_PRESSURE', color='red')
        ax.plot(data_true['DATEPRD'], data_true['AVG_DOWNHOLE_TEMPERATURE'], label='True AVG_DOWNHOLE_TEMPERATURE', color='green')
        ax.plot(data_predicted['DATEPRD'], data_predicted['Predicted_AVG_DOWNHOLE_TEMPERATURE'], label='Predicted AVG_DOWNHOLE_TEMPERATURE', color='orange')
        
        ax.set_title('Data Asli dan Prediksi AVG_DOWNHOLE_PRESSURE dan AVG_DOWNHOLE_TEMPERATURE vs Waktu')
        ax.set_xlabel('Tanggal')
        ax.set_ylabel('Nilai')
        ax.legend()
        ax.tick_params(axis='x', rotation=45)
        st.pyplot(fig)

        # Membuat plot interaktif dengan Plotly
        fig_plotly = px.line(data_predicted, x='DATEPRD', y=['Predicted_AVG_DOWNHOLE_PRESSURE', 'Predicted_AVG_DOWNHOLE_TEMPERATURE'],
                             labels={'value': 'Predicted Value', 'variable': 'Predictions'}, title="Predicted Values Over Time")
        st.plotly_chart(fig_plotly)

    # Evaluasi model
    def evaluate_model(model_pressure, model_temperature, X_train, y_train_pressure, y_train_temperature):
        st.write("Evaluasi model...")

        # Menghitung evaluasi untuk tekanan downhole
        y_pred_pressure = model_pressure.predict(X_train)
        st.write("Evaluasi untuk AVG_DOWNHOLE_PRESSURE:")
        st.write(f"R²: {r2_score(y_train_pressure, y_pred_pressure)}")
        st.write(f"MAE: {mean_absolute_error(y_train_pressure, y_pred_pressure)}")

        # Menghitung evaluasi untuk temperatur downhole
        y_pred_temperature = model_temperature.predict(X_train)
        st.write("Evaluasi untuk AVG_DOWNHOLE_TEMPERATURE:")
        st.write(f"R²: {r2_score(y_train_temperature, y_pred_temperature)}")
        st.write(f"MAE: {mean_absolute_error(y_train_temperature, y_pred_temperature)}")

        # Menampilkan metrik evaluasi dalam bentuk tabel
        metrics = pd.DataFrame({
            "Model": ["Pressure", "Temperature"],
            "R²": [r2_score(y_train_pressure, y_pred_pressure), r2_score(y_train_temperature, y_pred_temperature)],
            "MAE": [mean_absolute_error(y_train_pressure, y_pred_pressure), mean_absolute_error(y_train_temperature, y_pred_temperature)]
        })
        st.write("Model Evaluation Metrics:")
        st.write(metrics)

    # Fungsi untuk menyiapkan file CSV untuk diunduh
    def create_downloadable_csv(data):
        st.write("Membuat file CSV untuk unduhan...")
        
        # Membuat DataFrame untuk data yang akan diunduh
        download_data = data[['DATEPRD', 'Predicted_AVG_DOWNHOLE_PRESSURE', 'Predicted_AVG_DOWNHOLE_TEMPERATURE']]
        
        # Mengonversi DataFrame menjadi CSV
        csv = download_data.to_csv(index=False)
        st.download_button(
            label="Download Prediksi Tekanan dan Temperatur",
            data=csv,
            file_name="prediksi_downhole.csv",
            mime="text/csv"
        )

    # Upload file Excel
    uploaded_file = st.file_uploader("Pilih file Excel", type="xlsx")
    if uploaded_file is not None:
        # Langkah-langkah aplikasi
        data, X_train, y_train_pressure, y_train_temperature, features = load_and_process_data(uploaded_file)
        
        # Menampilkan distribusi data sebelum pelatihan model
        plot_distribution_before_model(data)
        
        # Melatih model dan prediksi
        model_pressure, model_temperature, data = train_and_predict(X_train, y_train_pressure, y_train_temperature, data, features)
        
        # Menampilkan visualisasi setelah pelatihan model
        plot_data_after_model(data)
        
        # Evaluasi model
        evaluate_model(model_pressure, model_temperature, X_train, y_train_pressure, y_train_temperature)
        
        # Menyiapkan file CSV untuk diunduh
        create_downloadable_csv(data)
