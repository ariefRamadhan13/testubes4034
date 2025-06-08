# home.py
import streamlit as st
import Tes1  # Mengimpor aplikasi Downhole dari Downhole1.py
import Tes2  # Mengimpor aplikasi ARIMA dari Tes2.py


def main():
    # Judul aplikasi utama
    st.title("Prediksi Produksi Minyak dan Parameter Reservoir")
    
    # Pilihan aplikasi yang ingin dijalankan
    app_choice = st.selectbox("Pilih Aplikasi", ["Prediksi Harga Minyak", "Analisis Downhole"])
    
    if app_choice == "Prediksi Harga Minyak":
        # Menjalankan aplikasi ARIMA dari Tes2.py
        Tes2.arima_app()
    elif app_choice == "Analisis Downhole":
        # Menjalankan aplikasi Downhole dari Downhole1.py
        Tes1.downhole_app()

if __name__ == "__main__":
    main()