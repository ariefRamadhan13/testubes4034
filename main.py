# home.py
import streamlit as st
import Tes1  # Mengimpor aplikasi Downhole dari Downhole1.py
import Tes2  # Mengimpor aplikasi ARIMA dari Tes2.py
import Tes3
import Tes4
from PIL import Image

img = Image.open("erik.png")
st.image(img,width = 1500)

def main():
    # Judul aplikasi utama
    st.title("Prediksi Harga Minyak, Downhole dan Litologi")
    
    # Pilihan aplikasi yang ingin dijalankan
    app_choice = st.selectbox("Pilih Aplikasi", ["Prediksi Harga Minyak (ARIMA Model)", "Prediksi Harga Minyak (LSTM Model)", "Analisis Downhole", "Lithology Prediction"])
    
    if app_choice == "Prediksi Harga Minyak (ARIMA Model)":
        # Menjalankan aplikasi ARIMA dari Tes2.py
        Tes2.arima_app()
    elif app_choice == "Prediksi Harga Minyak (LSTM Model)":
        Tes4.LSTM_app()
        # Menjalankan aplikasi Downhole dari Downhole1.py
    elif app_choice == "Analisis Downhole":
        # Menjalankan aplikasi Downhole dari Downhole1.py
        Tes1.downhole_app()
    elif app_choice == "Lithology Prediction":
        # Menjalankan aplikasi Downhole dari Downhole1.py
        Tes3.lithology_app()

if __name__ == "__main__":
    main()
