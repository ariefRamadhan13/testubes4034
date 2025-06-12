# home.py
import streamlit as st
import Tes1  # Mengimpor aplikasi Downhole dari Downhole1.py
import Tes2  # Mengimpor aplikasi ARIMA dari Tes2.py
import Tes3
import Tes4
import Tes5
from PIL import Image

img = Image.open("Logo_Institut_Teknologi_Bandung (2).png")
st.image(img,width = 1500)

def main():
    # Judul aplikasi utama
    st.title("Prediksi Harga Minyak, Downhole dan Litologi")
    
    # Pilihan aplikasi yang ingin dijalankan
    app_choice = st.selectbox("Pilih Aplikasi", ["Prediksi Harga Minyak (ARIMA Model)", "Prediksi Harga Minyak (LSTM Model)","Prediksi Harga Minyak (RNN Model)", "Analisis Downhole P&T (RF)", "Prediksi Litologi (RF)"])
    
    if app_choice == "Prediksi Harga Minyak (ARIMA Model)":
        # Menjalankan aplikasi ARIMA dari Tes2.py
        Tes2.arima_app()
    elif app_choice == "Prediksi Harga Minyak (LSTM Model)":
        Tes4.LSTM_app()
        # Menjalankan aplikasi Downhole dari Downhole1.py
    elif app_choice == "Prediksi Harga Minyak (RNN Model)":
        Tes5.RNN_app()
        # Menjalankan aplikasi Downhole dari Downhole1.py
    elif app_choice == "Analisis Downhole P&T (RF)":
        # Menjalankan aplikasi Downhole dari Downhole1.py
        Tes1.downhole_app()
    elif app_choice == "Prediksi Litologi (RF)":
        # Menjalankan aplikasi Downhole dari Downhole1.py
        Tes3.lithology_app()

if __name__ == "__main__":
    main()
