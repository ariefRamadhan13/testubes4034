def arima_app():
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import streamlit as st
    from statsmodels.tsa.arima.model import ARIMA
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    # Menambahkan judul aplikasi
    st.title("Prediksi Harga Minyak Menggunakan ARIMA dengan Grid Search")

    # Upload file Excel
    uploaded_file = st.file_uploader("Pilih file Excel", type=["xlsx"])

    if uploaded_file is not None:
        # Memuat data dari file Excel
        data = pd.read_excel(uploaded_file, sheet_name='Sheet1')

        # Mengonversi kolom 'Date' menjadi datetime dan menjadikannya index
        data['Date'] = pd.to_datetime(data['Date'])
        data.set_index('Date', inplace=True)

        # Menampilkan lima baris pertama dari data untuk memeriksa
        st.write("Data pertama:")
        st.write(data.head())

        # Mengisi missing values dengan rata-rata nilai sebelumnya dan sesudahnya
        data['Price'] = data['Price'].fillna(data['Price'].interpolate())

        # Menampilkan data setelah pengisian missing values
        st.write("Data setelah pengisian missing values:")
        st.write(data.head())

        # Menampilkan plot harga minyak asli
        st.write("Plot harga minyak asli:")
        plt.figure(figsize=(10, 6))
        plt.plot(data.index, data['Price'], label='Harga Minyak Brent', color='blue')
        plt.title('Harga Minyak Brent')
        plt.xlabel('Tanggal')
        plt.ylabel('Harga (USD)')
        plt.legend()
        st.pyplot(plt)

        # Mengecek stasioneritas data dengan plot ACF dan PACF
        st.write("Plot ACF dan PACF:")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        plot_acf(data['Price'], lags=12, ax=ax1)
        plot_pacf(data['Price'], lags=12, ax=ax2)
        st.pyplot(fig)

        # Input untuk nilai maksimum p, d, dan q
        p_max = st.number_input("Masukkan nilai maksimum untuk p (AR) yang akan diuji:", min_value=1, max_value=5, value=3)
        d_max = st.number_input("Masukkan nilai maksimum untuk d (Differencing) yang akan diuji:", min_value=0, max_value=2, value=1)
        q_max = st.number_input("Masukkan nilai maksimum untuk q (MA) yang akan diuji:", min_value=1, max_value=5, value=3)

        # Fungsi untuk mencari parameter terbaik (p, d, q) menggunakan Grid Search
        def grid_search_arima(data, p_values, d_values, q_values):
            best_aic = np.inf
            best_order = None
            best_model = None

            for p in p_values:
                for d in d_values:
                    for q in q_values:
                        try:
                            # Membangun dan melatih model ARIMA dengan parameter (p, d, q)
                            model = ARIMA(data, order=(p, d, q))
                            model_fit = model.fit()

                            # Menghitung AIC (Akaike Information Criterion)
                            aic = model_fit.aic

                            # Memilih model dengan AIC terkecil
                            if aic < best_aic:
                                best_aic = aic
                                best_order = (p, d, q)
                                best_model = model_fit
                        except:
                            continue

            return best_order, best_model

        # Mencari nilai terbaik untuk p, d, q menggunakan Grid Search
        p_values = range(0, p_max + 1)  # Uji p dari 0 sampai p_max
        d_values = range(0, d_max + 1)  # Uji d dari 0 sampai d_max
        q_values = range(0, q_max + 1)  # Uji q dari 0 sampai q_max

        best_order, best_model = grid_search_arima(data['Price'], p_values, d_values, q_values)

        # Menampilkan hasil model terbaik
        st.write(f"Best ARIMA Model Order: {best_order}")
        st.write(f"Best Model AIC: {best_model.aic}")

        # Input untuk jumlah langkah prediksi (forecast steps) oleh pengguna
        forecast_steps = st.number_input("Masukkan jumlah langkah prediksi (forecast steps):", min_value=1, max_value=12, value=6)

        # Membuat prediksi untuk data masa depan
        forecast = best_model.forecast(steps=forecast_steps)

        # Tampilkan hasil prediksi
        st.write(f'Prediksi Harga Minyak untuk {forecast_steps} hari ke depan:')
        st.write(forecast)

        # Membuat plot untuk hasil prediksi
        st.write("Plot hasil prediksi harga minyak:")
        plt.figure(figsize=(10, 6))
        plt.plot(data.index, data['Price'], label='Harga Minyak Asli', color='blue')
        plt.plot(pd.date_range(data.index[-1], periods=forecast_steps+1, freq='M')[1:], forecast, label='Prediksi Harga Minyak', color='red')
        plt.title('Prediksi Harga Minyak Menggunakan ARIMA')
        plt.xlabel('Tanggal')
        plt.ylabel('Harga (USD)')
        plt.legend()
        st.pyplot(plt)

        # Menghitung dan menampilkan error evaluasi model
        y_true = data['Price'].tail(forecast_steps)  # Ambil data terakhir untuk evaluasi
        mae = mean_absolute_error(y_true, forecast)
        rmse = np.sqrt(mean_squared_error(y_true, forecast))
        st.write(f'Mean Absolute Error (MAE): {mae}')
        st.write(f'Root Mean Squared Error (RMSE): {rmse}')
