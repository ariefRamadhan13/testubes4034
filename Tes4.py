def LSTM_app():    
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from sklearn.metrics import mean_squared_error, r2_score
    import streamlit as st
    from io import BytesIO

    # Fungsi untuk memuat dan memproses data
    def load_and_process_data(uploaded_file):
        df = pd.read_excel(uploaded_file)
        
        # Memeriksa apakah kolom yang diperlukan ada
        if 'Date' not in df.columns or 'Open' not in df.columns:
            st.error("Data tidak mengandung kolom 'Date' atau 'Open'")
            return None
        
        # Konversi tanggal menjadi format datetime
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        return df

    # Fungsi untuk membuat dataset dengan window 'look_back'
    def create_dataset(data, look_back):
        X, y = [], []
        for i in range(look_back, len(data)):
            X.append(data[i-look_back:i, 0])  # Menggunakan hanya fitur 'Open'
            y.append(data[i, 0])  # Target adalah harga 'Open'
        return np.array(X), np.array(y)

    # Fungsi untuk membangun dan melatih model LSTM
    def build_and_train_model(X_train, y_train, epochs):
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(units=1))

        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=1)
        
        return model

    # Fungsi untuk prediksi masa depan
    def predict_future(model, last_data, days_to_predict, scaler, look_back):
        predicted_prices = []
        for _ in range(days_to_predict):
            next_day_prediction = model.predict(last_data)
            predicted_prices.append(next_day_prediction[0, 0])
            # Update last_data dengan hasil prediksi
            last_data = np.append(last_data[:, 1:, :], next_day_prediction.reshape(1, 1, 1), axis=1)
        
        predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))
        return predicted_prices

    # Streamlit app
    def app():
        st.title("Prediksi Harga Minyak dengan LSTM")

        # Input file
        uploaded_file = st.file_uploader("Pilih file Excel", type="xlsx")
        if uploaded_file is not None:
            df = load_and_process_data(uploaded_file)
            if df is None:
                return

            # Menampilkan beberapa data pertama
            st.write(df.head())

            # Inputan parameter
            look_back = st.slider("Pilih Jumlah Lookback (Hari)", 30, 365, 365)
            epochs = st.slider("Pilih Jumlah Epoch", 1, 100, 50)
            days_to_predict = st.slider("Pilih Jumlah Hari untuk Prediksi", 1, 30, 10)

            # Tombol Apply untuk memulai proses setelah input parameter
            apply_button = st.button("Apply")

            if apply_button:
                # Menormalisasi hanya harga 'Open' untuk model LSTM
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_data = scaler.fit_transform(df[['Open']])

                # Persiapkan data
                train_size = int(len(scaled_data) * 0.7)
                train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]
                
                X_train, y_train = create_dataset(train_data, look_back)
                X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)  # Menggunakan 1 fitur (Open)

                # Membangun dan melatih model
                model = build_and_train_model(X_train, y_train, epochs)

                # Melakukan prediksi pada data testing
                X_test, y_test = create_dataset(test_data, look_back)
                X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

                predictions = model.predict(X_test)
                predictions = scaler.inverse_transform(predictions)
                y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

                # Evaluasi model
                rmse = np.sqrt(mean_squared_error(y_test, predictions))
                r2 = r2_score(y_test, predictions)

                st.write(f"RMSE: {rmse}")
                st.write(f"R²: {r2}")

                # Visualisasi hasil prediksi vs nilai aktual
                st.subheader('Visualisasi Prediksi Harga Minyak')
                plt.figure(figsize=(10, 6))
                plt.plot(df.index[train_size+look_back:], y_test, color='blue', label='Harga Minyak Aktual')
                plt.plot(df.index[train_size+look_back:], predictions, color='red', label='Prediksi Harga Minyak')
                plt.title('Prediksi Harga Minyak menggunakan LSTM')
                plt.xlabel('Tanggal')
                plt.ylabel('Harga Minyak')
                plt.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(plt)

                # Prediksi Harga Minyak Masa Depan
                st.subheader(f'Prediksi Harga Minyak {days_to_predict} Hari Ke Depan')
                last_data = scaled_data[-look_back:].reshape(1, look_back, 1)
                predicted_prices = predict_future(model, last_data, days_to_predict, scaler, look_back)

                future_dates = pd.date_range(df.index[-1], periods=days_to_predict+1, freq='D')[1:]

                # Visualisasi Prediksi Masa Depan
                plt.figure(figsize=(10, 6))
                plt.plot(df.index[-look_back:], df['Open'].iloc[-look_back:], color='blue', label='Harga Minyak Terakhir')
                plt.plot(future_dates, predicted_prices, color='green', label=f'Prediksi Harga Minyak {days_to_predict} Hari Ke Depan')
                plt.title(f'Prediksi Harga Minyak {days_to_predict} Hari Ke Depan')
                plt.xlabel('Tanggal')
                plt.ylabel('Harga Minyak')
                plt.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(plt)

                # Simpan Prediksi ke Excel
                predictions_df = pd.DataFrame({
                    'Tanggal': future_dates,
                    'Prediksi Harga Minyak': predicted_prices.flatten()
                })

                # Menggunakan BytesIO untuk menyimpan file dalam format Excel
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    predictions_df.to_excel(writer, index=False, sheet_name="Prediksi Harga Minyak")
                
                # Menyediakan tombol download untuk file hasil prediksi
                output.seek(0)
                st.download_button(
                    label="Unduh Prediksi Harga Minyak (Excel)",
                    data=output,
                    file_name="prediksi_harga_minyak.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )


    app()
