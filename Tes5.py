def RNN_app():
    import streamlit as st
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler
    from keras.models import Sequential
    from keras.layers import Dense, SimpleRNN, Dropout
    import io

    # Fungsi untuk memprediksi harga minyak setelah data tes terakhir
    def predict_future_prices_from_test(model, test_data, future_days, scaler, time_step=50):
        # Mengambil data terakhir dari data tes untuk prediksi
        input_data = test_data[-time_step:]
        input_data = np.reshape(input_data, (1, time_step, 1))  # Menyesuaikan shape data untuk input model
        
        future_predictions = []
        
        # Prediksi harga untuk 'future_days' ke depan
        for _ in range(future_days):
            pred = model.predict(input_data)  # Melakukan prediksi
            future_predictions.append(pred[0,0])  # Menyimpan prediksi dalam list
            input_data = np.append(input_data[:,1:,:], pred.reshape(1, 1, 1), axis=1)  # Update data input untuk prediksi selanjutnya
        
        # Mengubah hasil prediksi dari skala 0-1 ke harga asli
        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
        
        return future_predictions

    # Streamlit App
    st.title('Oil Price Prediction with RNN')

    # Step 1: Upload the Excel file
    uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

    if uploaded_file is not None:
        # Step 2: Read the Excel file
        data = pd.read_excel(uploaded_file, sheet_name='Sheet1')

        # Display basic info about the data
        st.subheader("Data Information")
        data_info = data.info()
        print(data_info)  # Show the data info
        
        # Step 3: Handle missing values by interpolation
        st.subheader("Filling Missing Values with Interpolation")
        data = data.interpolate(method='linear', axis=0)  # Interpolate missing values
        st.write("Missing values (nulls) are now filled. Below is the updated data info:")
        data_info_after_interpolation = data.info()
        st.text(data_info_after_interpolation)  # Show data info after interpolation
        
        # Show the first few rows of the data after interpolation
        st.write(data.head())

        # Step 4: Show Price vs Time and Normalized Price vs Time
        st.subheader("Price vs Time")
        fig, ax = plt.subplots(figsize=(16,6))  # Buat figure dan ax terlebih dahulu
        ax.plot(data['Date'], data['Price'], label='Price')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price USD ($/Barrel)')
        ax.set_title('Price vs Time')
        st.pyplot(fig)  # Pass the figure to st.pyplot()

        # Normalized Price vs Time
        scaler = MinMaxScaler(feature_range=(0, 1))
        normalized_price = scaler.fit_transform(data['Price'].values.reshape(-1, 1))
        
        st.subheader("Normalized Price vs Time")
        fig, ax = plt.subplots(figsize=(16,6))  # Buat figure dan ax terlebih dahulu
        ax.plot(data['Date'], normalized_price, label='Normalized Price', color='orange')
        ax.set_xlabel('Date')
        ax.set_ylabel('Normalized Price')
        ax.set_title('Normalized Price vs Time')
        st.pyplot(fig)  # Pass the figure to st.pyplot()

        # Step 5: Allow user to adjust the split ratio using a slider
        split_ratio = st.slider("Select the fraction of data for training", min_value=0.5, max_value=0.9, value=0.7)

        # Step 6: Prepare the dataset
        data['Date'] = pd.to_datetime(data['Date'])  # converting to date time object
        length_data = len(data)
        length_train = round(length_data * split_ratio)
        length_validation = length_data - length_train

        train_data = data[:length_train].iloc[:, :2]
        validation_data = data[length_train:].iloc[:, :2]

        dataset_train = train_data['Price'].values
        dataset_train = np.reshape(dataset_train, (-1, 1))

        # Normalize dataset
        scaler = MinMaxScaler(feature_range=(0, 1))
        dataset_train_scaled = scaler.fit_transform(dataset_train)

        # Prepare X_train and y_train
        X_train = []
        y_train = []
        time_step = 50

        for i in range(time_step, length_train):
            X_train.append(dataset_train_scaled[i - time_step:i, 0])
            y_train.append(dataset_train_scaled[i, 0])

        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        y_train = np.reshape(y_train, (y_train.shape[0], 1))

        # Step 7: Build the RNN model
        regressor = Sequential()
        regressor.add(SimpleRNN(units=50, activation="tanh", return_sequences=True, input_shape=(X_train.shape[1], 1)))
        regressor.add(Dropout(0.2))

        regressor.add(SimpleRNN(units=50, activation="tanh", return_sequences=True))
        regressor.add(Dropout(0.2))

        regressor.add(SimpleRNN(units=50, activation="tanh", return_sequences=True))
        regressor.add(Dropout(0.2))

        regressor.add(SimpleRNN(units=50))
        regressor.add(Dropout(0.2))

        regressor.add(Dense(units=1))
        regressor.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])

        # Step 8: Get user input for Epoch and prediction days
        epochs = st.slider("Select the number of epochs", min_value=5, max_value=100, value=25)
        future_days = st.slider("Select the number of days to predict", min_value=1, max_value=30, value=7)

        # Step 9: Train the model and display loss vs Epochs graph
        if st.button('Apply'):
            # Fit the RNN model
            history = regressor.fit(X_train, y_train, epochs=epochs, batch_size=32)

            # Plotting loss vs Epochs
            st.write("Training Loss vs Epochs:")
            fig, ax = plt.subplots(figsize=(10, 5))  # Buat figure dan ax terlebih dahulu
            ax.plot(history.history["loss"])
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Loss")
            ax.set_title("Simple RNN model, Loss vs Epoch")
            st.pyplot(fig)  # Pass the figure to st.pyplot()

            # Step 10: Predict the future prices
            last_test_data = scaler.fit_transform(validation_data.Price.values.reshape(-1, 1))
            predicted_prices = predict_future_prices_from_test(regressor, last_test_data, future_days, scaler, time_step)

            # Display the predictions
            st.write(f"Predicted prices for the next {future_days} days:")
            st.write(predicted_prices)

            # Plot predictions
            st.write(f"Predicted Oil Prices for the Next {future_days} Days")
            fig, ax = plt.subplots(figsize=(16, 6))  # Buat figure dan ax terlebih dahulu
            ax.plot(range(1, future_days + 1), predicted_prices, label="Predicted Prices", color='orange')
            ax.set_xlabel("Days after last test data")
            ax.set_ylabel("Price USD ($/Barrel)")
            ax.set_title(f"Predicted Oil Prices for the Next {future_days} Days")
            st.pyplot(fig)  # Pass the figure to st.pyplot()

            # Step 11: Prepare a downloadable Excel file with the predicted prices
            predicted_data = pd.DataFrame(predicted_prices, columns=["Predicted Price"])
            predicted_data["Date"] = pd.date_range(start=validation_data.Date.iloc[-1] + pd.Timedelta(days=1), periods=future_days)

            # Reorder columns: Column A should be Date, Column B should be Predicted Price
            predicted_data = predicted_data[["Date", "Predicted Price"]]

            # Save the predictions to Excel
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
                predicted_data.to_excel(writer, index=False, sheet_name="Predictions")

            output.seek(0)

            # Provide download button for the predictions Excel file
            st.download_button(
                label="Download Predicted Data",
                data=output,
                file_name="predicted_oil_prices.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
