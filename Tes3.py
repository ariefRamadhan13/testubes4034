import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as mno
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier

# Defining the lithology keys mapping
lithology_keys = {
    30000: 'Sandstone',
    65030: 'Sandstone/Shale',
    65000: 'Shale',
    80000: 'Marl',
    74000: 'Dolomite',
    70000: 'Limestone',
    70032: 'Chalk',
    88000: 'Halite',
    86000: 'Anhydrite',
    99000: 'Tuff',
    90000: 'Coal',
    93000: 'Basement'
}

def lithology_app():
    st.title("Lithology Classification App")
    
    # Upload file CSV
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    
    if uploaded_file is not None:
        # Membaca file CSV
        df = pd.read_csv(uploaded_file)

        # Menampilkan ringkasan statistik dan informasi DataFrame
        st.subheader("Data Overview")
        st.write(df.describe())
        st.write(df.info())

        # Menampilkan nilai unik dari kolom 'LITH' beserta keterangan lithologi
        st.subheader("Unique values in 'LITH' column with Lithology Description")
        unique_values = df['LITH'].unique()

        # Mengonversi nilai-nilai unik menjadi nama lithologi berdasarkan dictionary
        lithology_descriptions = [lithology_keys.get(value, "Unknown") for value in unique_values]

        # Menampilkan nilai unik dengan deskripsi lithologi
        for value, description in zip(unique_values, lithology_descriptions):
            st.write(f"Code: {value} -> Lithology: {description}")

        # Menampilkan jumlah missing values dalam bentuk tabel
        st.subheader("Missing Values Before Dropping")
        missing_before = df.isnull().sum().reset_index()
        missing_before.columns = ['Column', 'Missing Values']
        st.write(missing_before)  # Tampilkan jumlah missing values dalam bentuk tabel

        # Membuat plot missing values
        st.subheader("Missing Values Plot Before Dropping")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(missing_before['Column'], missing_before['Missing Values'], color='skyblue')
        ax.set_xlabel('Columns')
        ax.set_ylabel('Missing Values Count')
        ax.set_title('Missing Values per Column')
        ax.set_xticklabels(missing_before['Column'], rotation=90)
        st.pyplot(fig)  # Menampilkan plot

        # Mengecek jika ada missing values, lalu menghapusnya
        if missing_before['Missing Values'].any():
            df.dropna(inplace=True)

        # Menampilkan jumlah missing values setelah penghapusan
        st.subheader("Missing Values After Dropping")
        missing_after = df.isnull().sum().reset_index()
        missing_after.columns = ['Column', 'Missing Values']
        st.write(missing_after)  # Tampilkan jumlah missing values setelah penghapusan

        # Menampilkan plot missing values setelah penghapusan
        st.subheader("Missing Values Plot After Dropping")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(missing_after['Column'], missing_after['Missing Values'], color='salmon')
        ax.set_xlabel('Columns')
        ax.set_ylabel('Missing Values Count')
        ax.set_title('Missing Values per Column After Dropping')
        ax.set_xticklabels(missing_after['Column'], rotation=90)
        st.pyplot(fig)  # Menampilkan plot

        # Menampilkan pilihan kolom untuk fitur
        feature_columns = st.multiselect(
            'Select feature columns for prediction:',
            df.columns.tolist(),
            default=['RDEP', 'RHOB', 'GR', 'DTC']  # Default selected columns
        )

        # Memastikan 'LITH' ada dalam kolom yang tidak dipilih oleh pengguna
        if 'LITH' not in feature_columns:
            feature_columns.append('LITH')  # Menambahkan 'LITH' jika tidak ada

        # Memisahkan fitur (X) dan target (y)
        X = df[feature_columns[:-1]]  # Semua kolom kecuali 'LITH'
        y = df[feature_columns[-1]]   # Kolom terakhir sebagai target ('LITH')

        # Membagi dataset menjadi data latih dan data uji
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        # Membuat dan melatih model RandomForest
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)

        # Prediksi dengan data uji
        y_pred = clf.predict(X_test)

        # Menampilkan akurasi model
        accuracy = accuracy_score(y_test, y_pred)
        st.subheader(f"Model Accuracy: {accuracy:.2f}")

        # Menampilkan laporan klasifikasi
        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred))

        # Matriks kebingungannya
        cf_matrix = confusion_matrix(y_test, y_pred)

        # Menyusun label untuk confusion matrix
        labels = ["Sandstone", "Sandstone/Shale", "Marl", "Dolomite", "Limestone", "Chalk"]
        labels = sorted(labels)

        # Membuat heatmap untuk confusion matrix
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(10, 10))  # Membuat figure dan axis baru
        ax = sns.heatmap(cf_matrix, annot=True, cmap="Reds", fmt=".0f",
                         xticklabels=labels, yticklabels=labels)
        ax.set_title('Confusion Matrix with labels\n\n')
        ax.set_xlabel('\nPredicted Values')
        ax.set_ylabel('Actual Values ')
        st.pyplot(fig)  # Menampilkan heatmap dengan figure yang jelas

        # Menambahkan konversi angka lithologi ke nama lithologi
        results_df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': y_pred,
            'Predicted Lithology': [lithology_keys.get(value, "Unknown") for value in y_pred]  # Konversi angka ke lithologi
        })

        # Menyediakan tombol untuk mendownload hasil prediksi
        st.subheader("Download Prediction Results")
        csv = results_df.to_csv(index=False)  # Konversi DataFrame ke CSV
        st.download_button(
            label="Download Predicted Results",
            data=csv,
            file_name="predicted_results.csv",
            mime="text/csv"
        )
