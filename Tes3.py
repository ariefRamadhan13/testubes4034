import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as mno
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier

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

        # Menampilkan nilai unik dari kolom 'LITH'
        st.subheader("Unique values in 'LITH' column")
        st.write(df['LITH'].unique())

        # Visualisasi missing values sebelum penghapusan
        st.subheader("Missing Values Before Dropping")
        mno.bar(df)
        fig, ax = plt.subplots()
        ax = plt.gca()  # Mendapatkan axis saat ini
        st.pyplot(fig)

        # Menghapus baris dengan missing values
        df.dropna(inplace=True)

        # Visualisasi missing values setelah penghapusan
        st.subheader("Missing Values After Dropping")
        mno.bar(df)
        fig, ax = plt.subplots()
        ax = plt.gca()  # Mendapatkan axis saat ini
        st.pyplot(fig)

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

# Jika Anda ingin menjalankan aplikasi, pastikan untuk memanggil fungsi lithology_app()
# lithology_app()
