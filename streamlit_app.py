import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE

# Set style seaborn
sns.set(style="whitegrid")

st.title("Analisis Data Obesitas dengan Streamlit")

uploaded_file = st.file_uploader("Unggah file CSV dataset obesitas", type=["csv"])

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)

    st.header("Preview Dataset")
    st.dataframe(df.head())

    st.header("Informasi Dataset")
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    st.text(info_str)

    st.header("Statistik Deskriptif")
    st.dataframe(df.describe(include='all'))

    # Konversi kolom numerik yang bertipe object ke numerik
    numerical_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    for col in numerical_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    st.header("Visualisasi Data")

    # Plot distribusi usia dan berat badan
    fig1, axs1 = plt.subplots(1, 2, figsize=(12, 4))
    if 'Age' in df.columns:
        sns.histplot(df['Age'].dropna(), kde=True, bins=20, ax=axs1[0])
        axs1[0].set_title('Distribusi Usia')
    if 'Weight' in df.columns:
        sns.histplot(df['Weight'].dropna(), kde=True, bins=20, ax=axs1[1], color='orange')
        axs1[1].set_title('Distribusi Berat Badan')
    st.pyplot(fig1)

    # Boxplot tinggi badan berdasarkan gender
    if 'Height' in df.columns and 'Gender' in df.columns:
        fig2, ax2 = plt.subplots(figsize=(6,4))
        sns.boxplot(x='Gender', y='Height', data=df, ax=ax2)
        ax2.set_title('Tinggi Badan Berdasarkan Gender')
        st.pyplot(fig2)

    # Countplot kategori obesitas
    if 'NObeyesdad' in df.columns:
        fig3, ax3 = plt.subplots(figsize=(6,4))
        sns.countplot(y='NObeyesdad', data=df, order=df['NObeyesdad'].value_counts().index, ax=ax3)
        ax3.set_title('Distribusi Kategori Obesitas')
        st.pyplot(fig3)

    # Preprocessing Data
    st.header("Preprocessing Data")
    with st.spinner("Sedang membersihkan data..."):
        # Drop duplikat dan missing values
        df_clean = df.drop_duplicates().dropna()

        # Hapus outlier dengan metode IQR
        Q1 = df_clean[numerical_cols].quantile(0.25)
        Q3 = df_clean[numerical_cols].quantile(0.75)
        IQR = Q3 - Q1
        df_clean = df_clean[~((df_clean[numerical_cols] < (Q1 - 1.5 * IQR)) | (df_clean[numerical_cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

        st.success("Data berhasil dibersihkan.")
        st.write(f"Jumlah data setelah preprocessing: {df_clean.shape[0]} baris")

    # Encoding dan persiapan fitur dan target
    if 'NObeyesdad' in df_clean.columns:
        le = LabelEncoder()
        df_clean['NObeyesdad_enc'] = le.fit_transform(df_clean['NObeyesdad'])

        # One-hot encoding fitur kategorikal selain target
        cat_cols = df_clean.select_dtypes(include=['object']).columns.tolist()
        cat_cols = [c for c in cat_cols if c != 'NObeyesdad']
        df_encoded = pd.get_dummies(df_clean.drop(columns=['NObeyesdad']), columns=cat_cols, drop_first=True)

        X = df_encoded.drop(columns=['NObeyesdad_enc'])
        y = df_encoded['NObeyesdad_enc']

        # Tangani ketidakseimbangan data dengan SMOTE
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)

        # Standarisasi data numerik
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_res)

        st.header("Modeling dan Evaluasi")
        test_size = st.slider("Pilih proporsi data uji (%)", 10, 50, 20)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_res, test_size=test_size/100, random_state=42, stratify=y_res
        )

        # Pilihan model
        models = {
            'Decision Tree': DecisionTreeClassifier(random_state=42),
            'Random Forest': RandomForestClassifier(random_state=42),
            'SVM': SVC(random_state=42)
        }

        results = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            results[name] = {
                'Akurasi': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, average='weighted'),
                'Recall': recall_score(y_test, y_pred, average='weighted'),
                'F1 Score': f1_score(y_test, y_pred, average='weighted')
            }

        df_results = pd.DataFrame(results).T

        st.subheader("Hasil Evaluasi Model")
        st.dataframe(df_results.style.format("{:.4f}"))

        # Plot hasil evaluasi
        fig4, ax4 = plt.subplots(figsize=(8,5))
        df_results.plot(kind='bar', ax=ax4)
        ax4.set_title("Perbandingan Performansi Model")
        ax4.set_ylabel("Score")
        ax4.set_ylim([0,1])
        ax4.legend(loc='lower right')
        st.pyplot(fig4)

else:
    st.info("Silakan unggah file CSV dataset obesitas terlebih dahulu.")
