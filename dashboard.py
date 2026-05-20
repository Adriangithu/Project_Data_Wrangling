import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Dashboard Analisis Perilaku Nasabah",
    page_icon="🏦",
    layout="wide"
)


URL_LIVE_SPREADSHEET = "hasil_segmentasi_nasabah.csv"  

@st.cache_data
def load_data(url):
    try:
        df = pd.read_csv(url)
    except Exception as e:

        df = pd.read_csv('hasil_segmentasi_nasabah.csv')
    return df

df_clean = load_data(URL_LIVE_SPREADSHEET)


fitur_numerik = ['Jumlah_Transaksi', 'Usia_Nasabah', 'Durasi_Transaksi', 'Saldo_Akun']


X_model = df_clean[fitur_numerik]
y_model = df_clean['Kluster']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_model)

model_rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
model_rf.fit(X_model, y_model)

pemetaan_segmen = {
    0: 'High Spender (Transaksi Besar)',
    1: 'Saver (Mapan & Saldo Tinggi)',
    2: 'Young & Active (Muda & Saldo Rendah)'
}

st.title("🏦 Proyek Analisis Data: Analisis Perilaku & Segmentasi Nasabah")
st.markdown("""
Dashboard ini dirancang untuk memantau segmen profil keuangan nasabah secara real-time dan menyediakan fitur prediksi klasifikasi instan bagi nasabah baru.
""")

tab1, tab2 = st.tabs(["📊 Analisis Tren & Segmen", "🔮 Fitur Prediksi Nasabah Baru"])

with tab1:
    st.subheader("Ringkasan Statistik Keuangan")
    
    pekerjaan_pilihan = st.sidebar.selectbox(
        "Filter Berdasarkan Pekerjaan:", 
        ['Semua'] + list(df_clean['Pekerjaan_Nasabah'].unique())
    )
    
    df_filtered = df_clean.copy()
    if pekerjaan_pilihan != 'Semua':
        df_filtered = df_filtered[df_filtered['Pekerjaan_Nasabah'] == pekerjaan_pilihan]
        
    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Total Sampel Transaksi", f"{df_filtered.shape[0]} Data")
    kpi2.metric("Rata-rata Saldo Akun", f"${df_filtered['Saldo_Akun'].mean():,.2f}")
    kpi3.metric("Rata-rata Nilai Pengeluaran", f"${df_filtered['Jumlah_Transaksi'].mean():,.2f}")
    
    st.markdown("---")
    
    col_graph1, col_graph2 = st.columns(2)
    
    with col_graph1:
        st.write("### 📈 Pemetaan Kluster Finansial Nasabah")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.scatterplot(
            data=df_filtered, 
            x='Saldo_Akun', 
            y='Jumlah_Transaksi', 
            hue='Nama_Segmen', 
            palette='magma', 
            alpha=0.7, 
            ax=ax1
        )
        ax1.set_xlabel("Saldo Akun ($)")
        ax1.set_ylabel("Jumlah Transaksi ($)")
        st.pyplot(fig1)
        
    with col_graph2:
        st.write("### 📊 Rata-rata Pengeluaran per Segmen")
        fig2, ax2 = plt.subplots(figsize=(6, 4))

        rata_transaksi = df_filtered.groupby('Nama_Segmen')['Jumlah_Transaksi'].mean().reset_index()
        sns.barplot(data=rata_transaksi, x='Nama_Segmen', y='Jumlah_Transaksi', palette='Set2', ax=ax2)
        ax2.set_xlabel("Nama Segmen")
        ax2.set_ylabel("Rata-rata Pengeluaran ($)")
        plt.xticks(rotation=15)
        st.pyplot(fig2)

    st.markdown("---")
    st.write("### 📋 Preview Dataset Tersegmentasi")
    st.dataframe(df_filtered[['ID_Transaksi', 'Usia_Nasabah', 'Pekerjaan_Nasabah', 'Jumlah_Transaksi', 'Saldo_Akun', 'Nama_Segmen']])

with tab2:
    st.subheader("🔮 Klasifikasi Otomatis untuk Nasabah Baru")
    st.markdown("Masukkan parameter data di bawah ini untuk memprediksi secara instan kategori segmen nasabah baru tersebut.")
    
    col_in1, col_in2 = st.columns(2)
    with col_in1:
        input_usia = st.number_input("Masukkan Usia Nasabah:", min_value=17, max_value=100, value=30)
        input_saldo = st.number_input("Masukkan Saldo Akun ($):", min_value=0.0, value=5000.0)
    with col_in2:
        input_jumlah_tx = st.number_input("Masukkan Estimasi Nilai Transaksi ($):", min_value=0.0, value=250.0)
        input_durasi = st.number_input("Masukkan Durasi Sesi Transaksi (Detik):", min_value=0, value=120)
        
    if st.button("🚀 Prediksi Klasifikasi Segmen"):

        data_nasabah_baru = [[input_jumlah_tx, input_usia, input_durasi, input_saldo]]

        hasil_prediksi = model_rf.predict(data_nasabah_baru)[0]
        nama_hasil_segmen = pemetaan_segmen[hasil_prediksi]
        
        st.success(f"Hasil Analisis Model: Nasabah baru tersebut diklasifikasikan ke dalam kelompok **{nama_hasil_segmen}**")