import streamlit as st
import pandas as pd
import plotly.graph_objects as go

# 1. Sayfa Ayarları (Tam Ekran ve Başlık)
st.set_page_config(page_title="Cloud Cost Anomaly Detector", layout="wide", page_icon="☁️")

st.title("☁️ Automated Cloud Cost Anomaly Detection Dashboard")
st.markdown("**Proje Ekibi:** Emir Ata Karagenç, Tolga Türkmen, Zeki Zafer Aydınlı")
st.markdown("---")

# 2. İşlenmiş Veriyi Yükleme

def load_data():
    # Bir önceki aşamada kaydettiğimiz analizli veriyi okuyoruz
    df = pd.read_csv('Data/processed_cur_data.csv')
    return df

df = load_data()

# 3. Üst Kısım: Genel Fatura Özeti (Metrik Kartları)
st.header("📊 1. Genel Fatura Özeti")
total_cost = df['DailyCost_USD'].sum()
total_days = len(df)
avg_cost = df['DailyCost_USD'].mean()

col1, col2, col3 = st.columns(3)
col1.metric("Toplam İzlenen Gün", f"{total_days} Gün")
col2.metric("Toplam Bulut Harcaması", f"${total_cost:,.2f}")
col3.metric("Ortalama Günlük Maliyet", f"${avg_cost:,.2f}")

st.markdown("---")

# 4. Orta Kısım: İnteraktif Grafik ve Model Seçimi
st.header("📈 2. Anomali Tespiti ve Görselleştirme")

# Kullanıcının modeli seçebileceği açılır menü
model_choice = st.selectbox(
    "Görüntülemek İstediğiniz Anomali Tespit Algoritmasını Seçin:",
    ("STL Decomposition (Önerilen Model)", "Isolation Forest (Makine Öğrenmesi)", "Z-Score (İstatistiksel)")
)

# Seçime göre hangi veri sütununu kullanacağımızı belirliyoruz
if model_choice == "STL Decomposition (Önerilen Model)":
    anomaly_col = 'STL_Anomaly'
elif model_choice == "Isolation Forest (Makine Öğrenmesi)":
    anomaly_col = 'IsoForest_Anomaly'
else:
    anomaly_col = 'ZScore_Anomaly'

# Plotly ile İnteraktif Grafik Çizimi
fig = go.Figure()

# Normal harcama çizgisi
fig.add_trace(go.Scatter(x=df['Date'], y=df['DailyCost_USD'], 
                         mode='lines', name='Günlük Maliyet', 
                         line=dict(color='#1f77b4', width=2)))

# Anomalileri grafiğin üzerine kırmızı çarpı olarak ekliyoruz
anomalies = df[df[anomaly_col] == True]
fig.add_trace(go.Scatter(x=anomalies['Date'], y=anomalies['DailyCost_USD'], 
                         mode='markers', name='Tespit Edilen Anomali', 
                         marker=dict(color='red', size=12, symbol='x', line=dict(width=2, color='darkred'))))

fig.update_layout(
    title=f"{model_choice} ile Bulunan Maliyet Anomalileri",
    xaxis_title="Tarih",
    yaxis_title="Maliyet (USD)",
    hovermode="x unified",
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# 5. Alt Kısım: Anomali Detay Tablosu
st.header("🚨 3. Tespit Edilen Anomaliler (Detaylı Liste)")
st.write(f"Seçilen algoritma toplam **{len(anomalies)}** adet anomali tespit etti.")

# Ekranda sadece anlaşılır ve gerekli kolonları gösterelim
display_df = anomalies[['Date', 'DailyCost_USD', 'Residual', 'Is_Anomaly', anomaly_col]].copy()

# Sayıları daha şık görünmesi için formatlayalım
display_df['DailyCost_USD'] = display_df['DailyCost_USD'].apply(lambda x: f"${x:,.2f}")
display_df['Residual'] = display_df['Residual'].apply(lambda x: f"{x:,.2f}")

# True/False (Checkbox) değerlerini net okunur Emojili metinlere çevirelim
display_df['Is_Anomaly'] = display_df['Is_Anomaly'].map({True: '✅ Evet', False: '❌ Hayır'})
display_df[anomaly_col] = display_df[anomaly_col].map({True: '✅ Tespit Edildi', False: '❌ Kaçırıldı'})

display_df.rename(columns={
    'Date': 'Tarih', 
    'DailyCost_USD': 'Fatura Tutarı',
    'Residual': 'Artık/Sapma Değeri',
    'Is_Anomaly': 'Gerçekte Anomali mi?',
    anomaly_col: 'Model Tespiti'
}, inplace=True)

st.dataframe(display_df, use_container_width=True)