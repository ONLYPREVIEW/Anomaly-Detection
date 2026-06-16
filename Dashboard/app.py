import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import precision_score, recall_score, f1_score

# 1. Sayfa Ayarları (Tam Ekran ve Başlık)
st.set_page_config(page_title="Cloud Cost Anomaly Detector", layout="wide", page_icon="☁️")

st.title("☁️ Automated Cloud Cost Anomaly Detection Dashboard")
st.markdown("**Proje Ekibi:** Emir Ata Karagenç, Tolga Türkmen, Zeki Zafer Aydınlı")
st.markdown("**Proje Seviyesi:** Level 1 Standard - Phase 1 & Phase 2 Bütünleşik Sunumu")
st.markdown("---")

# 2. İşlenmiş Veriyi Yükleme
def load_data():
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
st.header("📈 2. Anomali Tespiti ve Model Karşılaştırması")

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

# --- PHASE 2: CANLI METRİK HESAPLAMA ---
y_true = df['Is_Anomaly']
y_pred = df[anomaly_col]

precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

# Metrikleri grafiğin üstünde gösterelim 
m1, m2, m3 = st.columns(3)
m1.metric("🎯 Precision (Kesinlik)", f"{precision:.2f}", help="Modelin bulduğu anomalilerin ne kadarının GERÇEK anomali olduğunu (Yanlış Alarmları) ölçer.")
m2.metric("🔍 Recall (Duyarlılık)", f"{recall:.2f}", help="Sistemdeki toplam gerçek anomalilerin ne kadarını YAKALAYABİLDİĞİMİZİ ölçer.")
m3.metric("🏆 F1-Score (Genel Başarı)", f"{f1:.2f}", help="Precision ve Recall değerlerinin Harmonik Ortalamasıdır. Sistemin genel dengesini gösterir.")

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


with st.expander(f"📚 {model_choice} - Matematiksel Arka Plan ve Çalışma Mantığı", expanded=False):
    if model_choice == "STL Decomposition (Önerilen Model)":
        st.markdown("""
        **Neden Öneriyoruz?** STL, zaman serisindeki mevsimselliği (hafta sonu düşüşlerini vb.) anladığı için en kararlı modeldir.
        - **Formül:** $Y_t = T_t + S_t + R_t$ (Fatura = Trend + Mevsimsellik + Sapma/Artık)
        - **Çalışma Mantığı:** Sistem $R_t$ (Sapma) değerini izole eder. Eğer bu değer serinin standart sapmasının 3 katını aşarsa proaktif olarak anomali uyarısı fırlatır.
        - **Sonuç:** Sinsi (Contextual) anomalileri mükemmel yakalar.
        """)
    elif model_choice == "Z-Score (İstatistiksel)":
        st.markdown("""
        **Model Değerlendirmesi:** Z-Score, faturanın ortalamadan ne kadar uzaklaştığına bakan basit bir istatistiksel yöntemdir.
        - **Formül:** $Z = (X - \mu) / \sigma$ 
        - **Zaafları:** Bu model verideki haftalık döngüleri (Mevsimsellik) anlayamaz. Bu yüzden hafta sonuna denk gelen sinsi bir maliyet artışını normal bir hafta içi faturası zannedip es geçebilir (Recall değerini düşürür).
        """)
    else:
        st.markdown("""
        **Model Değerlendirmesi:** Isolation Forest, veriyi rastgele kesmelerle izole etmeye çalışan ağaç tabanlı bir Makine Öğrenmesi algoritmasıdır.
        - **Formül:** $s(x, n) = 2^{-E(h(x)) / c(n)}$
        - **Zaafları:** Çok boyutlu karmaşık verilerde harika çalışsa da, bulut faturaları gibi tek boyutlu finansal zaman serilerinde **fazla hassas** davranabilir. Normal günlere de "Anomali" diyerek yanlış alarm (False Positive) üretebilir (Precision değerini düşürür).
        """)

st.markdown("---")

# 5. Alt Kısım: Anomali Detay Tablosu
st.header("🚨 3. Tespit Edilen ve Kaçırılan Anomaliler (Detaylı Liste)")


table_data = df[(df['Is_Anomaly'] == True) | (df[anomaly_col] == True)].copy()

st.write(f"Seçilen algoritma grafikte **{len(anomalies)}** adet anomali tespit etti. Aşağıdaki tabloda modelin başarı durumunu detaylı görebilirsiniz.")


display_df = table_data[['Date', 'DailyCost_USD', 'Residual', 'Is_Anomaly', anomaly_col]].copy()


display_df['DailyCost_USD'] = display_df['DailyCost_USD'].apply(lambda x: f"${x:,.2f}")
display_df['Residual'] = display_df['Residual'].apply(lambda x: f"{x:,.2f}")


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