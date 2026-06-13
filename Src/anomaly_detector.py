import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
from sklearn.ensemble import IsolationForest
import os

def detect_anomalies_all_methods(data_path='Data/synthetic_cur_data.csv'):
    # 1. Veriyi yükle
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # --- YÖNTEM 1: STL DECOMPOSITION ---
    stl = STL(df['DailyCost_USD'], period=7, robust=True)
    res = stl.fit()
    df['Residual'] = res.resid

    std_dev = df['Residual'].std()
    threshold = 3 * std_dev
    df['STL_Anomaly'] = np.abs(df['Residual']) > threshold

    # --- YÖNTEM 2: Z-SCORE (Basit İstatistiksel Yöntem) ---
    mean_cost = df['DailyCost_USD'].mean()
    std_cost = df['DailyCost_USD'].std()
    df['Z_Score'] = (df['DailyCost_USD'] - mean_cost) / std_cost
    df['ZScore_Anomaly'] = np.abs(df['Z_Score']) > 3 # Z-skoru 3'ten büyükse anomali

    # --- YÖNTEM 3: ISOLATION FOREST (Makine Öğrenmesi) ---
    # Maliyet verisini modele uygun 2D dizi formuna getiriyoruz
    X = df[['DailyCost_USD']].values
    
    # contamination: Veri içindeki tahmini anomali oranı (yaklaşık %1.5 - %2 diyelim)
    iso_forest = IsolationForest(contamination=0.015, random_state=42)
    df['IsoForest_Label'] = iso_forest.fit_predict(X)
    
    # Isolation Forest normal veriye 1, anomaliye -1 der. Biz bunu True/False yapalım:
    df['IsoForest_Anomaly'] = df['IsoForest_Label'] == -1

    # Sonuçları Kaydet
    output_path = 'Data/processed_cur_data.csv'
    df.reset_index().to_csv(output_path, index=False)
    
    print("✅ Tüm algoritmalar (STL, Z-Score, Isolation Forest) çalıştırıldı.")
    print("-" * 40)
    print("📊 ALGORİTMA ANOMALİ TESPİT SONUÇLARI:")
    print(f"Sisteme Enjekte Edilen Gerçek Anomali Sayısı: {df['Is_Anomaly'].sum()}")
    print(f"STL Decomposition Tespiti: {df['STL_Anomaly'].sum()}")
    print(f"Z-Score Tespiti: {df['ZScore_Anomaly'].sum()}")
    print(f"Isolation Forest Tespiti: {df['IsoForest_Anomaly'].sum()}")
    print("-" * 40)
    print(f"Tüm veriler karşılaştırma için kaydedildi: {output_path}")

if __name__ == "__main__":
    detect_anomalies_all_methods()