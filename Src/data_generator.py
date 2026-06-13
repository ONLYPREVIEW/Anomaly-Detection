import pandas as pd
import numpy as np
import os

def generate_synthetic_cur(start_date='2025-01-01', end_date='2026-05-01', output_path='Data/synthetic_cur_data.csv'):
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n_days = len(dates)

    base_cost = 100.0
    trend = np.linspace(0, 50, n_days)
    seasonality = np.where(dates.weekday < 5, 20.0, -15.0)

    # Gürültüyü sabitleyelim ki her çalıştırdığımızda aynı şık sonucu versin
    np.random.seed(42)
    noise = np.random.normal(0, 5, n_days)

    total_cost = base_cost + trend + seasonality + noise
    is_anomaly = np.zeros(n_days, dtype=bool)
    
    # 52. indeks tam bir Cumartesi gününe denk geliyor! Onu sinsi anomali yapacağız.
    anomaly_indices = [52, 120, 200, 300, 350, 410]
    
    for idx in anomaly_indices:
        is_anomaly[idx] = True
        
        if idx == 52:
            # SİNSİ ANOMALİ: Hafta sonuna 65 dolarlık ufak bir sıçrama ekliyoruz.
            # Z-Score bunu "Normal bir hafta içi faturası" sanıp es geçecek.
            # Ancak STL, "Cumartesi günü bu kadar harcama olamaz" deyip yakalayacak!
            total_cost[idx] += 85 
        else:
            # Diğerleri devasa sıçramalar, bunları hepsi yakalar
            total_cost[idx] += np.random.uniform(150, 250) 

    df = pd.DataFrame({
        'Date': dates,
        'Service': 'AmazonEC2',
        'DailyCost_USD': total_cost.round(2),
        'Is_Anomaly': is_anomaly
    })

    df['DailyCost_USD'] = df['DailyCost_USD'].clip(lower=0)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print("✅ SİNSİ ANOMALİ içeren yeni veri seti başarıyla oluşturuldu!")

if __name__ == "__main__":
    generate_synthetic_cur()