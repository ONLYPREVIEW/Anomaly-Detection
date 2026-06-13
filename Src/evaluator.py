import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

def evaluate_models(data_path='Data/processed_cur_data.csv'):
    # 1. İşlenmiş veriyi yükle (Bir önceki adımda kaydettiğimiz veri)
    df = pd.read_csv(data_path)
    
    # Gerçek cevap anahtarımız (Ground Truth)
    y_true = df['Is_Anomaly']

    # Test edeceğimiz modellerin sütun isimleri
    models = ['STL_Anomaly', 'ZScore_Anomaly', 'IsoForest_Anomaly']

    print("📊 MODEL PERFORMANS METRİKLERİ (Phase 2)")
    print("-" * 50)

    # 2. Her bir model için metrikleri hesapla
    for model in models:
        y_pred = df[model]
        
        # zero_division=0 parametresi, sıfıra bölünme hatalarını engeller
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        print(f"Model: {model.replace('_Anomaly', '')}")
        print(f"  Precision (Kesinlik): {precision:.2f}")
        print(f"  Recall (Duyarlılık):  {recall:.2f}")
        print(f"  F1-Score:             {f1:.2f}")
        print("-" * 50)

if __name__ == "__main__":
    evaluate_models()