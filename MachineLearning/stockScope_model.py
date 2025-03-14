import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
import lightgbm as lgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice
import warnings
import logging

# Suppress all warnings and logging
warnings.filterwarnings('ignore')
logging.getLogger('lightgbm').setLevel(logging.ERROR)

class StockPredictor:
    def __init__(self):
        self.data = None
        self.X = None
        self.y = None
        
        # Model parametreleri
        self.rf_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        self.xgb_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.8, 0.9, 1.0]
        }
        
        self.lgb_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'num_leaves': [31, 50, 70]
        }
        
        # Base modeller
        self.models = {
            'random_forest': RandomForestClassifier(),
            'xgboost': xgb.XGBClassifier(verbosity=0),
            'lightgbm': lgb.LGBMClassifier(verbose=-1)
        }
        
        self.best_models = {}
        self.ensemble_model = None
        self.feature_selector = None
        self.feature_columns = None
        self.scaler = StandardScaler()
    
    def add_technical_indicators(self):
        """Yeni teknik göstergeler ekle"""
        # Bollinger Bands
        bb = BollingerBands(close=self.data['Close'])
        self.data['BB_high'] = bb.bollinger_hband()
        self.data['BB_low'] = bb.bollinger_lband()
        self.data['BB_mid'] = bb.bollinger_mavg()
        
        # Stochastic Oscillator
        stoch = StochasticOscillator(
            high=self.data['High'],
            low=self.data['Low'],
            close=self.data['Close']
        )
        self.data['Stoch_k'] = stoch.stoch()
        self.data['Stoch_d'] = stoch.stoch_signal()
        
        # VWAP
        vwap = VolumeWeightedAveragePrice(
            high=self.data['High'],
            low=self.data['Low'],
            close=self.data['Close'],
            volume=self.data['Volume']
        )
        self.data['VWAP'] = vwap.volume_weighted_average_price()
        
        # Trend Strength
        self.data['Trend_Strength'] = abs(self.data['EMA4'] - self.data['EMA8'])
        
        # Price Range
        self.data['Price_Range'] = self.data['High'] - self.data['Low']
        
        # Volume Trend
        self.data['Volume_MA5'] = self.data['Volume'].rolling(window=5).mean()
        self.data['Volume_Trend'] = self.data['Volume'] / self.data['Volume_MA5']
    
    def load_data(self, file_path):
        """Veri setini yükle ve hazırla"""
        self.data = pd.read_csv(file_path)
        
        # Tarih sütununu datetime'a çevir
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        
        # Yeni teknik göstergeleri ekle
        self.add_technical_indicators()
        
        # Özellik sütunlarını belirle (Date ve Label hariç)
        self.feature_columns = [col for col in self.data.columns 
                              if col not in ['Date', 'Label']]
        
        # NaN değerleri temizle
        self.data = self.data.dropna()
        
        # Özellikleri ve hedef değişkeni ayır
        self.X = self.data[self.feature_columns]
        self.y = self.data['Label']
        
        # Özellik seçimi
        base_model = RandomForestClassifier(n_estimators=100)
        self.feature_selector = SelectFromModel(base_model, prefit=False)
        
        # Numpy array'e çevir
        X_array = np.array(self.X)
        self.X = self.feature_selector.fit_transform(X_array, self.y)
        
        # Seçilen özellikleri göster
        selected_mask = self.feature_selector.get_support()
        selected_features = [col for idx, col in enumerate(self.feature_columns) if selected_mask[idx]]
        
        # Özellikleri ölçeklendir
        self.X = self.scaler.fit_transform(self.X)
    
    def train_models(self, test_size=0.2):
        """Tüm modelleri eğit ve ensemble model oluştur"""
        # Veriyi eğitim ve test setlerine ayır
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, shuffle=True
        )
        
        # Her model için GridSearchCV ile en iyi parametreleri bul
        for name, model in self.models.items():
            # Model parametrelerini seç
            if name == 'random_forest':
                params = self.rf_params
            elif name == 'xgboost':
                params = self.xgb_params
            else:
                params = self.lgb_params
            
            # GridSearchCV ile en iyi parametreleri bul
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=params,
                cv=5,
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            self.best_models[name] = grid_search.best_estimator_
        
        # Ensemble model oluştur ve eğit
        self.ensemble_model = VotingClassifier(
            estimators=[
                (name, model) for name, model in self.best_models.items()
            ],
            voting='soft'
        )
        
        self.ensemble_model.fit(X_train, y_train)
        
        # Test seti üzerinde tahmin yap
        y_pred = self.ensemble_model.predict(X_test)
        
        # Metrikleri hesapla ve döndür
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        metrics = {
            'accuracy': accuracy,
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1': report['weighted avg']['f1-score']
        }
        
        # Sonuçları göster
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-score: {metrics['f1']:.4f}")
        
        return metrics
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Karmaşıklık matrisini görselleştir"""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Karmaşıklık Matrisi')
        plt.ylabel('Gerçek Değer')
        plt.xlabel('Tahmin')
        plt.savefig('confusion_matrix.png')
        plt.close()
    
    def plot_feature_importance(self):
        """Özellik önemliliklerini görselleştir"""
        importances = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.best_model.feature_importances_
        })
        importances = importances.sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=importances)
        plt.title('Özellik Önemlilikleri')
        plt.savefig('feature_importance.png')
        plt.close()
    
    def save_model(self, model_path='ensemble_model.joblib'):
        """Ensemble modeli kaydet"""
        if self.ensemble_model is not None:
            joblib.dump({
                'ensemble_model': self.ensemble_model,
                'feature_selector': self.feature_selector,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns
            }, model_path)
            print(f"\nModel kaydedildi: {model_path}")
    
    def predict_next_week(self, current_data):
        """Gelecek hafta için tahmin yap"""
        if self.ensemble_model is None:
            raise ValueError("Model henüz eğitilmemiş!")
        
        # Yeni teknik göstergeleri ekle
        for col in self.feature_columns:
            if col not in current_data.columns:
                print(f"Uyarı: {col} özelliği eksik, hesaplanıyor...")
        
        # Özellik seçimi uygula
        selected_data = self.feature_selector.transform(current_data[self.feature_columns])
        
        # Veriyi ölçeklendir
        scaled_data = self.scaler.transform(selected_data)
        
        # Tahmin yap
        prediction = self.ensemble_model.predict(scaled_data)
        probability = self.ensemble_model.predict_proba(scaled_data)
        
        return prediction[0], probability[0]

# Ana kod
if __name__ == "__main__":
    # İterasyon sayısı
    n_iterations = 3
    
    # Her metrik için sonuçları sakla
    results = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    print("Model eğitimi başlıyor...\n")
    
    for i in range(n_iterations):
        print(f"\n{i+1}. İterasyon:")
        print("-" * 20)
        
        # Model nesnesini oluştur
        predictor = StockPredictor()
        
        # Veriyi yükle
        predictor.load_data('../AAPL_with_indicators.csv')
        
        # Modelleri eğit ve sonuçları al
        metrics = predictor.train_models()
        
        # Sonuçları sakla
        for metric, value in metrics.items():
            results[metric].append(value)
    
    # Ortalama sonuçları göster
    print("\nOrtalama Sonuçlar:")
    print("-" * 20)
    for metric in results:
        values = results[metric]
        mean = sum(values) / len(values)
        std = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
        print(f"{metric.title()}: {mean:.4f} (±{std:.4f})")