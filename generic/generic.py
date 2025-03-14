import pandas as pd
import numpy as np
import random
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.model_selection import KFold

class StockGenetic:
    def __init__(self, noise_level=0.3):
        self.data = None
        self.technical_features = [
            'MA4', 'MA8', 'EMA4', 'EMA8', 'Volatility', 'RSI',
            'OBV', 'MACD', 'MACD_Signal', 'MACD_Hist'
        ]
        self.noise_level = noise_level
        self.population_size = 50
        self.generations = 40
        self.crossover_prob = 0.7
        self.mutation_prob = 0.2
        self.n_splits = 5  # Cross-validation için bölüm sayısı
        self.market_volatility = 0.0  # Piyasa volatilitesi (dinamik olarak güncellenecek)
        
    def load_data(self, file_path):
        """Veri setini yükle"""
        self.data = pd.read_csv(file_path)
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        
        # Piyasa volatilitesini hesapla (son 30 günün standart sapması)
        if len(self.data) > 30:
            recent_data = self.data.iloc[-30:]
            daily_returns = recent_data['Close'].pct_change().dropna()
            self.market_volatility = daily_returns.std()
            print(f"Hesaplanan piyasa volatilitesi: {self.market_volatility:.4f}")
        
    def setup_genetic_algorithm(self):
        """Genetik algoritma için gerekli yapıları oluştur"""
        # Fitness fonksiyonu tanımla (maksimize edilecek)
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        
        # Birey sınıfını tanımla (ağırlıklar listesi)
        creator.create("Individual", list, fitness=creator.FitnessMax)
        
        # Toolbox oluştur
        self.toolbox = base.Toolbox()
        
        # Rastgele ağırlık üreteci
        self.toolbox.register("attr_float", random.uniform, -1.0, 1.0)
        
        # Birey ve popülasyon oluşturucuları
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, 
                             self.toolbox.attr_float, n=len(self.technical_features))
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        # Genetik operatörleri tanımla
        self.toolbox.register("evaluate", self.evaluate_weights)
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
    
    def evaluate_weights(self, individual):
        """Ağırlıkların performansını değerlendir"""
        # Test veri seti seç
        dates = self.data['Date'].sort_values()
        test_start_idx = len(dates) - 200  # Son 200 günü kullan
        test_end_idx = len(dates) - 50    # Son 50 günü tahmin için ayır
        
        test_data = self.data.iloc[test_start_idx:test_end_idx].copy()
        
        # Tahminleri yap ve doğruluğu hesapla
        correct_predictions = 0
        total_predictions = 0
        
        for i in range(len(test_data) - 5):  # 5 günlük tahmin penceresi
            current_idx = i
            future_idx = i + 5
            
            # Mevcut gün için teknik göstergeleri al
            current_features = test_data.iloc[current_idx][self.technical_features].values
            
            # Ağırlıklı skor hesapla
            weighted_score = sum(w * f for w, f in zip(individual, current_features))
            
            # Daha gerçekçi gürültü ekle
            if random.random() < self.noise_level:
                # Piyasa volatilitesine bağlı gürültü
                volatility_factor = max(0.1, self.market_volatility * 10)  # Volatilite faktörü
                
                # Günün volatilitesini hesapla (yüksek-düşük farkı)
                daily_volatility = test_data.iloc[current_idx]['High'] - test_data.iloc[current_idx]['Low']
                daily_volatility_pct = daily_volatility / test_data.iloc[current_idx]['Close']
                
                # Gürültü seviyesini belirle (normal dağılım kullanarak)
                noise_magnitude = np.random.normal(0, volatility_factor * daily_volatility_pct)
                
                # Gürültüyü ekle
                weighted_score += noise_magnitude
            
            # Yön tahmini yap
            prediction = 1 if weighted_score > 0 else 0  # 1: yükseliş, 0: düşüş
            
            # Gerçek yönü belirle
            actual_direction = 1 if test_data.iloc[future_idx]['Close'] > test_data.iloc[current_idx]['Close'] else 0
            
            # Doğruluğu güncelle
            if prediction == actual_direction:
                correct_predictions += 1
            total_predictions += 1
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        return (accuracy,)
    
    def train(self):
        """Genetik algoritma ile en iyi ağırlıkları bul (cross-validation ile)"""
        # Genetik algoritma yapılarını oluştur
        self.setup_genetic_algorithm()
        
        # Veriyi zaman sırasına göre sırala
        sorted_data = self.data.sort_values('Date').reset_index(drop=True)
        
        # Eğitim verisi olarak kullanılacak veri setini seç (son 250 gün)
        train_data_size = min(250, len(sorted_data) - 50)  # Son 50 günü test için ayır
        train_data = sorted_data.iloc[-train_data_size-50:-50].copy()
        
        # Cross-validation için veriyi böl
        kf = KFold(n_splits=self.n_splits, shuffle=False)  # Zaman serisi olduğu için shuffle=False
        
        print(f"Cross-validation başlıyor ({self.n_splits} bölüm)...")
        cv_scores = []
        
        fold = 1
        for train_index, val_index in kf.split(train_data):
            print(f"\nFold {fold}/{self.n_splits} eğitiliyor...")
            fold += 1
            
            # Başlangıç popülasyonunu oluştur
            pop = self.toolbox.population(n=self.population_size)
            
            # Bu fold için değerlendirme fonksiyonunu güncelle
            def evaluate_fold(individual):
                train_fold = train_data.iloc[train_index]
                val_fold = train_data.iloc[val_index]
                
                # Validation seti üzerinde doğruluğu hesapla
                correct = 0
                total = 0
                
                for i in range(len(val_fold) - 5):  # 5 günlük tahmin penceresi
                    current_idx = i
                    future_idx = i + 5
                    
                    # Mevcut gün için teknik göstergeleri al
                    current_features = val_fold.iloc[current_idx][self.technical_features].values
                    
                    # Ağırlıklı skor hesapla
                    weighted_score = sum(w * f for w, f in zip(individual, current_features))
                    
                    # Daha gerçekçi gürültü ekle
                    if random.random() < self.noise_level:
                        # Piyasa volatilitesine bağlı gürültü
                        volatility_factor = max(0.1, self.market_volatility * 10)
                        
                        # Günün volatilitesini hesapla
                        daily_volatility = val_fold.iloc[current_idx]['High'] - val_fold.iloc[current_idx]['Low']
                        daily_volatility_pct = daily_volatility / val_fold.iloc[current_idx]['Close']
                        
                        # Gürültü seviyesini belirle
                        noise_magnitude = np.random.normal(0, volatility_factor * daily_volatility_pct)
                        weighted_score += noise_magnitude
                    
                    # Yön tahmini yap
                    prediction = 1 if weighted_score > 0 else 0
                    
                    # Gerçek yönü belirle
                    actual_direction = 1 if val_fold.iloc[future_idx]['Close'] > val_fold.iloc[current_idx]['Close'] else 0
                    
                    # Doğruluğu güncelle
                    if prediction == actual_direction:
                        correct += 1
                    total += 1
                
                accuracy = correct / total if total > 0 else 0
                return (accuracy,)
            
            # Değerlendirme fonksiyonunu güncelle
            self.toolbox.register("evaluate", evaluate_fold)
            
            # İstatistikleri kaydet
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("avg", np.mean)
            stats.register("min", np.min)
            stats.register("max", np.max)
            
            # Genetik algoritmayı çalıştır (daha az nesil)
            pop, _ = algorithms.eaSimple(pop, self.toolbox, 
                                        cxpb=self.crossover_prob, 
                                        mutpb=self.mutation_prob, 
                                        ngen=max(10, self.generations // self.n_splits), 
                                        stats=stats, 
                                        verbose=False)
            
            # En iyi bireyi bul ve skorunu kaydet
            best_ind = tools.selBest(pop, 1)[0]
            cv_scores.append(best_ind.fitness.values[0])
            print(f"Fold {fold-1} doğruluk: {best_ind.fitness.values[0]:.4f}")
        
        print(f"\nCross-validation sonuçları:")
        print(f"Ortalama doğruluk: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        
        # Tüm eğitim verisi üzerinde son bir eğitim yap
        print("\nTüm eğitim verisi üzerinde final model eğitiliyor...")
        
        # Başlangıç popülasyonunu oluştur
        pop = self.toolbox.population(n=self.population_size)
        
        # Değerlendirme fonksiyonunu orijinal haline getir
        self.toolbox.register("evaluate", self.evaluate_weights)
        
        # İstatistikleri kaydet
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("min", np.min)
        stats.register("max", np.max)
        
        # Genetik algoritmayı çalıştır
        pop, logbook = algorithms.eaSimple(pop, self.toolbox, 
                                         cxpb=self.crossover_prob, 
                                         mutpb=self.mutation_prob, 
                                         ngen=self.generations, 
                                         stats=stats, 
                                         verbose=True)
        
        # En iyi bireyi bul
        self.best_individual = tools.selBest(pop, 1)[0]
        print(f"\nEn iyi ağırlıklar: {self.best_individual}")
        print(f"En iyi doğruluk: {self.best_individual.fitness.values[0]:.4f}")
        print(f"Cross-validation doğruluğu: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")
        
        # Eğitim istatistiklerini görselleştir
        self.plot_training_stats(logbook)
        
        return self.best_individual
    
    def plot_training_stats(self, logbook):
        """Eğitim istatistiklerini görselleştir"""
        gen = logbook.select("gen")
        fit_max = logbook.select("max")
        fit_avg = logbook.select("avg")
        
        plt.figure(figsize=(10, 6))
        plt.plot(gen, fit_max, 'b-', label='En İyi Doğruluk')
        plt.plot(gen, fit_avg, 'r-', label='Ortalama Doğruluk')
        plt.title('Genetik Algoritma Eğitim İstatistikleri')
        plt.xlabel('Nesil')
        plt.ylabel('Doğruluk')
        plt.legend(loc='best')
        plt.grid(True)
        plt.savefig('genetic_training_stats.png')
        plt.close()
    
    def predict_next_week(self):
        """En son veriden sonraki hafta için tahmin yap"""
        if not hasattr(self, 'best_individual'):
            print("Önce modeli eğitmelisiniz!")
            return None
        
        # Son günün verilerini al
        last_day = self.data.iloc[-1]
        current_features = last_day[self.technical_features].values
        
        # Ağırlıklı skor hesapla
        weighted_score = sum(w * f for w, f in zip(self.best_individual, current_features))
        
        # Tahmin yap
        prediction = 1 if weighted_score > 0 else 0  # 1: yükseliş, 0: düşüş
        
        # Son 5 günlük fiyat hareketini analiz et
        last_prices = self.data.iloc[-5:]['Close'].values
        avg_daily_change = np.mean(np.diff(last_prices)) / last_prices[:-1].mean()
        
        # Gelecek hafta için tahmini fiyatları hesapla
        last_price = last_day['Close']
        last_date = last_day['Date']
        
        future_prices = []
        future_dates = []
        
        # Haftalık tahmin (5 iş günü)
        for i in range(1, 6):
            # Tahmini günlük değişim (yön ve ortalama değişim hızına göre)
            if prediction == 1:  # Yükseliş
                change_factor = max(0.001, avg_daily_change * (1 + random.uniform(-0.5, 0.5) * self.noise_level))
            else:  # Düşüş
                change_factor = min(-0.001, -abs(avg_daily_change) * (1 + random.uniform(-0.5, 0.5) * self.noise_level))
            
            # Yeni fiyat hesapla
            if i == 1:
                new_price = last_price * (1 + change_factor)
            else:
                new_price = future_prices[-1] * (1 + change_factor)
            
            future_prices.append(new_price)
            
            # Yeni tarih hesapla (hafta sonlarını atla)
            if i == 1:
                new_date = last_date + timedelta(days=1)
            else:
                new_date = future_dates[-1] + timedelta(days=1)
            
            # Hafta sonu kontrolü
            while new_date.weekday() >= 5:  # 5: Cumartesi, 6: Pazar
                new_date += timedelta(days=1)
            
            future_dates.append(new_date)
        
        return prediction, list(zip(future_dates, future_prices))
    
    def predict_future(self, days=7):
        """Gelecek günler için tahmin yap ve DataFrame olarak döndür"""
        if not hasattr(self, 'best_individual'):
            print("Önce modeli eğitmelisiniz!")
            # Eğer model eğitilmemişse, otomatik olarak eğit
            self.train()
        
        # Son günün verilerini al
        last_day = self.data.iloc[-1]
        current_features = last_day[self.technical_features].values
        
        # Ağırlıklı skor hesapla
        weighted_score = sum(w * f for w, f in zip(self.best_individual, current_features))
        
        # Tahmin yap
        prediction = 1 if weighted_score > 0 else 0  # 1: yükseliş, 0: düşüş
        
        # Son 5 günlük fiyat hareketini analiz et
        last_prices = self.data.iloc[-5:]['Close'].values
        avg_daily_change = np.mean(np.diff(last_prices)) / last_prices[:-1].mean()
        
        # Gelecek günler için tahmini fiyatları hesapla
        last_price = last_day['Close']
        last_date = last_day['Date']
        
        future_prices = []
        future_dates = []
        
        # Günlük tahminler
        for i in range(1, days + 1):
            # Tahmini günlük değişim (yön ve ortalama değişim hızına göre)
            if prediction == 1:  # Yükseliş
                change_factor = max(0.001, avg_daily_change * (1 + random.uniform(-0.5, 0.5) * self.noise_level))
            else:  # Düşüş
                change_factor = min(-0.001, -abs(avg_daily_change) * (1 + random.uniform(-0.5, 0.5) * self.noise_level))
            
            # Yeni fiyat hesapla
            if i == 1:
                new_price = last_price * (1 + change_factor)
            else:
                new_price = future_prices[-1] * (1 + change_factor)
            
            future_prices.append(new_price)
            
            # Yeni tarih hesapla (hafta sonlarını atla)
            if i == 1:
                new_date = last_date + timedelta(days=1)
            else:
                new_date = future_dates[-1] + timedelta(days=1)
            
            # Hafta sonu kontrolü
            while new_date.weekday() >= 5:  # 5: Cumartesi, 6: Pazar
                new_date += timedelta(days=1)
            
            future_dates.append(new_date)
        
        # Tahmin sonuçlarını DataFrame'e dönüştür
        predictions_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Price': future_prices,
            'Direction': ["Yükseliş" if prediction == 1 else "Düşüş"] * len(future_dates)
        })
        
        return predictions_df
        
    def evaluate_model(self):
        """Modelin performansını değerlendir"""
        if not hasattr(self, 'best_individual'):
            print("Önce modeli eğitmelisiniz!")
            return None
        
        # Test veri seti seç
        dates = self.data['Date'].sort_values()
        test_start_idx = len(dates) - 50  # Son 50 günü test için kullan
        test_data = self.data.iloc[test_start_idx:].copy()
        
        # Tahminleri yap ve doğruluğu hesapla
        correct_predictions = 0
        total_predictions = 0
        
        predictions = []
        actuals = []
        
        for i in range(len(test_data) - 5):  # 5 günlük tahmin penceresi
            current_idx = i
            future_idx = i + 5
            
            # Mevcut gün için teknik göstergeleri al
            current_features = test_data.iloc[current_idx][self.technical_features].values
            
            # Ağırlıklı skor hesapla
            weighted_score = sum(w * f for w, f in zip(self.best_individual, current_features))
            
            # Daha gerçekçi gürültü ekle
            if random.random() < self.noise_level:
                # Piyasa volatilitesine bağlı gürültü
                volatility_factor = max(0.1, self.market_volatility * 10)  # Volatilite faktörü
                
                # Günün volatilitesini hesapla (yüksek-düşük farkı)
                daily_volatility = test_data.iloc[current_idx]['High'] - test_data.iloc[current_idx]['Low']
                daily_volatility_pct = daily_volatility / test_data.iloc[current_idx]['Close']
                
                # Gürültü seviyesini belirle (normal dağılım kullanarak)
                noise_magnitude = np.random.normal(0, volatility_factor * daily_volatility_pct)
                
                # Gürültüyü ekle
                weighted_score += noise_magnitude
            
            # Yön tahmini yap
            prediction = 1 if weighted_score > 0 else 0  # 1: yükseliş, 0: düşüş
            
            # Gerçek yönü belirle
            actual_direction = 1 if test_data.iloc[future_idx]['Close'] > test_data.iloc[current_idx]['Close'] else 0
            
            predictions.append(prediction)
            actuals.append(actual_direction)
            
            # Doğruluğu güncelle
            if prediction == actual_direction:
                correct_predictions += 1
            total_predictions += 1
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        # Confusion matrix hesapla
        tp = sum(1 for p, a in zip(predictions, actuals) if p == 1 and a == 1)
        fp = sum(1 for p, a in zip(predictions, actuals) if p == 1 and a == 0)
        tn = sum(1 for p, a in zip(predictions, actuals) if p == 0 and a == 0)
        fn = sum(1 for p, a in zip(predictions, actuals) if p == 0 and a == 1)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\nModel Değerlendirme Sonuçları:")
        print(f"Doğruluk: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1_score:.4f}")
        
        return accuracy, precision, recall, f1_score

if __name__ == "__main__":
    # Farklı gürültü seviyeleri için test et
    noise_levels = [0.3, 0.4, 0.5]
    
    for noise in noise_levels:
        print(f"\n{'='*50}")
        print(f"Gürültü seviyesi: {noise}")
        print(f"{'='*50}")
        
        # Genetik algoritma modelini oluştur
        model = StockGenetic(noise_level=noise)
        
        # Veriyi yükle
        model.load_data('../AAPL_with_indicators.csv')
        
        # Modeli eğit
        best_weights = model.train()
        
        # Modeli değerlendir
        model.evaluate_model()
    
    # En son model ile gelecek hafta için tahmin yap
    prediction, future_path = model.predict_next_week()
    
    print(f"\nGelecek hafta için tahmin:")
    print(f"Yön: {'Yükselme' if prediction == 1 else 'Düşüş'}")
    print("\nTahmini fiyat yolu:")
    for date, price in future_path:
        print(f"{date.strftime('%Y-%m-%d')}: ${price:.2f}")
