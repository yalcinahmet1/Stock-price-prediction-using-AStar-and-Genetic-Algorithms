import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

class MarkovChainStockPredictor:
    def __init__(self, order=1, discretization_bins=10):
        """
        Markov zinciri tabanlı hisse senedi fiyat tahmini modeli.
        
        Args:
            order (int): Markov zincirinin derecesi (kaç önceki durumu dikkate alacağı)
            discretization_bins (int): Fiyat değişimlerini kaç ayrık duruma böleceğimiz
        """
        self.order = order
        self.bins = discretization_bins
        self.transitions = defaultdict(lambda: defaultdict(int))
        self.bin_edges = None
        self.bin_labels = None
        self.state_sequence = []
        self.price_changes = None
        self.test_data = None
        self.train_data = None
        
    def load_data(self, file_path):
        """
        Veri setini yükle ve hazırla
        
        Args:
            file_path (str): CSV dosyasının yolu
        """
        try:
            # Veri setini yükle
            data = pd.read_csv(file_path)
            
            # Tarihi datetime formatına çevir
            data['Date'] = pd.to_datetime(data['Date'])
            
            # Fiyat değişimlerini hesapla (yüzde olarak)
            data['PriceChange'] = data['Close'].pct_change() * 100
            
            # NaN değerleri temizle
            data = data.dropna()
            
            # Veriyi eğitim ve test olarak ayır
            self.train_data, self.test_data = train_test_split(
                data, test_size=0.2, shuffle=False
            )
            
            print(f"Veri yüklendi: {len(data)} satır, {len(self.train_data)} eğitim, {len(self.test_data)} test")
            
            # Fiyat değişimlerini ayrık durumlara dönüştür
            self._discretize_price_changes()
            
            return True
        except Exception as e:
            print(f"Veri yükleme hatası: {e}")
            return False
    
    def _discretize_price_changes(self):
        """
        Fiyat değişimlerini ayrık durumlara dönüştür
        """
        # Eğitim verisi üzerinde bin'leri oluştur
        self.price_changes = self.train_data['PriceChange'].values
        self.bin_edges = np.linspace(
            np.percentile(self.price_changes, 1),
            np.percentile(self.price_changes, 99),
            self.bins + 1
        )
        
        # Bin etiketlerini oluştur
        self.bin_labels = [f"Bin_{i}" for i in range(self.bins)]
        
        # Eğitim verisini ayrık durumlara dönüştür
        train_states = np.digitize(self.price_changes, self.bin_edges[:-1])
        self.train_data['State'] = [self.bin_labels[min(s-1, self.bins-1)] for s in train_states]
        
        # Test verisini ayrık durumlara dönüştür
        test_price_changes = self.test_data['PriceChange'].values
        test_states = np.digitize(test_price_changes, self.bin_edges[:-1])
        self.test_data['State'] = [self.bin_labels[min(s-1, self.bins-1)] for s in test_states]
        
        # Durum dizisini oluştur
        self.state_sequence = self.train_data['State'].tolist()
        
        print(f"Fiyat değişimleri {self.bins} ayrık duruma dönüştürüldü")
        
    def train(self):
        """
        Markov zinciri modelini eğit
        """
        # Geçiş matrisini oluştur
        for i in range(len(self.state_sequence) - self.order):
            current_state = tuple(self.state_sequence[i:i+self.order])
            next_state = self.state_sequence[i+self.order]
            self.transitions[current_state][next_state] += 1
        
        # Geçiş olasılıklarını hesapla
        for current_state, next_states in self.transitions.items():
            total = sum(next_states.values())
            for next_state, count in next_states.items():
                self.transitions[current_state][next_state] = count / total
        
        print(f"{self.order}. dereceden Markov zinciri modeli eğitildi")
        print(f"Toplam durum sayısı: {len(self.transitions)}")
        
    def predict_next_state(self, current_state):
        """
        Verilen mevcut duruma göre bir sonraki durumu tahmin et
        
        Args:
            current_state (tuple): Mevcut durum
            
        Returns:
            str: Tahmin edilen bir sonraki durum
        """
        if current_state not in self.transitions:
            # Eğer mevcut durum eğitim verisinde yoksa, en sık görülen durumu döndür
            most_common_state = max(set(self.state_sequence), key=self.state_sequence.count)
            return most_common_state
        
        # Olasılıklara göre bir sonraki durumu seç
        next_states = self.transitions[current_state]
        states = list(next_states.keys())
        probabilities = list(next_states.values())
        
        return np.random.choice(states, p=probabilities)
    
    def predict(self):
        """
        Test verisi üzerinde tahminler yap
        
        Returns:
            pd.DataFrame: Tahminleri içeren DataFrame
        """
        predictions = []
        test_states = self.test_data['State'].tolist()
        
        for i in range(len(test_states) - self.order):
            current_state = tuple(test_states[i:i+self.order])
            actual_next_state = test_states[i+self.order]
            predicted_next_state = self.predict_next_state(current_state)
            
            predictions.append({
                'Date': self.test_data['Date'].iloc[i+self.order],
                'Actual': actual_next_state,
                'Predicted': predicted_next_state,
                'Correct': actual_next_state == predicted_next_state
            })
        
        return pd.DataFrame(predictions)
    
    def evaluate(self, predictions):
        """
        Model performansını değerlendir
        
        Args:
            predictions (pd.DataFrame): Tahminleri içeren DataFrame
            
        Returns:
            dict: Performans metrikleri
        """
        y_true = predictions['Actual']
        y_pred = predictions['Predicted']
        
        # Doğruluk oranı
        accuracy = accuracy_score(y_true, y_pred)
        
        # Yön tahmini (yukarı/aşağı)
        direction_true = [self._get_direction(state) for state in y_true]
        direction_pred = [self._get_direction(state) for state in y_pred]
        direction_accuracy = accuracy_score(direction_true, direction_pred)
        
        # Diğer metrikler
        precision = precision_score(direction_true, direction_pred, average='weighted', zero_division=0)
        recall = recall_score(direction_true, direction_pred, average='weighted', zero_division=0)
        f1 = f1_score(direction_true, direction_pred, average='weighted', zero_division=0)
        
        metrics = {
            'accuracy': accuracy,
            'direction_accuracy': direction_accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        print("\nPerformans Metrikleri:")
        print(f"Durum Doğruluğu: {accuracy:.4f}")
        print(f"Yön Doğruluğu: {direction_accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        return metrics
    
    def _get_direction(self, state):
        """
        Durumun yönünü belirle (yukarı/aşağı)
        
        Args:
            state (str): Durum etiketi
            
        Returns:
            int: 1 (yukarı) veya 0 (aşağı)
        """
        # Bin indeksini al
        bin_index = int(state.split('_')[1])
        
        # Orta noktadan büyükse yukarı, değilse aşağı
        if bin_index >= self.bins // 2:
            return 1  # yukarı
        else:
            return 0  # aşağı
    
    def plot_transition_matrix(self):
        """
        Geçiş matrisini görselleştir
        """
        # Sadece birinci dereceden Markov zincirleri için
        if self.order != 1:
            print("Geçiş matrisi görselleştirmesi sadece 1. derece Markov zincirleri için destekleniyor")
            return
        
        # Geçiş matrisini oluştur
        states = sorted(list(set(self.state_sequence)))
        n_states = len(states)
        transition_matrix = np.zeros((n_states, n_states))
        
        state_to_idx = {state: i for i, state in enumerate(states)}
        
        for current_state, next_states in self.transitions.items():
            if len(current_state) == 1:  # 1. derece
                i = state_to_idx[current_state[0]]
                for next_state, prob in next_states.items():
                    j = state_to_idx[next_state]
                    transition_matrix[i, j] = prob
        
        # Görselleştir
        plt.figure(figsize=(10, 8))
        sns.heatmap(transition_matrix, annot=True, cmap='YlGnBu', xticklabels=states, yticklabels=states)
        plt.title('Markov Zinciri Geçiş Olasılıkları')
        plt.xlabel('Bir Sonraki Durum')
        plt.ylabel('Mevcut Durum')
        plt.tight_layout()
        plt.show()
    
    def plot_predictions(self, predictions):
        """
        Tahminleri görselleştir
        
        Args:
            predictions (pd.DataFrame): Tahminleri içeren DataFrame
        """
        plt.figure(figsize=(12, 6))
        
        # Doğru ve yanlış tahminleri ayır
        correct = predictions[predictions['Correct']]
        incorrect = predictions[~predictions['Correct']]
        
        # Zaman serisi grafiği
        plt.scatter(correct['Date'], [1] * len(correct), color='green', label='Doğru Tahmin', alpha=0.6)
        plt.scatter(incorrect['Date'], [0] * len(incorrect), color='red', label='Yanlış Tahmin', alpha=0.6)
        
        plt.title('Markov Zinciri Tahmin Performansı')
        plt.ylabel('Tahmin Doğruluğu')
        plt.yticks([0, 1], ['Yanlış', 'Doğru'])
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Karışıklık matrisi
        y_true = [self._get_direction(state) for state in predictions['Actual']]
        y_pred = [self._get_direction(state) for state in predictions['Predicted']]
        
        cm = pd.crosstab(pd.Series(y_true, name='Gerçek'), 
                         pd.Series(y_pred, name='Tahmin'))
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Yön Tahmini Karışıklık Matrisi')
        plt.tight_layout()
        plt.show()
    
    def predict_future(self, days=5):
        """
        Gelecekteki fiyat hareketlerini tahmin et
        
        Args:
            days (int): Kaç gün ilerisi için tahmin yapılacağı
            
        Returns:
            pd.DataFrame: Gelecek tahminlerini içeren DataFrame
        """
        # Son durumları al
        last_states = self.test_data['State'].iloc[-self.order:].tolist()
        current_state = tuple(last_states)
        
        # Son kapanış fiyatını al
        last_close = self.test_data['Close'].iloc[-1]
        
        future_predictions = []
        future_dates = pd.date_range(
            start=self.test_data['Date'].iloc[-1] + pd.Timedelta(days=1),
            periods=days,
            freq='B'  # İş günleri
        )
        
        for i, date in enumerate(future_dates):
            # Bir sonraki durumu tahmin et
            next_state = self.predict_next_state(current_state)
            
            # Durumun ortalama fiyat değişimini hesapla
            bin_index = int(next_state.split('_')[1])
            bin_start = self.bin_edges[bin_index]
            bin_end = self.bin_edges[bin_index + 1]
            avg_change = (bin_start + bin_end) / 2
            
            # Yeni fiyatı hesapla
            new_price = last_close * (1 + avg_change / 100)
            last_close = new_price
            
            # Yön belirle
            direction = "Yükseliş" if self._get_direction(next_state) == 1 else "Düşüş"
            
            future_predictions.append({
                'Date': date,
                'Predicted_State': next_state,
                'Predicted_Change': avg_change,
                'Predicted_Price': new_price,
                'Direction': direction
            })
            
            # Durumu güncelle
            current_state = current_state[1:] + (next_state,)
        
        return pd.DataFrame(future_predictions)

# Ana kod
if __name__ == "__main__":
    # Markov zinciri modelini oluştur
    print("Markov Zinciri Hisse Senedi Tahmini Başlıyor...\n")
    
    # Farklı dereceler için modelleri test et
    for order in [1, 2, 3]:
        print(f"\n{order}. Derece Markov Zinciri Modeli:\n{'-'*40}")
        
        # Modeli oluştur
        model = MarkovChainStockPredictor(order=order, discretization_bins=10)
        
        # Veriyi yükle
        model.load_data('../AAPL_with_indicators.csv')
        
        # Modeli eğit
        model.train()
        
        # Tahmin yap
        predictions = model.predict()
        
        # Performansı değerlendir
        metrics = model.evaluate(predictions)
        
        # Geçiş matrisini görselleştir (sadece 1. derece için)
        if order == 1:
            model.plot_transition_matrix()
        
        # Tahminleri görselleştir
        model.plot_predictions(predictions)
        
        # Gelecek tahmini yap
        future = model.predict_future(days=5)
        print("\nGelecek 5 İş Günü İçin Tahminler:")
        print(future[['Date', 'Predicted_Price', 'Direction']])
    
    print("\nMarkov Zinciri Analizi Tamamlandı.")
