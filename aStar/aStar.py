import pandas as pd
import numpy as np
import random
from collections import defaultdict
from queue import PriorityQueue

class StockNode:
    def __init__(self, date, price, features):
        self.date = date
        self.price = price
        self.features = features  # Teknik göstergeler
        self.g_score = float('inf')  # Başlangıçtan bu noktaya olan maliyet
        self.f_score = float('inf')  # Toplam tahmini maliyet (g_score + h_score)
        self.came_from = None
        
    def __lt__(self, other):
        return self.f_score < other.f_score

class StockAStar:
    def __init__(self, noise_level=0.3):
        self.data = None
        self.nodes = {}
        self.technical_features = [
            'MA4', 'MA8', 'EMA4', 'EMA8', 'Volatility', 'RSI',
            'OBV', 'MACD', 'MACD_Signal', 'MACD_Hist'
        ]
        self.noise_level = noise_level  # Tahmin gürültü seviyesi (0-1 arası)
    
    def load_data(self, file_path):
        """Veri setini yükle ve node'ları oluştur"""
        self.data = pd.read_csv(file_path)
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        
        # Her tarih için bir node oluştur
        for _, row in self.data.iterrows():
            features = row[self.technical_features].to_dict()
            node = StockNode(row['Date'], row['Close'], features)
            self.nodes[row['Date']] = node
    
    def heuristic(self, current_node, target_date=None):
        """Hedef node'a olan tahmini maliyeti hesapla"""
        # Teknik göstergeleri kullanarak karmaşık bir skor hesapla
        
        # 1. Trend analizi
        short_trend = current_node.features['MA4'] - current_node.features['MA8']
        trend_strength = abs(short_trend) / current_node.features['MA8']
        trend_score = np.sign(short_trend) * trend_strength
        
        # 2. Momentum analizi
        rsi = current_node.features['RSI']
        rsi_score = 0
        if rsi > 70:  # Aşırı alım
            rsi_score = -1
        elif rsi < 30:  # Aşırı satım
            rsi_score = 1
        else:
            rsi_score = (rsi - 50) / 20  # -1 ile 1 arasında normalize et
        
        # 3. MACD analizi
        macd_diff = current_node.features['MACD'] - current_node.features['MACD_Signal']
        macd_strength = abs(macd_diff) / abs(current_node.features['MACD'])
        macd_score = np.sign(macd_diff) * macd_strength if current_node.features['MACD'] != 0 else 0
        
        # 4. Hacim analizi
        volume_trend = np.sign(current_node.features['OBV'])
        
        # Ağırlıklı skor hesapla
        base_score = (
            0.35 * trend_score +
            0.25 * rsi_score +
            0.25 * macd_score +
            0.15 * volume_trend
        )
        
        # Volatilite ile riski ayarla
        risk_factor = 1 + (current_node.features['Volatility'] * 2)
        
        # Gürültü ekle - piyasa belirsizliğini temsil eder
        noise = random.uniform(-self.noise_level, self.noise_level)
        
        # Final skor
        final_score = (abs(base_score) * risk_factor) + noise
        
        return final_score
    
    def get_neighbors(self, date):
        """Verilen tarihin komşu node'larını döndür"""
        neighbors = []
        current_idx = self.data[self.data['Date'] == date].index[0]
        current_node = self.nodes[date]
        
        # İleri doğru 5 günlük komşuları al
        for i in range(1, 6):
            if current_idx + i < len(self.data):
                neighbor_date = self.data.iloc[current_idx + i]['Date']
                neighbor_node = self.nodes[neighbor_date]
                
                # Komşu seçim kriterleri
                price_change = abs(neighbor_node.price - current_node.price) / current_node.price
                volume_change = abs(neighbor_node.features['OBV'] - current_node.features['OBV'])
                
                # Daha esnek filtre kriterleri
                if price_change <= 0.15:  # %15'e kadar fiyat değişimine izin ver
                    neighbors.append(neighbor_node)
        
        # En az bir komşu olduğundan emin ol
        if not neighbors and current_idx + 1 < len(self.data):
            next_date = self.data.iloc[current_idx + 1]['Date']
            neighbors.append(self.nodes[next_date])
        
        return neighbors
    
    def reconstruct_path(self, current):
        """En iyi yolu yeniden oluştur"""
        path = []
        while current:
            path.append((current.date, current.price))
            current = current.came_from
        return path[::-1]
    
    def predict_next_price(self, start_date, prediction_window=5):
        """A* algoritması ile gelecek fiyatı tahmin et"""
        start_node = self.nodes[pd.Timestamp(start_date)]
        start_idx = self.data[self.data['Date'] == start_date].index[0]
        
        if start_idx + prediction_window >= len(self.data):
            return None, None
        
        target_date = self.data.iloc[start_idx + prediction_window]['Date']
        
        # A* algoritması başlat
        open_set = PriorityQueue()
        open_set.put((0, start_node))
        
        # Tüm node'ların g_score ve f_score'larını sıfırla
        for node in self.nodes.values():
            node.g_score = float('inf')
            node.f_score = float('inf')
            node.came_from = None
        
        start_node.g_score = 0
        start_node.f_score = self.heuristic(start_node)
        
        # Ziyaret edilen node'ları takip et
        visited = set()
        
        while not open_set.empty():
            current = open_set.get()[1]
            visited.add(current.date)
            
            if current.date == target_date:
                path = self.reconstruct_path(current)
                
                # Teknik göstergelere dayalı yön tahmini yap
                # Gerçek fiyat farkı yerine teknik göstergeleri kullan
                
                # Son node'un özelliklerini al
                # path bir tuple listesi olduğu için, node'un kendisini kullanmalıyız
                last_date = path[-1][0]  # Son tuple'ın ilk elemanı (tarih)
                last_node_obj = self.nodes[last_date]  # Tarihten node nesnesini al
                
                # Trend sinyalleri
                ma_trend = last_node_obj.features['MA4'] > last_node_obj.features['MA8']
                ema_trend = last_node_obj.features['EMA4'] > last_node_obj.features['EMA8']
                
                # Momentum sinyalleri
                rsi_signal = last_node_obj.features['RSI'] > 50
                macd_signal = last_node_obj.features['MACD'] > last_node_obj.features['MACD_Signal']
                
                # Sinyalleri birleştir
                signals = [ma_trend, ema_trend, rsi_signal, macd_signal]
                positive_signals = sum(signals)
                
                # Gürültü ekle
                if random.random() < self.noise_level:
                    # Rastgele bir yön seç
                    direction = random.randint(0, 1)
                else:
                    # Sinyallerin çoğunluğuna göre yön belirle
                    direction = 1 if positive_signals >= len(signals) / 2 else 0
                
                return direction, path
            
            for neighbor in self.get_neighbors(current.date):
                if neighbor.date in visited:
                    continue
                    
                # Yol maliyetini artır
                tentative_g_score = current.g_score + 1
                
                if tentative_g_score < neighbor.g_score:
                    neighbor.came_from = current
                    neighbor.g_score = tentative_g_score
                    neighbor.f_score = tentative_g_score + self.heuristic(neighbor)
                    open_set.put((neighbor.f_score, neighbor))
        
        return None, None

    def predict_future(self, days=7):
        """Gelecek günler için tahmin yap ve DataFrame olarak döndür"""
        from datetime import timedelta
        
        # Son günün tarihini al
        last_date = self.data['Date'].max()
        
        # Son günden tahmin yap
        direction, path = self.predict_next_price(last_date)
        
        if direction is None or path is None:
            # Eğer tahmin yapılamazsa, son 5 günün ortalama değişimini kullan
            last_prices = self.data.iloc[-5:]['Close'].values
            avg_daily_change = np.mean(np.diff(last_prices)) / last_prices[:-1].mean()
            direction = 1 if avg_daily_change > 0 else 0
            
            # Son fiyat ve tarihi al
            last_price = self.data.iloc[-1]['Close']
            last_date = self.data.iloc[-1]['Date']
            
            # Basit bir tahmin yap
            future_prices = []
            future_dates = []
            
            for i in range(1, days + 1):
                # Tahmini günlük değişim
                if direction == 1:  # Yükseliş
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
        else:
            # A* algoritmasının bulduğu yolu kullan
            # Ancak bu sadece bir sonraki gün için tahmin içeriyor, bunu genişletelim
            
            # Son fiyat ve tarihi al
            last_price = path[-1][1]  # Son tuple'ın ikinci elemanı (fiyat)
            last_date = path[-1][0]   # Son tuple'ın ilk elemanı (tarih)
            
            # Son 5 günlük fiyat hareketini analiz et
            last_prices = [p for _, p in path[-5:]] if len(path) >= 5 else [p for _, p in path]
            if len(last_prices) >= 2:
                avg_daily_change = np.mean(np.diff(last_prices)) / np.mean(last_prices[:-1])
            else:
                # Yeterli veri yoksa, veri setinden hesapla
                last_prices = self.data.iloc[-5:]['Close'].values
                avg_daily_change = np.mean(np.diff(last_prices)) / last_prices[:-1].mean()
            
            # Gelecek günler için tahmin
            future_prices = []
            future_dates = []
            
            # Mevcut path'ten gelen tahminleri ekle (varsa)
            for date, price in path:
                if date > self.data['Date'].max():
                    future_dates.append(date)
                    future_prices.append(price)
            
            # Kalan günler için tahmin yap
            remaining_days = days - len(future_prices)
            
            if remaining_days > 0:
                for i in range(1, remaining_days + 1):
                    # Tahmini günlük değişim
                    if direction == 1:  # Yükseliş
                        change_factor = max(0.001, avg_daily_change * (1 + random.uniform(-0.5, 0.5) * self.noise_level))
                    else:  # Düşüş
                        change_factor = min(-0.001, -abs(avg_daily_change) * (1 + random.uniform(-0.5, 0.5) * self.noise_level))
                    
                    # Yeni fiyat hesapla
                    if future_prices:
                        new_price = future_prices[-1] * (1 + change_factor)
                    else:
                        new_price = last_price * (1 + change_factor)
                    
                    future_prices.append(new_price)
                    
                    # Yeni tarih hesapla (hafta sonlarını atla)
                    if future_dates:
                        new_date = future_dates[-1] + timedelta(days=1)
                    else:
                        new_date = last_date + timedelta(days=1)
                    
                    # Hafta sonu kontrolü
                    while new_date.weekday() >= 5:  # 5: Cumartesi, 6: Pazar
                        new_date += timedelta(days=1)
                    
                    future_dates.append(new_date)
        
        # Tahmin sonuçlarını DataFrame'e dönüştür
        predictions_df = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Price': future_prices,
            'Direction': ["Yükseliş" if direction == 1 else "Düşüş"] * len(future_dates)
        })
        
        return predictions_df
    
    def find_path(self):
        """A* algoritması ile en iyi yolu bul ve döndür"""
        # Son günün tarihini al
        last_date = self.data['Date'].max()
        
        # Son günden 5 gün öncesini başlangıç olarak al
        start_idx = len(self.data) - 6  # Son 5 günlük veriyi kullanmak için 6 gün öncesinden başla
        if start_idx < 0:
            start_idx = 0
        
        start_date = self.data.iloc[start_idx]['Date']
        
        # Tahmin yap
        direction, path = self.predict_next_price(start_date)
        
        return path, direction
        
    def evaluate_predictions(self, start_date, end_date, window=5):
        """Belirli bir tarih aralığında tahminleri değerlendir"""
        current_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)
        
        predictions = []
        actuals = []
        
        while current_date < end_date:
            direction, path = self.predict_next_price(current_date, window)
            
            if direction is not None:
                current_idx = self.data[self.data['Date'] == current_date].index[0]
                future_idx = current_idx + window
                
                if future_idx < len(self.data):
                    actual_direction = 1 if self.data.iloc[future_idx]['Close'] > self.data.iloc[current_idx]['Close'] else 0
                    predictions.append(direction)
                    actuals.append(actual_direction)
            
            # Bir sonraki tarihe geç
            current_date_idx = self.data[self.data['Date'] == current_date].index[0]
            if current_date_idx + 1 < len(self.data):
                current_date = self.data.iloc[current_date_idx + 1]['Date']
            else:
                break
        
        # Doğruluk hesapla
        correct = sum(p == a for p, a in zip(predictions, actuals))
        accuracy = correct / len(predictions) if predictions else 0
        
        return accuracy

if __name__ == "__main__":
    # Çeşitli gürültü seviyelerini test et
    noise_levels = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    print("Farklı gürültü seviyeleri için test sonuçları:")
    
    for noise in noise_levels:
        # A* modelini oluştur
        model = StockAStar(noise_level=noise)
        
        # Veriyi yükle
        model.load_data('../AAPL_with_indicators.csv')
        
        # Test tarihi aralığını belirle (son 200 gün)
        dates = model.data['Date'].sort_values()
        test_start = dates.iloc[-200]
        test_end = dates.iloc[-50]  # Son 50 günü tahmin için ayır
        
        # Tahminleri değerlendir
        accuracy = model.evaluate_predictions(test_start, test_end)
        print(f"Gürültü seviyesi {noise:.1f} için doğruluk: {accuracy:.4f}")
    
    print("\n")
    
    # En uygun gürültü seviyesi ile son model
    model = StockAStar(noise_level=0.5)
    model.load_data('../AAPL_with_indicators.csv')
    
    # Son durumdan gelecek tahmini yap
    dates = model.data['Date'].sort_values()
    last_date = dates.iloc[-6]  # Son 5 günlük veriyi kullanmak için 6 gün öncesinden başla
    direction, path = model.predict_next_price(last_date)
    
    print(f"Son tarihten tahmin:")
    if direction is not None and path is not None:
        print(f"Yön: {'Yükselme' if direction == 1 else 'Düşüş'}")
        print("\nTahmin yolu:")
        for date, price in path:
            print(f"{date.strftime('%Y-%m-%d')}: ${price:.2f}")
    else:
        print("Tahmin yapılamadı.")