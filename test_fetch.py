import yfinance as yf
import pandas as pd
import numpy as np
import time

# Ana kod
ticker_symbol = "AAPL"

for attempt in range(3):  # Maksimum 3 deneme yap
    try:
        # Hisse senedi verisini indir (10 yıllık, haftalık veriler)
        print(f"{ticker_symbol} için veriler çekiliyor...")
        data = yf.download(ticker_symbol, start="1999-01-01", end="2024-01-01", interval="1wk")
        
        # Veri başarılı şekilde çekilmiş mi kontrol et
        if data.empty:
            print(f"No data found for {ticker_symbol}. Retrying... (Attempt {attempt+1}/3)")
            time.sleep(2)  # 2 saniye bekleyerek tekrar dene
            continue
        
        # Veri çerçevesinin yapısını göster
        print("\nVeri çerçevesinin yapısı:")
        print(data.info())
        print("\nİlk 5 satır:")
        print(data.head())
        
        # Eğer çok seviyeli sütunlar varsa, düzleştir
        if isinstance(data.columns, pd.MultiIndex):
            print("Çok seviyeli sütunlar düzleştiriliyor...")
            # Sütun adlarını düzleştir
            data.columns = [col[0] for col in data.columns]
        
        # Tarihi indeksten sütuna çevir
        data = data.reset_index()
        
        print(f"\nTeknik göstergeler ekleniyor...")
        
        # MA4 ve MA8 (4 ve 8 haftalık hareketli ortalamalar)
        data['MA4'] = data['Close'].rolling(window=4).mean()
        data['MA8'] = data['Close'].rolling(window=8).mean()
        
        # EMA4 ve EMA8 (4 ve 8 haftalık üssel hareketli ortalamalar)
        data['EMA4'] = data['Close'].ewm(span=4, adjust=False).mean()
        data['EMA8'] = data['Close'].ewm(span=8, adjust=False).mean()
        
        # Volatilite (20 haftalık standart sapma)
        data['Volatility'] = data['Close'].rolling(window=20).std()
        
        # RSI (Göreceli Güç Endeksi)
        delta = data['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        data['RSI'] = 100 - (100 / (1 + rs))
        
        # OBV (On-Balance Volume)
        data['OBV'] = np.where(data['Close'] > data['Close'].shift(1), data['Volume'], 
                      np.where(data['Close'] < data['Close'].shift(1), -data['Volume'], 0)).cumsum()
        
        # MACD (Moving Average Convergence Divergence)
        ema_fast = data['Close'].ewm(span=12, adjust=False).mean()
        ema_slow = data['Close'].ewm(span=26, adjust=False).mean()
        data['MACD'] = ema_fast - ema_slow
        data['MACD_Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
        data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']
        
        # Label (Gelecek hafta yükselecek mi düşecek mi)
        data['Next_Week_Close'] = data['Close'].shift(-1)  # Bir sonraki haftanın kapanış fiyatı
        data['Label'] = (data['Next_Week_Close'] > data['Close']).astype(int)  # 1: Yükselme, 0: Düşme
        
        # Geçici sütunları kaldır
        data = data.drop(['Next_Week_Close'], axis=1)
        
        # NaN değerleri temizle
        data = data.dropna()
        
        # CSV'ye kaydet
        csv_filename = f"{ticker_symbol}_with_indicators.csv"
        data.to_csv(csv_filename, index=False)
        print(f"{csv_filename} dosyası oluşturuldu.")
        
        # İlk 5 satırı göster
        print("\nVeri önizleme (ilk 5 satır):")
        print(data.head())
        
        # Sütun bilgilerini göster
        print("\nVeri sütunları:")
        for column in data.columns:
            print(f"- {column}")
            
        print(f"\nToplam {len(data)} satır veri kaydedildi.")
        break  # Başarıyla çekildiyse döngüden çık

    except Exception as e:
        print(f"Error fetching data for {ticker_symbol}: {e}")
        print(f"Hata detayı: {type(e).__name__}")
        time.sleep(2)  # 2 saniye bekleyerek tekrar dene
