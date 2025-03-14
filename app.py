import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os
import sys
import io
import contextlib

# Modülleri import et
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from generic.generic import StockGenetic
from aStar.aStar import StockAStar
from MachineLearning.stockScope_model import StockPredictor
from Markov.markov import MarkovChainStockPredictor

# Sayfa yapılandırması
st.set_page_config(
    page_title="StockScope",
    page_icon="📈"
)

# Başlık ve açıklama
st.title("📈 StockScope - Hisse Senedi Tahmin Aracı")

# Algoritma seçimi
algorithm = st.selectbox(
    "Tahmin Algoritması Seçin",
    ["Genetik Algoritma", "A* Algoritması", "Makine Öğrenimi", "Markov Zinciri"],
    index=0
)

# Veriyi yükle
@st.cache_data
def load_data(file_path):
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    return data

try:
    # Sabit veri dosyası
    data_file = 'AAPL_with_indicators.csv'
    data = load_data(data_file)
    
    # Son 30 günlük veriyi göster
    st.subheader("Son 30 Günlük Fiyat Grafiği")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data['Date'].iloc[-30:],
        y=data['Close'].iloc[-30:],
        mode='lines',
        name='Kapanış Fiyatı',
        line=dict(color='blue')
    ))
    fig.update_layout(
        xaxis_title='Tarih',
        yaxis_title='Fiyat',
        hovermode='x unified'
    )
    st.plotly_chart(fig)
    
    # Tahmin fonksiyonları
    def run_genetic_algorithm():
        st.subheader("🧬 Genetik Algoritma Tahminleri")
        
        # Terminal çıktıları için bir alan oluştur
        console_output = st.empty()
        
        with st.spinner("Genetik algoritma çalışıyor..."):
            # Model parametreleri
            col1, col2, col3 = st.columns(3)
            with col1:
                population_size = st.slider("Popülasyon Büyüklüğü", 50, 200, 100, 10)
            with col2:
                generations = st.slider("Nesil Sayısı", 10, 100, 30, 5)
            with col3:
                noise_level = st.slider("Gürültü Seviyesi", 0.1, 1.0, 0.3, 0.1)
            
            # Terminal çıktılarını yakala
            with capture_output() as (stdout, stderr):
                # Modeli oluştur ve eğit
                model = StockGenetic()
                model.load_data(data_file)
                model.noise_level = noise_level
                model.population_size = population_size
                model.generations = generations
                
                # Eğitim
                stats = model.train()
                
                # Tahminler
                predictions = model.predict_future(days=prediction_days)
            
            # Terminal çıktılarını göster
            stdout_output = stdout.getvalue()
            stderr_output = stderr.getvalue()
            
            if stdout_output or stderr_output:
                with st.expander("Terminal Çıktıları", expanded=True):
                    if stdout_output:
                        st.subheader("Standart Çıktı")
                        st.code(stdout_output)
                    if stderr_output:
                        st.subheader("Hata Çıktısı")
                        st.code(stderr_output)
            
            # Eğitim istatistikleri
            st.subheader("Eğitim İstatistikleri")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(stats["max"], label="En İyi Skor")
            ax.plot(stats["avg"], label="Ortalama Skor")
            ax.set_xlabel("Nesil")
            ax.set_ylabel("Skor")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
            
            # Tahminler
            st.subheader("Tahmin Sonuçları")
            
            # Tahmin tablosu
            st.dataframe(predictions)
            
            # Tahmin grafiği
            fig = go.Figure()
            
            # Geçmiş veriler
            fig.add_trace(go.Scatter(
                x=data['Date'].iloc[-30:],
                y=data['Close'].iloc[-30:],
                mode='lines',
                name='Geçmiş Fiyat',
                line=dict(color='blue')
            ))
            
            # Tahmin verileri
            fig.add_trace(go.Scatter(
                x=predictions['Date'],
                y=predictions['Predicted_Price'],
                mode='lines+markers',
                name='Tahmin',
                line=dict(color='red', dash='dash')
            ))
            
            fig.update_layout(
                title='Genetik Algoritma Tahminleri',
                xaxis_title='Tarih',
                yaxis_title='Fiyat',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            return predictions
    
    def run_astar_algorithm():
        st.subheader("🔍 A* Algoritması Tahminleri")
        
        # Terminal çıktıları için bir alan oluştur
        console_output = st.empty()
        
        with st.spinner("A* algoritması çalışıyor..."):
            # Model parametreleri
            col1, col2 = st.columns(2)
            with col1:
                max_iterations = st.slider("Maksimum İterasyon", 100, 1000, 500, 50)
            with col2:
                heuristic_weight = st.slider("Sezgisel Ağırlık", 0.1, 2.0, 1.0, 0.1)
            
            # Terminal çıktılarını yakala
            with capture_output() as (stdout, stderr):
                # Modeli oluştur ve eğit
                model = StockAStar()
                model.load_data(data_file)
                model.max_iterations = max_iterations
                model.heuristic_weight = heuristic_weight
                
                # Eğitim
                path, cost = model.find_path()
                
                # Tahminler
                predictions = model.predict_future(days=prediction_days)
            
            # Terminal çıktılarını göster
            stdout_output = stdout.getvalue()
            stderr_output = stderr.getvalue()
            
            if stdout_output or stderr_output:
                with st.expander("Terminal Çıktıları", expanded=True):
                    if stdout_output:
                        st.subheader("Standart Çıktı")
                        st.code(stdout_output)
                    if stderr_output:
                        st.subheader("Hata Çıktısı")
                        st.code(stderr_output)
            
            # Sonuçlar
            st.subheader("A* Sonuçları")
            if cost is not None:
                st.write(f"Bulunan yolun maliyeti: {cost:.4f}")
            else:
                st.warning("Yol bulunamadı.")
            
            # Tahminler
            st.subheader("Tahmin Sonuçları")
            
            # Tahmin tablosu
            st.dataframe(predictions)
            
            # Tahmin grafiği
            fig = go.Figure()
            
            # Geçmiş veriler
            fig.add_trace(go.Scatter(
                x=data['Date'].iloc[-30:],
                y=data['Close'].iloc[-30:],
                mode='lines',
                name='Geçmiş Fiyat',
                line=dict(color='blue')
            ))
            
            # Tahmin verileri
            fig.add_trace(go.Scatter(
                x=predictions['Date'],
                y=predictions['Predicted_Price'],
                mode='lines+markers',
                name='Tahmin',
                line=dict(color='green', dash='dash')
            ))
            
            fig.update_layout(
                title='A* Algoritması Tahminleri',
                xaxis_title='Tarih',
                yaxis_title='Fiyat',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            return predictions
    
    def run_machine_learning():
        st.subheader("🤖 Makine Öğrenimi Tahminleri")
        
        # Terminal çıktıları için bir alan oluştur
        console_output = st.empty()
        
        with st.spinner("Makine öğrenimi modeli çalışıyor..."):
            # Terminal çıktılarını yakala
            with capture_output() as (stdout, stderr):
                # Modeli oluştur ve eğit
                model = StockPredictor()
                model.load_data(data_file)
                
                # Eğitim
                metrics = model.train_models()
                
                # Tahminler
                predictions = model.predict_future(days=prediction_days)
            
            # Terminal çıktılarını göster
            stdout_output = stdout.getvalue()
            stderr_output = stderr.getvalue()
            
            if stdout_output or stderr_output:
                with st.expander("Terminal Çıktıları", expanded=True):
                    if stdout_output:
                        st.subheader("Standart Çıktı")
                        st.code(stdout_output)
                    if stderr_output:
                        st.subheader("Hata Çıktısı")
                        st.code(stderr_output)
            
            # Sonuçlar
            st.subheader("Model Performansı")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Doğruluk", f"{metrics['accuracy']:.4f}")
            col2.metric("Precision", f"{metrics['precision']:.4f}")
            col3.metric("Recall", f"{metrics['recall']:.4f}")
            col4.metric("F1 Skoru", f"{metrics['f1']:.4f}")
            
            # Tahminler
            st.subheader("Tahmin Sonuçları")
            
            # Tahmin tablosu
            st.dataframe(predictions)
            
            # Tahmin grafiği
            fig = go.Figure()
            
            # Geçmiş veriler
            fig.add_trace(go.Scatter(
                x=data['Date'].iloc[-30:],
                y=data['Close'].iloc[-30:],
                mode='lines',
                name='Geçmiş Fiyat',
                line=dict(color='blue')
            ))
            
            # Tahmin verileri
            fig.add_trace(go.Scatter(
                x=predictions['Date'],
                y=predictions['Predicted_Price'],
                mode='lines+markers',
                name='Tahmin',
                line=dict(color='purple', dash='dash')
            ))
            
            fig.update_layout(
                title='Makine Öğrenimi Tahminleri',
                xaxis_title='Tarih',
                yaxis_title='Fiyat',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            return predictions
    
    def run_markov_chain():
        st.subheader("⛓️ Markov Zinciri Tahminleri")
        
        # Terminal çıktıları için bir alan oluştur
        console_output = st.empty()
        
        with st.spinner("Markov zinciri modeli çalışıyor..."):
            # Model parametreleri
            col1, col2 = st.columns(2)
            with col1:
                order = st.slider("Markov Zinciri Derecesi", 1, 3, 1, 1)
            with col2:
                bins = st.slider("Ayrık Durum Sayısı", 5, 20, 10, 1)
            
            # Terminal çıktılarını yakala
            with capture_output() as (stdout, stderr):
                # Modeli oluştur ve eğit
                model = MarkovChainStockPredictor(order=order, discretization_bins=bins)
                model.load_data(data_file)
                model.train()
                
                # Tahminler
                predictions = model.predict()
                metrics = model.evaluate(predictions)
                
                # Gelecek tahminleri
                future = model.predict_future(days=prediction_days)
            
            # Terminal çıktılarını göster
            stdout_output = stdout.getvalue()
            stderr_output = stderr.getvalue()
            
            if stdout_output or stderr_output:
                with st.expander("Terminal Çıktıları", expanded=True):
                    if stdout_output:
                        st.subheader("Standart Çıktı")
                        st.code(stdout_output)
                    if stderr_output:
                        st.subheader("Hata Çıktısı")
                        st.code(stderr_output)
            
            # Sonuçlar
            st.subheader("Model Performansı")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Durum Doğruluğu", f"{metrics['accuracy']:.4f}")
            col2.metric("Yön Doğruluğu", f"{metrics['direction_accuracy']:.4f}")
            col3.metric("Precision", f"{metrics['precision']:.4f}")
            col4.metric("F1 Skoru", f"{metrics['f1']:.4f}")
            
            # Gelecek tahminleri
            st.subheader("Tahmin Sonuçları")
            
            # Tahmin tablosu
            st.dataframe(future)
            
            # Tahmin grafiği
            fig = go.Figure()
            
            # Geçmiş veriler
            fig.add_trace(go.Scatter(
                x=data['Date'].iloc[-30:],
                y=data['Close'].iloc[-30:],
                mode='lines',
                name='Geçmiş Fiyat',
                line=dict(color='blue')
            ))
            
            # Tahmin verileri
            fig.add_trace(go.Scatter(
                x=future['Date'],
                y=future['Predicted_Price'],
                mode='lines+markers',
                name='Tahmin',
                line=dict(color='orange', dash='dash')
            ))
            
            fig.update_layout(
                title='Markov Zinciri Tahminleri',
                xaxis_title='Tarih',
                yaxis_title='Fiyat',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Geçiş matrisi (sadece 1. derece için)
            if order == 1:
                st.subheader("Markov Zinciri Geçiş Matrisi")
                model.plot_transition_matrix()
            
            return future
    
    # Terminal çıktılarını yakalamak için fonksiyon
    @contextlib.contextmanager
    def capture_output():
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        sys.stdout = stdout_buffer
        sys.stderr = stderr_buffer
        try:
            yield stdout_buffer, stderr_buffer
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
    
    # Tahmin butonu
    if st.button("1 Haftalık Tahmin Yap"):
        st.subheader(f"{algorithm} ile 1 Haftalık Tahmin")
        
        # Terminal çıktıları için bir alan oluştur
        console_output = st.empty()
        
        with st.spinner("Tahmin yapılıyor..."):
            # Sabit 7 günlük tahmin
            prediction_days = 7
            
            # Terminal çıktılarını yakala
            with capture_output() as (stdout, stderr):
                if algorithm == "Genetik Algoritma":
                    # Genetik algoritma
                    model = StockGenetic()
                    model.load_data(data_file)
                    model.train()
                    predictions = model.predict_future(days=prediction_days)
                    color = 'red'
                    
                elif algorithm == "A* Algoritması":
                    # A* algoritması
                    model = StockAStar()
                    model.load_data(data_file)
                    model.find_path()
                    predictions = model.predict_future(days=prediction_days)
                    color = 'green'
                    
                elif algorithm == "Makine Öğrenimi":
                    # Makine öğrenimi
                    model = StockPredictor()
                    model.load_data(data_file)
                    model.train_models()
                    predictions = model.predict_future(days=prediction_days)
                    color = 'purple'
                    
                elif algorithm == "Markov Zinciri":
                    # Markov zinciri
                    model = MarkovChainStockPredictor()
                    model.load_data(data_file)
                    model.train()
                    predictions = model.predict_future(days=prediction_days)
                    color = 'orange'
            
            # Terminal çıktılarını göster
            stdout_output = stdout.getvalue()
            stderr_output = stderr.getvalue()
            
            if stdout_output or stderr_output:
                with st.expander("Terminal Çıktıları", expanded=True):
                    if stdout_output:
                        st.subheader("Standart Çıktı")
                        st.code(stdout_output)
                    if stderr_output:
                        st.subheader("Hata Çıktısı")
                        st.code(stderr_output)
            
            # Tahmin tablosu
            st.dataframe(predictions)
            
            # Tahmin grafiği
            fig = go.Figure()
            
            # Geçmiş veriler
            fig.add_trace(go.Scatter(
                x=data['Date'].iloc[-30:],
                y=data['Close'].iloc[-30:],
                mode='lines',
                name='Geçmiş Fiyat',
                line=dict(color='blue')
            ))
            
            # Tahmin verileri
            fig.add_trace(go.Scatter(
                x=predictions['Date'],
                y=predictions['Predicted_Price'],
                mode='lines+markers',
                name='Tahmin',
                line=dict(color=color, dash='dash')
            ))
            
            fig.update_layout(
                xaxis_title='Tarih',
                yaxis_title='Fiyat',
                hovermode='x unified'
            )
            
            st.plotly_chart(fig)

except Exception as e:
    st.error(f"Bir hata oluştu: {e}")
    st.info("İpucu: Veri dosyasının doğru konumda olduğundan emin olun.")

# Altbilgi
st.markdown("---")
st.markdown("""
### StockScope Hakkında

StockScope, farklı algoritmalar kullanarak hisse senedi fiyat tahminleri yapan bir araçtır.

- **Genetik Algoritma**: Evrimsel bir yaklaşımla optimal tahmin parametrelerini bulur
- **A* Algoritması**: Heuristic tabanlı bir arama algoritması kullanarak en iyi tahmin yolunu bulur
- **Makine Öğrenimi**: Ensemble öğrenme teknikleri kullanarak tahminler yapar
- **Markov Zinciri**: Olasılıksal bir model kullanarak fiyat hareketlerini tahmin eder
""")
