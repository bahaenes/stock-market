#!/usr/bin/env python3
"""
Gelişmiş tahmin modellerini test et
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from app import create_app
from app.services import stock_service, prediction_service, news_service
import logging

# Logging konfigürasyonu
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_improved_services():
    print("🚀 Gelişmiş Servisler Testi\n")
    
    # Flask context oluştur
    app = create_app()
    
    with app.app_context():
        # Test ticker'ları
        test_tickers = ["AAPL", "MSFT", "NVDA"]
        
        for ticker in test_tickers:
            print(f"📊 Test ediliyor: {ticker}")
            
            try:
                # Veri çek
                data = stock_service.get_stock_data(ticker, period="3mo")
                
                if data is not None and not data.empty:
                    print(f"   ✅ VERİ BAŞARILI: {len(data)} kayıt")
                    print(f"   📈 Son fiyat: {data['Close'].iloc[-1]:.2f}")
                    
                    # Teknik analiz test et
                    indicators = stock_service.calculate_technical_indicators(data)
                    if indicators is not None:
                        print(f"   📊 Teknik analiz başarılı")
                        if 'RSI' in indicators and not indicators['RSI'].empty:
                            print(f"   📈 RSI: {indicators['RSI'].iloc[-1]:.2f}")
                    else:
                        print(f"   ⚠️  Teknik analiz başarısız")
                    
                    # Yeni tahmin modeli test et
                    print(f"   🤖 Gelişmiş tahmin modeli test ediliyor...")
                    prediction = prediction_service.predict_stock_price(
                        ticker, data, prediction_days=5
                    )
                    
                    if prediction:
                        print(f"   ✅ TAHMİN BAŞARILI")
                        print(f"   🎯 Model: {prediction['model_name']}")
                        print(f"   📈 Güven: {prediction['confidence']:.2f}")
                        print(f"   💰 Son tahmin: {prediction['predictions']['predicted_price'].iloc[-1]:.2f}")
                        
                        if 'individual_models' in prediction:
                            print(f"   🔧 Kullanılan modeller: {prediction['individual_models']}")
                    else:
                        print(f"   ❌ TAHMİN BAŞARISIZ")
                    
                    # Haber analizi test et (güvenli mod)
                    try:
                        print(f"   📰 Haber analizi test ediliyor...")
                        # FinBERT initialize etmeyi dene
                        finbert_success = news_service.initialize_finbert()
                        if finbert_success:
                            print(f"   ✅ FinBERT başarıyla yüklendi")
                        else:
                            print(f"   ⚠️  FinBERT yüklenemedi, VADER kullanılacak")
                        
                        # Test metni
                        test_text = f"{ticker} stock price is rising due to strong earnings report"
                        sentiment = news_service.get_sentiment_finbert(test_text)
                        print(f"   💭 Test sentiment: {sentiment}")
                        
                    except Exception as e:
                        print(f"   ⚠️  Haber analizi atlandı: {e}")
                        
                else:
                    print(f"   ❌ VERİ BAŞARISIZ")
                    
            except Exception as e:
                print(f"   💥 HATA: {e}")
            
            print()

if __name__ == "__main__":
    test_improved_services() 