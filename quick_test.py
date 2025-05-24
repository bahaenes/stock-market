#!/usr/bin/env python3
"""
Hızlı sistem testi
"""

print("🚀 Sistem Testi Başlatılıyor...\n")

# Temel importları test et
try:
    import pandas as pd
    import numpy as np
    import yfinance as yf
    print("✅ Temel kütüphaneler yüklü")
except ImportError as e:
    print(f"❌ Temel kütüphane hatası: {e}")

# ML kütüphanelerini test et
try:
    import lightgbm as lgb
    print("✅ LightGBM yüklü")
except ImportError:
    print("❌ LightGBM yok")

try:
    from prophet import Prophet
    print("✅ Prophet yüklü")
except ImportError:
    print("❌ Prophet yok")

try:
    from sklearn.ensemble import RandomForestRegressor
    print("✅ Scikit-learn yüklü")
except ImportError:
    print("❌ Scikit-learn yok")

# Transformers test et
try:
    from transformers import AutoTokenizer
    print("✅ Transformers yüklü")
except ImportError:
    print("❌ Transformers yok")

# Flask test et
try:
    from flask import Flask
    print("✅ Flask yüklü")
except ImportError:
    print("❌ Flask yok")

# Hızlı veri testi
try:
    print("\n📊 Veri Testi:")
    ticker = "AAPL"
    data = yf.download(ticker, period="5d", progress=False)
    if not data.empty:
        print(f"✅ {ticker} verisi alındı: {len(data)} kayıt")
        print(f"📈 Son fiyat: ${data['Close'].iloc[-1]:.2f}")
    else:
        print("❌ Veri alınamadı")
except Exception as e:
    print(f"❌ Veri testi hatası: {e}")

print("\n�� Test Tamamlandı!") 