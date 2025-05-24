# 📈 Finans Analiz Aracı - Türkiye Finans Piyasaları

Bu proje, Flask kullanılarak geliştirilmiş gelişmiş bir web uygulamasıdır. Kullanıcılara hisse senedi verilerini analiz etme, modern makine öğrenmesi modelleri ile fiyat tahmini yapma, teknik göstergeleri görüntüleme ve haber duyarlılık analizi yapma imkanı sunar.

## ✨ Özellikler

### 📊 Veri Analizi
- **Kapsamlı Hisse Senedi Listesi:** BIST ve ABD borsalarından popüler hisse senetleri
- **Gerçek Zamanlı Veri:** Yahoo Finance API ile güncel veriler
- **Teknik Göstergeler:** RSI, MACD, Bollinger Bantları, hareketli ortalamalar

### 🤖 Modern ML Modelleri
- **LightGBM:** Gradient boosting ile yüksek performanslı tahmin
- **Prophet:** Facebook'un zaman serisi analiz modeli
- **RandomForest:** Ensemble öğrenme yöntemi
- **Ensemble Modeling:** Birden fazla modelin birleşimi ile daha güvenilir tahminler

### 📰 Duyarlılık Analizi
- **FinBERT:** Finans alanına özel BERT modeli ile haber analizi
- **VADER Sentiment:** Güvenli fallback analiz
- **Güncel Haberler:** NewsAPI entegrasyonu

### 🛡️ Güvenilirlik ve Performans
- **Kapsamlı Error Handling:** Güvenli fallback mekanizmaları
- **Timezone Handling:** Global timezone desteği
- **Önbellekleme:** Hızlı veri erişimi
- **İş Günü Hesaplaması:** Gerçekçi tahmin tarihleri

### 🎨 Kullanıcı Arayüzü
- **Dinamik Grafikleme:** Plotly ile interaktif grafikler
- **Responsive Design:** Mobil uyumlu tasarım
- **Türkçe Arayüz:** Tam Türkçe destek
- **Gerçek Zamanlı Güncelleme:** Anlık veri güncellemeleri

## 🚀 Kurulum

### Gereksinimler
- Python 3.9+ (Python 3.13 test edildi)
- pip
- İnternet bağlantısı

### 1. Projeyi Klonlayın
```bash
git clone https://github.com/username/stock-market.git
cd stock-market
```

### 2. Sanal Ortam Oluşturun
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### 3. Bağımlılıkları Yükleyin
```bash
# Python 3.13 için optimize edilmiş
pip install -r requirements.txt

# Alternatif olarak
pip install -r requirements-py313.txt
```

### 4. Ortam Değişkenlerini Ayarlayın
`.env` dosyası oluşturun:
```env
NEWS_API_KEY=your_news_api_key_here
FLASK_ENV=development
SECRET_KEY=your_secret_key_here
```

### 5. Uygulamayı Başlatın
```bash
python run.py
```

## 💻 Kullanım

1. **Web Arayüzü:** `http://127.0.0.1:5000/` adresine gidin
2. **Hisse Seçimi:** Kenar çubuğundan hisse senedi seçin
3. **Analiz Periyodu:** İstediğiniz zaman dilimini belirleyin
4. **Tahmin Süresi:** Kaç günlük tahmin istediğinizi seçin
5. **Analiz Et:** Kapsamlı analizi görüntüleyin

### Desteklenen Hisse Senetleri
- **BIST:** AKBNK.IS, GARAN.IS, TUPRS.IS, BIMAS.IS, THYAO.IS
- **ABD:** AAPL, GOOGL, MSFT, TSLA, AMZN, NVDA
- **Ve daha fazlası...**

## 🛠️ Teknolojiler

### Backend
- **Flask:** Web framework
- **SQLAlchemy:** Database ORM
- **Pandas:** Veri manipülasyonu
- **NumPy:** Numerik hesaplamalar

### Machine Learning
- **LightGBM:** Microsoft'un gradient boosting kütüphanesi
- **Prophet:** Facebook'un zaman serisi kütüphanesi
- **Scikit-learn:** RandomForest ve diğer ML araçları
- **Transformers:** Hugging Face FinBERT modeli

### Veri Kaynakları
- **yfinance:** Yahoo Finance API
- **NewsAPI:** Güncel haber verileri
- **ta:** Teknik analiz göstergeleri

### Frontend
- **Plotly:** İnteraktif grafikler
- **Bootstrap:** Responsive UI
- **JavaScript:** Dinamik içerik

## 🔧 Yapılandırma

### Model Ayarları
```python
# config.py içinde
PREDICTION_MODELS = ['lightgbm', 'prophet', 'randomforest']
ENSEMBLE_WEIGHTS = {'lightgbm': 0.4, 'prophet': 0.4, 'randomforest': 0.2}
CACHE_MAX_AGE_SECONDS = 3600
```

### Error Handling
```python
SAFE_MODE = True  # Hataları graceful handle et
DEBUG_MODE = False  # Production için False
LOG_LEVEL = 'INFO'
```

## 🧪 Test Etme

```bash
# Tüm sistem testleri
python final_test.py

# Timezone testleri
python test_timezone_fixes.py

# Model testleri
python test_improved_models.py
```

## 📊 Model Performansı

| Model | Accuracy | Speed | Reliability |
|-------|----------|-------|-------------|
| LightGBM | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Prophet | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| RandomForest | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Ensemble | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

## 🚨 Sorun Giderme

### Yaygın Hatalar

1. **Timezone Hatası**
   ```bash
   # Timezone fonksiyonlarını test edin
   python test_timezone_fixes.py
   ```

2. **Model Import Hatası**
   ```bash
   # Bağımlılıkları tekrar yükleyin
   pip install --upgrade -r requirements.txt
   ```

3. **API Limit Hatası**
   ```
   # Demo mod otomatik aktif olur
   Gerçek veriler yerine demo veriler kullanılır
   ```

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/AmazingFeature`)
3. Commit yapın (`git commit -m 'Add some AmazingFeature'`)
4. Push yapın (`git push origin feature/AmazingFeature`)
5. Pull Request açın

## 📝 Changelog

### v2.0.0 (2025-05-24)
- ✅ Modern ML modelleri (LightGBM, Prophet)
- ✅ Kapsamlı error handling sistemi
- ✅ Timezone-aware datetime işlemleri
- ✅ İş günü hesaplaması
- ✅ Ensemble modeling
- ✅ FinBERT sentiment analizi

### v1.0.0 (2025-01-01)
- 🎉 İlk sürüm
- ⚡ XGBoost model
- 📊 Temel teknik analiz

## 📄 Lisans

Bu proje MIT Lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakınız.

## 👨‍💻 Geliştirici

**bahaenes** - [GitHub Profile](https://github.com/bahaenes)

## 🙏 Teşekkürler

- Yahoo Finance - Finansal veri API
- Hugging Face - FinBERT modeli
- Microsoft - LightGBM
- Facebook - Prophet
- NewsAPI - Haber verileri

---

⭐ Bu projeyi beğendiyseniz star vermeyi unutmayın! 