# Finans Analiz Aracı

Bu proje, Flask kullanılarak geliştirilmiş bir web uygulamasıdır. Kullanıcılara hisse senedi verilerini analiz etme, teknik göstergeleri görüntüleme, ilgili haberler hakkında duyarlılık analizi yapma ve XGBoost modeli ile fiyat tahmini alma imkanı sunar.

## Özellikler

-   **Kapsamlı Hisse Senedi Listesi:** BIST ve ABD borsalarından popüler hisse senetlerini içerir.
-   **Dinamik Grafikleme:** Plotly kullanılarak çizgi ve mum grafikleri ile fiyat hareketleri, hareketli ortalamalar (SMA), Bollinger Bantları, RSI ve MACD göstergeleri görselleştirilir.
-   **Temel ve Teknik Göstergeler:** F/K oranı, piyasa değeri, işlem hacmi gibi temel verilerin yanı sıra RSI, MACD gibi teknik göstergelerin son değerleri sunulur.
-   **Haber Duyarlılık Analizi:** NewsAPI üzerinden çekilen güncel haber başlıkları ve açıklamaları FinBERT (Hugging Face Transformers tabanlı) modeli ile analiz edilerek pozitif, negatif veya nötr duyarlılıkları belirlenir.
-   **Fiyat Tahmini:** XGBoost modeli kullanılarak seçilen hisse senedi için kısa vadeli fiyat tahmini yapılır.
-   **Detaylı Yorumlama:** Teknik göstergeler, haber duyarlılığı ve model tahminleri birleştirilerek kapsamlı bir analiz özeti ve yapay zeka destekli yorumlar sunulur.
-   **Kullanıcı Dostu Arayüz:** Kenar çubuğunda aranabilir hisse senedi listesi ve ana içerik alanında analiz sonuçları net bir şekilde gösterilir.
-   **Önbellekleme:** Sık erişilen hisse senedi bilgileri ve verileri için basit bir önbellekleme mekanizması içerir.

## Kurulum

1.  **Proje Dosyalarını İndirin:**
    Bu repoyu klonlayın veya dosyaları ZIP olarak indirin.

2.  **Sanal Ortam Oluşturun (Önerilir):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # Linux/macOS için
    # venv\Scripts\activate  # Windows için
    ```

3.  **Bağımlılıkları Yükleyin:**
    Proje ana dizinindeyken aşağıdaki komutu çalıştırın:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Ortam Değişkenlerini Ayarlayın:**
    Haber analizi özelliğini kullanmak için bir NewsAPI anahtarına ihtiyacınız olacaktır.
    Proje ana dizininde `.env` adında bir dosya oluşturun ve içine aşağıdaki satırı ekleyin (kendi API anahtarınızla değiştirin):
    ```
    NEWS_API_KEY=YOUR_NEWS_API_KEY
    ```
    Eğer NewsAPI anahtarınız yoksa, haber analizi bölümü atlanacaktır.

## Kullanım

1.  Proje ana dizinindeyken Flask uygulamasını başlatın:
    ```bash
    python app.py
    ```
2.  Web tarayıcınızda `http://127.0.0.1:5000/` adresine gidin.
3.  Kenar çubuğundaki listeden bir hisse senedi seçin veya arama kutusunu kullanarak hisse kodu/piyasa ile arama yapın.
4.  Alternatif olarak, ana formdaki "Hisse Kodu" alanına doğrudan bir kod girin (Örn: `GARAN.IS`, `AAPL`).
5.  İstediğiniz periyodu ve grafik türünü seçin.
6.  "Analiz Et" butonuna tıklayın.

## Teknolojiler

-   **Backend:** Flask (Python)
-   **Veri Kaynağı (Hisse Senetleri):** yfinance
-   **Veri Kaynağı (Haberler):** NewsAPI
-   **Veri Analizi ve Manipülasyonu:** Pandas
-   **Teknik Göstergeler:** ta
-   **Grafikleme:** Plotly
-   **Makine Öğrenimi (Duyarlılık Analizi):** Hugging Face Transformers (FinBERT), PyTorch
-   **Makine Öğrenimi (Fiyat Tahmini):** XGBoost
-   **Duyarlılık Analizi (Alternatif/Basit):** VADER Sentiment
-   **Ortam Değişkenleri:** python-dotenv

## Katkıda Bulunma

Katkılarınız her zaman beklerim! Lütfen bir issue açın veya bir pull request gönderin.

## Lisans

Bu proje MIT Lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakınız. 