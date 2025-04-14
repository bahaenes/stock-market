# Finans Analiz ve Tahmin Aracı

Bu proje, Flask kullanılarak geliştirilmiş bir web uygulamasıdır. Belirli bir hisse senedi için temel ve teknik göstergeleri analiz eder, ilgili haberleri çeker, duyarlılık analizi yapar ve Facebook Prophet modeli ile kısa vadeli fiyat tahmini sunar.

## Özellikler

*   Popüler BIST ve ABD hisseleri için kenar çubuğundan hızlı erişim.
*   Seçilebilir periyotlarda hisse senedi fiyat grafiği (Kapanış, SMA 20, SMA 50).
*   Facebook Prophet ile gelecek 30 gün için fiyat tahmini ve güven aralığı görselleştirmesi.
*   Teknik gösterge grafikleri (RSI, MACD).
*   Temel göstergeler (F/K, Piyasa Değeri, Hacim, 52 Hafta Zirve/Dip vb.).
*   Teknik göstergelerin son değerleri (RSI, MACD).
*   Analist tavsiyesi (varsa).
*   İlgili güncel haberler (NewsAPI - Son 7 gün).
*   Haber başlıkları üzerinden duyarlılık analizi (VADER).
*   Teknik göstergeler, haber duyarlılığı ve Prophet tahminini içeren birleşik özet yorum.
*   **Yasal Uyarı:** Tüm analizler ve tahminler yalnızca bilgilendirme amaçlıdır, yatırım tavsiyesi değildir.

## Kurulum

1.  **Depoyu Klonlayın:**
    ```bash
    git clone https://github.com/KULLANICI_ADINIZ/DEPO_ADINIZ.git
    cd DEPO_ADINIZ
    ```
    *(KULLANICI_ADINIZ ve DEPO_ADINIZ kısımlarını kendi bilgilerinizle değiştirin)*

2.  **(Önerilen) Sanal Ortam Oluşturun ve Aktif Edin:**
    ```bash
    python -m venv venv
    # Windows:
    venv\Scripts\activate
    # macOS/Linux:
    source venv/bin/activate
    ```

3.  **Gerekli Kütüphaneleri Kurun:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **NewsAPI Anahtarını Ayarlayın:**
    Haber analizi özelliğinin çalışması için bir [NewsAPI.org](https://newsapi.org/) anahtarına ihtiyacınız var.
    *   **Yöntem 1 (Önerilen): Ortam Değişkeni:** Uygulamayı çalıştırmadan önce terminalinizde aşağıdaki komutlardan uygun olanı çalıştırın (`YOUR_API_KEY` kısmını kendi anahtarınızla değiştirin):
        ```bash
        # Windows (Komut İstemi): set NEWS_API_KEY=YOUR_API_KEY
        # Windows (PowerShell): $env:NEWS_API_KEY="YOUR_API_KEY"
        # macOS/Linux: export NEWS_API_KEY="YOUR_API_KEY"
        ```
    *   **Yöntem 2: `.env` Dosyası:** Proje ana dizininde `.env` adında bir dosya oluşturun ve içine şunu yazın:
        ```
        NEWS_API_KEY=YOUR_API_KEY
        ```
        *Unutmayın: `.env` dosyası `.gitignore` içinde olmalıdır.*

## Çalıştırma

Uygulamayı başlatmak için proje ana dizinindeyken terminalde şu komutu çalıştırın:
```bash
python app.py
```
Ardından tarayıcınızda `http://127.0.0.1:5000/` (veya terminalde belirtilen başka bir adres) adresine gidin.

## Olası İyileştirmeler

*   Prophet modeli için parametre optimizasyonu.
*   Daha fazla teknik gösterge eklemek (Bollinger Bantları vb.).
*   Türkçe haber kaynakları ve Türkçe duyarlılık analizi entegrasyonu.
*   Kullanıcıların kendi portföylerini takip edebilmesi.
*   Daha gelişmiş ML modelleri (LSTM vb.) denemek.

## Lisans

(Buraya bir lisans belirtebilirsiniz, örn: MIT Lisansı) 