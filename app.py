from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import plotly.io as pio
import numbers # Sayısal tipleri kontrol etmek için
import ta # Teknik analiz kütüphanesi
from plotly.subplots import make_subplots # Subplot için
import requests # Haber API için
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer # Duyarlılık analizi için
import os # API anahtarını ortam değişkeninden okumak için (önerilen)
from datetime import datetime, timedelta # Tarih işlemleri için
import xgboost as xgb # XGBoost eklendi
from dotenv import load_dotenv # .env dosyasını yüklemek için
import logging # logging modülünü import et

# Hugging Face Transformers ve PyTorch importları
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F # Softmax için

load_dotenv() # .env dosyasındaki değişkenleri ortam değişkeni olarak yükle

app = Flask(__name__)

# --- Kapsamlı Hisse Senedi Listesi ---
# Bu listeyi daha da genişletebilir veya dinamik bir kaynaktan alabilirsiniz.
COMPREHENSIVE_STOCK_LIST = [
    # BIST
    {"ticker": "GARAN.IS", "name": "Garanti BBVA", "market": "BIST"},
    {"ticker": "THYAO.IS", "name": "Türk Hava Yolları", "market": "BIST"},
    {"ticker": "EREGL.IS", "name": "Erdemir", "market": "BIST"},
    {"ticker": "TUPRS.IS", "name": "Tüpraş", "market": "BIST"},
    {"ticker": "AKBNK.IS", "name": "Akbank", "market": "BIST"},
    {"ticker": "SISE.IS", "name": "Şişecam", "market": "BIST"},
    {"ticker": "KCHOL.IS", "name": "Koç Holding", "market": "BIST"},
    {"ticker": "BIMAS.IS", "name": "Bim Mağazaları", "market": "BIST"},
    {"ticker": "PETKM.IS", "name": "Petkim", "market": "BIST"},
    {"ticker": "SAHOL.IS", "name": "Sabancı Holding", "market": "BIST"},
    {"ticker": "ASELS.IS", "name": "Aselsan", "market": "BIST"},
    {"ticker": "PGSUS.IS", "name": "Pegasus", "market": "BIST"},
    {"ticker": "FROTO.IS", "name": "Ford Otosan", "market": "BIST"},
    {"ticker": "TOASO.IS", "name": "Tofaş Oto. Fab.", "market": "BIST"},
    {"ticker": "ARCLK.IS", "name": "Arçelik", "market": "BIST"},
    {"ticker": "YKBNK.IS", "name": "Yapı Kredi Bankası", "market": "BIST"},
    {"ticker": "ISCTR.IS", "name": "İş Bankası (C)", "market": "BIST"},
    {"ticker": "TCELL.IS", "name": "Turkcell", "market": "BIST"},
    {"ticker": "SASA.IS", "name": "SASA Polyester", "market": "BIST"},
    {"ticker": "HEKTS.IS", "name": "Hektaş", "market": "BIST"},
    # ABD
    {"ticker": "AAPL", "name": "Apple Inc.", "market": "ABD"},
    {"ticker": "MSFT", "name": "Microsoft Corp.", "market": "ABD"},
    {"ticker": "GOOGL", "name": "Alphabet Inc. (C)", "market": "ABD"},
    {"ticker": "GOOG", "name": "Alphabet Inc. (A)", "market": "ABD"},
    {"ticker": "AMZN", "name": "Amazon.com Inc.", "market": "ABD"},
    {"ticker": "NVDA", "name": "NVIDIA Corp.", "market": "ABD"},
    {"ticker": "TSLA", "name": "Tesla Inc.", "market": "ABD"},
    {"ticker": "META", "name": "Meta Platforms Inc.", "market": "ABD"},
    {"ticker": "BRK-B", "name": "Berkshire Hathaway (B)", "market": "ABD"},
    {"ticker": "JPM", "name": "JPMorgan Chase & Co.", "market": "ABD"},
    {"ticker": "JNJ", "name": "Johnson & Johnson", "market": "ABD"},
    {"ticker": "V", "name": "Visa Inc.", "market": "ABD"},
    {"ticker": "PG", "name": "Procter & Gamble Co.", "market": "ABD"},
    {"ticker": "MA", "name": "Mastercard Inc.", "market": "ABD"},
    {"ticker": "UNH", "name": "UnitedHealth Group Inc.", "market": "ABD"},
    {"ticker": "HD", "name": "Home Depot, Inc.", "market": "ABD"},
    {"ticker": "BAC", "name": "Bank of America Corp.", "market": "ABD"},
    {"ticker": "PFE", "name": "Pfizer Inc.", "market": "ABD"},
    {"ticker": "DIS", "name": "Walt Disney Co.", "market": "ABD"},
    {"ticker": "NFLX", "name": "Netflix, Inc.", "market": "ABD"},
]
# --- Kapsamlı Hisse Senedi Listesi Sonu ---

# --- FinBERT Model ve Tokenizer Global Değişkenleri ---
finbert_tokenizer = None
finbert_model = None
FINBERT_MODEL_NAME = "ProsusAI/finbert"
# --- FinBERT Model ve Tokenizer Global Değişkenleri Sonu ---

def initialize_finbert():
    """FinBERT modelini ve tokenizer'ını yükler."""
    global finbert_tokenizer, finbert_model
    # Sadece model daha önce yüklenmediyse yükle
    if finbert_model is None and finbert_tokenizer is None:
        try:
            app.logger.info(f"Attempting to load FinBERT model and tokenizer: {FINBERT_MODEL_NAME}")
            finbert_tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_NAME)
            finbert_model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL_NAME)
            # Modeli değerlendirme moduna al (dropout vb. katmanları devre dışı bırakır)
            if finbert_model:
                 finbert_model.eval()
            app.logger.info(f"FinBERT model and tokenizer '{FINBERT_MODEL_NAME}' loaded successfully.")
        except Exception as e:
            app.logger.error(f"Failed to load FinBERT model or tokenizer '{FINBERT_MODEL_NAME}': {e}", exc_info=True)
            # Hata durumunda global değişkenleri None olarak bırak
            finbert_tokenizer = None
            finbert_model = None
    else:
        app.logger.info("FinBERT model and tokenizer already loaded.")

# Uygulama başlatıldığında FinBERT modelini yüklemeyi dene
with app.app_context(): # app.logger'ı kullanabilmek için context içinde çağır
    initialize_finbert()

# --- Haber API Ayarları ---
NEWS_API_URL = 'https://newsapi.org/v2/everything'
# Ortam değişkeninden API anahtarını al ve app.config'e ata
app.config['NEWS_API_KEY'] = os.environ.get('NEWS_API_KEY') # Kullanıcı değişikliği geri alındı, NEWS_API_KEY kullanılıyor
if not app.config['NEWS_API_KEY']: # Kullanıcı değişikliği geri alındı
    app.logger.warning("UYARI: NEWS_API_KEY ortam değişkeni ayarlanmamış. Haber analizi atlanacak.")
# --- Haber API Ayarları Sonu ---

# --- Basit Önbellek (Cache) ---
# Bu, sunucu çalıştığı sürece verileri bellekte tutar.
# Daha gelişmiş çözümler için Flask-Caching gibi eklentiler kullanılabilir.
CACHE_MAX_AGE_SECONDS = 300 # 5 dakika
_info_cache = {}
_data_cache = {}
_lgbm_forecast_cache = {} # LightGBM tahminleri için önbellek
_xgb_forecast_cache = {} # XGBoost tahminleri için önbellek
# --- Basit Önbellek Sonu ---

# --- Helper Fonksiyonlar (Global Kapsamda) ---

def get_numeric_value(data_dict, key):
    value = data_dict.get(key)
    if value is None: return None
    if isinstance(value, pd.Series):
        if not value.empty:
            try: element = value.iloc[0]; return float(element)
            except (ValueError, TypeError, IndexError): return None
        else: return None
    try: return float(value)
    except (ValueError, TypeError): return None

def format_indicator(value, format_spec=".2f"):
    if value is not None:
        try: return f"{value:{format_spec}}"
        except (ValueError, TypeError): return "Veri yok"
    return "Veri yok"

def format_market_cap(value):
    if value is not None:
        try: return f"{value / 1e9:.2f} Milyar $"
        except (ValueError, TypeError): return "Veri yok"
    return "Veri yok"

def format_volume(value):
    if value is not None:
        try: return f"{value / 1e6:.2f} Milyon"
        except (ValueError, TypeError): return "Veri yok"
    return "Veri yok"

# --- Teknik Yorum Yardımcı Fonksiyonları ---
def get_sma_comment(last_close, last_sma20, last_sma50):
    if last_close is not None and last_sma20 is not None and last_sma50 is not None:
        if last_close > last_sma20 and last_close > last_sma50:
            return "Fiyat, kısa ve orta vadeli hareketli ortalamaların üzerinde."
        elif last_close < last_sma20 and last_close < last_sma50:
            return "Fiyat, kısa ve orta vadeli hareketli ortalamaların altında."
        else:
            return "Fiyat, hareketli ortalamalara göre karışık bir görünümde."
    return "Hareketli ortalamalar için yeterli veri yok."

def get_rsi_comment(last_rsi):
    if last_rsi is not None:
        if last_rsi > 70:
            return f"RSI ({format_indicator(last_rsi)}) aşırı alım bölgesinde, olası bir düzeltmeye işaret edebilir."
        elif last_rsi < 30:
            return f"RSI ({format_indicator(last_rsi)}) aşırı satım bölgesinde, olası bir tepki yükselişine işaret edebilir."
        else:
            return f"RSI ({format_indicator(last_rsi)}) nötr bölgede."
    return "RSI hesaplanamadı."

def get_macd_comment(last_macd, last_macd_signal, prev_macd, prev_macd_signal):
    if last_macd is not None and last_macd_signal is not None and prev_macd is not None and prev_macd_signal is not None:
        if last_macd > last_macd_signal and prev_macd <= prev_macd_signal:
            return "MACD çizgisi sinyal çizgisini yukarı kesti (Al sinyali)."
        elif last_macd < last_macd_signal and prev_macd >= prev_macd_signal:
            return "MACD çizgisi sinyal çizgisini aşağı kesti (Sat sinyali)."
        elif last_macd > last_macd_signal:
            return "MACD çizgisi sinyal çizgisinin üzerinde (Pozitif momentum)."
        else: # last_macd <= last_macd_signal (ve yeni bir aşağı kesişim değilse)
            return "MACD çizgisi sinyal çizgisinin altında (Negatif momentum)."
    return "MACD yorumu için yeterli veri yok."

# --- Bollinger Bantları Yorum Fonksiyonu ---
def get_bollinger_comment(last_close, last_bb_high, last_bb_low, last_bb_mavg, last_bb_pband, last_bb_wband, prev_bb_wband):
    if None in [last_close, last_bb_high, last_bb_low, last_bb_mavg, last_bb_pband, last_bb_wband]:
        return "Bollinger Bantları yorumu için yeterli veri yok."

    comment = ""
    # Fiyatın Konumu
    if last_close > last_bb_high:
        comment += f"Fiyat ({format_indicator(last_close)}) Bollinger Üst Bandının ({format_indicator(last_bb_high)}) üzerinde, aşırı alım durumu olabilir. "
    elif last_close < last_bb_low:
        comment += f"Fiyat ({format_indicator(last_close)}) Bollinger Alt Bandının ({format_indicator(last_bb_low)}) altında, aşırı satım durumu olabilir. "
    elif abs(last_close - last_bb_high) < (last_bb_high - last_bb_mavg) * 0.1: # Üst banda çok yakınsa
        comment += f"Fiyat ({format_indicator(last_close)}) Bollinger Üst Bandına ({format_indicator(last_bb_high)}) çok yakın. "
    elif abs(last_close - last_bb_low) < (last_bb_mavg - last_bb_low) * 0.1: # Alt banda çok yakınsa
        comment += f"Fiyat ({format_indicator(last_close)}) Bollinger Alt Bandına ({format_indicator(last_bb_low)}) çok yakın. "
    else:
        comment += f"Fiyat ({format_indicator(last_close)}) Bollinger Bantları arasında ({format_indicator(last_bb_low)} - {format_indicator(last_bb_high)}). "

    # %B Yorumu (last_bb_pband 0-1 arası bir değerdir)
    if last_bb_pband > 0.95:
        comment += f"%B ({format_indicator(last_bb_pband, '.2%')}) çok yüksek, üst banda yakınlığı teyit ediyor. "
    elif last_bb_pband < 0.05:
        comment += f"%B ({format_indicator(last_bb_pband, '.2%')}) çok düşük, alt banda yakınlığı teyit ediyor. "

    # Bant Genişliği Yorumu
    if prev_bb_wband is not None:
        if last_bb_wband < prev_bb_wband and last_bb_wband < 0.10: # %10'un altındaysa dar kabul edilebilir (eşik değere bağlı)
            comment += f"Bant genişliği ({format_indicator(last_bb_wband, '.2%')}) daralıyor ve düşük seviyede, olası bir sıkışma ve ardından sert bir hareket gelebilir (Squeeze). "
        elif last_bb_wband > prev_bb_wband:
            comment += f"Bant genişliği ({format_indicator(last_bb_wband, '.2%')}) genişliyor, artan volatiliteye işaret ediyor. "
        else:
            comment += f"Bant genişliği ({format_indicator(last_bb_wband, '.2%')}) stabil görünüyor. "
    else:
        comment += f"Bant genişliği ({format_indicator(last_bb_wband, '.2%')}). "
        
    return comment.strip()
# --- Bollinger Bantları Yorum Fonksiyonu Sonu ---

# --- LightGBM için Özellik Mühendisliği ---
def create_features_for_lgbm(df_series, target_col='y', n_lags=7, window_size=7):
    """
    Zaman serisi verisinden LightGBM için özellikler oluşturur.
    """
    df = pd.DataFrame(df_series.copy())
    df.rename(columns={df_series.name: target_col}, inplace=True) # Seriyi DataFrame'e çevirip sütun adını belirliyoruz

    # Gecikmeli Değerler (Lagged Features)
    for lag in range(1, n_lags + 1):
        df[f'lag_{lag}'] = df[target_col].shift(lag)

    # Kayar Pencere Özellikleri (Rolling Window Features)
    df[f'rolling_mean_{window_size}'] = df[target_col].shift(1).rolling(window=window_size).mean() # Dünkü değeri dahil etmeden
    df[f'rolling_std_{window_size}'] = df[target_col].shift(1).rolling(window=window_size).std()

    # Tarih Özellikleri
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['dayofyear'] = df.index.dayofyear
    df['weekofyear'] = df.index.isocalendar().week.astype(int) # weekofyear'ı integer yap

    df.dropna(inplace=True) # NaN içeren satırları kaldır (lag ve rolling window'dan dolayı oluşur)
    return df
# --- LightGBM için Özellik Mühendisliği Sonu ---

# --- FinBERT Duyarlılık Analizi Fonksiyonu ---
def get_sentiment_finbert(text):
    """Verilen metin için FinBERT kullanarak duyarlılık analizi yapar."""
    if not finbert_model or not finbert_tokenizer:
        app.logger.warning("FinBERT model or tokenizer not available. Skipping FinBERT sentiment analysis.")
        return "Nötr", 0.0 # Model yoksa varsayılan bir değer döndür

    try:
        inputs = finbert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad(): # Gradyan hesaplamalarını devre dışı bırak
            logits = finbert_model(**inputs).logits
        
        probabilities = F.softmax(logits, dim=1).squeeze() # squeeze() ile [[]] yerine [] elde et
        # Modelin etiketleri: ['positive', 'negative', 'neutral'] sırasına göre olabilir.
        # ProsusAI/finbert için bu sıra genellikle bu şekildedir, kontrol etmekte fayda var.
        # Veya model.config.id2label kullanarak dinamik olarak alabiliriz.
        id2label = {0: 'positive', 1: 'negative', 2: 'neutral'} # Varsayılan, ProsusAI/finbert için bu sıra olmayabilir!
        # Doğru etiket sırasını model config'den alalım:
        if hasattr(finbert_model, 'config') and hasattr(finbert_model.config, 'id2label'):
            id2label = finbert_model.config.id2label
        else: # Fallback, eğer config'den okuyamazsak
            app.logger.warning("Could not read id2label from FinBERT model config. Using default label order.")
            # Bu varsayılan sıra ProsusAI/finbert için doğru DEĞİL. Genellikle ['positive', 'negative', 'neutral'] şeklindedir.
            # ProsusAI/finbert için doğru sıra: Positive: 0, Negative: 1, Neutral: 2 (logitlere göre)
            # Ancak softmax sonrası en yüksek olasılıklı indeksi alıp bu id2label ile eşleştireceğiz.
            # FinBERT (ProsusAI) genellikle 3 etiket döndürür: positive, negative, neutral.
            # Bu etiketlerin modelin çıktısındaki sıralaması önemlidir.
            # Genellikle: index 0 -> positive, index 1 -> negative, index 2 -> neutral
            # Bu varsayımı kontrol etmemiz gerek. Model kartında bu bilgi olmalı.
            # HuggingFace model kartına göre `ProsusAI/finbert` için etiketler: LABEL_0: positive, LABEL_1: negative, LABEL_2: neutral
            pass # id2label varsayılanını kullan
            
        # En yüksek olasılıklı sınıfı bul
        predicted_class_id = torch.argmax(probabilities).item()
        sentiment_category = id2label.get(predicted_class_id, "Nötr") # Bilinmeyen ID için Nötr

        # Sayısal skor (average_sentiment için)
        if sentiment_category == 'positive': score = 0.75 # Daha güçlü pozitif etki
        elif sentiment_category == 'negative': score = -0.75 # Daha güçlü negatif etki
        else: score = 0.1 # Nötr için hafif pozitif bir bias (veya 0.0)
        
        # Debug için olasılıkları loglayabiliriz
        # app.logger.debug(f"FinBERT probs: Pos={probabilities[0]:.4f}, Neg={probabilities[1]:.4f}, Neu={probabilities[2]:.4f} -> {sentiment_category} ({score})")

        return sentiment_category, score

    except Exception as e:
        app.logger.error(f"Error during FinBERT sentiment analysis for text '{text[:100]}...': {e}", exc_info=True)
        return "Nötr", 0.0 # Hata durumunda varsayılan
# --- FinBERT Duyarlılık Analizi Fonksiyonu Sonu ---

# --- Kapsamlı Özet Oluşturma Fonksiyonu ---
def generate_comprehensive_summary(ticker, period_text, price_change, clean_close, end_date, 
                                   sma_comment, rsi_comment, macd_comment, bollinger_comment, 
                                   sentiment_summary, xgb_summary, indicators, disclaimer):
    
    # Başlangıç ve Genel Durum
    summary_parts = []
    summary_parts.append(f"**{ticker} Hisse Senedi Analizi ({end_date} itibarıyla)**")
    summary_parts.append(f"İncelenen {period_text} döneminde hisse senedi %{price_change:.2f} oranında bir değişim göstermiştir. Son kapanış fiyatı {format_indicator(clean_close)} seviyesindedir.")

    # Teknik Görünüm Bölümü
    summary_parts.append("\n**--- Teknik Göstergeler Detaylı Yorumu ---**")
    # Her bir yorumu ayrı satırlara ekleyelim ve başına işaret koyalım
    summary_parts.append(f"*   **Hareketli Ortalamalar (SMA):** {sma_comment}")
    summary_parts.append(f"*   **Göreceli Güç Endeksi (RSI):** {rsi_comment}")
    summary_parts.append(f"*   **MACD (Hareketli Ortalama Yakınsama/Iraksama):** {macd_comment}")
    summary_parts.append(f"*   **Bollinger Bantları:** {bollinger_comment}")

    # Haber Duyarlılığı
    summary_parts.append("\n**--- Piyasa Duyarlılığı (Haberler) ---**")
    summary_parts.append(sentiment_summary)

    # Tahmin Modeli
    summary_parts.append("\n**--- Geleceğe Yönelik Fiyat Tahmini (XGBoost Modeli) ---**")
    summary_parts.append(xgb_summary)
    
    # Temel Göstergelerden Bazıları
    summary_parts.append("\n**--- Bazı Önemli Temel Göstergeler ---**")
    fk_ratio_val = indicators.get('Fiyat/Kazanç (F/K)', 'Veri Yok')
    piyasa_degeri_val = indicators.get('Piyasa Değeri', 'Veri Yok')
    summary_parts.append(f"*   Fiyat/Kazanç Oranı (F/K): {fk_ratio_val} (Şirketin karına göre fiyatının ne kadar olduğunu gösterir. Düşük olması 'ucuz' olduğu anlamına gelebilir, ancak sektör ortalamasıyla karşılaştırılmalıdır.)")
    summary_parts.append(f"*   Piyasa Değeri: {piyasa_degeri_val} (Şirketin toplam değerini ifade eder.)")


    # --- "AKILLI" YORUM BÖLÜMÜ (GEMINI ETKİSİ) ---
    summary_parts.append("\n**--- Genel Değerlendirme ve Olası Senaryolar (Yapay Zeka Destekli Yorum) ---**")
    
    # Genel Trend ve Momentum
    trend_strength = "belirgin değil"
    if "üzerinde" in sma_comment and "pozitif momentum" in macd_comment.lower():
        trend_strength = "güçlü bir yükseliş"
    elif "altında" in sma_comment and "negatif momentum" in macd_comment.lower():
        trend_strength = "belirgin bir düşüş"
    elif "karışık" in sma_comment:
        trend_strength = "karışık veya yatay bir"
        
    summary_parts.append(f"**Genel Trend:** Mevcut teknik göstergeler, hisse senedinin {trend_strength} trend içerisinde olduğunu işaret ediyor. ")

    # Aşırı Alım/Satım ve Düzeltme Potansiyeli
    overbought_risk = False
    oversold_opportunity = False
    if "aşırı alım" in rsi_comment:
        summary_parts.append(f"**RSI Sinyali:** RSI göstergesi, hissenin 'aşırı alım' bölgesinde olduğunu belirtiyor. Bu, fiyatın kısa vadede biraz fazla yükselmiş olabileceğine ve bir düzeltme (geri çekilme) yaşanabileceğine işaret edebilir. ")
        if "Bollinger Üst Bandının üzerinde" in bollinger_comment or "Bollinger Üst Bandına çok yakın" in bollinger_comment:
            summary_parts.append("Bu durum, fiyatın Bollinger Üst Bandı'na yakın veya üzerinde olmasıyla da destekleniyor, ki bu da genellikle aşırı bir hareket olarak yorumlanır.")
            overbought_risk = True
        elif "Bollinger Bantları arasında" in bollinger_comment :
             summary_parts.append("Ancak fiyat hala Bollinger Bantları içerisinde hareket ediyor, bu da aşırı durumun henüz teyit edilmediğini gösterebilir.")
    
    if "aşırı satım" in rsi_comment:
        summary_parts.append(f"**RSI Sinyali:** RSI göstergesi, hissenin 'aşırı satım' bölgesinde olduğunu belirtiyor. Bu, fiyatın kısa vadede biraz fazla düşmüş olabileceğine ve bir tepki yükselişi yaşanabileceğine işaret edebilir. ")
        if "Bollinger Alt Bandının altında" in bollinger_comment or "Bollinger Alt Bandına çok yakın" in bollinger_comment:
            summary_parts.append("Bu durum, fiyatın Bollinger Alt Bandı'na yakın veya altında olmasıyla da destekleniyor, ki bu da genellikle aşırı bir hareket olarak yorumlanır.")
            oversold_opportunity = True
        elif "Bollinger Bantları arasında" in bollinger_comment :
             summary_parts.append("Ancak fiyat hala Bollinger Bantları içerisinde hareket ediyor, bu da aşırı durumun henüz teyit edilmediğini gösterebilir.")

    # MACD Sinyalleri ve Trend Teyidi
    if "Al sinyali" in macd_comment:
        summary_parts.append(f"**MACD Sinyali:** MACD göstergesi, kısa süre önce bir 'Al' sinyali üretmiş. Bu, genellikle yükseliş momentumunun arttığına veya başlayabileceğine dair bir işarettir.")
        if "Pozitif" in sentiment_summary:
            summary_parts.append("Haberlerdeki genel duyarlılığın da pozitif olması bu olası yükseliş sinyalini destekleyebilir.")
        if overbought_risk:
            summary_parts.append("Ancak, RSI'ın aşırı alım bölgesinde olması nedeniyle, bu 'Al' sinyaline rağmen temkinli olmak ve olası bir geri çekilmeyi göz önünde bulundurmak faydalı olabilir.")
    elif "Sat sinyali" in macd_comment:
        summary_parts.append(f"**MACD Sinyali:** MACD göstergesi, kısa süre önce bir 'Sat' sinyali üretmiş. Bu, genellikle düşüş momentumunun arttığına veya başlayabileceğine dair bir işarettir.")
        if "Negatif" in sentiment_summary:
            summary_parts.append("Haberlerdeki genel duyarlılığın da negatif olması bu olası düşüş sinyalini destekleyebilir.")
        if oversold_opportunity:
            summary_parts.append("Ancak, RSI'ın aşırı satım bölgesinde olması nedeniyle, bu 'Sat' sinyaline rağmen olası bir tepki alımı fırsatı sunabilir.")

    # Volatilite ve Beklentiler (Bollinger Bant Genişliği)
    if "daralıyor ve düşük seviyede" in bollinger_comment or "Squeeze" in bollinger_comment:
        summary_parts.append(f"**Volatilite (Oynaklık):** Bollinger Bantları'nın daralması (Squeeze durumu), piyasada bir sıkışma ve düşük bir oynaklık olduğunu gösteriyor. Bu tür durumlar genellikle, yakın gelecekte her iki yöne de (yukarı veya aşağı) sert bir fiyat hareketinin habercisi olabilir. Bu hareketin yönü genellikle önemli bir haber veya gelişmeyle tetiklenebilir.")
    elif "genişliyor, artan volatiliteye işaret ediyor" in bollinger_comment:
        summary_parts.append(f"**Volatilite (Oynaklık):** Bollinger Bantları'nın genişlemesi, piyasada oynaklığın arttığını gösteriyor. Bu, fiyatların daha büyük aralıklarda hareket edebileceği anlamına gelir ve genellikle belirsizliğin veya önemli piyasa hareketlerinin olduğu dönemlerde görülür.")

    # Haber ve Tahmin Entegrasyonu
    if "Pozitif" in sentiment_summary and "fiyatın ortalama" in xgb_summary and not overbought_risk:
        if clean_close < float(xgb_summary.split("ortalama ")[1].split(" ")[0].replace(',', '.')): # Basit bir sayısal karşılaştırma
             summary_parts.append(f"**Pozitif Sinyallerin Birleşimi:** Hem haber duyarlılığının pozitif olması hem de XGBoost modelinin gelecekte mevcut fiyattan daha yüksek bir seviye öngörmesi, hisse için olumlu bir tablo çiziyor. Teknik göstergeler de aşırı bir alım durumu göstermiyorsa, bu durum yükseliş potansiyelini destekleyebilir.")
    elif "Negatif" in sentiment_summary and "fiyatın ortalama" in xgb_summary and not oversold_opportunity:
        if clean_close > float(xgb_summary.split("ortalama ")[1].split(" ")[0].replace(',', '.')): # Basit bir sayısal karşılaştırma
            summary_parts.append(f"**Negatif Sinyallerin Birleşimi:** Hem haber duyarlılığının negatif olması hem de XGBoost modelinin gelecekte mevcut fiyattan daha düşük bir seviye öngörmesi, hisse için dikkatli olunması gereken bir tablo çiziyor. Teknik göstergeler de aşırı bir satım durumu göstermiyorsa, bu durum düşüş potansiyelini destekleyebilir.")

    # Sonuç ve Öneri (Genelleyici)
    summary_parts.append(f"\n**Özetle:** {ticker} hissesi için teknik göstergeler, haber akışı ve model tahminleri birlikte değerlendirildiğinde, yatırımcıların dikkat etmesi gereken çeşitli sinyaller bulunmaktadır. Özellikle RSI'ın mevcut durumu, MACD sinyalleri ve Bollinger Bantları'nın işaret ettiği volatilite seviyesi yakından takip edilmelidir.")
    if overbought_risk:
        summary_parts.append("Mevcut aşırı alım sinyalleri göz önüne alındığında, yeni pozisyon açmadan önce bir düzeltme beklenmesi veya risk yönetimine ekstra özen gösterilmesi düşünülebilir.")
    elif oversold_opportunity:
        summary_parts.append("Mevcut aşırı satım sinyalleri, kısa vadeli bir tepki alımı fırsatı sunabilir, ancak genel trend ve haber akışı teyit için izlenmelidir.")
    
    summary_parts.append(disclaimer)
    return "\n".join(summary_parts) # HTML'de düzgün görünmesi için \n kullandım, render_template bunu <br> yapabilir.
# --- Kapsamlı Özet Oluşturma Fonksiyonu Sonu ---

@app.route('/', methods=['GET'])
def index():
    # Başlangıçta sadece ana sayfayı göster, grafik yok
    # Hisse senedi listesini şablona gönder
    return render_template('index.html', chart_html=None, stock_list=COMPREHENSIVE_STOCK_LIST)

@app.route('/analyze', methods=['POST'])
def analyze():
    ticker = request.form.get('ticker')
    period = request.form.get('period', '1y')
    chart_type = request.form.get('chart_type', 'line') # Grafik türünü al, varsayılan 'line'

    news_list = [] # Başlangıçta boş liste
    average_sentiment = 0 # Başlangıçta sıfır
    xgb_forecast_df = None # XGBoost tahminleri için yeni DataFrame
    forecast_to_plot = None # Görselleştirilecek tahmini burada başlat
    future_periods = 30 # Sabit tahmin periyodu
    N_LAGS_LGBM = 7 # LightGBM için lag sayısı
    ROLLING_WINDOW_LGBM = 7 # LightGBM için kayar pencere boyutu

    if not ticker:
        return render_template('index.html', error="Lütfen bir hisse senedi kodu girin.")

    # Temel göstergeleri ve hisse adı gibi bilgileri al (Haber API sorgusu için gerekli)
    now = datetime.now()
    stock_info = None
    cache_key_info = f"{ticker}_info"

    if cache_key_info in _info_cache:
        cached_entry = _info_cache[cache_key_info]
        if (now - cached_entry['timestamp']).total_seconds() < CACHE_MAX_AGE_SECONDS:
            app.logger.info(f"{ticker} için bilgiler önbellekten alındı.")
            stock_info = cached_entry['data']
        else:
            app.logger.info(f"{ticker} için önbellek süresi doldu (info).")
            del _info_cache[cache_key_info] # Süresi dolan kaydı sil

    if stock_info is None:
        try:
            stock_info_data = yf.Ticker(ticker).info
            if stock_info_data and stock_info_data.get('regularMarketPrice') is not None: # Temel bir kontrol
                stock_info = stock_info_data
                _info_cache[cache_key_info] = {'data': stock_info, 'timestamp': now}
                app.logger.info(f"{ticker} için bilgiler yfinance'ten çekildi ve önbelleğe alındı.")
            else:
                # yfinance bazen boş veya eksik 'info' döndürebilir
                app.logger.warning(f"yf.Ticker({ticker}).info beklenen veriyi döndürmedi.")
                return render_template('index.html', error=f"'{ticker}' için temel bilgiler alınamadı veya eksik. Kod geçerli mi?")
        except Exception as e:
            app.logger.error(f"Hisse bilgisi alınırken hata (yf.Ticker({ticker}).info): {e}")
            return render_template('index.html', error=f"'{ticker}' için temel bilgiler alınamadı. Kod geçerli mi?")

    # --- Haber Analizi (API anahtarı varsa) ---
    if app.config.get('NEWS_API_KEY'): # app.config'den oku
        query = stock_info.get('longName', ticker)
        from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        news_params = {
            'q': query, 'apiKey': app.config['NEWS_API_KEY'], 'language': 'en',
            'pageSize': 10, 'sortBy': 'publishedAt', 'from': from_date
        }
        try:
            response = requests.get(NEWS_API_URL, params=news_params, timeout=10) # timeout eklendi
            response.raise_for_status()
            news_data = response.json()
            if news_data.get('status') == 'ok' and news_data.get('totalResults', 0) > 0:
                sentiment_scores = []
                for article in news_data.get('articles', []):
                    title = article.get('title')
                    description = article.get('description')
                    url = article.get('url')
                    
                    # Analiz için metin seçimi: Varsa description, yoksa title
                    text_to_analyze = description if description else title

                    if text_to_analyze and url: # Analiz edilecek metin ve URL varsa devam et
                        # FinBERT ile duyarlılık analizi
                        sentiment_category, compound_score = get_sentiment_finbert(text_to_analyze)
                        
                        sentiment_scores.append(compound_score) # FinBERT'ten gelen sayısal skoru kullan
                        news_list.append({
                            'title': title, # Başlığı hala saklayalım, göstermek için
                            'url': url,
                            'sentiment_score': compound_score,
                            'sentiment_category': sentiment_category,
                            'analyzed_text': text_to_analyze # Hangi metnin analiz edildiğini de ekleyebiliriz (opsiyonel)
                        })
                if sentiment_scores:
                    average_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        except requests.exceptions.RequestException as e:
            app.logger.error(f"NewsAPI isteği başarısız: {e}") # print yerine logger
        except Exception as e:
            app.logger.error(f"Haber işlenirken hata: {e}") # print yerine logger
    else:
        # Bu uyarı zaten başlangıçta verildi, burada tekrarlamaya gerek yok.
        # app.logger.info("\nUYARI: NEWS_API_KEY ortam değişkeni ayarlanmamış. Haber analizi atlanıyor.\n")
        pass
    # --- Haber Analizi Sonu ---

    # --- Ana Analiz Bloğu (Hisse verisi, Teknik analiz, Yorum vb.) ---
    try:
        # --- Hisse Senedi Verisi Çekme ---
        stock_data = None
        cache_key_data = f"{ticker}_{period}_data"
        if cache_key_data in _data_cache:
            cached_entry = _data_cache[cache_key_data]
            if (now - cached_entry['timestamp']).total_seconds() < CACHE_MAX_AGE_SECONDS:
                app.logger.info(f"{ticker} ({period}) için hisse verisi önbellekten alındı.")
                stock_data = cached_entry['data']
            else:
                app.logger.info(f"{ticker} ({period}) için önbellek süresi doldu (data).")
                del _data_cache[cache_key_data]

        if stock_data is None:
            stock_data_downloaded = yf.download(ticker, period=period)
            if not stock_data_downloaded.empty:
                 stock_data = stock_data_downloaded
                 _data_cache[cache_key_data] = {'data': stock_data, 'timestamp': now}
                 app.logger.info(f"{ticker} ({period}) için hisse verisi yfinance'ten çekildi ve önbelleğe alındı.")
            else:
                app.logger.warning(f"{ticker} için yf.download boş DataFrame döndürdü.")
        
        if stock_data is None or stock_data.empty or len(stock_data) < 50: # stock_data None olabilir
            return render_template('index.html', error=f"{ticker} için yeterli veri bulunamadı veya periyot çok kısa.")
        
        # --- Temel Göstergeler ve Veri Ön İşleme ---
        # 'Close' sütununu MultiIndex kullanarak seç ve temizle
        close_col_name = ('Close', ticker.upper()) # Doğru MultiIndex sütun adı
        if close_col_name not in stock_data.columns:
             # Bazen yfinance tek ticker için MultiIndex kullanmayabilir
             if 'Close' in stock_data.columns:
                 close_col_name = 'Close' 
             else:
                 return render_template('index.html', error=f"'{ticker}' için 'Close' sütunu bulunamadı.")

        close_series = pd.to_numeric(stock_data[close_col_name], errors='coerce')
        close_series = close_series.dropna() # Sayısal olmayan veya boş değerleri kaldır

        # Temizlenmiş verinin teknik analiz için yeterli olup olmadığını kontrol et
        if len(close_series) < (N_LAGS_LGBM + ROLLING_WINDOW_LGBM): # LGBM için de yeterli veri lazım
            return render_template('index.html', error=f"{ticker} için analiz yapılamadı (Ağaç tabanlı model için yetersiz temizlenmiş veri: {len(close_series)}). Periyodu uzatmayı deneyin.")

        # --- XGBoost Model Eğitme ve Tahmin (LightGBM yerine) ---
        # Önbellek anahtarını oluştur
        xgb_cache_key = f"{ticker}_{period}_{future_periods}_{N_LAGS_LGBM}_{ROLLING_WINDOW_LGBM}_xgb_forecast"
        now_for_cache = datetime.now() # Önbellek zaman damgası için

        if xgb_cache_key in _xgb_forecast_cache:
            cached_entry = _xgb_forecast_cache[xgb_cache_key]
            if (now_for_cache - cached_entry['timestamp']).total_seconds() < CACHE_MAX_AGE_SECONDS:
                app.logger.info(f"{xgb_cache_key} için XGBoost tahmini önbellekten alındı.")
                forecast_to_plot = cached_entry['data']
            else:
                app.logger.info(f"{xgb_cache_key} için XGBoost önbellek süresi doldu.")
                del _xgb_forecast_cache[xgb_cache_key]

        if forecast_to_plot is None: 
            app.logger.info(f"{xgb_cache_key} için XGBoost tahmini hesaplanıyor.")
            try:
                # Özellik Mühendisliği (create_features_for_lgbm fonksiyonunu kullanmaya devam ediyoruz)
                features_df = create_features_for_lgbm(close_series.rename('y'), target_col='y', n_lags=N_LAGS_LGBM, window_size=ROLLING_WINDOW_LGBM)

                if features_df.empty or len(features_df) < 20: 
                     raise ValueError("Özellik mühendisliği sonrası yeterli veri kalmadı.")

                X = features_df.drop('y', axis=1)
                y = features_df['y']

                # XGBoost Modelini Eğit
                xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42) # Temel parametreler
                xgb_model.fit(X, y)

                # Gelecek Tahminleri için Özyinelemeli Strateji (LightGBM ile aynı mantık)
                future_predictions = []
                current_features_df = features_df.copy() 
                last_known_date = current_features_df.index[-1]
                
                temp_series_for_rolling = close_series.rename('y').copy() 

                for i in range(future_periods):
                    last_features = current_features_df.drop('y', axis=1).iloc[[-1]].copy() 
                    next_date = last_known_date + timedelta(days=i + 1) 

                    if i > 0:
                        last_features[f'lag_1'] = future_predictions[-1]
                        for lag_num in range(N_LAGS_LGBM, 1, -1):
                            if f'lag_{lag_num-1}' in last_features.columns:
                                 last_features[f'lag_{lag_num}'] = last_features[f'lag_{lag_num-1}']
                            else: 
                                 last_features[f'lag_{lag_num}'] = pd.NA 
                        
                        if future_predictions: # Kayar pencere özelliklerini güncelle
                            combined_series = close_series.rename('y').copy()
                            predicted_dates = pd.date_range(start=last_known_date + timedelta(days=1), periods=len(future_predictions), freq=combined_series.index.freq or 'B')
                            predicted_series = pd.Series(future_predictions, index=predicted_dates)
                            series_for_rolling_calc = pd.concat([combined_series, predicted_series])
                            relevant_series_for_next_pred = series_for_rolling_calc[series_for_rolling_calc.index < next_date]

                            if len(relevant_series_for_next_pred) >= ROLLING_WINDOW_LGBM:
                                last_features[f'rolling_mean_{ROLLING_WINDOW_LGBM}'] = relevant_series_for_next_pred.shift(0).rolling(window=ROLLING_WINDOW_LGBM).mean().iloc[-1]
                                last_features[f'rolling_std_{ROLLING_WINDOW_LGBM}'] = relevant_series_for_next_pred.shift(0).rolling(window=ROLLING_WINDOW_LGBM).std().iloc[-1]
                            else:
                                last_features[f'rolling_mean_{ROLLING_WINDOW_LGBM}'] = pd.NA
                                last_features[f'rolling_std_{ROLLING_WINDOW_LGBM}'] = pd.NA
                    
                    last_features['dayofweek'] = next_date.dayofweek
                    last_features['month'] = next_date.month
                    last_features['year'] = next_date.year
                    last_features['dayofyear'] = next_date.dayofyear
                    last_features['weekofyear'] = next_date.isocalendar().week

                    next_pred = xgb_model.predict(last_features)[0]
                    future_predictions.append(float(next_pred)) # XGBoost bazen numpy float döndürebilir, Python float'a çevir

                future_dates = pd.date_range(start=last_known_date + timedelta(days=1), periods=future_periods, freq=close_series.index.freq or 'B')
                xgb_forecast_df_temp = pd.DataFrame({'ds': future_dates, 'yhat': future_predictions})
                forecast_to_plot = xgb_forecast_df_temp.set_index('ds')
                
                _xgb_forecast_cache[xgb_cache_key] = {'data': forecast_to_plot, 'timestamp': now_for_cache}
                app.logger.info(f"{xgb_cache_key} için XGBoost tahmini hesaplandı ve önbelleğe alındı.")

            except Exception as model_e: # Daha genel bir hata adı
                app.logger.error(f"XGBoost modeli hatası: {model_e}", exc_info=True)
                forecast_to_plot = None
        # --- XGBoost Model Sonu ---

        # Temizlenmiş Series üzerinden göstergeleri hesapla
        sma_20 = ta.trend.sma_indicator(close_series, window=20)
        sma_50 = ta.trend.sma_indicator(close_series, window=50)
        rsi = ta.momentum.rsi(close_series, window=14)
        macd_obj = ta.trend.MACD(close_series) # macd objesini alalım
        macd_val = macd_obj.macd()
        macd_signal = macd_obj.macd_signal()
        macd_hist = macd_obj.macd_diff()

        # Bollinger Bantları Hesaplama
        bb_window = 20
        bb_std_dev = 2
        bollinger = ta.volatility.BollingerBands(close_series, window=bb_window, window_dev=bb_std_dev)
        bb_high = bollinger.bollinger_hband() # Üst Bant
        bb_low = bollinger.bollinger_lband()  # Alt Bant
        bb_mavg = bollinger.bollinger_mavg() # Orta Bant (SMA20 ile aynı olmalı)
        bb_pband = bollinger.bollinger_pband() # %B (Fiyatın bantlara göre konumu)
        bb_wband = bollinger.bollinger_wband() # Bant Genişliği

        # Hesaplanan göstergeleri orijinal DataFrame'e geri ata (indeks ile eşleşecek)
        stock_data['SMA_20'] = sma_20
        stock_data['SMA_50'] = sma_50
        stock_data['RSI'] = rsi
        stock_data['MACD'] = macd_val
        stock_data['MACD_Signal'] = macd_signal
        stock_data['MACD_Hist'] = macd_hist
        stock_data['BB_High'] = bb_high
        stock_data['BB_Low'] = bb_low
        stock_data['BB_MAVG'] = bb_mavg # Orta bandı da ekleyelim, teyit için veya SMA20 yerine kullanılabilir
        stock_data['BB_PBAND'] = bb_pband # %B değerini ekle
        stock_data['BB_WBAND'] = bb_wband # Bant genişliğini ekle
        # --- Teknik Analiz Sonu ---
        
        # --- Veri Ön İşleme (Devamı - Kapanış fiyatı için) ---
        clean_pe = get_numeric_value(stock_info, 'forwardPE')
        clean_market_cap = get_numeric_value(stock_info, 'marketCap')
        clean_volume = get_numeric_value(stock_info, 'volume')
        clean_high = get_numeric_value(stock_info, 'fiftyTwoWeekHigh')
        clean_low = get_numeric_value(stock_info, 'fiftyTwoWeekLow')
        # Temizlenmiş serinin son değerini al
        clean_close = close_series.iloc[-1] if not close_series.empty else None 
        # --- Veri Ön İşleme Sonu ---

        # Analist Tavsiyesini al
        recommendation = stock_info.get('recommendationKey', 'Veri yok').replace('_', ' ').title()

        # Göstergeleri oluştur (Temizlenmiş değerlerle)
        indicators = {
            'Fiyat/Kazanç (F/K)': format_indicator(clean_pe),
            'Piyasa Değeri': format_market_cap(clean_market_cap),
            'Günlük İşlem Hacmi': format_volume(clean_volume),
            'Son Kapanış Fiyatı': format_indicator(clean_close),
            '52 Hafta En Yüksek': format_indicator(clean_high),
            '52 Hafta En Düşük': format_indicator(clean_low),
            'Analist Tavsiyesi': recommendation # Analist tavsiyesini ekle
        }
        # Teknik göstergeleri ayrı bir sözlüğe alalım
        tech_indicators = {
            'RSI (14)': format_indicator(stock_data['RSI'].iloc[-1] if not stock_data.empty and 'RSI' in stock_data.columns and not pd.isna(stock_data['RSI'].iloc[-1]) else None),
            'MACD': format_indicator(stock_data['MACD'].iloc[-1] if not stock_data.empty and 'MACD' in stock_data.columns and not pd.isna(stock_data['MACD'].iloc[-1]) else None),
            'MACD Sinyal': format_indicator(stock_data['MACD_Signal'].iloc[-1] if not stock_data.empty and 'MACD_Signal' in stock_data.columns and not pd.isna(stock_data['MACD_Signal'].iloc[-1]) else None),
            'BB Üst': format_indicator(stock_data['BB_High'].iloc[-1] if not stock_data.empty and 'BB_High' in stock_data.columns and not pd.isna(stock_data['BB_High'].iloc[-1]) else None),
            'BB Alt': format_indicator(stock_data['BB_Low'].iloc[-1] if not stock_data.empty and 'BB_Low' in stock_data.columns and not pd.isna(stock_data['BB_Low'].iloc[-1]) else None),
            'BB %B': format_indicator(stock_data['BB_PBAND'].iloc[-1] if not stock_data.empty and 'BB_PBAND' in stock_data.columns and not pd.isna(stock_data['BB_PBAND'].iloc[-1]) else None),
            'BB Genişlik': format_indicator(stock_data['BB_WBAND'].iloc[-1] if not stock_data.empty and 'BB_WBAND' in stock_data.columns and not pd.isna(stock_data['BB_WBAND'].iloc[-1]) else None),
        }

        # Özet bilgi oluştur (Temizlenmiş kapanış fiyatını ve teknik yorumu kullan)
        summary_close = clean_close if clean_close is not None else 0
        # Başlangıç fiyatını temizlenmiş seriden al
        summary_start_close = close_series.iloc[0] if not close_series.empty else 0 
        price_change = ((summary_close - summary_start_close) / summary_start_close) * 100 if summary_start_close != 0 else 0

        start_date = close_series.index[0].strftime('%Y-%m-%d') # Temizlenmiş seriden al
        end_date = close_series.index[-1].strftime('%Y-%m-%d') # Temizlenmiş seriden al
        period_text = f"{start_date} - {end_date}" # Periyot metnini başlangıç/bitiş olarak değiştir

        # --- Geliştirilmiş Teknik Yorum --- 
        technical_comment = ""
        last_close = clean_close
        last_sma20 = stock_data['SMA_20'].iloc[-1] if not stock_data.empty and 'SMA_20' in stock_data.columns and not pd.isna(stock_data['SMA_20'].iloc[-1]) else None
        last_sma50 = stock_data['SMA_50'].iloc[-1] if not stock_data.empty and 'SMA_50' in stock_data.columns and not pd.isna(stock_data['SMA_50'].iloc[-1]) else None
        last_rsi = stock_data['RSI'].iloc[-1] if not stock_data.empty and 'RSI' in stock_data.columns and not pd.isna(stock_data['RSI'].iloc[-1]) else None
        last_macd = stock_data['MACD'].iloc[-1] if not stock_data.empty and 'MACD' in stock_data.columns and not pd.isna(stock_data['MACD'].iloc[-1]) else None
        last_macd_signal = stock_data['MACD_Signal'].iloc[-1] if not stock_data.empty and 'MACD_Signal' in stock_data.columns and not pd.isna(stock_data['MACD_Signal'].iloc[-1]) else None
        prev_macd = stock_data['MACD'].iloc[-2] if len(stock_data) > 1 and 'MACD' in stock_data.columns and not pd.isna(stock_data['MACD'].iloc[-2]) else None
        prev_macd_signal = stock_data['MACD_Signal'].iloc[-2] if len(stock_data) > 1 and 'MACD_Signal' in stock_data.columns and not pd.isna(stock_data['MACD_Signal'].iloc[-2]) else None

        # Bollinger Bantları için son ve bir önceki değerler
        last_bb_high = stock_data['BB_High'].iloc[-1] if not stock_data.empty and 'BB_High' in stock_data.columns and not pd.isna(stock_data['BB_High'].iloc[-1]) else None
        last_bb_low = stock_data['BB_Low'].iloc[-1] if not stock_data.empty and 'BB_Low' in stock_data.columns and not pd.isna(stock_data['BB_Low'].iloc[-1]) else None
        last_bb_mavg = stock_data['BB_MAVG'].iloc[-1] if not stock_data.empty and 'BB_MAVG' in stock_data.columns and not pd.isna(stock_data['BB_MAVG'].iloc[-1]) else None
        last_bb_pband = stock_data['BB_PBAND'].iloc[-1] if not stock_data.empty and 'BB_PBAND' in stock_data.columns and not pd.isna(stock_data['BB_PBAND'].iloc[-1]) else None
        last_bb_wband = stock_data['BB_WBAND'].iloc[-1] if not stock_data.empty and 'BB_WBAND' in stock_data.columns and not pd.isna(stock_data['BB_WBAND'].iloc[-1]) else None
        prev_bb_wband = stock_data['BB_WBAND'].iloc[-2] if len(stock_data) > 1 and 'BB_WBAND' in stock_data.columns and not pd.isna(stock_data['BB_WBAND'].iloc[-2]) else None

        # SMA Yorumu (Yeni fonksiyonu kullanarak)
        sma_comment = get_sma_comment(last_close, last_sma20, last_sma50)

        # RSI Yorumu (Yeni fonksiyonu kullanarak)
        rsi_comment = get_rsi_comment(last_rsi)

        # MACD Yorumu (Yeni fonksiyonu kullanarak)
        macd_comment = get_macd_comment(last_macd, last_macd_signal, prev_macd, prev_macd_signal)

        # Bollinger Bantları Yorumu (Yeni fonksiyonu kullanarak)
        bollinger_comment = get_bollinger_comment(last_close, last_bb_high, last_bb_low, last_bb_mavg, last_bb_pband, last_bb_wband, prev_bb_wband)

        technical_comment = f"{sma_comment} {rsi_comment} {macd_comment} {bollinger_comment}" # Ichimoku yorumu çıkarıldı
        # --- Teknik Yorum Sonu --- 

        # --- Yorumlama ve Özet ---
        # Haber duyarlılığını yoruma ekle
        sentiment_summary = ""
        if news_list:
            if average_sentiment >= 0.05: sentiment_summary = f"Son haberlerde genel duyarlılık pozitif ({format_indicator(average_sentiment)})."
            elif average_sentiment <= -0.05: sentiment_summary = f"Son haberlerde genel duyarlılık negatif ({format_indicator(average_sentiment)})."
            else: sentiment_summary = f"Son haberlerde genel duyarlılık nötr ({format_indicator(average_sentiment)})."
        elif not app.config.get('NEWS_API_KEY'):
             sentiment_summary = "Haber analizi yapılamadı (API anahtarı eksik)."
        else: # API anahtarı var ama haber bulunamadı
             sentiment_summary = "İlgili güncel haber bulunamadı."
        
        # XGBoost tahminini yoruma ekle (LightGBM yerine)
        xgb_summary = ""
        if forecast_to_plot is not None and not forecast_to_plot.empty:
            last_forecast_date = forecast_to_plot.index[-1].strftime('%Y-%m-%d')
            last_yhat = forecast_to_plot['yhat'].iloc[-1]
            xgb_summary = f"XGBoost modeli, önümüzdeki {future_periods} gün için ({last_forecast_date}'e kadar) fiyatın ortalama {format_indicator(last_yhat)} civarında olabileceğini öngörüyor."
        else:
            xgb_summary = "Zaman serisi tahmini (XGBoost) yapılamadı."

        # Disclaimer
        disclaimer = "\n\nUYARI: Bu analizler ve yorumlar yalnızca bilgilendirme amaçlıdır ve yatırım tavsiyesi niteliği taşımaz. Finansal kararlarınızı vermeden önce kendi araştırmanızı yapmanız ve/veya profesyonel bir danışmana başvurmanız önerilir.\n"

        summary = generate_comprehensive_summary(
            ticker=ticker, 
            period_text=period_text, 
            price_change=price_change, 
            clean_close=clean_close, 
            end_date=end_date,
            sma_comment=sma_comment, 
            rsi_comment=rsi_comment, 
            macd_comment=macd_comment, 
            bollinger_comment=bollinger_comment,
            sentiment_summary=sentiment_summary, 
            xgb_summary=xgb_summary,
            indicators=indicators,
            disclaimer=disclaimer
        )

        # --- Plotly Grafik Oluşturma (Subplots ile) ---
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.03, 
                            row_heights=[0.6, 0.2, 0.2],
                            subplot_titles=(f'{ticker} Fiyat Grafiği ({chart_type.capitalize()})', 'RSI', 'MACD'))

        # Ana Fiyat Grafiği (Row 1)
        if chart_type == 'candlestick':
            fig.add_trace(go.Candlestick(x=stock_data.index,
                                           open=stock_data['Open'],
                                           high=stock_data['High'],
                                           low=stock_data['Low'],
                                           close=stock_data[close_col_name], # Temizlenmiş kapanış sütun adı
                                           name='Mum Grafiği',
                                           increasing_line_color= '#27AE60', decreasing_line_color= '#E74C3C'), 
                          row=1, col=1)
        else: # Varsayılan olarak çizgi grafik
            fig.add_trace(go.Scatter(x=close_series.index, y=close_series, mode='lines', name='Kapanış', line=dict(color='#5DADE2', width=2)), row=1, col=1)

        # Hareketli Ortalamalar (Her iki grafik türü için de eklenebilir)
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['SMA_20'], mode='lines', name='SMA 20', line=dict(color='#F39C12', width=1.2, dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['SMA_50'], mode='lines', name='SMA 50', line=dict(color='#A569BD', width=1.2, dash='dash')), row=1, col=1)

        # Bollinger Bantlarını Grafiğe Ekle
        if 'BB_High' in stock_data.columns and 'BB_Low' in stock_data.columns:
            fig.add_trace(go.Scatter(
                x=stock_data.index.tolist() + stock_data.index.tolist()[::-1], 
                y=stock_data['BB_High'].tolist() + stock_data['BB_Low'].tolist()[::-1], 
                fill='toself',
                fillcolor='rgba(133, 193, 233, 0.2)', # Daha yumuşak bir Bollinger dolgu rengi
                line=dict(color='rgba(255,255,255,0)'), 
                hoverinfo="skip", 
                showlegend=False,
                name='Bollinger Band Aralığı'
            ), row=1, col=1)
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['BB_High'], mode='lines', name='BB Üst Bant', line=dict(color='rgba(133, 193, 233, 0.5)', width=1, dash='dash')), row=1, col=1)
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['BB_Low'], mode='lines', name='BB Alt Bant', line=dict(color='rgba(133, 193, 233, 0.5)', width=1, dash='dash')), row=1, col=1)

        # XGBoost tahminini ekle (eğer varsa)
        if forecast_to_plot is not None and not forecast_to_plot.empty:
             fig.add_trace(go.Scatter(x=forecast_to_plot.index, y=forecast_to_plot['yhat'], mode='lines', name='Tahmin (XGBoost)', line=dict(color='#E74C3C', width=1.5, dash='dashdot')), row=1, col=1)
        
        # RSI Grafiği (Row 2)
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['RSI'], mode='lines', name='RSI', line=dict(color='#A569BD', width=1.5)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", line_width=1, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", line_width=1, row=2, col=1)
        fig.add_hrect(y0=70, y1=100, line_width=0, fillcolor='rgba(255,0,0,0.1)', opacity=0.2, row=2, col=1) # Aşırı alım bölgesi
        fig.add_hrect(y0=0, y1=30, line_width=0, fillcolor='rgba(0,255,0,0.1)', opacity=0.2, row=2, col=1) # Aşırı satım bölgesi

        # MACD Grafiği (Row 3)
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MACD'], mode='lines', name='MACD', line=dict(color='#2ECC71', width=1.5)), row=3, col=1)
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MACD_Signal'], mode='lines', name='MACD Sinyal', line=dict(color='#E74C3C', width=1.5)), row=3, col=1)
        # MACD Histogramı için bar chart
        colors = ['#27AE60' if val >= 0 else '#E74C3C' for val in stock_data['MACD_Hist']]
        fig.add_trace(go.Bar(x=stock_data.index, y=stock_data['MACD_Hist'], name='MACD Histogram', marker_color=colors), row=3, col=1)

        # Grafik Layout Ayarları (Daha Modern Görünüm)
        fig.update_layout(
            title_text=f'{ticker} Detaylı Analiz Grafiği',
            height=800, 
            showlegend=True,
            legend=dict(
                orientation="v", # Dikey efsane
                yanchor="top",
                y=0.98, # Grafiğin üst kısmına yakın
                xanchor="left", # Efsaneyi x konumuna göre sola yasla
                x=1.01, # Grafiğin hemen sağına (grafik genişliğinin %101'i)
                bgcolor="rgba(255,255,255,0.9)", # Hafif saydam beyaz arka plan
                bordercolor="#E5E5E5",
                borderwidth=1,
                traceorder="normal" # Ekleme sırasına göre göster
            ),
            plot_bgcolor='#FFFFFF', 
            paper_bgcolor='#FFFFFF', 
            font=dict(family="Arial, sans-serif", size=11, color="#333333"),
            xaxis=dict(showgrid=True, gridcolor='#E5E5E5', rangeslider=dict(visible=False)), # X ekseni grid çizgileri ve rangeslider kapatıldı
            yaxis=dict(showgrid=True, gridcolor='#E5E5E5'),
            yaxis2=dict(showgrid=True, gridcolor='#E5E5E5'), # RSI için
            yaxis3=dict(showgrid=True, gridcolor='#E5E5E5'), # MACD için
            # Genel grafik kenar boşlukları
            margin=dict(l=50, r=50, t=100, b=50),
            # Mum grafiği için rangeslider'ı özellikle kapatalım (subplotlarda iyi çalışmayabilir)
            xaxis_rangeslider_visible=False
        )

        # X eksenindeki tarih formatını ve hover formatını iyileştirme (isteğe bağlı)
        fig.update_xaxes(tickformat='%Y-%m-%d', hoverformat='%Y-%m-%d (%a)')

        # Subplot başlıklarını güncelle
        fig.layout.annotations[0].update(text=f'{ticker} Fiyat Grafiği ({chart_type.capitalize()})')
        fig.layout.annotations[1].update(text='Göreceli Güç Endeksi (RSI)')
        fig.layout.annotations[2].update(text='MACD & Histogram')

        # Prophet tahminini yoruma ekle
        prophet_summary = ""
        if forecast_to_plot is not None:
            last_forecast_date = forecast_to_plot.index[-1].strftime('%Y-%m-%d')
            last_yhat = forecast_to_plot['yhat'].iloc[-1]
            prophet_summary = f"Prophet modeli, önümüzdeki {future_periods} gün için ({last_forecast_date}'e kadar) fiyatın {format_indicator(last_yhat)} civarında olabileceğini öngörüyor."
        else:
            prophet_summary = "Zaman serisi tahmini yapılamadı."

        # Grafiği, göstergeleri ve özeti içeren sayfayı göster
        # Hisse senedi listesini analiz sonuçları sayfası için de gönderelim (opsiyonel, ama tutarlılık için iyi olabilir)
        return render_template('index.html', chart_html=pio.to_html(fig, full_html=False, include_plotlyjs='cdn'), 
                               indicators=indicators, tech_indicators=tech_indicators, summary=summary, 
                               news_list=news_list, average_sentiment=average_sentiment, 
                               stock_list=COMPREHENSIVE_STOCK_LIST) # Analiz sonrası da listeyi gönder

    except Exception as e:
        app.logger.error(f"Ana analiz bloğunda hata: {e}", exc_info=True) # print yerine logger, exc_info traceback ekler
        error_message = f"Analiz sırasında bir hata oluştu. Lütfen tekrar deneyin."
        # Hata durumunda da listeyi gönderelim ki sayfa yapısı bozulmasın
        return render_template('index.html', error=error_message, stock_list=COMPREHENSIVE_STOCK_LIST)

if __name__ == '__main__':
    # Geliştirme sunucusu için debug logging seviyesini ayarlayalım
    # Production'da bu seviye INFO veya WARNING olabilir.
    if app.debug:
        app.logger.setLevel(logging.DEBUG) # logging modülünü import etmeyi unutmayalım
    else:
        app.logger.setLevel(logging.INFO)
    
    app.run(debug=True) # debug=True geliştirme sırasında hataları görmemizi sağlar 