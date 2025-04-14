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
from prophet import Prophet # Prophet modelini import et
from dotenv import load_dotenv # .env dosyasını yüklemek için

load_dotenv() # .env dosyasındaki değişkenleri ortam değişkeni olarak yükle

app = Flask(__name__)

# --- Haber API Ayarları ---
NEWS_API_URL = 'https://newsapi.org/v2/everything'
# --- Haber API Ayarları Sonu ---

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

@app.route('/', methods=['GET'])
def index():
    # Başlangıçta sadece ana sayfayı göster, grafik yok
    return render_template('index.html', chart_html=None)

@app.route('/analyze', methods=['POST'])
def analyze():
    ticker = request.form.get('ticker')
    period = request.form.get('period', '1y')
    news_list = [] # Başlangıçta boş liste
    average_sentiment = 0 # Başlangıçta sıfır
    prophet_forecast = None # Prophet tahminini burada başlat
    forecast_to_plot = None # Görselleştirilecek tahmini burada başlat
    future_periods = 30 # Sabit tahmin periyodu

    if not ticker:
        return render_template('index.html', error="Lütfen bir hisse senedi kodu girin.")

    # Ortam değişkeninden API anahtarını al
    NEWS_API_KEY = os.environ.get('NEWS_API_KEY')

    # Temel göstergeleri ve hisse adı gibi bilgileri al (Haber API sorgusu için gerekli)
    try:
        stock_info = yf.Ticker(ticker).info
    except Exception as e:
        print(f"Hisse bilgisi alınırken hata (yf.Ticker({ticker}).info): {e}")
        return render_template('index.html', error=f"'{ticker}' için temel bilgiler alınamadı. Kod geçerli mi?")

    # --- Haber Analizi (API anahtarı varsa) ---
    if NEWS_API_KEY:
        sentiment_analyzer = SentimentIntensityAnalyzer()
        query = stock_info.get('longName', ticker)
        from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        news_params = {
            'q': query, 'apiKey': NEWS_API_KEY, 'language': 'en',
            'pageSize': 10, 'sortBy': 'publishedAt', 'from': from_date
        }
        try:
            response = requests.get(NEWS_API_URL, params=news_params)
            response.raise_for_status()
            news_data = response.json()
            if news_data.get('status') == 'ok' and news_data.get('totalResults', 0) > 0:
                sentiment_scores = []
                for article in news_data.get('articles', []):
                    title = article.get('title')
                    url = article.get('url')
                    if title and url:
                        vs = sentiment_analyzer.polarity_scores(title)
                        compound_score = vs['compound']
                        sentiment_scores.append(compound_score)
                        if compound_score >= 0.05: sentiment_category = 'Pozitif'
                        elif compound_score <= -0.05: sentiment_category = 'Negatif'
                        else: sentiment_category = 'Nötr'
                        news_list.append({
                            'title': title, 'url': url,
                            'sentiment_score': compound_score,
                            'sentiment_category': sentiment_category
                        })
                if sentiment_scores:
                    average_sentiment = sum(sentiment_scores) / len(sentiment_scores)
        except requests.exceptions.RequestException as e:
            print(f"NewsAPI isteği başarısız: {e}")
        except Exception as e:
            print(f"Haber işlenirken hata: {e}")
    else:
        print("\nUYARI: NEWS_API_KEY ortam değişkeni ayarlanmamış. Haber analizi atlanıyor.\n")
    # --- Haber Analizi Sonu ---

    # --- Ana Analiz Bloğu (Hisse verisi, Teknik analiz, Yorum vb.) ---
    try:
        # --- Hisse Senedi Verisi Çekme ---
        stock_data = yf.download(ticker, period=period)
        if stock_data.empty or len(stock_data) < 50:
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
        if len(close_series) < 50: # En uzun periyot (SMA 50) için en az 50 veri noktası gerekli
            return render_template('index.html', error=f"{ticker} için teknik analiz yapılamadı (yetersiz temizlenmiş veri). Periyodu uzatmayı deneyin.")

        # --- Prophet Model Eğitme ve Tahmin --- 
        try:
            # Prophet için DataFrame'i doğrudan oluştur
            prophet_df = pd.DataFrame({
                'ds': close_series.index, 
                'y': close_series.values
            })
            
            # # --- Debug: prophet_df kontrolü (Oluşturma sonrası) --- # Kaldırıldı
            # print("--- Prophet DF after direct creation ---")
            # print(f"Columns: {prophet_df.columns}")
            # print(prophet_df.head())
            # print("-------------------------------------")
            # # --- End Debug ---

            model = Prophet(interval_width=0.95)
            model.fit(prophet_df) # Şimdi ds ve y sütunları olmalı
            future = model.make_future_dataframe(periods=future_periods)
            prophet_forecast = model.predict(future)
            forecast_to_plot = prophet_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(future_periods)
            forecast_to_plot = forecast_to_plot.set_index('ds') # İndeksi ds yap

        except Exception as pe:
            print(f"Prophet modeli hatası: {pe}")
            prophet_forecast = None 
            forecast_to_plot = None
        # --- Prophet Model Sonu ---

        # Temizlenmiş Series üzerinden göstergeleri hesapla
        sma_20 = ta.trend.sma_indicator(close_series, window=20)
        sma_50 = ta.trend.sma_indicator(close_series, window=50)
        rsi = ta.momentum.rsi(close_series, window=14)
        macd = ta.trend.MACD(close_series)
        macd_val = macd.macd()
        macd_signal = macd.macd_signal()
        macd_hist = macd.macd_diff()

        # Hesaplanan göstergeleri orijinal DataFrame'e geri ata (indeks ile eşleşecek)
        # Atama için basit sütun isimleri kullanabiliriz, sorun olmaz.
        stock_data['SMA_20'] = sma_20
        stock_data['SMA_50'] = sma_50
        stock_data['RSI'] = rsi
        stock_data['MACD'] = macd_val
        stock_data['MACD_Signal'] = macd_signal
        stock_data['MACD_Hist'] = macd_hist
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
            'MACD Sinyal': format_indicator(stock_data['MACD_Signal'].iloc[-1] if not stock_data.empty and 'MACD_Signal' in stock_data.columns and not pd.isna(stock_data['MACD_Signal'].iloc[-1]) else None)
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
        last_sma20 = stock_data['SMA_20'].iloc[-1] if not stock_data.empty and 'SMA_20' in stock_data.columns else None
        last_sma50 = stock_data['SMA_50'].iloc[-1] if not stock_data.empty and 'SMA_50' in stock_data.columns else None
        last_rsi = stock_data['RSI'].iloc[-1] if not stock_data.empty and 'RSI' in stock_data.columns else None
        last_macd = stock_data['MACD'].iloc[-1] if not stock_data.empty and 'MACD' in stock_data.columns else None
        last_macd_signal = stock_data['MACD_Signal'].iloc[-1] if not stock_data.empty and 'MACD_Signal' in stock_data.columns else None
        prev_macd = stock_data['MACD'].iloc[-2] if len(stock_data) > 1 and 'MACD' in stock_data.columns else None
        prev_macd_signal = stock_data['MACD_Signal'].iloc[-2] if len(stock_data) > 1 and 'MACD_Signal' in stock_data.columns else None

        # SMA Yorumu
        sma_comment = ""
        if last_close is not None and last_sma20 is not None and last_sma50 is not None:
            if last_close > last_sma20 and last_close > last_sma50:
                sma_comment = "Fiyat, kısa ve orta vadeli hareketli ortalamaların üzerinde."
            elif last_close < last_sma20 and last_close < last_sma50:
                sma_comment = "Fiyat, kısa ve orta vadeli hareketli ortalamaların altında."
            else:
                sma_comment = "Fiyat, hareketli ortalamalara göre karışık bir görünümde."
        else:
            sma_comment = "Hareketli ortalamalar için yeterli veri yok."

        # RSI Yorumu
        rsi_comment = ""
        if last_rsi is not None:
            if last_rsi > 70:
                rsi_comment = f"RSI ({format_indicator(last_rsi)}) aşırı alım bölgesinde, olası bir düzeltmeye işaret edebilir."
            elif last_rsi < 30:
                rsi_comment = f"RSI ({format_indicator(last_rsi)}) aşırı satım bölgesinde, olası bir tepki yükselişine işaret edebilir."
            else:
                rsi_comment = f"RSI ({format_indicator(last_rsi)}) nötr bölgede."
        else:
            rsi_comment = "RSI hesaplanamadı."

        # MACD Yorumu
        macd_comment = ""
        if last_macd is not None and last_macd_signal is not None and prev_macd is not None and prev_macd_signal is not None:
            if last_macd > last_macd_signal and prev_macd <= prev_macd_signal:
                macd_comment = "MACD çizgisi sinyal çizgisini yukarı kesti (Al sinyali)."
            elif last_macd < last_macd_signal and prev_macd >= prev_macd_signal:
                macd_comment = "MACD çizgisi sinyal çizgisini aşağı kesti (Sat sinyali)."
            elif last_macd > last_macd_signal:
                 macd_comment = "MACD çizgisi sinyal çizgisinin üzerinde (Pozitif momentum)."
            else:
                 macd_comment = "MACD çizgisi sinyal çizgisinin altında (Negatif momentum)."
        else:
            macd_comment = "MACD yorumu için yeterli veri yok."

        technical_comment = f"{sma_comment} {rsi_comment} {macd_comment}"
        # --- Teknik Yorum Sonu --- 

        # --- Yorumlama ve Özet ---
        # Haber duyarlılığını yoruma ekle
        sentiment_summary = ""
        if news_list:
            if average_sentiment >= 0.05: sentiment_summary = f"Son haberlerde genel duyarlılık pozitif ({format_indicator(average_sentiment)})."
            elif average_sentiment <= -0.05: sentiment_summary = f"Son haberlerde genel duyarlılık negatif ({format_indicator(average_sentiment)})."
            else: sentiment_summary = f"Son haberlerde genel duyarlılık nötr ({format_indicator(average_sentiment)})."
        elif not NEWS_API_KEY:
             sentiment_summary = "Haber analizi yapılamadı (API anahtarı eksik)."
        else: # API anahtarı var ama haber bulunamadı
             sentiment_summary = "İlgili güncel haber bulunamadı."
        
        # Prophet tahminini yoruma ekle
        prophet_summary = ""
        if prophet_forecast is not None:
            last_forecast_date = prophet_forecast['ds'].iloc[-1].strftime('%Y-%m-%d')
            last_yhat = prophet_forecast['yhat'].iloc[-1]
            last_yhat_lower = prophet_forecast['yhat_lower'].iloc[-1]
            last_yhat_upper = prophet_forecast['yhat_upper'].iloc[-1]
            prophet_summary = f"Prophet modeli, önümüzdeki {future_periods} gün için ({last_forecast_date}'e kadar) fiyatın {format_indicator(last_yhat)} civarında olabileceğini öngörüyor (Tahmin aralığı: {format_indicator(last_yhat_lower)} - {format_indicator(last_yhat_upper)})." 
        else:
            prophet_summary = "Zaman serisi tahmini yapılamadı."

        # Disclaimer
        disclaimer = "\n\nUYARI: Bu analizler ve yorumlar yalnızca bilgilendirme amaçlıdır ve yatırım tavsiyesi niteliği taşımaz. Finansal kararlarınızı vermeden önce kendi araştırmanızı yapmanız ve/veya profesyonel bir danışmana başvurmanız önerilir.\n"

        summary = f"""
        {ticker} hisse senedi {period_text} döneminde %{price_change:.2f} değişim göstermiştir.
        Son kapanış fiyatı {format_indicator(clean_close)} ({end_date} itibarıyla) olarak gerçekleşmiştir.\n
        Teknik Görünüm: {technical_comment}\n
        Haber Duyarlılığı: {sentiment_summary}\n
        Tahmin Modeli: {prophet_summary}
        {disclaimer}
        """

        # --- Plotly Grafik Oluşturma (Subplots ile) ---
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.03, # Dikey boşluğu azalt
                            row_heights=[0.6, 0.2, 0.2], # Ana grafik daha büyük
                            subplot_titles=('Fiyat, Ortalamalar ve Tahmin', 'RSI', 'MACD')) # Alt grafik başlıkları

        # Ana Fiyat Grafiği (Row 1)
        fig.add_trace(go.Scatter(x=close_series.index, y=close_series, mode='lines', name='Kapanış', line=dict(color='#1f77b4', width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['SMA_20'], mode='lines', name='SMA 20', line=dict(color='#ff7f0e', width=1, dash='dot')), row=1, col=1)
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['SMA_50'], mode='lines', name='SMA 50', line=dict(color='#2ca02c', width=1, dash='dash')), row=1, col=1)

        # Prophet tahminini ve güven aralığını ekle (eğer varsa)
        if forecast_to_plot is not None:
             # Güven aralığı (shaded area)
             fig.add_trace(go.Scatter(
                 x=forecast_to_plot.index.tolist() + forecast_to_plot.index.tolist()[::-1], # x değerleri ileri ve geri
                 y=forecast_to_plot['yhat_upper'].tolist() + forecast_to_plot['yhat_lower'].tolist()[::-1], # y değerleri üst ve alt sınır
                 fill='toself', 
                 fillcolor='rgba(255, 193, 7, 0.2)', # Sarımsı şeffaf dolgu
                 line=dict(color='rgba(255,255,255,0)'), # Kenar çizgisi olmasın
                 hoverinfo="skip", # Hover'da görünmesin
                 showlegend=False,
                 name='Tahmin Aralığı'
             ), row=1, col=1)
            # Tahmin çizgisi (yhat)
             fig.add_trace(go.Scatter(x=forecast_to_plot.index, y=forecast_to_plot['yhat'], mode='lines', name='Tahmin (Prophet)', line=dict(color='#ffc107', dash='dashdot')), row=1, col=1)

        # RSI Grafiği (Row 2)
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['RSI'], mode='lines', name='RSI', line=dict(color='#9467bd')), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="#d62728", line_width=1, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="#2ca02c", line_width=1, row=2, col=1)

        # MACD Grafiği (Row 3)
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MACD'], mode='lines', name='MACD', line=dict(color='#1f77b4')), row=3, col=1)
        fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data['MACD_Signal'], mode='lines', name='Sinyal', line=dict(color='#ff7f0e')), row=3, col=1)
        colors = ['#2ca02c' if val >= 0 else '#d62728' for val in stock_data['MACD_Hist']] # Yeşil/Kırmızı renkler
        fig.add_trace(go.Bar(x=stock_data.index, y=stock_data['MACD_Hist'], name='Histogram', marker_color=colors, marker_line_width=0), row=3, col=1) # Çizgileri kaldır

        # Grafik Düzenlemeleri
        title_period_text = f"{start_date} - {end_date}" # Periyot metnini kullan
        fig.update_layout(
            title_text=f'{ticker} Fiyat ve Teknik Göstergeler ({title_period_text})',
            height=750, # Yüksekliği biraz azalt
            xaxis_rangeslider_visible=False,
            template='plotly_white',
            legend_title_text='Gösterge',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), # Legend'ı üste taşı
            margin=dict(l=50, r=50, t=100, b=50) # Kenar boşluklarını ayarla
        )
        # Eksen Başlıkları ve Grid
        fig.update_yaxes(title_text="Fiyat", row=1, col=1, gridcolor='#e5e5e5')
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=2, col=1, gridcolor='#e5e5e5')
        fig.update_yaxes(title_text="MACD", row=3, col=1, gridcolor='#e5e5e5')
        fig.update_xaxes(gridcolor='#e5e5e5') # Tüm x eksenleri için grid rengi
        # --- Grafik Oluşturma Sonu ---

        chart_html = pio.to_html(fig, full_html=False, include_plotlyjs='cdn')

        # Grafiği, göstergeleri ve özeti içeren sayfayı göster
        return render_template('index.html', chart_html=chart_html, indicators=indicators, tech_indicators=tech_indicators, summary=summary, news_list=news_list, average_sentiment=average_sentiment)

    except Exception as e:
        print(f"Ana analiz bloğunda hata: {e}")
        import traceback
        traceback.print_exc()
        error_message = f"Analiz sırasında bir hata oluştu: {e}. Lütfen tekrar deneyin."
        return render_template('index.html', error=error_message)


if __name__ == '__main__':
    app.run(debug=True) # debug=True geliştirme sırasında hataları görmemizi sağlar 