from flask import render_template, request, flash, current_app, jsonify, redirect, url_for
from app.main import bp
from app.services import stock_service, news_service, prediction_service, chart_service
from app.utils.formatters import *
from datetime import datetime
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Varsayılan hisse senetleri listesi
DEFAULT_STOCKS = [
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
    # ABD Hisseleri
    {"ticker": "AAPL", "name": "Apple Inc.", "market": "US"},
    {"ticker": "MSFT", "name": "Microsoft Corp.", "market": "US"},
    {"ticker": "GOOGL", "name": "Alphabet Inc. (C)", "market": "US"},
    {"ticker": "AMZN", "name": "Amazon.com Inc.", "market": "US"},
    {"ticker": "NVDA", "name": "NVIDIA Corp.", "market": "US"},
    {"ticker": "TSLA", "name": "Tesla Inc.", "market": "US"},
    {"ticker": "META", "name": "Meta Platforms Inc.", "market": "US"},
    {"ticker": "JPM", "name": "JPMorgan Chase & Co.", "market": "US"},
    {"ticker": "V", "name": "Visa Inc.", "market": "US"},
    {"ticker": "JNJ", "name": "Johnson & Johnson", "market": "US"},
]

@bp.route('/')
def index():
    """Ana sayfa."""
    try:
        # Veritabanından hisse listesini al, yoksa varsayılan listeyi kullan
        stock_list = stock_service.get_stock_list()
        if not stock_list:
            stock_list = DEFAULT_STOCKS
            
        # Genel piyasa duyarlılığını al
        market_sentiment = news_service.get_market_sentiment()
        
        return render_template('index.html', 
                             stock_list=stock_list,
                             market_sentiment=market_sentiment)
                             
    except Exception as e:
        logger.error(f"Ana sayfa hatası: {e}")
        return render_template('index.html', 
                             stock_list=DEFAULT_STOCKS,
                             error="Sayfa yüklenirken bir hata oluştu.")

@bp.route('/analyze', methods=['POST'])
def analyze():
    """Hisse senedi analizi yap."""
    try:
        # Form verilerini al
        ticker = request.form.get('ticker', '').strip().upper()
        period = request.form.get('period', '1y')
        chart_type = request.form.get('chart_type', 'line')
        
        if not ticker:
            flash('Lütfen bir hisse senedi kodu girin.', 'error')
            return render_template('index.html', stock_list=DEFAULT_STOCKS)
        
        logger.info(f"Analiz başlatıldı: {ticker}, {period}, {chart_type}")
        
        # Hisse senedi bilgilerini al veya oluştur
        stock = stock_service.get_or_create_stock(ticker)
        if not stock:
            flash(f'{ticker} için hisse senedi bilgileri alınamadı.', 'error')
            return render_template('index.html', stock_list=DEFAULT_STOCKS)
        
        # Hisse senedi verilerini çek
        stock_data = stock_service.get_stock_data(ticker, period)
        if stock_data is None or stock_data.empty:
            flash(f'{ticker} için yeterli veri bulunamadı.', 'error')
            return render_template('index.html', stock_list=DEFAULT_STOCKS)
        
        # Teknik göstergeleri hesapla
        indicators = stock_service.calculate_technical_indicators(stock_data)
        if indicators is None:
            flash(f'{ticker} için teknik analiz yapılamadı.', 'error')
            return render_template('index.html', stock_list=DEFAULT_STOCKS)
        
        # Hisse senedi temel bilgilerini al
        stock_info = stock_service.get_stock_info(ticker)
        
        # Haber analizi yap
        news_analysis = news_service.get_stock_news_analysis(
            stock.name, ticker, days_back=7
        )
        
        # Fiyat tahmini yap - yeni gelişmiş model
        prediction_result = prediction_service.predict_stock_price(
            ticker, stock_data, prediction_days=7
        )
        
        # Temel göstergeleri formatla
        basic_indicators = {}
        if stock_info:
            basic_indicators = {
                'F/K Oranı': format_indicator(get_numeric_value(stock_info, 'forwardPE')),
                'Piyasa Değeri': format_market_cap(get_numeric_value(stock_info, 'marketCap')),
                'Günlük Hacim': format_volume(get_numeric_value(stock_info, 'volume')),
                'Son Kapanış': format_price(stock_data['Close'].iloc[-1] if not stock_data['Close'].empty else None),
                '52 Hafta Yüksek': format_price(get_numeric_value(stock_info, 'fiftyTwoWeekHigh')),
                '52 Hafta Düşük': format_price(get_numeric_value(stock_info, 'fiftyTwoWeekLow')),
                'Temettü Verimi': format_percentage(get_numeric_value(stock_info, 'dividendYield')),
                'Beta': format_indicator(get_numeric_value(stock_info, 'beta')),
            }
        
        # Teknik göstergeleri formatla
        tech_indicators = {}
        if indicators:
            current_rsi = indicators['RSI'].iloc[-1] if not indicators['RSI'].empty else None
            current_macd = indicators['MACD'].iloc[-1] if not indicators['MACD'].empty else None
            current_macd_signal = indicators['MACD_Signal'].iloc[-1] if not indicators['MACD_Signal'].empty else None
            current_sma20 = indicators['SMA_20'].iloc[-1] if not indicators['SMA_20'].empty else None
            current_sma50 = indicators['SMA_50'].iloc[-1] if not indicators['SMA_50'].empty else None
            current_bb_upper = indicators['BB_High'].iloc[-1] if not indicators['BB_High'].empty else None
            current_bb_lower = indicators['BB_Low'].iloc[-1] if not indicators['BB_Low'].empty else None
            
            tech_indicators = {
                'RSI (14)': format_technical_signal('RSI', current_rsi)[0],
                'MACD': format_indicator(current_macd),
                'MACD Sinyal': format_indicator(current_macd_signal),
                'SMA 20': format_price(current_sma20),
                'SMA 50': format_price(current_sma50),
                'BB Üst': format_price(current_bb_upper),
                'BB Alt': format_price(current_bb_lower),
                'Stochastic K': format_indicator(indicators['Stoch_K'].iloc[-1] if not indicators['Stoch_K'].empty else None),
                'Williams %R': format_indicator(indicators['Williams_R'].iloc[-1] if not indicators['Williams_R'].empty else None),
                'ATR': format_indicator(indicators['ATR'].iloc[-1] if not indicators['ATR'].empty else None),
            }
        
        # Grafik oluştur
        chart_html = chart_service.create_stock_chart(
            stock_data, indicators, prediction_result, chart_type, ticker
        )
        
        # Analiz özetini oluştur
        summary = generate_analysis_summary(
            stock, stock_data['Close'], indicators, news_analysis, prediction_result, period
        )
        
        # Analizi veritabanına kaydet
        try:
            price_info = {
                'current_price': stock_data['Close'].iloc[-1] if not stock_data['Close'].empty else None,
                'price_change': calculate_price_change(stock_data['Close'])
            }
            
            stock_service.save_analysis(
                stock.id, period, chart_type, 
                format_indicators_for_db(indicators),
                price_info,
                {'average': news_analysis['average_sentiment'], 'count': news_analysis['total_count']},
                {'price': prediction_result['predictions']['predicted_price'].iloc[-1] if prediction_result else None,
                 'confidence': prediction_result['confidence'] if prediction_result else None}
            )
        except Exception as e:
            logger.warning(f"Analiz kaydedilemedi: {e}")
        
        return render_template('analysis.html',
                             ticker=ticker,
                             stock=stock,
                             chart_html=chart_html,
                             basic_indicators=basic_indicators,
                             tech_indicators=tech_indicators,
                             news_analysis=news_analysis,
                             prediction_result=prediction_result,
                             summary=summary,
                             stock_list=DEFAULT_STOCKS)
        
    except Exception as e:
        logger.error(f"Analiz hatası: {e}")
        flash('Analiz sırasında bir hata oluştu. Lütfen tekrar deneyin.', 'error')
        return render_template('index.html', stock_list=DEFAULT_STOCKS)

@bp.route('/compare')
def compare():
    """Hisse senedi karşılaştırma sayfası."""
    try:
        stock_list = stock_service.get_stock_list()
        if not stock_list:
            stock_list = DEFAULT_STOCKS
            
        return render_template('compare.html', stock_list=stock_list)
        
    except Exception as e:
        logger.error(f"Karşılaştırma sayfası hatası: {e}")
        return render_template('compare.html', stock_list=DEFAULT_STOCKS)

@bp.route('/compare_stocks', methods=['POST'])
def compare_stocks():
    """Seçilen hisse senetlerini karşılaştır."""
    try:
        tickers = request.form.getlist('tickers')
        period = request.form.get('period', '1y')
        
        if len(tickers) < 2:
            flash('En az 2 hisse senedi seçiniz.', 'error')
            return redirect(url_for('main.compare'))
        
        if len(tickers) > 6:
            flash('En fazla 6 hisse senedi seçebilirsiniz.', 'error')
            return redirect(url_for('main.compare'))
        
        # Her hisse için veri çek
        stocks_data = {}
        for ticker in tickers:
            data = stock_service.get_stock_data(ticker.upper(), period)
            if data is not None and not data.empty:
                stocks_data[ticker.upper()] = data
        
        if not stocks_data:
            flash('Seçilen hisse senetleri için veri bulunamadı.', 'error')
            return redirect(url_for('main.compare'))
        
        # Karşılaştırma grafiği oluştur
        comparison_chart = chart_service.create_comparison_chart(stocks_data, period)
        
        # Performans hesapla
        performance_data = calculate_comparison_performance(stocks_data)
        
        return render_template('comparison_result.html',
                             comparison_chart=comparison_chart,
                             performance_data=performance_data,
                             selected_tickers=list(stocks_data.keys()),
                             period=period)
        
    except Exception as e:
        logger.error(f"Karşılaştırma hatası: {e}")
        flash('Karşılaştırma sırasında bir hata oluştu.', 'error')
        return redirect(url_for('main.compare'))

@bp.route('/search_stocks')
def search_stocks():
    """AJAX ile hisse senedi ara."""
    try:
        query = request.args.get('q', '').strip()
        if len(query) < 2:
            return jsonify([])
        
        results = stock_service.search_stocks(query)
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Hisse arama hatası: {e}")
        return jsonify([])

def generate_analysis_summary(stock, close_series, indicators, news_analysis, prediction_result, period):
    """Analiz özetini oluştur."""
    try:
        summary_parts = []
        
        if not close_series.empty:
            current_price = close_series.iloc[-1]
            start_price = close_series.iloc[0]
            price_change_pct = ((current_price - start_price) / start_price) * 100
            
            summary_parts.append(f"{stock.ticker} ({stock.name}) son {period} döneminde %{price_change_pct:+.2f} değişim gösterdi.")
        
        # Teknik analiz özeti
        if indicators:
            tech_summary = []
            
            # RSI analizi
            current_rsi = indicators['RSI'].iloc[-1] if not indicators['RSI'].empty else None
            if current_rsi:
                if current_rsi > 70:
                    tech_summary.append("RSI aşırı alım bölgesinde")
                elif current_rsi < 30:
                    tech_summary.append("RSI aşırı satım bölgesinde")
                else:
                    tech_summary.append("RSI nötr bölgede")
            
            # MACD analizi
            current_macd = indicators['MACD'].iloc[-1] if not indicators['MACD'].empty else None
            current_macd_signal = indicators['MACD_Signal'].iloc[-1] if not indicators['MACD_Signal'].empty else None
            if current_macd and current_macd_signal:
                if current_macd > current_macd_signal:
                    tech_summary.append("MACD pozitif momentum gösteriyor")
                else:
                    tech_summary.append("MACD negatif momentum gösteriyor")
            
            if tech_summary:
                summary_parts.append("Teknik göstergeler: " + ", ".join(tech_summary) + ".")
        
        # Haber analizi özeti
        if news_analysis['total_count'] > 0:
            sentiment_score = news_analysis['average_sentiment']
            if sentiment_score > 0.1:
                sentiment_text = "pozitif"
            elif sentiment_score < -0.1:
                sentiment_text = "negatif"
            else:
                sentiment_text = "nötr"
            
            summary_parts.append(f"Son {news_analysis['total_count']} haberde genel duyarlılık {sentiment_text}.")
        
        # Tahmin özeti
        if prediction_result:
            summary_parts.append(prediction_service.get_prediction_summary(prediction_result))
        
        summary_parts.append("\nUYARI: Bu analizler yalnızca bilgilendirme amaçlıdır ve yatırım tavsiyesi değildir.")
        
        return "\n".join(summary_parts)
        
    except Exception as e:
        logger.error(f"Özet oluşturma hatası: {e}")
        return "Analiz özeti oluşturulamadı."

def calculate_price_change(close_series):
    """Fiyat değişimini hesapla."""
    if close_series.empty or len(close_series) < 2:
        return None
    
    try:
        current = close_series.iloc[-1]
        previous = close_series.iloc[0]
        return ((current - previous) / previous) * 100
    except:
        return None

def format_indicators_for_db(indicators):
    """Göstergeleri veritabanı için formatla."""
    formatted = {}
    
    if indicators:
        for key, series in indicators.items():
            if series is not None and not series.empty:
                formatted[key] = {'current': series.iloc[-1]}
    
    return formatted

def calculate_comparison_performance(stocks_data):
    """Karşılaştırma performansını hesapla."""
    performance = {}
    
    for ticker, data in stocks_data.items():
        try:
            close_col = 'Close'
            if isinstance(data.columns, pd.MultiIndex):
                close_cols = [col for col in data.columns if col[0] == 'Close']
                if close_cols:
                    close_col = close_cols[0]
            
            close_series = data[close_col].dropna()
            if not close_series.empty and len(close_series) >= 2:
                total_return = ((close_series.iloc[-1] / close_series.iloc[0]) - 1) * 100
                performance[ticker] = {
                    'total_return': total_return,
                    'current_price': close_series.iloc[-1],
                    'start_price': close_series.iloc[0]
                }
        except Exception as e:
            logger.error(f"Performans hesaplama hatası ({ticker}): {e}")
            performance[ticker] = {
                'total_return': 0,
                'current_price': 0,
                'start_price': 0
            }
    
    return performance 