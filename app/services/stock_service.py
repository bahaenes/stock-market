import yfinance as yf
import pandas as pd
import ta
from datetime import datetime, timedelta
from app.models import Stock, Analysis
from app import db
from flask import current_app
import logging
import time
import random
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

# Rate limiting için global değişkenler
_last_request_time = 0
_request_count = 0
_rate_limit_reset_time = 0

# Önbellek için global değişkenler
_info_cache = {}
_data_cache = {}
_demo_data_cache = {}

def configure_yfinance_session():
    """YFinance için özel session yapılandırması."""
    session = requests.Session()
    
    # Retry stratejisi
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # Headers
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'application/json',
        'Accept-Language': 'en-US,en;q=0.9',
    })
    
    return session

def wait_for_rate_limit():
    """Rate limit kontrolü - istekler arası bekleme."""
    global _last_request_time, _request_count, _rate_limit_reset_time
    
    current_time = time.time()
    
    # Rate limit reset kontrolü (her dakika)
    if current_time > _rate_limit_reset_time:
        _request_count = 0
        _rate_limit_reset_time = current_time + 60
    
    # İstek sayısı kontrolü (dakikada maksimum 10 istek)
    if _request_count >= 10:
        sleep_time = _rate_limit_reset_time - current_time
        if sleep_time > 0:
            logger.info(f"Rate limit doldu, {sleep_time:.1f} saniye bekleniyor...")
            time.sleep(sleep_time)
            _request_count = 0
            _rate_limit_reset_time = time.time() + 60
    
    # İstekler arası minimum bekleme (1-3 saniye arası rastgele)
    time_since_last = current_time - _last_request_time
    min_wait = random.uniform(1.0, 3.0)
    
    if time_since_last < min_wait:
        sleep_time = min_wait - time_since_last
        logger.debug(f"İstekler arası {sleep_time:.1f} saniye bekleniyor...")
        time.sleep(sleep_time)
    
    _last_request_time = time.time()
    _request_count += 1

def create_demo_data(ticker):
    """Rate limit sorunları için demo veri oluştur."""
    logger.info(f"Demo veri oluşturuluyor: {ticker}")
    
    # Bugünden 252 gün geriye (1 yıl) rastgele fiyat verisi
    end_date = datetime.now()
    start_date = end_date - timedelta(days=252)
    
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    dates = dates[dates.weekday < 5]  # Sadece iş günleri
    
    # Başlangıç fiyatı
    base_price = random.uniform(50, 500)
    
    # Rastgele yürüyüş ile fiyat oluştur
    prices = []
    current_price = base_price
    
    for _ in range(len(dates)):
        # -2% ile +2% arası rastgele değişim
        change = random.uniform(-0.02, 0.02)
        current_price *= (1 + change)
        prices.append(current_price)
    
    # OHLCV verisi oluştur
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        high = close * random.uniform(1.001, 1.03)
        low = close * random.uniform(0.97, 0.999)
        open_price = close * random.uniform(0.98, 1.02)
        volume = random.randint(100000, 10000000)
        
        data.append({
            'Date': date,
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('Date', inplace=True)
    
    return df

def get_stock_data(ticker, period='1y'):
    """Hisse senedi verilerini çek (rate limiting ile)."""
    cache_key = f"{ticker}_{period}_data"
    now = datetime.now()
    
    # Önbellekten kontrol et
    if cache_key in _data_cache:
        cached_entry = _data_cache[cache_key]
        cache_age = (now - cached_entry['timestamp']).total_seconds()
        cache_max_age = getattr(current_app.config, 'CACHE_MAX_AGE_SECONDS', 3600)
        if cache_age < cache_max_age:
            logger.info(f"{ticker} için veri önbellekten alındı")
            return cached_entry['data']
        else:
            del _data_cache[cache_key]
    
    # Demo veri kontrolü
    if cache_key in _demo_data_cache:
        logger.info(f"{ticker} için demo veri kullanılıyor")
        return _demo_data_cache[cache_key]
    
    try:
        # Rate limit bekle
        wait_for_rate_limit()
        
        # YFinance session konfigürasyonu
        session = configure_yfinance_session()
        
        # Veri çek
        logger.info(f"{ticker} için veri çekiliyor...")
        
        # Önce basit download dene
        stock_data = yf.download(ticker, period=period, session=session, progress=False)
        
        if stock_data.empty:
            # Alternatif olarak Ticker objesi dene
            stock_obj = yf.Ticker(ticker, session=session)
            stock_data = stock_obj.history(period=period)
        
        if not stock_data.empty:
            _data_cache[cache_key] = {
                'data': stock_data,
                'timestamp': now
            }
            logger.info(f"{ticker} için yeni veri çekildi ve önbelleğe alındı ({len(stock_data)} kayıt)")
            return stock_data
        else:
            logger.warning(f"{ticker} için veri bulunamadı, demo veri oluşturuluyor")
            demo_data = create_demo_data(ticker)
            _demo_data_cache[cache_key] = demo_data
            return demo_data
            
    except requests.exceptions.HTTPError as e:
        if '429' in str(e):
            logger.warning(f"Rate limit aşıldı ({ticker}), demo veri oluşturuluyor")
            demo_data = create_demo_data(ticker)
            _demo_data_cache[cache_key] = demo_data
            return demo_data
        else:
            logger.error(f"HTTP hatası ({ticker}): {e}")
            return None
    except Exception as e:
        logger.error(f"Hisse verisi çekilirken hata ({ticker}): {e}")
        # Acil durum için demo veri
        demo_data = create_demo_data(ticker)
        _demo_data_cache[cache_key] = demo_data
        return demo_data

def get_stock_info(ticker):
    """Hisse senedi temel bilgilerini çek (rate limiting ile)."""
    cache_key = f"{ticker}_info"
    now = datetime.now()
    
    # Önbellekten kontrol et
    if cache_key in _info_cache:
        cached_entry = _info_cache[cache_key]
        cache_age = (now - cached_entry['timestamp']).total_seconds()
        cache_max_age = getattr(current_app.config, 'CACHE_MAX_AGE_SECONDS', 3600)
        if cache_age < cache_max_age:
            logger.info(f"{ticker} için bilgi önbellekten alındı")
            return cached_entry['data']
        else:
            del _info_cache[cache_key]
    
    try:
        # Rate limit bekle
        wait_for_rate_limit()
        
        # YFinance session konfigürasyonu
        session = configure_yfinance_session()
        
        # Bilgi çek
        stock_obj = yf.Ticker(ticker, session=session)
        stock_info = stock_obj.info
        
        if stock_info and len(stock_info) > 3:  # En az birkaç alan olmalı
            _info_cache[cache_key] = {
                'data': stock_info,
                'timestamp': now
            }
            logger.info(f"{ticker} için yeni bilgi çekildi ve önbelleğe alındı")
            return stock_info
        else:
            logger.warning(f"{ticker} için bilgi bulunamadı")
            # Temel bilgi oluştur
            return {
                'longName': ticker,
                'symbol': ticker,
                'market': get_market_from_ticker(ticker)
            }
            
    except requests.exceptions.HTTPError as e:
        if '429' in str(e):
            logger.warning(f"Rate limit aşıldı ({ticker} bilgisi)")
            return {
                'longName': ticker,
                'symbol': ticker,
                'market': get_market_from_ticker(ticker)
            }
        else:
            logger.error(f"HTTP hatası ({ticker} bilgisi): {e}")
            return None
    except Exception as e:
        logger.error(f"Hisse bilgisi çekilirken hata ({ticker}): {e}")
        return {
            'longName': ticker,
            'symbol': ticker,
            'market': get_market_from_ticker(ticker)
        }

def calculate_technical_indicators(stock_data):
    """Teknik göstergeleri hesapla."""
    if stock_data is None or stock_data.empty:
        return None
    
    try:
        # Close sütununu belirle
        close_col = 'Close'
        high_col = 'High'
        low_col = 'Low'
        
        if isinstance(stock_data.columns, pd.MultiIndex):
            close_cols = [col for col in stock_data.columns if col[0] == 'Close']
            high_cols = [col for col in stock_data.columns if col[0] == 'High']
            low_cols = [col for col in stock_data.columns if col[0] == 'Low']
            
            if close_cols:
                close_col = close_cols[0]
            if high_cols:
                high_col = high_cols[0]
            if low_cols:
                low_col = low_cols[0]
        
        close_series = pd.to_numeric(stock_data[close_col], errors='coerce').dropna()
        high_series = pd.to_numeric(stock_data[high_col], errors='coerce').dropna()
        low_series = pd.to_numeric(stock_data[low_col], errors='coerce').dropna()
        
        if len(close_series) < 50:
            logger.warning("Teknik analiz için yeterli veri yok")
            return None
        
        # Teknik göstergeleri hesapla
        indicators = {}
        
        # Hareketli ortalamalar
        indicators['SMA_20'] = ta.trend.sma_indicator(close_series, window=20)
        indicators['SMA_50'] = ta.trend.sma_indicator(close_series, window=50)
        
        # RSI
        indicators['RSI'] = ta.momentum.rsi(close_series, window=14)
        
        # MACD
        macd_obj = ta.trend.MACD(close_series)
        indicators['MACD'] = macd_obj.macd()
        indicators['MACD_Signal'] = macd_obj.macd_signal()
        indicators['MACD_Hist'] = macd_obj.macd_diff()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(close_series, window=20, window_dev=2)
        indicators['BB_High'] = bb.bollinger_hband()
        indicators['BB_Low'] = bb.bollinger_lband()
        indicators['BB_MAVG'] = bb.bollinger_mavg()
        
        # Stochastic
        indicators['Stoch_K'] = ta.momentum.stoch(high_series, low_series, close_series)
        indicators['Stoch_D'] = ta.momentum.stoch_signal(high_series, low_series, close_series)
        
        # Williams %R
        indicators['Williams_R'] = ta.momentum.williams_r(high_series, low_series, close_series)
        
        # Average True Range (ATR)
        indicators['ATR'] = ta.volatility.average_true_range(high_series, low_series, close_series)
        
        logger.info(f"Teknik göstergeler başarıyla hesaplandı")
        return indicators
    
    except Exception as e:
        logger.error(f"Teknik göstergeler hesaplanırken hata: {e}")
        return None

def get_or_create_stock(ticker, name=None, market=None):
    """Veritabanından hisse senedi al veya oluştur."""
    stock = Stock.query.filter_by(ticker=ticker.upper()).first()
    
    if not stock:
        # Yeni hisse senedi oluştur
        stock_info = get_stock_info(ticker)
        if stock_info:
            stock = Stock(
                ticker=ticker.upper(),
                name=name or stock_info.get('longName', ticker),
                market=market or get_market_from_ticker(ticker),
                sector=stock_info.get('sector'),
                industry=stock_info.get('industry'),
                market_cap=stock_info.get('marketCap')
            )
            db.session.add(stock)
            try:
                db.session.commit()
                logger.info(f"Yeni hisse senedi oluşturuldu: {ticker}")
            except Exception as e:
                db.session.rollback()
                logger.error(f"Hisse senedi oluşturulurken hata: {e}")
                return None
    
    return stock

def get_market_from_ticker(ticker):
    """Ticker'dan borsa bilgisini çıkar."""
    if '.IS' in ticker.upper():
        return 'BIST'
    elif any(exchange in ticker.upper() for exchange in ['.L', '.TO', '.V']):
        return 'OTHER'
    else:
        return 'US'

def save_analysis(stock_id, period, chart_type, indicators, price_info, sentiment_data=None, prediction_data=None):
    """Analiz sonuçlarını veritabanına kaydet."""
    try:
        analysis = Analysis(
            stock_id=stock_id,
            period=period,
            chart_type=chart_type,
            rsi=indicators.get('RSI', {}).get('current') if indicators.get('RSI') else None,
            macd=indicators.get('MACD', {}).get('current') if indicators.get('MACD') else None,
            macd_signal=indicators.get('MACD_Signal', {}).get('current') if indicators.get('MACD_Signal') else None,
            sma_20=indicators.get('SMA_20', {}).get('current') if indicators.get('SMA_20') else None,
            sma_50=indicators.get('SMA_50', {}).get('current') if indicators.get('SMA_50') else None,
            current_price=price_info.get('current_price'),
            price_change=price_info.get('price_change'),
            sentiment_score=sentiment_data.get('average') if sentiment_data else None,
            news_count=sentiment_data.get('count') if sentiment_data else None,
            predicted_price=prediction_data.get('price') if prediction_data else None,
            prediction_confidence=prediction_data.get('confidence') if prediction_data else None
        )
        
        db.session.add(analysis)
        db.session.commit()
        logger.info(f"Analiz kaydedildi: Stock ID {stock_id}")
        
    except Exception as e:
        logger.error(f"Analiz kaydedilirken hata: {e}")
        db.session.rollback()

def get_stock_list():
    """Veritabanından hisse senetleri listesini al."""
    try:
        stocks = Stock.query.limit(50).all()
        return [{'ticker': s.ticker, 'name': s.name, 'market': s.market} for s in stocks]
    except Exception as e:
        logger.error(f"Hisse listesi alınırken hata: {e}")
        return []

def search_stocks(query):
    """Hisse senetlerinde arama yap."""
    try:
        query = f"%{query.upper()}%"
        stocks = Stock.query.filter(
            (Stock.ticker.like(query)) | (Stock.name.like(query))
        ).limit(10).all()
        
        return [{'ticker': s.ticker, 'name': s.name, 'market': s.market} for s in stocks]
    except Exception as e:
        logger.error(f"Hisse arama hatası: {e}")
        return [] 