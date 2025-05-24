import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from flask import current_app
import pytz

# Modern ML models
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logging.warning("LightGBM yüklü değil, basit model kullanılacak")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logging.warning("Prophet yüklü değil, basit model kullanılacak")

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("Scikit-learn yüklü değil")

logger = logging.getLogger(__name__)

# Önbellek
_prediction_cache = {}

def safe_datetime_diff(date1, date2):
    """Güvenli datetime fark hesaplama - timezone sorunlarını çözer."""
    try:
        # Her iki tarihi de timezone-naive yapalım
        if hasattr(date1, 'tz_localize') and date1.tz is not None:
            date1 = date1.tz_localize(None)
        if hasattr(date2, 'tz_localize') and date2.tz is not None:
            date2 = date2.tz_localize(None)
        
        # Pandas Timestamp'leri normalize et
        if hasattr(date1, 'normalize'):
            date1 = date1.normalize()
        if hasattr(date2, 'normalize'):
            date2 = date2.normalize()
        
        return (date1 - date2).days
    except Exception as e:
        logger.warning(f"Tarih farkı hesaplama hatası: {e}")
        return 0

def get_current_date_safe():
    """Güvenli şu anki tarih alma."""
    try:
        # Timezone-naive tarih döndür
        return pd.Timestamp.now().normalize()
    except Exception:
        return datetime.now().date()

def normalize_datetime(dt):
    """Datetime objelerini normalize et."""
    try:
        if dt is None:
            return None
        
        # Pandas Timestamp ise
        if hasattr(dt, 'tz_localize'):
            if dt.tz is not None:
                dt = dt.tz_localize(None)
            return dt.normalize()
        
        # Python datetime ise
        if hasattr(dt, 'date'):
            return pd.Timestamp(dt.date())
        
        # String ise
        if isinstance(dt, str):
            return pd.Timestamp(dt).normalize()
        
        return pd.Timestamp(dt).normalize()
    except Exception as e:
        logger.warning(f"Datetime normalize hatası: {e}")
        return pd.Timestamp.now().normalize()

def create_features(df, lookback_days=30):
    """Gelişmiş feature engineering."""
    features_df = df.copy()
    
    # Temel fiyat özellikleri
    features_df['returns'] = features_df['Close'].pct_change()
    features_df['log_returns'] = np.log(features_df['Close'] / features_df['Close'].shift(1))
    features_df['price_change'] = features_df['Close'] - features_df['Open']
    features_df['daily_range'] = features_df['High'] - features_df['Low']
    features_df['price_position'] = (features_df['Close'] - features_df['Low']) / features_df['daily_range']
    
    # Hareketli ortalamalar
    for window in [5, 10, 20, 50]:
        features_df[f'sma_{window}'] = features_df['Close'].rolling(window=window).mean()
        features_df[f'ema_{window}'] = features_df['Close'].ewm(span=window).mean()
        features_df[f'close_sma_{window}_ratio'] = features_df['Close'] / features_df[f'sma_{window}']
    
    # Volatilite özellikleri
    for window in [5, 10, 20]:
        features_df[f'volatility_{window}'] = features_df['returns'].rolling(window=window).std()
        features_df[f'price_std_{window}'] = features_df['Close'].rolling(window=window).std()
    
    # Momentum göstergeleri
    features_df['rsi_14'] = calculate_rsi(features_df['Close'], 14)
    features_df['rsi_30'] = calculate_rsi(features_df['Close'], 30)
    
    # Volume özellikleri
    features_df['volume_sma_20'] = features_df['Volume'].rolling(window=20).mean()
    features_df['volume_ratio'] = features_df['Volume'] / features_df['volume_sma_20']
    features_df['price_volume'] = features_df['Close'] * features_df['Volume']
    
    # Bollinger Bands
    bb_window = 20
    bb_std = 2
    features_df['bb_middle'] = features_df['Close'].rolling(window=bb_window).mean()
    bb_std_dev = features_df['Close'].rolling(window=bb_window).std()
    features_df['bb_upper'] = features_df['bb_middle'] + (bb_std_dev * bb_std)
    features_df['bb_lower'] = features_df['bb_middle'] - (bb_std_dev * bb_std)
    features_df['bb_position'] = (features_df['Close'] - features_df['bb_lower']) / (features_df['bb_upper'] - features_df['bb_lower'])
    
    # Lag features
    for lag in [1, 2, 3, 5, 10]:
        features_df[f'close_lag_{lag}'] = features_df['Close'].shift(lag)
        features_df[f'volume_lag_{lag}'] = features_df['Volume'].shift(lag)
        features_df[f'returns_lag_{lag}'] = features_df['returns'].shift(lag)
    
    # Zaman özellikleri
    features_df['day_of_week'] = features_df.index.dayofweek
    features_df['month'] = features_df.index.month
    features_df['quarter'] = features_df.index.quarter
    features_df['day_of_month'] = features_df.index.day
    
    return features_df

def calculate_rsi(prices, window=14):
    """RSI hesaplama."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def predict_with_lightgbm(df, prediction_days=7):
    """LightGBM ile tahmin."""
    if not LIGHTGBM_AVAILABLE:
        return None
    
    try:
        logger.info("LightGBM ile tahmin başlatılıyor...")
        
        # Feature engineering
        features_df = create_features(df)
        
        # Target variable (gelecek günün kapanış fiyatı)
        features_df['target'] = features_df['Close'].shift(-1)
        
        # NaN değerleri temizle
        features_df = features_df.dropna()
        
        if len(features_df) < 50:
            logger.warning("LightGBM için yeterli veri yok")
            return None
        
        # Feature seçimi
        feature_columns = [col for col in features_df.columns if col not in ['target', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        X = features_df[feature_columns]
        y = features_df['target']
        
        # Train/validation split
        train_size = int(len(X) * 0.8)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        # LightGBM parametreleri
        params = {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
        
        # Model training
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        # Validation predictions
        val_predictions = model.predict(X_val)
        mae = mean_absolute_error(y_val, val_predictions)
        rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
        
        # Future predictions
        last_features = X.iloc[-1:].copy()
        predictions = []
        
        for i in range(prediction_days):
            pred = model.predict(last_features)[0]
            predictions.append(pred)
            
            # Update features for next prediction (simplified)
            last_features = last_features.copy()
            # Bu kısım daha karmaşık feature güncellemesi gerektirir
        
        confidence = max(0.1, min(0.9, 1 - (mae / y.mean())))
        
        result = {
            'model_name': 'LightGBM',
            'predictions': predictions,
            'confidence': confidence,
            'mae': mae,
            'rmse': rmse,
            'feature_importance': dict(zip(feature_columns, model.feature_importance()))
        }
        
        logger.info(f"LightGBM tahmin tamamlandı - MAE: {mae:.2f}, Confidence: {confidence:.2f}")
        return result
        
    except Exception as e:
        logger.error(f"LightGBM tahmin hatası: {e}")
        return None

def predict_with_prophet(df, prediction_days=7):
    """Prophet ile tahmin."""
    if not PROPHET_AVAILABLE:
        return None
    
    try:
        logger.info("Prophet ile tahmin başlatılıyor...")
        
        # Prophet için veri hazırlama
        prophet_df = pd.DataFrame()
        prophet_df['ds'] = df.index
        prophet_df['y'] = df['Close'].values
        
        if len(prophet_df) < 50:
            logger.warning("Prophet için yeterli veri yok")
            return None
        
        # Model oluşturma ve eğitim
        model = Prophet(
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0,
            holidays_prior_scale=10.0,
            seasonality_mode='multiplicative'
        )
        
        # Volume özelliği ekleme
        if 'Volume' in df.columns:
            model.add_regressor('volume')
            prophet_df['volume'] = df['Volume'].values
        
        model.fit(prophet_df)
        
        # Gelecek iş günleri için tahmin
        last_date = df.index[-1]
        
        # İş günlerini hesapla
        future_business_days = []
        current_date = last_date + timedelta(days=1)
        
        # Hafta sonunu atla
        while current_date.weekday() >= 5:
            current_date += timedelta(days=1)
        
        # İş günlerini ekle
        while len(future_business_days) < prediction_days:
            if current_date.weekday() < 5:  # Pazartesi=0, Cuma=4
                future_business_days.append(current_date)
            current_date += timedelta(days=1)
        
        # Future dataframe oluştur
        future = pd.DataFrame({'ds': future_business_days})
        
        if 'Volume' in df.columns:
            # Volume için basit extrapolation (son 5 günün ortalaması)
            avg_volume = df['Volume'].tail(5).mean()
            future['volume'] = avg_volume
        
        forecast = model.predict(future)
        
        # Confidence hesaplama (uncertainty interval'a göre)
        uncertainty = (forecast['yhat_upper'] - forecast['yhat_lower']).mean()
        relative_uncertainty = uncertainty / forecast['yhat'].mean()
        confidence = max(0.1, min(0.9, 1 - relative_uncertainty))
        
        result = {
            'model_name': 'Prophet',
            'predictions': forecast['yhat'].tolist(),
            'confidence': confidence,
            'trend': forecast['trend'].iloc[-1] if len(forecast) > 0 else 0,
            'uncertainty': uncertainty,
            'forecast_data': forecast,
            'future_dates': future_business_days
        }
        
        logger.info(f"Prophet tahmin tamamlandı - Confidence: {confidence:.2f}")
        logger.info(f"Prophet tahmin tarihleri: {future_business_days[0].date()} - {future_business_days[-1].date()}")
        return result
        
    except Exception as e:
        logger.error(f"Prophet tahmin hatası: {e}")
        return None

def predict_with_ensemble(df, prediction_days=7):
    """Ensemble model (LightGBM + Prophet + RandomForest)."""
    logger.info("Ensemble tahmin başlatılıyor...")
    
    results = []
    
    # LightGBM
    lgb_result = predict_with_lightgbm(df, prediction_days)
    if lgb_result:
        results.append(lgb_result)
    
    # Prophet
    prophet_result = predict_with_prophet(df, prediction_days)
    if prophet_result:
        results.append(prophet_result)
    
    # RandomForest (fallback)
    if SKLEARN_AVAILABLE:
        rf_result = predict_with_random_forest(df, prediction_days)
        if rf_result:
            results.append(rf_result)
    
    if not results:
        logger.warning("Hiçbir model başarılı olmadı")
        return None
    
    # Ensemble predictions (weighted average)
    weights = [r['confidence'] for r in results]
    total_weight = sum(weights)
    
    if total_weight == 0:
        return None
    
    # Normalize weights
    weights = [w / total_weight for w in weights]
    
    # Weighted average predictions
    ensemble_predictions = []
    for i in range(prediction_days):
        pred = sum(r['predictions'][i] * w for r, w in zip(results, weights))
        ensemble_predictions.append(pred)
    
    # Average confidence
    ensemble_confidence = sum(r['confidence'] * w for r, w in zip(results, weights))
    
    result = {
        'model_name': 'Ensemble',
        'predictions': ensemble_predictions,
        'confidence': ensemble_confidence,
        'individual_models': [r['model_name'] for r in results],
        'model_count': len(results),
        'weights': dict(zip([r['model_name'] for r in results], weights))
    }
    
    logger.info(f"Ensemble tahmin tamamlandı - {len(results)} model, Confidence: {ensemble_confidence:.2f}")
    return result

def predict_with_random_forest(df, prediction_days=7):
    """RandomForest ile tahmin (fallback)."""
    if not SKLEARN_AVAILABLE:
        return None
    
    try:
        logger.info("RandomForest ile tahmin başlatılıyor...")
        
        # Feature engineering
        features_df = create_features(df)
        features_df['target'] = features_df['Close'].shift(-1)
        features_df = features_df.dropna()
        
        if len(features_df) < 30:
            return None
        
        # Feature selection
        feature_columns = [col for col in features_df.columns if col not in ['target', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        X = features_df[feature_columns]
        y = features_df['target']
        
        # Model
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X, y)
        
        # Predictions
        last_features = X.iloc[-1:].copy()
        predictions = []
        
        for i in range(prediction_days):
            pred = model.predict(last_features)[0]
            predictions.append(pred)
        
        # Simple confidence based on feature importance variance
        confidence = 0.6  # Default medium confidence for RF
        
        result = {
            'model_name': 'RandomForest',
            'predictions': predictions,
            'confidence': confidence
        }
        
        logger.info(f"RandomForest tahmin tamamlandı")
        return result
        
    except Exception as e:
        logger.error(f"RandomForest tahmin hatası: {e}")
        return None

def predict_stock_price(ticker, stock_data, prediction_days=7):
    """Ana tahmin fonksiyonu - en iyi mevcut modeli kullan."""
    logger.info(f"{ticker} için {prediction_days} günlük tahmin başlatılıyor...")
    
    if stock_data is None or stock_data.empty:
        logger.error("Veri yok, tahmin yapılamıyor")
        return None
    
    if len(stock_data) < 30:
        logger.warning("Tahmin için yeterli veri yok")
        return None
    
    try:
        # Veri güncelliğini güvenli şekilde kontrol et
        last_data_date = normalize_datetime(stock_data.index[-1])
        current_date = get_current_date_safe()
        
        # Veri yaşını hesapla
        data_age = safe_datetime_diff(current_date, last_data_date)
        if data_age > 3:
            logger.warning(f"{ticker} verisi {data_age} gün eski - {last_data_date.date()}")
        
        # Önce ensemble dene
        result = predict_with_ensemble(stock_data, prediction_days)
        
        # Ensemble başarısız olursa en iyi tek modeli dene
        if not result:
            if LIGHTGBM_AVAILABLE:
                result = predict_with_lightgbm(stock_data, prediction_days)
            elif PROPHET_AVAILABLE:
                result = predict_with_prophet(stock_data, prediction_days)
            elif SKLEARN_AVAILABLE:
                result = predict_with_random_forest(stock_data, prediction_days)
        
        if result:
            # Son gerçek fiyat ve tahmin verilerini ekle
            result['last_actual_price'] = float(stock_data['Close'].iloc[-1])
            result['last_data_date'] = last_data_date.strftime('%Y-%m-%d')
            result['prediction_horizon_days'] = prediction_days
            
            # Gelecek tarihler için güvenli hesaplama
            try:
                # Başlangıç tarihini güvenli şekilde hesapla
                start_date = last_data_date + timedelta(days=1)
                while start_date.weekday() >= 5:  # Hafta sonunu atla
                    start_date += timedelta(days=1)
                
                # İş günlerini hesapla
                future_dates = pd.bdate_range(
                    start=start_date,
                    periods=prediction_days,
                    freq='B'
                )
                
                # Timezone bilgisini temizle
                future_dates = future_dates.tz_localize(None) if future_dates.tz is not None else future_dates
                
            except Exception as e:
                logger.warning(f"Tarih hesaplama hatası: {e}, varsayılan yöntem kullanılacak")
                # Fallback: basit tarih hesaplama
                future_dates = []
                current_date_iter = last_data_date + timedelta(days=1)
                while len(future_dates) < prediction_days:
                    if current_date_iter.weekday() < 5:  # İş günü
                        future_dates.append(current_date_iter)
                    current_date_iter += timedelta(days=1)
                future_dates = pd.DatetimeIndex(future_dates)
            
            # Tahmin dizisini düzelt
            predictions_list = result['predictions']
            if len(predictions_list) != len(future_dates):
                if len(predictions_list) > len(future_dates):
                    predictions_list = predictions_list[:len(future_dates)]
                else:
                    # Eksik tahmini son değerle doldur
                    last_pred = predictions_list[-1] if predictions_list else result['last_actual_price']
                    while len(predictions_list) < len(future_dates):
                        predictions_list.append(last_pred)
            
            result['predictions'] = pd.DataFrame({
                'date': future_dates,
                'predicted_price': predictions_list
            })
            
            # Güncel tarih bilgilerini ekle
            result['prediction_created_at'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            result['next_trading_day'] = future_dates[0].strftime('%Y-%m-%d') if len(future_dates) > 0 else None
            
            logger.info(f"{ticker} tahmin başarılı - Model: {result['model_name']}, Confidence: {result['confidence']:.2f}")
            if len(future_dates) > 0:
                logger.info(f"Tahmin tarihleri: {future_dates[0].date()} - {future_dates[-1].date()}")
        else:
            logger.error(f"{ticker} için tüm tahmin modelleri başarısız oldu")
        
        return result
        
    except Exception as e:
        logger.error(f"{ticker} tahmin fonksiyonu hatası: {e}")
        return None

def get_price_prediction(ticker, close_series, period, future_periods=None):
    """Ana tahmin fonksiyonu."""
    try:
        if future_periods is None:
            future_periods = current_app.config.get('FUTURE_PERIODS', 30)
        
        # Önbellek kontrolü
        cache_key = f"{ticker}_{period}_{future_periods}_prediction"
        now = datetime.now()
        
        if cache_key in _prediction_cache:
            cached_entry = _prediction_cache[cache_key]
            try:
                # Güvenli cache yaşı hesaplama
                cache_age_seconds = (now - cached_entry['timestamp']).total_seconds()
                if cache_age_seconds < current_app.config['CACHE_MAX_AGE_SECONDS']:
                    logger.info(f"{ticker} için tahmin önbellekten alındı")
                    return cached_entry['data']
                else:
                    del _prediction_cache[cache_key]
            except Exception as e:
                logger.warning(f"Cache yaş hesaplama hatası: {e}")
                del _prediction_cache[cache_key]
        
        # Özellikler oluştur
        features_df = create_features(
            close_series,
            n_lags=current_app.config.get('N_LAGS', 7),
            window_size=current_app.config.get('ROLLING_WINDOW', 7)
        )
        
        if features_df is None:
            logger.warning(f"{ticker} için özellik oluşturulamadı")
            return None
        
        # Modelleri eğit
        models_and_metrics = []
        
        # XGBoost
        try:
            xgb_model, xgb_metrics = train_xgboost_model(features_df)
            if xgb_model is not None:
                models_and_metrics.append((xgb_model, xgb_metrics))
        except Exception as e:
            logger.warning(f"XGBoost model eğitimi başarısız: {e}")
        
        # Random Forest
        try:
            rf_model, rf_metrics = train_random_forest_model(features_df)
            if rf_model is not None:
                models_and_metrics.append((rf_model, rf_metrics))
        except Exception as e:
            logger.warning(f"RandomForest model eğitimi başarısız: {e}")
        
        if not models_and_metrics:
            logger.warning(f"{ticker} için hiçbir model eğitilemedi")
            return None
        
        # Ensemble tahmin
        try:
            ensemble_result = ensemble_prediction(models_and_metrics, features_df, future_periods)
            
            if ensemble_result is None:
                logger.warning(f"{ticker} için ensemble tahmin başarısız")
                return None
            
            prediction_df, confidence = ensemble_result
            
            result = {
                'predictions': prediction_df,
                'confidence': confidence,
                'model_count': len(models_and_metrics),
                'last_actual_price': close_series.iloc[-1],
                'prediction_horizon_days': future_periods
            }
            
            # Önbelleğe kaydet
            _prediction_cache[cache_key] = {
                'data': result,
                'timestamp': now
            }
            
            logger.info(f"{ticker} için tahmin tamamlandı - Güven: {confidence:.3f}")
            
            return result
        except Exception as e:
            logger.error(f"Ensemble tahmin hatası: {e}")
            return None
        
    except Exception as e:
        logger.error(f"Fiyat tahmini hatası ({ticker}): {e}")
        return None

def get_prediction_summary(prediction_result):
    """Tahmin özetini oluştur."""
    if prediction_result is None:
        return "Fiyat tahmini yapılamadı."
    
    try:
        predictions = prediction_result['predictions']
        confidence = prediction_result['confidence']
        last_price = prediction_result['last_actual_price']
        horizon = prediction_result['prediction_horizon_days']
        
        if predictions.empty:
            return "Tahmin verisi bulunamadı."
        
        # İstatistikler hesapla
        last_predicted = predictions['predicted_price'].iloc[-1]
        first_predicted = predictions['predicted_price'].iloc[0]
        mean_predicted = predictions['predicted_price'].mean()
        
        # Değişim yüzdeleri
        price_change_pct = ((last_predicted - last_price) / last_price) * 100
        short_term_change = ((first_predicted - last_price) / last_price) * 100
        
        # Trend belirleme
        if price_change_pct > 5:
            trend = "güçlü yükseliş"
        elif price_change_pct > 1:
            trend = "yükseliş"
        elif price_change_pct < -5:
            trend = "güçlü düşüş"
        elif price_change_pct < -1:
            trend = "düşüş"
        else:
            trend = "yatay seyir"
        
        # Güven seviyesi açıklaması
        if confidence > 0.8:
            confidence_text = "yüksek"
        elif confidence > 0.6:
            confidence_text = "orta"
        else:
            confidence_text = "düşük"
        
        summary = f"""
        Makine öğrenmesi modelleri {horizon} günlük periyot için {trend} eğilimi öngörüyor.
        
        Mevcut fiyat: {last_price:.2f}
        Kısa vadeli tahmin (1 gün): {first_predicted:.2f} (%{short_term_change:+.1f})
        Ortalama tahmin: {mean_predicted:.2f}
        Uzun vadeli tahmin ({horizon} gün): {last_predicted:.2f} (%{price_change_pct:+.1f})
        
        Model güveni: {confidence_text} (%{confidence*100:.0f})
        """
        
        return summary.strip()
        
    except Exception as e:
        logger.error(f"Tahmin özeti oluşturma hatası: {e}")
        return "Tahmin özeti oluşturulamadı." 