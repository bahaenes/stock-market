"""
Kapsamlı Error Handling Sistemi
"""

import logging
import traceback
from datetime import datetime
from functools import wraps
import pandas as pd
from flask import current_app
import sys

logger = logging.getLogger(__name__)

class FinanceAppError(Exception):
    """Ana uygulama hatası."""
    pass

class DataFetchError(FinanceAppError):
    """Veri çekme hatası."""
    pass

class PredictionError(FinanceAppError):
    """Tahmin hatası."""
    pass

class TimezoneError(FinanceAppError):
    """Timezone hatası."""
    pass

def safe_execute(operation_name="Bilinmeyen işlem", fallback_value=None, raise_on_error=False):
    """Güvenli işlem yürütme decorator."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_details = {
                    'operation': operation_name,
                    'function': func.__name__,
                    'error_type': type(e).__name__,
                    'error_message': str(e),
                    'timestamp': datetime.now().isoformat(),
                    'args_count': len(args),
                    'kwargs_count': len(kwargs)
                }
                
                logger.error(f"❌ {operation_name} hatası: {e}")
                logger.debug(f"Hata detayları: {error_details}")
                
                if raise_on_error:
                    raise
                
                return fallback_value
        return wrapper
    return decorator

def handle_timezone_error(func):
    """Timezone hatalarını yakalama decorator."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (TypeError, ValueError) as e:
            if any(keyword in str(e).lower() for keyword in ['timezone', 'tz', 'utc', 'naive', 'aware']):
                logger.warning(f"Timezone hatası yakalandı: {e}")
                # Timezone hatası durumunda güvenli fallback
                try:
                    # Basit pandas işlemi dene
                    return pd.Timestamp.now().normalize()
                except:
                    return datetime.now().date()
            else:
                raise
    return wrapper

def handle_data_error(func):
    """Veri hatalarını yakalama decorator."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = str(e).lower()
            
            # API limit hataları
            if any(keyword in error_msg for keyword in ['429', 'rate limit', 'too many requests']):
                logger.warning("API rate limit hatası - demo veri kullanılacak")
                raise DataFetchError("API rate limit aşıldı")
            
            # Network hataları
            elif any(keyword in error_msg for keyword in ['connection', 'network', 'timeout', 'dns']):
                logger.warning("Network hatası - demo veri kullanılacak")
                raise DataFetchError("Network bağlantı hatası")
            
            # Yahoo Finance spesifik hataları
            elif any(keyword in error_msg for keyword in ['yahoo', 'yfinance', 'json']):
                logger.warning("Yahoo Finance hatası - demo veri kullanılacak")
                raise DataFetchError("Yahoo Finance veri hatası")
            
            else:
                raise
    return wrapper

def log_system_info():
    """Sistem bilgilerini logla."""
    try:
        info = {
            'python_version': sys.version,
            'pandas_version': pd.__version__,
            'timestamp': datetime.now().isoformat(),
            'platform': sys.platform,
        }
        
        try:
            import yfinance
            info['yfinance_version'] = yfinance.__version__
        except:
            info['yfinance_version'] = 'Unknown'
        
        try:
            import flask
            info['flask_version'] = flask.__version__
        except:
            info['flask_version'] = 'Unknown'
        
        logger.info(f"Sistem bilgileri: {info}")
        
    except Exception as e:
        logger.error(f"Sistem bilgisi toplama hatası: {e}")

def create_error_response(error, operation="İşlem"):
    """Standardize edilmiş hata response'u oluştur."""
    error_data = {
        'success': False,
        'operation': operation,
        'error_type': type(error).__name__,
        'error_message': str(error),
        'timestamp': datetime.now().isoformat(),
        'suggestions': []
    }
    
    # Hata tipine göre öneriler ekle
    if isinstance(error, DataFetchError):
        error_data['suggestions'] = [
            "Biraz bekleyip tekrar deneyin",
            "İnternet bağlantınızı kontrol edin",
            "Farklı bir hisse senedi deneyin"
        ]
    elif isinstance(error, PredictionError):
        error_data['suggestions'] = [
            "Daha fazla tarihi veri gereken hisse senedi seçin",
            "Farklı tahmin süresi deneyin",
            "Sistem yöneticisine başvurun"
        ]
    elif isinstance(error, TimezoneError):
        error_data['suggestions'] = [
            "Sistem tarihi ayarlarını kontrol edin",
            "Tarayıcı önbelleğini temizleyin"
        ]
    else:
        error_data['suggestions'] = [
            "Sayfayı yenileyin",
            "Biraz bekleyip tekrar deneyin",
            "Sorun devam ederse sistem yöneticisine başvurun"
        ]
    
    return error_data

class ErrorLogger:
    """Gelişmiş hata loglama sınıfı."""
    
    @staticmethod
    def log_error(error, context=None, user_action=None):
        """Detaylı hata loglama."""
        error_record = {
            'timestamp': datetime.now().isoformat(),
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context or {},
            'user_action': user_action,
            'traceback': traceback.format_exc() if hasattr(error, '__traceback__') else None
        }
        
        logger.error(f"🔥 Detaylı hata kaydı: {error_record}")
        
        return error_record
    
    @staticmethod
    def log_recovery(recovery_action, success=True):
        """Hata düzeltme işlemlerini logla."""
        recovery_record = {
            'timestamp': datetime.now().isoformat(),
            'action': recovery_action,
            'success': success,
        }
        
        if success:
            logger.info(f"✅ Hata düzeltme başarılı: {recovery_record}")
        else:
            logger.warning(f"⚠️ Hata düzeltme başarısız: {recovery_record}")
        
        return recovery_record

# Uygulama başlangıcında sistem bilgilerini logla
def initialize_error_handling():
    """Error handling sistemini başlat."""
    logger.info("🛡️ Error handling sistemi başlatılıyor...")
    log_system_info()
    logger.info("✅ Error handling sistemi aktif") 