import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Base configuration."""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-key-please-change-in-production')
    NEWS_API_KEY = os.environ.get('NEWS_API_KEY')
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'sqlite:///finans_analiz.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Security settings
    WTF_CSRF_TIME_LIMIT = 3600  # 1 hour
    FORCE_HTTPS = os.environ.get('FORCE_HTTPS', 'false').lower() == 'true'
    SESSION_COOKIE_SECURE = FORCE_HTTPS
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = 'Lax'
    PERMANENT_SESSION_LIFETIME = 86400  # 24 hours
    
    # Mail settings
    MAIL_SERVER = os.environ.get('MAIL_SERVER', 'localhost')
    MAIL_PORT = int(os.environ.get('MAIL_PORT', 587))
    MAIL_USE_TLS = os.environ.get('MAIL_USE_TLS', 'true').lower() == 'true'
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')
    MAIL_DEFAULT_SENDER = os.environ.get('MAIL_DEFAULT_SENDER', 'noreply@stockanalysis.com')
    
    # Cache settings
    CACHE_MAX_AGE_SECONDS = 300  # 5 dakika
    
    # Model settings - güvenli konfigürasyon
    FINBERT_MODEL_NAME = os.environ.get('FINBERT_MODEL_NAME', 'ProsusAI/finbert')
    FINBERT_ENABLED = os.environ.get('FINBERT_ENABLED', 'false').lower() == 'true'
    MODEL_CACHE_DIR = './.model_cache'
    
    # Sentiment analiz ayarları
    USE_FINBERT = True  # FinBERT kullanmayı dene
    FALLBACK_TO_VADER = True  # Başarısız olursa VADER kullan
    
    # API settings
    NEWS_API_URL = 'https://newsapi.org/v2/everything'
    
    # Prediction settings - gelişmiş model ayarları
    FUTURE_PERIODS = 7  # Varsayılan tahmin günü
    PREDICTION_MODELS = ['lightgbm', 'prophet', 'random_forest']  # Kullanılacak modeller
    
    # ML Model ayarları
    LIGHTGBM_PARAMS = {
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
    
    # Rate limiting
    YFINANCE_RATE_LIMIT = 10  # Dakikada maksimum istek
    YFINANCE_MIN_DELAY = 1.0  # İstekler arası minimum bekleme (saniye)
    YFINANCE_MAX_DELAY = 3.0  # İstekler arası maksimum bekleme (saniye)
    
    # Error handling
    ENABLE_DEMO_DATA = True  # API hatası durumunda demo veri oluştur
    DEMO_DATA_SEED = 42  # Tutarlı demo veri için seed
    
    # Logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = os.environ.get('DEV_DATABASE_URL', 'sqlite:///finans_analiz_dev.db')
    
    # Development için daha toleranslı ayarlar
    FINBERT_ENABLED = False  # Development'da FinBERT'i devre dışı bırak
    USE_FINBERT = False
    YFINANCE_RATE_LIMIT = 20  # Development için daha yüksek limit
    LOG_LEVEL = 'DEBUG'

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'postgresql://user:pass@localhost/finans_analiz')
    
    # Production için güvenli ayarlar
    FINBERT_ENABLED = os.environ.get('FINBERT_ENABLED', 'true').lower() == 'true'
    USE_FINBERT = True
    YFINANCE_RATE_LIMIT = 5  # Production için düşük limit
    LOG_LEVEL = 'INFO'
    
    # Production security
    FORCE_HTTPS = True
    SESSION_COOKIE_SECURE = True

class TestingConfig(Config):
    """Testing configuration."""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'
    
    # Testing için hızlı ayarlar
    FINBERT_ENABLED = False
    USE_FINBERT = False
    ENABLE_DEMO_DATA = True
    CACHE_MAX_AGE_SECONDS = 1  # Test için kısa cache

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
} 