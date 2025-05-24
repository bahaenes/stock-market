from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from config import config
import logging
import os

db = SQLAlchemy()

def create_app(config_name=None):
    app = Flask(__name__)
    
    # Configuration
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'development')
    
    app.config.from_object(config[config_name])
    
    # Logging yapılandırması
    configure_logging(app)
    
    # Database initialization
    db.init_app(app)
    
    # Error handling sistemi başlat
    try:
        from app.utils.error_handler import initialize_error_handling
        with app.app_context():
            initialize_error_handling()
    except Exception as e:
        app.logger.warning(f"Error handling sistemi başlatılamadı: {e}")
    
    # Register Blueprints
    from app.main import bp as main_bp
    app.register_blueprint(main_bp)
    
    # Global error handlers
    register_error_handlers(app)
    
    # Database tables oluştur
    with app.app_context():
        try:
            db.create_all()
            app.logger.info("✅ Database tabloları oluşturuldu")
        except Exception as e:
            app.logger.error(f"❌ Database tablo oluşturma hatası: {e}")
    
    app.logger.info("🚀 Flask uygulaması başarıyla başlatıldı")
    
    return app

def configure_logging(app):
    """Logging yapılandırması."""
    log_level = getattr(app.config, 'LOG_LEVEL', 'INFO')
    
    # Log formatı
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level))
    console_handler.setFormatter(formatter)
    
    # File handler (opsiyonel)
    try:
        if not os.path.exists('logs'):
            os.makedirs('logs')
        
        file_handler = logging.FileHandler('logs/app.log')
        file_handler.setLevel(logging.WARNING)
        file_handler.setFormatter(formatter)
        
        app.logger.addHandler(file_handler)
    except Exception as e:
        print(f"File logging kurulamadı: {e}")
    
    app.logger.addHandler(console_handler)
    app.logger.setLevel(getattr(logging, log_level))
    
    # Diğer loggerları da yapılandır
    for logger_name in ['app.services', 'app.main', 'app.utils']:
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, log_level))
        logger.addHandler(console_handler)

def register_error_handlers(app):
    """Global hata işleyicilerini kaydet."""
    
    @app.errorhandler(404)
    def page_not_found(error):
        app.logger.warning(f"404 hatası: {error}")
        return "Sayfa bulunamadı", 404
    
    @app.errorhandler(500)
    def internal_error(error):
        app.logger.error(f"500 hatası: {error}")
        db.session.rollback()
        return "İç sunucu hatası", 500
    
    @app.errorhandler(Exception)
    def handle_unexpected_error(error):
        """Beklenmeyen hataları yakala."""
        try:
            from app.utils.error_handler import ErrorLogger, create_error_response
            
            # Hata kaydı
            ErrorLogger.log_error(
                error, 
                context={'url': getattr(app, 'current_request', {}).get('url', 'unknown')},
                user_action='request'
            )
            
            # Güvenli response oluştur
            error_response = create_error_response(error, "Web isteği")
            
            app.logger.error(f"🚨 Beklenmeyen hata: {error_response}")
            
            # Production'da basit mesaj, development'da detay
            if app.config.get('DEBUG', False):
                return f"Hata detayları: {error_response}", 500
            else:
                return "Bir hata oluştu. Lütfen tekrar deneyin.", 500
                
        except Exception as handler_error:
            app.logger.critical(f"💥 Error handler hatası: {handler_error}")
            return "Kritik sistem hatası", 500 