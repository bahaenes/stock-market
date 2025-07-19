from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
from flask import current_app
import jwt
from time import time
from app import db, login

class Stock(db.Model):
    """Hisse senedi modeli."""
    __tablename__ = 'stocks'
    
    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(20), unique=True, nullable=False, index=True)
    name = db.Column(db.String(100), nullable=False)
    market = db.Column(db.String(10), nullable=False)
    sector = db.Column(db.String(50))
    industry = db.Column(db.String(100))
    market_cap = db.Column(db.Float)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # İlişkiler
    portfolio_items = db.relationship('Portfolio', backref='stock', lazy='dynamic')
    watchlist_items = db.relationship('Watchlist', backref='stock', lazy='dynamic')
    analyses = db.relationship('Analysis', backref='stock', lazy='dynamic')
    
    def __repr__(self):
        return f'<Stock {self.ticker}: {self.name}>'

@login.user_loader
def load_user(id):
    """Flask-Login user loader."""
    return User.query.get(int(id))


class User(UserMixin, db.Model):
    """Kullanıcı modeli with authentication."""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(128))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_seen = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    
    # İlişkiler
    portfolios = db.relationship('Portfolio', backref='user', lazy='dynamic')
    watchlists = db.relationship('Watchlist', backref='user', lazy='dynamic')
    
    def __repr__(self):
        return f'<User {self.username}>'
    
    def set_password(self, password):
        """Set password hash."""
        self.password_hash = generate_password_hash(password)
    
    def check_password(self, password):
        """Check password hash."""
        return check_password_hash(self.password_hash, password)
    
    def get_reset_password_token(self, expires_in=600):
        """Generate password reset token."""
        return jwt.encode(
            {'reset_password': self.id, 'exp': time() + expires_in},
            current_app.config['SECRET_KEY'], 
            algorithm='HS256'
        )
    
    @staticmethod
    def verify_reset_password_token(token):
        """Verify password reset token."""
        try:
            id = jwt.decode(
                token, 
                current_app.config['SECRET_KEY'],
                algorithms=['HS256']
            )['reset_password']
        except:
            return
        return User.query.get(id)
    
    def to_dict(self):
        """Convert user to dictionary."""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'created_at': self.created_at.isoformat() + 'Z',
            'last_seen': self.last_seen.isoformat() + 'Z' if self.last_seen else None
        }

class Portfolio(db.Model):
    """Portföy modeli."""
    __tablename__ = 'portfolios'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    stock_id = db.Column(db.Integer, db.ForeignKey('stocks.id'), nullable=False)
    quantity = db.Column(db.Float, nullable=False)
    purchase_price = db.Column(db.Float, nullable=False)
    purchase_date = db.Column(db.Date, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Portfolio {self.user.username}: {self.stock.ticker}>'

class Watchlist(db.Model):
    """İzleme listesi modeli."""
    __tablename__ = 'watchlists'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    stock_id = db.Column(db.Integer, db.ForeignKey('stocks.id'), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Unique constraint
    __table_args__ = (db.UniqueConstraint('user_id', 'stock_id', name='unique_user_stock'),)
    
    def __repr__(self):
        return f'<Watchlist {self.user.username}: {self.stock.ticker}>'

class Analysis(db.Model):
    """Analiz geçmişi modeli."""
    __tablename__ = 'analyses'
    
    id = db.Column(db.Integer, primary_key=True)
    stock_id = db.Column(db.Integer, db.ForeignKey('stocks.id'), nullable=False)
    period = db.Column(db.String(10), nullable=False)
    chart_type = db.Column(db.String(20), nullable=False)
    
    # Teknik göstergeler
    rsi = db.Column(db.Float)
    macd = db.Column(db.Float)
    macd_signal = db.Column(db.Float)
    sma_20 = db.Column(db.Float)
    sma_50 = db.Column(db.Float)
    
    # Fiyat bilgileri
    current_price = db.Column(db.Float)
    price_change = db.Column(db.Float)
    
    # Duyarlılık analizi
    news_sentiment = db.Column(db.Float)
    news_count = db.Column(db.Integer)
    
    # Tahmin
    predicted_price = db.Column(db.Float)
    prediction_confidence = db.Column(db.Float)
    
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Analysis {self.stock.ticker}: {self.created_at}>'

class Alert(db.Model):
    """Fiyat uyarıları modeli."""
    __tablename__ = 'alerts'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    stock_id = db.Column(db.Integer, db.ForeignKey('stocks.id'), nullable=False)
    alert_type = db.Column(db.String(20), nullable=False)  # 'above', 'below', 'change'
    target_price = db.Column(db.Float)
    percentage_change = db.Column(db.Float)
    is_active = db.Column(db.Boolean, default=True)
    is_triggered = db.Column(db.Boolean, default=False)
    triggered_at = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    user = db.relationship('User', backref='alerts')
    stock = db.relationship('Stock', backref='alerts')
    
    def __repr__(self):
        return f'<Alert {self.user.username}: {self.stock.ticker}>' 