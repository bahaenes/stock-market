from datetime import datetime
from app import db

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

class User(db.Model):
    """Kullanıcı modeli."""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(128))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    last_seen = db.Column(db.DateTime, default=datetime.utcnow)
    
    # İlişkiler
    portfolios = db.relationship('Portfolio', backref='user', lazy='dynamic')
    watchlists = db.relationship('Watchlist', backref='user', lazy='dynamic')
    
    def __repr__(self):
        return f'<User {self.username}>'

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