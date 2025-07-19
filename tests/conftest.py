"""
Test configuration and fixtures.
"""

import pytest
import tempfile
import os
from app import create_app, db
from app.models import User, Stock
from config import TestingConfig


@pytest.fixture(scope='session')
def app():
    """Create and configure a test Flask application."""
    # Create a temporary database file
    db_fd, db_path = tempfile.mkstemp()
    
    # Configure test app
    test_config = TestingConfig()
    test_config.SQLALCHEMY_DATABASE_URI = f'sqlite:///{db_path}'
    test_config.WTF_CSRF_ENABLED = False  # Disable CSRF for testing
    
    app = create_app('testing')
    app.config.from_object(test_config)
    
    with app.app_context():
        db.create_all()
        
        # Create test data
        create_test_data()
        
        yield app
        
        # Cleanup
        db.drop_all()
        os.close(db_fd)
        os.unlink(db_path)


@pytest.fixture(scope='function')
def client(app):
    """Create a test client."""
    return app.test_client()


@pytest.fixture(scope='function')
def runner(app):
    """Create a test CLI runner."""
    return app.test_cli_runner()


@pytest.fixture
def auth_client(client):
    """Create a test client with authenticated user."""
    # Register and login a test user
    response = client.post('/auth/register', data={
        'username': 'testuser',
        'email': 'test@example.com',
        'password': 'testpassword123',
        'password2': 'testpassword123'
    })
    
    response = client.post('/auth/login', data={
        'username': 'testuser',
        'password': 'testpassword123'
    })
    
    return client


@pytest.fixture
def test_user():
    """Create a test user."""
    user = User(username='testuser', email='test@example.com')
    user.set_password('testpassword123')
    db.session.add(user)
    db.session.commit()
    return user


@pytest.fixture
def test_stock():
    """Create a test stock."""
    stock = Stock(
        ticker='AAPL',
        name='Apple Inc.',
        market='NASDAQ'
    )
    db.session.add(stock)
    db.session.commit()
    return stock


@pytest.fixture
def sample_stock_data():
    """Sample stock data for testing."""
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    # Generate 30 days of sample data
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=30),
        end=datetime.now(),
        freq='B'  # Business days only
    )
    
    # Generate realistic price data
    base_price = 150.0
    price_changes = np.random.normal(0, 0.02, len(dates))  # 2% daily volatility
    prices = [base_price]
    
    for change in price_changes[1:]:
        new_price = prices[-1] * (1 + change)
        prices.append(new_price)
    
    data = pd.DataFrame({
        'Open': [p * np.random.uniform(0.98, 1.02) for p in prices],
        'High': [p * np.random.uniform(1.01, 1.05) for p in prices],
        'Low': [p * np.random.uniform(0.95, 0.99) for p in prices],
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    }, index=dates)
    
    return data


def create_test_data():
    """Create initial test data."""
    # Create test stocks
    stocks = [
        Stock(ticker='AAPL', name='Apple Inc.', market='NASDAQ'),
        Stock(ticker='GOOGL', name='Alphabet Inc.', market='NASDAQ'),
        Stock(ticker='MSFT', name='Microsoft Corporation', market='NASDAQ'),
        Stock(ticker='AKBNK.IS', name='Akbank T.A.S.', market='BIST'),
        Stock(ticker='GARAN.IS', name='Türkiye Garanti Bankası A.Ş.', market='BIST'),
    ]
    
    for stock in stocks:
        db.session.add(stock)
    
    # Create test admin user
    admin = User(username='admin', email='admin@test.com')
    admin.set_password('admin123')
    db.session.add(admin)
    
    db.session.commit()


@pytest.fixture
def mock_yfinance_data(monkeypatch):
    """Mock yfinance data for testing."""
    import pandas as pd
    import numpy as np
    
    class MockTicker:
        def __init__(self, symbol):
            self.symbol = symbol
        
        def history(self, period="1mo", interval="1d"):
            """Return mock historical data."""
            dates = pd.date_range(
                start="2024-01-01",
                end="2024-01-31",
                freq='B'
            )
            
            data = pd.DataFrame({
                'Open': np.random.uniform(100, 200, len(dates)),
                'High': np.random.uniform(100, 200, len(dates)),
                'Low': np.random.uniform(100, 200, len(dates)),
                'Close': np.random.uniform(100, 200, len(dates)),
                'Volume': np.random.randint(1000000, 10000000, len(dates))
            }, index=dates)
            
            return data
        
        @property
        def info(self):
            """Return mock stock info."""
            return {
                'symbol': self.symbol,
                'shortName': f'Mock {self.symbol}',
                'longName': f'Mock {self.symbol} Corporation',
                'currentPrice': 150.0,
                'marketCap': 2000000000000,
                'sector': 'Technology',
                'industry': 'Software'
            }
    
    def mock_ticker(symbol):
        return MockTicker(symbol)
    
    monkeypatch.setattr('yfinance.Ticker', mock_ticker)


@pytest.fixture
def mock_news_api(monkeypatch):
    """Mock news API responses."""
    def mock_get(*args, **kwargs):
        class MockResponse:
            def __init__(self):
                self.status_code = 200
            
            def json(self):
                return {
                    'status': 'ok',
                    'totalResults': 2,
                    'articles': [
                        {
                            'title': 'Test News Article 1',
                            'description': 'This is a positive test article',
                            'publishedAt': '2024-01-15T10:00:00Z',
                            'url': 'https://example.com/article1'
                        },
                        {
                            'title': 'Test News Article 2',
                            'description': 'This is a negative test article',
                            'publishedAt': '2024-01-15T11:00:00Z',
                            'url': 'https://example.com/article2'
                        }
                    ]
                }
        
        return MockResponse()
    
    monkeypatch.setattr('requests.get', mock_get)