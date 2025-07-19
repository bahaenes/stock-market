"""
Unit tests for database models.
"""

import pytest
from app.models import User, Stock, Portfolio, Watchlist, Analysis
from app import db
import jwt
from time import time


@pytest.mark.unit
class TestUser:
    """Test User model."""
    
    def test_user_creation(self, app):
        """Test user creation."""
        with app.app_context():
            user = User(username='testuser', email='test@example.com')
            assert user.username == 'testuser'
            assert user.email == 'test@example.com'
            assert user.password_hash is None
    
    def test_password_hashing(self, app):
        """Test password hashing and verification."""
        with app.app_context():
            user = User(username='testuser', email='test@example.com')
            user.set_password('testpassword')
            
            assert user.password_hash is not None
            assert user.password_hash != 'testpassword'
            assert user.check_password('testpassword') is True
            assert user.check_password('wrongpassword') is False
    
    def test_password_reset_token(self, app):
        """Test password reset token generation and verification."""
        with app.app_context():
            user = User(username='testuser', email='test@example.com')
            user.set_password('testpassword')
            db.session.add(user)
            db.session.commit()
            
            # Generate token
            token = user.get_reset_password_token()
            assert token is not None
            
            # Verify token
            verified_user = User.verify_reset_password_token(token)
            assert verified_user.id == user.id
            
            # Test invalid token
            invalid_user = User.verify_reset_password_token('invalid_token')
            assert invalid_user is None
    
    def test_user_to_dict(self, app):
        """Test user serialization."""
        with app.app_context():
            user = User(username='testuser', email='test@example.com')
            db.session.add(user)
            db.session.commit()
            
            user_dict = user.to_dict()
            assert user_dict['username'] == 'testuser'
            assert user_dict['email'] == 'test@example.com'
            assert 'id' in user_dict
            assert 'password_hash' not in user_dict


@pytest.mark.unit
class TestStock:
    """Test Stock model."""
    
    def test_stock_creation(self, app):
        """Test stock creation."""
        with app.app_context():
            stock = Stock(
                ticker='AAPL',
                name='Apple Inc.',
                market='NASDAQ'
            )
            
            assert stock.ticker == 'AAPL'
            assert stock.name == 'Apple Inc.'
            assert stock.market == 'NASDAQ'
    
    def test_stock_repr(self, app):
        """Test stock string representation."""
        with app.app_context():
            stock = Stock(
                ticker='AAPL',
                name='Apple Inc.',
                market='NASDAQ'
            )
            
            assert str(stock) == '<Stock AAPL: Apple Inc.>'


@pytest.mark.unit
class TestPortfolio:
    """Test Portfolio model."""
    
    def test_portfolio_creation(self, app, test_user, test_stock):
        """Test portfolio creation."""
        with app.app_context():
            portfolio = Portfolio(
                user_id=test_user.id,
                stock_id=test_stock.id,
                quantity=10.0,
                purchase_price=150.0
            )
            
            assert portfolio.quantity == 10.0
            assert portfolio.purchase_price == 150.0


@pytest.mark.unit
class TestWatchlist:
    """Test Watchlist model."""
    
    def test_watchlist_creation(self, app, test_user, test_stock):
        """Test watchlist creation."""
        with app.app_context():
            watchlist = Watchlist(
                user_id=test_user.id,
                stock_id=test_stock.id
            )
            
            db.session.add(watchlist)
            db.session.commit()
            
            assert watchlist.user_id == test_user.id
            assert watchlist.stock_id == test_stock.id
    
    def test_watchlist_unique_constraint(self, app, test_user, test_stock):
        """Test watchlist unique constraint."""
        with app.app_context():
            # Add first watchlist item
            watchlist1 = Watchlist(
                user_id=test_user.id,
                stock_id=test_stock.id
            )
            db.session.add(watchlist1)
            db.session.commit()
            
            # Try to add duplicate - should fail
            watchlist2 = Watchlist(
                user_id=test_user.id,
                stock_id=test_stock.id
            )
            db.session.add(watchlist2)
            
            with pytest.raises(Exception):
                db.session.commit()


@pytest.mark.unit
class TestAnalysis:
    """Test Analysis model."""
    
    def test_analysis_creation(self, app, test_stock):
        """Test analysis creation."""
        with app.app_context():
            analysis = Analysis(
                stock_id=test_stock.id,
                period='1mo',
                chart_type='candlestick',
                rsi=65.5,
                current_price=150.0,
                predicted_price=155.0,
                prediction_confidence=0.85
            )
            
            assert analysis.rsi == 65.5
            assert analysis.current_price == 150.0
            assert analysis.predicted_price == 155.0
            assert analysis.prediction_confidence == 0.85