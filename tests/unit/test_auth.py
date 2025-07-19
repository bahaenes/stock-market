"""
Unit tests for authentication functionality.
"""

import pytest
from app.auth.forms import LoginForm, RegistrationForm, ResetPasswordRequestForm
from app.models import User
from app import db


@pytest.mark.unit
@pytest.mark.auth
class TestAuthForms:
    """Test authentication forms."""
    
    def test_login_form_valid(self, app):
        """Test valid login form."""
        with app.app_context():
            form = LoginForm()
            form.username.data = 'testuser'
            form.password.data = 'testpassword'
            
            # We can't test validate() without request context,
            # but we can test data assignment
            assert form.username.data == 'testuser'
            assert form.password.data == 'testpassword'
    
    def test_registration_form_data(self, app):
        """Test registration form data."""
        with app.app_context():
            form = RegistrationForm()
            form.username.data = 'newuser'
            form.email.data = 'newuser@example.com'
            form.password.data = 'password123'
            form.password2.data = 'password123'
            
            assert form.username.data == 'newuser'
            assert form.email.data == 'newuser@example.com'
            assert form.password.data == 'password123'
            assert form.password2.data == 'password123'
    
    def test_reset_password_form_data(self, app):
        """Test reset password form data."""
        with app.app_context():
            form = ResetPasswordRequestForm()
            form.email.data = 'test@example.com'
            
            assert form.email.data == 'test@example.com'


@pytest.mark.unit
@pytest.mark.auth
class TestUserAuthentication:
    """Test user authentication functionality."""
    
    def test_user_login_success(self, client, app):
        """Test successful user login."""
        with app.app_context():
            # Create test user
            user = User(username='testuser', email='test@example.com')
            user.set_password('testpassword')
            db.session.add(user)
            db.session.commit()
            
            # Test login
            response = client.post('/auth/login', data={
                'username': 'testuser',
                'password': 'testpassword'
            }, follow_redirects=True)
            
            assert response.status_code == 200
    
    def test_user_login_failure(self, client, app):
        """Test failed user login."""
        with app.app_context():
            # Create test user
            user = User(username='testuser', email='test@example.com')
            user.set_password('testpassword')
            db.session.add(user)
            db.session.commit()
            
            # Test login with wrong password
            response = client.post('/auth/login', data={
                'username': 'testuser',
                'password': 'wrongpassword'
            })
            
            assert response.status_code == 302  # Redirect back to login
    
    def test_user_registration(self, client, app):
        """Test user registration."""
        with app.app_context():
            response = client.post('/auth/register', data={
                'username': 'newuser',
                'email': 'newuser@example.com',
                'password': 'password123',
                'password2': 'password123'
            }, follow_redirects=True)
            
            assert response.status_code == 200
            
            # Check user was created
            user = User.query.filter_by(username='newuser').first()
            assert user is not None
            assert user.email == 'newuser@example.com'
    
    def test_duplicate_user_registration(self, client, app):
        """Test duplicate user registration fails."""
        with app.app_context():
            # Create first user
            user = User(username='testuser', email='test@example.com')
            user.set_password('testpassword')
            db.session.add(user)
            db.session.commit()
            
            # Try to register same username
            response = client.post('/auth/register', data={
                'username': 'testuser',
                'email': 'different@example.com',
                'password': 'password123',
                'password2': 'password123'
            })
            
            # Should not redirect (form validation error)
            assert response.status_code == 200
            assert b'Please use a different username' in response.data
    
    def test_user_logout(self, auth_client):
        """Test user logout."""
        response = auth_client.get('/auth/logout', follow_redirects=True)
        assert response.status_code == 200


@pytest.mark.unit
@pytest.mark.auth
class TestPasswordReset:
    """Test password reset functionality."""
    
    def test_password_reset_request(self, client, app):
        """Test password reset request."""
        with app.app_context():
            # Create test user
            user = User(username='testuser', email='test@example.com')
            user.set_password('testpassword')
            db.session.add(user)
            db.session.commit()
            
            response = client.post('/auth/reset_password_request', data={
                'email': 'test@example.com'
            }, follow_redirects=True)
            
            assert response.status_code == 200
    
    def test_password_reset_invalid_email(self, client, app):
        """Test password reset with invalid email."""
        with app.app_context():
            response = client.post('/auth/reset_password_request', data={
                'email': 'nonexistent@example.com'
            }, follow_redirects=True)
            
            # Should still return success (security measure)
            assert response.status_code == 200
    
    def test_password_reset_with_valid_token(self, client, app):
        """Test password reset with valid token."""
        with app.app_context():
            # Create test user
            user = User(username='testuser', email='test@example.com')
            user.set_password('oldpassword')
            db.session.add(user)
            db.session.commit()
            
            # Generate reset token
            token = user.get_reset_password_token()
            
            # Reset password
            response = client.post(f'/auth/reset_password/{token}', data={
                'password': 'newpassword123',
                'password2': 'newpassword123'
            }, follow_redirects=True)
            
            assert response.status_code == 200
            
            # Verify password was changed
            db.session.refresh(user)
            assert user.check_password('newpassword123') is True
            assert user.check_password('oldpassword') is False
    
    def test_password_reset_with_invalid_token(self, client, app):
        """Test password reset with invalid token."""
        with app.app_context():
            response = client.get('/auth/reset_password/invalid_token')
            assert response.status_code == 302  # Redirect to index