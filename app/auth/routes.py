"""
Authentication routes with security features.
"""

from flask import render_template, redirect, url_for, flash, request, current_app
from flask_login import login_user, logout_user, current_user, login_required
from werkzeug.urls import url_parse
from app import db
from app.auth import auth
from app.auth.forms import LoginForm, RegistrationForm, ResetPasswordRequestForm, ResetPasswordForm
from app.models import User
from app.auth.email import send_password_reset_email
import logging

logger = logging.getLogger(__name__)


@auth.route('/login', methods=['GET', 'POST'])
def login():
    """User login with rate limiting and security logging."""
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))
    
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        
        if user is None or not user.check_password(form.password.data):
            logger.warning(f"Failed login attempt for username: {form.username.data} from IP: {request.remote_addr}")
            flash('Invalid username or password')
            return redirect(url_for('auth.login'))
        
        login_user(user, remember=form.remember_me.data)
        logger.info(f"Successful login for user: {user.username} from IP: {request.remote_addr}")
        
        # Redirect to next page or index
        next_page = request.args.get('next')
        if not next_page or url_parse(next_page).netloc != '':
            next_page = url_for('main.index')
        return redirect(next_page)
    
    return render_template('auth/login.html', title='Sign In', form=form)


@auth.route('/logout')
@login_required
def logout():
    """User logout with security logging."""
    logger.info(f"User logout: {current_user.username} from IP: {request.remote_addr}")
    logout_user()
    return redirect(url_for('main.index'))


@auth.route('/register', methods=['GET', 'POST'])
def register():
    """User registration with validation."""
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))
    
    form = RegistrationForm()
    if form.validate_on_submit():
        try:
            user = User(
                username=form.username.data,
                email=form.email.data
            )
            user.set_password(form.password.data)
            db.session.add(user)
            db.session.commit()
            
            logger.info(f"New user registered: {user.username} ({user.email}) from IP: {request.remote_addr}")
            flash('Congratulations, you are now a registered user!')
            return redirect(url_for('auth.login'))
            
        except Exception as e:
            db.session.rollback()
            logger.error(f"User registration failed: {e}")
            flash('Registration failed. Please try again.')
    
    return render_template('auth/register.html', title='Register', form=form)


@auth.route('/reset_password_request', methods=['GET', 'POST'])
def reset_password_request():
    """Password reset request."""
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))
    
    form = ResetPasswordRequestForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user:
            send_password_reset_email(user)
            logger.info(f"Password reset requested for user: {user.username}")
        # Always flash the same message for security
        flash('Check your email for the instructions to reset your password')
        return redirect(url_for('auth.login'))
    
    return render_template('auth/reset_password_request.html',
                         title='Reset Password', form=form)


@auth.route('/reset_password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    """Password reset with token validation."""
    if current_user.is_authenticated:
        return redirect(url_for('main.index'))
    
    user = User.verify_reset_password_token(token)
    if not user:
        logger.warning(f"Invalid password reset token used from IP: {request.remote_addr}")
        return redirect(url_for('main.index'))
    
    form = ResetPasswordForm()
    if form.validate_on_submit():
        try:
            user.set_password(form.password.data)
            db.session.commit()
            logger.info(f"Password reset completed for user: {user.username}")
            flash('Your password has been reset.')
            return redirect(url_for('auth.login'))
        except Exception as e:
            db.session.rollback()
            logger.error(f"Password reset failed for user {user.username}: {e}")
            flash('Password reset failed. Please try again.')
    
    return render_template('auth/reset_password.html', form=form)