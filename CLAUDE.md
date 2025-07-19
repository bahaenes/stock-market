# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
```bash
python run.py                    # Start Flask development server on localhost:5000
./start.sh                      # Automated startup script (Linux/macOS)
start.bat                       # Automated startup script (Windows)
```

### Testing
```bash
python final_test.py            # Comprehensive system tests
python test_timezone_fixes.py   # Timezone-specific tests
python test_improved_models.py  # ML model tests
python test_updated_predictions.py  # Prediction system tests
python quick_test.py            # Quick functionality tests
```

### Environment Setup
```bash
python -m venv venv             # Create virtual environment
source venv/bin/activate        # Activate (Linux/macOS)
venv\Scripts\activate           # Activate (Windows)
pip install -r requirements.txt # Install dependencies
pip install -r requirements-py313.txt  # Python 3.13 optimized dependencies
```

## Application Architecture

### Core Flask Structure
- **app/__init__.py**: Flask app factory with configuration management
- **run.py**: Main entry point with database initialization and FinBERT setup
- **config.py**: Multi-environment configuration (development/production/testing)

### Service Layer Architecture
- **app/services/prediction_service.py**: ML prediction engine with ensemble modeling (LightGBM, Prophet, RandomForest)
- **app/services/stock_service.py**: Yahoo Finance data fetching with rate limiting
- **app/services/news_service.py**: FinBERT sentiment analysis with VADER fallback
- **app/services/chart_service.py**: Plotly chart generation

### Data Models (SQLAlchemy)
- **Stock**: Ticker symbols, market info, relationships
- **User**: Authentication and user management  
- **Portfolio**: User stock holdings
- **Watchlist**: User stock tracking
- **Analysis**: Historical analysis storage
- **Alert**: Price alert system

### ML Model Configuration
The prediction system uses ensemble modeling with configurable weights:
- **LightGBM**: Primary gradient boosting model (40% weight)
- **Prophet**: Time series forecasting (40% weight) 
- **RandomForest**: Fallback ensemble member (20% weight)

Models automatically fallback when dependencies are unavailable, with comprehensive error handling.

### Error Handling System
- **app/utils/error_handler.py**: Centralized error handling with safe execution decorators
- **Graceful Degradation**: Demo data generation when APIs fail
- **Timezone Safety**: All datetime operations use timezone-aware utilities

### Configuration Environments
- **Development**: Debug enabled, FinBERT disabled, higher API limits
- **Production**: Optimized for deployment, PostgreSQL support
- **Testing**: In-memory SQLite, fast cache, demo data enabled

## Important Implementation Notes

### ML Model Dependencies
Models have optional imports with graceful fallbacks:
```python
# Check availability before using
if LIGHTGBM_AVAILABLE:
    result = predict_with_lightgbm(data)
```

### Timezone Handling
All datetime operations use safe utilities:
```python
from app.services.prediction_service import normalize_datetime, safe_datetime_diff
```

### Business Day Calculations
Predictions automatically account for weekends and use business day frequency for realistic forecasting.

### Caching Strategy
- 5-minute cache for development (CACHE_MAX_AGE_SECONDS = 300)
- In-memory caching with timestamp validation
- Model predictions cached per ticker/period combination

### Environment Variables
Required `.env` configuration:
```
NEWS_API_KEY=your_news_api_key
FLASK_ENV=development
SECRET_KEY=your_secret_key
```

### Database Initialization
The application automatically:
1. Creates database tables on startup
2. Initializes FinBERT model (if enabled)
3. Populates default Turkish and US stock symbols

### Supported Stock Markets
- **BIST (Turkish)**: AKBNK.IS, GARAN.IS, TUPRS.IS, BIMAS.IS, THYAO.IS
- **US Markets**: AAPL, GOOGL, MSFT, TSLA, AMZN, NVDA

The application is designed for extensibility with comprehensive error handling, making it safe to add new features or modify existing functionality.