# Stock Market Analysis Tool - Turkey Financial Markets

A modern Flask web application for analyzing stock data, making price predictions using machine learning models, viewing technical indicators, and performing news sentiment analysis.

## Features

### Data Analysis
- Comprehensive stock list from BIST and US markets
- Real-time data via Yahoo Finance API
- Technical indicators: RSI, MACD, Bollinger Bands, moving averages

### Machine Learning Models
- LightGBM: High-performance gradient boosting predictions
- Prophet: Facebook's time series analysis model
- RandomForest: Ensemble learning method
- Ensemble Modeling: Combined predictions for improved reliability

### Sentiment Analysis
- FinBERT: Finance-specific BERT model for news analysis
- VADER Sentiment: Safe fallback analysis
- Current News: NewsAPI integration

### Reliability and Performance
- Comprehensive error handling with graceful fallbacks
- Timezone handling with global timezone support
- Caching for fast data access
- Business day calculations for realistic prediction dates

### User Interface
- Dynamic plotting with interactive Plotly charts
- Responsive mobile-friendly design
- Full Turkish language support
- Real-time data updates

## Installation

### Requirements
- Python 3.9+ (Python 3.13 tested)
- pip
- Internet connection

### 1. Clone the Repository
```bash
git clone https://github.com/username/stock-market.git
cd stock-market
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### 3. Install Dependencies
```bash
# Optimized for Python 3.13
pip install -r requirements.txt

# Alternative
pip install -r requirements-py313.txt
```

### 4. Set Environment Variables
Create a `.env` file:
```env
NEWS_API_KEY=your_news_api_key_here
FLASK_ENV=development
SECRET_KEY=your_secret_key_here
```

### 5. Start the Application
```bash
python run.py
```

## Usage

1. **Web Interface:** Navigate to `http://127.0.0.1:5000/`
2. **Stock Selection:** Choose a stock from the sidebar
3. **Analysis Period:** Select your desired time frame
4. **Prediction Duration:** Choose how many days to predict
5. **Analyze:** View comprehensive analysis

### Supported Stocks
- **BIST:** AKBNK.IS, GARAN.IS, TUPRS.IS, BIMAS.IS, THYAO.IS
- **US:** AAPL, GOOGL, MSFT, TSLA, AMZN, NVDA
- **And more...**

## Technologies

### Backend
- Flask: Web framework
- SQLAlchemy: Database ORM
- Pandas: Data manipulation
- NumPy: Numerical computations

### Machine Learning
- LightGBM: Microsoft's gradient boosting library
- Prophet: Facebook's time series library
- Scikit-learn: RandomForest and other ML tools
- Transformers: Hugging Face FinBERT model

### Data Sources
- yfinance: Yahoo Finance API
- NewsAPI: Current news data
- ta: Technical analysis indicators

### Frontend
- Plotly: Interactive charts
- Bootstrap: Responsive UI
- JavaScript: Dynamic content

## Configuration

### Model Settings
```python
# In config.py
PREDICTION_MODELS = ['lightgbm', 'prophet', 'randomforest']
ENSEMBLE_WEIGHTS = {'lightgbm': 0.4, 'prophet': 0.4, 'randomforest': 0.2}
CACHE_MAX_AGE_SECONDS = 3600
```

### Error Handling
```python
SAFE_MODE = True  # Handle errors gracefully
DEBUG_MODE = False  # False for production
LOG_LEVEL = 'INFO'
```

## Testing

```bash
# Comprehensive system tests
python final_test.py

# Timezone tests
python test_timezone_fixes.py

# Model tests
python test_improved_models.py
```

## Model Performance

| Model | Accuracy | Speed | Reliability |
|-------|----------|-------|-------------|
| LightGBM | High | High | Very High |
| Prophet | High | Medium | High |
| RandomForest | Medium | Very High | High |
| Ensemble | Very High | Medium | Very High |

## Troubleshooting

### Common Issues

1. **Timezone Error**
   ```bash
   # Test timezone functions
   python test_timezone_fixes.py
   ```

2. **Model Import Error**
   ```bash
   # Reinstall dependencies
   pip install --upgrade -r requirements.txt
   ```

3. **API Limit Error**
   ```
   # Demo mode automatically activates
   Demo data is used instead of real data
   ```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Changelog

### v2.0.0 (2025-05-24)
- Modern ML models (LightGBM, Prophet)
- Comprehensive error handling system
- Timezone-aware datetime operations
- Business day calculations
- Ensemble modeling
- FinBERT sentiment analysis

### v1.0.0 (2025-01-01)
- Initial release
- XGBoost model
- Basic technical analysis

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Developer

**bahaenes** - [GitHub Profile](https://github.com/bahaenes)

## Acknowledgments

- Yahoo Finance - Financial data API
- Hugging Face - FinBERT model
- Microsoft - LightGBM
- Facebook - Prophet
- NewsAPI - News data