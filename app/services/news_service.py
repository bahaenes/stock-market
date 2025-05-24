import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime, timedelta
from flask import current_app
import logging

# FinBERT için güvenli import
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    import torch.nn.functional as F
    TRANSFORMERS_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("Transformers kütüphanesi başarıyla yüklendi")
except ImportError as e:
    TRANSFORMERS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"Transformers kütüphanesi yüklenemedi: {e}")
except Exception as e:
    TRANSFORMERS_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning(f"Transformers import hatası: {e}")

# Global değişkenler
finbert_tokenizer = None
finbert_model = None
vader_analyzer = SentimentIntensityAnalyzer()

def initialize_finbert():
    """FinBERT modelini ve tokenizer'ını yükler (güvenli versiyon)."""
    global finbert_tokenizer, finbert_model
    
    if not TRANSFORMERS_AVAILABLE:
        logger.warning("Transformers kütüphanesi mevcut değil, VADER kullanılacak")
        return False
    
    if finbert_model is None and finbert_tokenizer is None:
        try:
            model_name = getattr(current_app.config, 'FINBERT_MODEL_NAME', 'ProsusAI/finbert')
            logger.info(f"FinBERT modeli yükleniyor: {model_name}")
            
            # Önce cache kontrol et
            try:
                finbert_tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir="./.model_cache",
                    local_files_only=False
                )
                finbert_model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    cache_dir="./.model_cache",
                    local_files_only=False
                )
            except Exception as cache_error:
                logger.warning(f"Model cache hatası, alternatif yüklenecek: {cache_error}")
                # Alternatif model veya basit sentiment analizi kullan
                finbert_tokenizer = None
                finbert_model = None
                return False
            
            if finbert_model:
                finbert_model.eval()
                logger.info("FinBERT modeli başarıyla yüklendi")
                return True
            
        except Exception as e:
            logger.error(f"FinBERT modeli yüklenirken kritik hata: {e}")
            finbert_tokenizer = None
            finbert_model = None
            return False
    
    return finbert_model is not None

def get_sentiment_finbert(text):
    """FinBERT ile duyarlılık analizi yap (güvenli versiyon)."""
    global finbert_tokenizer, finbert_model
    
    # FinBERT mevcut değilse direkt VADER kullan
    if not TRANSFORMERS_AVAILABLE or finbert_model is None or finbert_tokenizer is None:
        return get_sentiment_vader(text)
    
    try:
        # Metni temizle ve kısalt
        text = str(text).strip()
        if len(text) == 0:
            return "neutral", 0.0
        
        # Çok uzun metinleri kısalt
        if len(text) > 500:
            text = text[:500]
        
        # Metni tokenize et
        inputs = finbert_tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=512
        )
        
        # Model çıktısını al
        with torch.no_grad():
            outputs = finbert_model(**inputs)
            predictions = F.softmax(outputs.logits, dim=-1)
        
        # Sınıfları al: [negatif, nötr, pozitif]
        scores = predictions[0].tolist()
        negative_score = scores[0]
        neutral_score = scores[1]
        positive_score = scores[2]
        
        # Compound score hesapla (-1 ile +1 arası)
        compound_score = positive_score - negative_score
        
        # Kategori belirle
        if compound_score >= 0.05:
            category = "positive"
        elif compound_score <= -0.05:
            category = "negative"
        else:
            category = "neutral"
        
        return category, compound_score
        
    except Exception as e:
        logger.error(f"FinBERT analizi hatası: {e}")
        # Hata durumunda VADER'a geç
        return get_sentiment_vader(text)

def get_sentiment_vader(text):
    """VADER ile duyarlılık analizi yap."""
    try:
        scores = vader_analyzer.polarity_scores(text)
        compound = scores['compound']
        
        if compound >= 0.05:
            category = "positive"
        elif compound <= -0.05:
            category = "negative"
        else:
            category = "neutral"
        
        return category, compound
        
    except Exception as e:
        logger.error(f"VADER analizi hatası: {e}")
        return "neutral", 0.0

def get_news_data(query, days_back=7, page_size=10):
    """NewsAPI'den haber verilerini çek."""
    api_key = current_app.config.get('NEWS_API_KEY')
    
    if not api_key:
        logger.warning("NEWS_API_KEY bulunamadı")
        return []
    
    try:
        url = current_app.config['NEWS_API_URL']
        from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        
        params = {
            'q': query,
            'apiKey': api_key,
            'language': 'en',
            'pageSize': page_size,
            'sortBy': 'publishedAt',
            'from': from_date
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if data.get('status') == 'ok' and data.get('totalResults', 0) > 0:
            return data.get('articles', [])
        else:
            logger.info(f"{query} için haber bulunamadı")
            return []
            
    except requests.exceptions.RequestException as e:
        logger.error(f"NewsAPI isteği başarısız: {e}")
        return []
    except Exception as e:
        logger.error(f"Haber verisi işlenirken hata: {e}")
        return []

def analyze_news_sentiment(articles):
    """Haber listesinin duyarlılığını analiz et."""
    if not articles:
        return {
            'articles': [],
            'average_sentiment': 0.0,
            'sentiment_distribution': {'positive': 0, 'neutral': 0, 'negative': 0},
            'total_count': 0
        }
    
    analyzed_articles = []
    sentiment_scores = []
    sentiment_counts = {'positive': 0, 'neutral': 0, 'negative': 0}
    
    for article in articles:
        title = article.get('title', '')
        description = article.get('description', '')
        url = article.get('url', '')
        published_at = article.get('publishedAt', '')
        
        # Analiz için metni seç
        text_to_analyze = description if description else title
        
        if text_to_analyze and url:
            # Duyarlılık analizi yap
            category, score = get_sentiment_finbert(text_to_analyze)
            
            sentiment_scores.append(score)
            sentiment_counts[category] += 1
            
            analyzed_articles.append({
                'title': title,
                'description': description,
                'url': url,
                'published_at': published_at,
                'sentiment_category': category,
                'sentiment_score': score,
                'analyzed_text': text_to_analyze
            })
    
    # Ortalama duyarlılık hesapla
    average_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0
    
    return {
        'articles': analyzed_articles,
        'average_sentiment': average_sentiment,
        'sentiment_distribution': sentiment_counts,
        'total_count': len(analyzed_articles)
    }

def get_stock_news_analysis(stock_name, ticker, days_back=7):
    """Hisse senedi için haber analizi yap."""
    try:
        # Farklı arama terimleri dene
        search_queries = [stock_name, ticker.replace('.IS', '')]
        all_articles = []
        
        for query in search_queries:
            if query:
                articles = get_news_data(query, days_back=days_back)
                all_articles.extend(articles)
        
        # Tekrar eden haberleri temizle (URL'ye göre)
        seen_urls = set()
        unique_articles = []
        for article in all_articles:
            url = article.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_articles.append(article)
        
        # Duyarlılık analizi yap
        analysis_result = analyze_news_sentiment(unique_articles[:20])  # En fazla 20 haber
        
        logger.info(f"{ticker} için {len(analysis_result['articles'])} haber analiz edildi")
        
        return analysis_result
        
    except Exception as e:
        logger.error(f"Haber analizi hatası ({ticker}): {e}")
        return {
            'articles': [],
            'average_sentiment': 0.0,
            'sentiment_distribution': {'positive': 0, 'neutral': 0, 'negative': 0},
            'total_count': 0
        }

def get_market_sentiment():
    """Genel piyasa duyarlılığını al."""
    try:
        # Genel piyasa haberlerini çek
        market_terms = ["stock market", "financial markets", "economy", "inflation", "interest rates"]
        all_articles = []
        
        for term in market_terms:
            articles = get_news_data(term, days_back=3, page_size=5)
            all_articles.extend(articles)
        
        # Tekrar eden haberleri temizle
        seen_urls = set()
        unique_articles = []
        for article in all_articles:
            url = article.get('url', '')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_articles.append(article)
        
        # Duyarlılık analizi yap
        analysis_result = analyze_news_sentiment(unique_articles[:30])
        
        return {
            'sentiment_score': analysis_result['average_sentiment'],
            'article_count': analysis_result['total_count'],
            'distribution': analysis_result['sentiment_distribution']
        }
        
    except Exception as e:
        logger.error(f"Piyasa duyarlılığı analizi hatası: {e}")
        return {
            'sentiment_score': 0.0,
            'article_count': 0,
            'distribution': {'positive': 0, 'neutral': 0, 'negative': 0}
        } 