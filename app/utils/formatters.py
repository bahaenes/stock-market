import pandas as pd
import numbers
from datetime import datetime, timedelta

def get_numeric_value(data_dict, key):
    """Sözlükten sayısal değer al."""
    value = data_dict.get(key)
    if value is None:
        return None
    
    if isinstance(value, pd.Series):
        if not value.empty:
            try:
                element = value.iloc[0]
                return float(element)
            except (ValueError, TypeError, IndexError):
                return None
        else:
            return None
    
    try:
        return float(value)
    except (ValueError, TypeError):
        return None

def format_indicator(value, format_spec=".2f"):
    """Teknik gösterge değerini formatla."""
    if value is not None:
        try:
            return f"{value:{format_spec}}"
        except (ValueError, TypeError):
            return "Veri yok"
    return "Veri yok"

def format_market_cap(value):
    """Piyasa değerini formatla."""
    if value is not None:
        try:
            if value >= 1e12:
                return f"{value / 1e12:.2f} Trilyon $"
            elif value >= 1e9:
                return f"{value / 1e9:.2f} Milyar $"
            elif value >= 1e6:
                return f"{value / 1e6:.2f} Milyon $"
            else:
                return f"{value:,.0f} $"
        except (ValueError, TypeError):
            return "Veri yok"
    return "Veri yok"

def format_volume(value):
    """İşlem hacmini formatla."""
    if value is not None:
        try:
            if value >= 1e9:
                return f"{value / 1e9:.2f} Milyar"
            elif value >= 1e6:
                return f"{value / 1e6:.2f} Milyon"
            elif value >= 1e3:
                return f"{value / 1e3:.2f} Bin"
            else:
                return f"{value:,.0f}"
        except (ValueError, TypeError):
            return "Veri yok"
    return "Veri yok"

def format_price(value, currency="₺"):
    """Fiyatı formatla."""
    if value is not None:
        try:
            return f"{value:.2f} {currency}"
        except (ValueError, TypeError):
            return "Veri yok"
    return "Veri yok"

def format_percentage(value, decimal_places=2):
    """Yüzde değerini formatla."""
    if value is not None:
        try:
            return f"{value:.{decimal_places}f}%"
        except (ValueError, TypeError):
            return "Veri yok"
    return "Veri yok"

def format_change(current, previous, format_type="percentage"):
    """Değişim oranını hesapla ve formatla."""
    if current is None or previous is None or previous == 0:
        return "Veri yok"
    
    try:
        change = current - previous
        if format_type == "percentage":
            change_pct = (change / previous) * 100
            sign = "+" if change_pct >= 0 else ""
            return f"{sign}{change_pct:.2f}%"
        elif format_type == "absolute":
            sign = "+" if change >= 0 else ""
            return f"{sign}{change:.2f}"
        else:
            return f"{change:.2f}"
    except (ValueError, TypeError, ZeroDivisionError):
        return "Veri yok"

def format_date(date_obj, format_str="%Y-%m-%d"):
    """Tarihi formatla."""
    if date_obj is None:
        return "Veri yok"
    
    try:
        if isinstance(date_obj, str):
            # Eğer string ise datetime'a çevir
            date_obj = pd.to_datetime(date_obj)
        
        if isinstance(date_obj, (datetime, pd.Timestamp)):
            return date_obj.strftime(format_str)
        else:
            return str(date_obj)
    except Exception:
        return "Veri yok"

def format_time_ago(date_obj):
    """Tarihten şimdiye kadar geçen süreyi formatla."""
    if date_obj is None:
        return "Veri yok"
    
    try:
        if isinstance(date_obj, str):
            date_obj = pd.to_datetime(date_obj)
        
        if isinstance(date_obj, (datetime, pd.Timestamp)):
            now = datetime.now()
            if date_obj.tzinfo:
                # Timezone aware ise naive'e çevir
                date_obj = date_obj.replace(tzinfo=None)
            
            diff = now - date_obj
            
            if diff.days > 0:
                return f"{diff.days} gün önce"
            elif diff.seconds >= 3600:
                hours = diff.seconds // 3600
                return f"{hours} saat önce"
            elif diff.seconds >= 60:
                minutes = diff.seconds // 60
                return f"{minutes} dakika önce"
            else:
                return "Az önce"
        else:
            return "Veri yok"
    except Exception:
        return "Veri yok"

def format_risk_level(volatility):
    """Risk seviyesini volatiliteye göre formatla."""
    if volatility is None:
        return "Belirlenemiyor"
    
    try:
        vol = float(volatility)
        if vol < 0.15:
            return "Düşük Risk"
        elif vol < 0.25:
            return "Orta Risk"
        elif vol < 0.35:
            return "Yüksek Risk"
        else:
            return "Çok Yüksek Risk"
    except (ValueError, TypeError):
        return "Belirlenemiyor"

def truncate_text(text, max_length=100, suffix="..."):
    """Metni belirtilen uzunlukta kes."""
    if text is None:
        return ""
    
    text = str(text)
    if len(text) <= max_length:
        return text
    else:
        return text[:max_length - len(suffix)] + suffix

def format_sentiment_label(sentiment_score, sentiment_category=None):
    """Duyarlılık skorunu etiket olarak formatla."""
    if sentiment_score is None:
        return "Belirsiz", "neutral"
    
    try:
        score = float(sentiment_score)
        
        if sentiment_category:
            if sentiment_category.lower() == "positive":
                return "Pozitif", "positive"
            elif sentiment_category.lower() == "negative":
                return "Negatif", "negative"
            else:
                return "Nötr", "neutral"
        else:
            # Skora göre belirle
            if score >= 0.1:
                return "Pozitif", "positive"
            elif score <= -0.1:
                return "Negatif", "negative"
            else:
                return "Nötr", "neutral"
    
    except (ValueError, TypeError):
        return "Belirsiz", "neutral"

def format_technical_signal(indicator_name, value, thresholds=None):
    """Teknik gösterge sinyalini formatla."""
    if value is None:
        return "Veri yok", "neutral"
    
    try:
        val = float(value)
        
        if indicator_name.upper() == "RSI":
            if val >= 70:
                return "Aşırı Alım", "negative"
            elif val <= 30:
                return "Aşırı Satım", "positive"
            else:
                return "Nötr", "neutral"
        
        elif indicator_name.upper() == "MACD":
            if val > 0:
                return "Al Sinyali", "positive"
            else:
                return "Sat Sinyali", "negative"
        
        elif "SMA" in indicator_name.upper():
            # Bu durumda fiyat ile karşılaştırma gerekli
            return f"{val:.2f}", "neutral"
        
        else:
            # Genel threshold kontrolü
            if thresholds:
                if val >= thresholds.get('overbought', float('inf')):
                    return "Aşırı Alım", "negative"
                elif val <= thresholds.get('oversold', float('-inf')):
                    return "Aşırı Satım", "positive"
                else:
                    return "Nötr", "neutral"
            else:
                return f"{val:.2f}", "neutral"
    
    except (ValueError, TypeError):
        return "Veri yok", "neutral" 