import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import yfinance as yf
import pandas as pd

# 1. Configuración de Activos
tickers_nyse = ['MELI', 'YPF', 'AAPL', 'NVDA', 'GOOGL', 'AMZN', 'TSLA', 'META']
tickers_byma = ['GGAL.BA', 'YPFD.BA', 'PAMP.BA', 'BMA.BA', 'TXAR.BA']
all_tickers = tickers_nyse + tickers_byma

def calcular_indicadores(df):
    """Calcula las métricas técnicas sobre el DataFrame"""
    # Tendencias (Medias Móviles)
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # RSI (14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD (12, 26, 9)
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Bandas de Bollinger (20, 2)
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['Std_20'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['SMA_20'] + (2 * df['Std_20'])
    df['BB_Lower'] = df['SMA_20'] - (2 * df['Std_20'])
    
    return df

def analizar_mercado(ticker):
    """Descarga data y desglosa las señales de cada indicador"""
    try:
        df = yf.download(ticker, period='1y', progress=False)
        if df.empty or len(df) < 200:
            return None
        
        df = calcular_indicadores(df)
        
        # Extracción de los últimos valores en formato escalar
        precio = float(df['Close'].iloc[-1])
        sma_200 = float(df['SMA_200'].iloc[-1])
        sma_50 = float(df['SMA_50'].iloc[-1])
        rsi = float(df['RSI'].iloc[-1])
        macd = float(df['MACD'].iloc[-1])
        macd_sig = float(df['MACD_Signal'].iloc[-1])
        bb_upper = float(df['BB_Upper'].iloc[-1])
        bb_lower = float(df['BB_Lower'].iloc[-1])
        
        # LÓGICA DE SEÑALES INDIVIDUALES
        
        # 1. Régimen Macro (SMA 200)
        regimen = "🟢 BULL" if precio > sma_200 else "🔴 BEAR"
        
        # 2. Señal RSI (14)
        if rsi < 30:
            s_rsi = "🛒 COMPRA (Sobrevendido)"
        elif rsi > 70:
            s_rsi = "💰 VENTA (Sobrecomprado)"
        else:
            s_rsi = f"⏳ Neutro ({rsi:.1f})"
            
        # 3. Señal MACD (Cruce de líneas)
        s_macd = "🚀 COMPRA (Cruce Alza)" if macd > macd_sig else "📉 VENTA (Cruce Baja)"
        
        # 4. Señal Bandas de Bollinger (Extremos de volatilidad)
        if precio <= bb_lower:
            s_bb = "🛒 COMPRA (Piso Banda)"
        elif precio >= bb_upper:
            s_bb = "💰 VENTA (Techo Banda)"
        else:
            s_bb = "⏳ Neutro (Dentro de Banda)"
            
        # 5. Señal SMA 50 (Tendencia Mediano Plazo)
        s_sma50 = "📈 COMPRA (Arriba SMA50)" if precio > sma_50 else "📉 VENTA (Abajo SMA50)"
        
        # Contexto de Zona (Basado en la fuerza del RSI)
        zona = "⚖️ Trading (Lateral)" if 40 <= rsi <= 60 else "🔥 Impulso / Tendencia"

        return {
            "Activo": ticker.replace('.BA', ' (BYMA)') if '.BA' in ticker else f"{ticker} (NYSE)",
            "Precio": f"${precio:.2f}",
            "Régimen (200)": regimen,
            "Zona": zona,
            "Señal RSI": s_rsi,
            "Señal MACD": s_macd,
            "Señal Bollinger": s_bb,
            "Señal SMA 50": s_sma50
        }
    except Exception as e:
        print(f"Error procesando {ticker}: {e}")
        return None

# 2. Procesamiento en bucle
resultados = []
for t in all_tickers:
    res = analizar_mercado(t)
    if res:
        resultados.append(res)

df_reporte = pd.DataFrame(resultados)

# 3. Formateo del Mail en HTML
html_table = df_reporte.to_html(index=False, classes='table', border=0)

html_content = f"""
<html>
<head>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; color: #1e293b; }}
        h2 {{ color: #0f172a; border-bottom: 2px solid #3b82f6; padding-bottom: 8px; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 15px; font-size: 13px; }}
        th {{ background-color: #0f172a; color: #ffffff; padding: 10px; text-align: left; font-weight: 600; }}
        td {{ padding: 10px; border-bottom: 1px solid #e2e8f0; }}
        tr:nth-child(even) {{ background-color: #f8fafc; }}
        tr:hover {{ background-color: #f1f5f9; }}
    </style>
</head>
<body>
    <h2>📊 Reporte Técnico de Confluencia Horaria</h2>
    <p>Análisis individual por indicador para activos de NYSE y BYMA:</p>
    {html_table}
    <br>
    <p style="font-size: 11px; color: #94a3b8;">Generado automáticamente vía GitHub Actions.</p>
</body>
</html>
"""

# 4. Envío SMTP
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
EMAIL_TO = os.getenv("EMAIL_TO")

if all([SMTP_USER, SMTP_PASSWORD, EMAIL_TO]):
    msg = MIMEMultipart()
    msg['From'] = SMTP_USER
    msg['To'] = EMAIL_TO
    msg['Subject'] = "📈 Reporte Multi-Indicador: Señales Desglosadas"
    msg.attach(MIMEText(html_content, 'html'))
    
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)
        print("✉️ Reporte multi-indicador enviado correctamente.")
    except Exception as e:
        print(f"❌ Error al enviar el mail: {e}")
else:
    print("⚠️ Variables SMTP no detectadas. Muestra en consola:")
    print(df_reporte.to_string())
