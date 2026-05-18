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
    """Calcula las métricas técnicas fundamentales sobre el DataFrame"""
    # Medias Móviles de Tendencia
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # RSI (14 períodos)
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
    
    # Componentes para Bandas de Bollinger y Desviación Estándar (20 períodos)
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['Std_20'] = df['Close'].rolling(window=20).std()
    
    return df

def analizar_mercado(ticker):
    """Descarga datos, calcula señales e interpreta métricas con formato cuantitativo"""
    try:
        # Descargamos 1 año de data para garantizar la correcta carga de la SMA 200
        df = yf.download(ticker, period='1y', progress=False)
        if df.empty or len(df) < 200:
            return None
        
        df = calcular_indicadores(df)
        
        # Extracción de valores de la última fila disponible
        precio = float(df['Close'].iloc[-1])
        sma_200 = float(df['SMA_200'].iloc[-1])
        sma_50 = float(df['SMA_50'].iloc[-1])
        rsi = float(df['RSI'].iloc[-1])
        macd = float(df['MACD'].iloc[-1])
        macd_sig = float(df['MACD_Signal'].iloc[-1])
        sma_20 = float(df['SMA_20'].iloc[-1])
        std_20 = float(df['Std_20'].iloc[-1])
        
        # 1. Formateo de Activo y Moneda según su Mercado
        if '.BA' in ticker:
            moneda = f"${precio:.2f}"
            nombre_activo = ticker.replace('.BA', ' (BYMA)')
        else:
            moneda = f"U$D {precio:.2f}"
            nombre_activo = f"{ticker} (NYSE)"

        # 2. Régimen de Mercado (Macro Tendencia)
        regimen = "BULL" if precio > sma_200 else "BEAR"
        
        # 3. Estado de la Zona según RSI
        zona = "Trading" if 40 <= rsi <= 60 else "Tendencia"
        
        # --- PROCESAMIENTO CUANTITATIVO DE SEÑALES ---
        
        # Señal A: RSI
        if rsi < 30:
            s_rsi = f"({rsi:.1f}) COMPRA"
        elif rsi > 70:
            s_rsi = f"({rsi:.1f}) VENTA"
        else:
            s_rsi = f"({rsi:.1f}) NEUTRO"
            
        # Señal B: MACD (Filtro de Ruido / Neutralidad si las líneas comprimen a menos de 0.05% del precio)
        diff_macd = macd - macd_sig
        if abs(diff_macd) < (precio * 0.0005):
            s_macd = f"({diff_macd:+.2f}) NEUTRO"
        elif macd > macd_sig:
            s_macd = f"({diff_macd:+.2f}) COMPRA"
        else:
            s_macd = f"({diff_macd:+.2f}) VENTA"
        
        # Señal C: Bollinger Basado en Desviaciones Estándar (Z-Score)
        z_score = (precio - sma_20) / std_20 if std_20 != 0 else 0
        signo_de = "+" if z_score >= 0 else ""
        valor_de = f"{signo_de}{z_score:.1f}DE"
        
        if z_score <= -2.0:
            s_bb = f"({valor_de}) COMPRA"
        elif z_score >= 2.0:
            s_bb = f"({valor_de}) VENTA"
        else:
            s_bb = f"({valor_de}) NEUTRO"
            
        # Señal D: SMA 50 (Filtro de cercanía, Neutro si está a menos de 1% de la media)
        distancia_sma50 = ((precio - sma_50) / sma_50) * 100
        if abs(distancia_sma50) < 1.0:
            s_sma50 = f"({distancia_sma50:+.1f}%) NEUTRO"
        elif precio > sma_50:
            s_sma50 = f"({distancia_sma50:+.1f}%) COMPRA"
        else:
            s_sma50 = f"({distancia_sma50:+.1f}%) VENTA"

        return [nombre_activo, moneda, regimen, zona, s_rsi, s_macd, s_bb, s_sma50]
    except Exception as e:
        print(f"Error procesando el activo {ticker}: {e}")
        return None

# 2. Ejecución del Analizador en Bucle
columnas = ["Activo", "Precio", "Régimen (200)", "Zona", "RSI", "MACD", "Bollinger", "SMA 50"]
resultados = []
for t in all_tickers:
    res = analizar_mercado(t)
    if res:
        resultados.append(res)

# 3. Construcción Dinámica de la Tabla HTML con Clases CSS Condicionales
html_rows = ""
for fila in resultados:
    html_rows += "<tr>"
    for i, celda in enumerate(fila):
        clase_css = ""
        texto_upper = str(celda).upper()
        
        # Aplicamos colores desde la columna Régimen (2) en adelante
        if i >= 2: 
            if "COMPRA" in texto_upper or "BULL" in texto_upper:
                clase_css = ' class="bg-verde"'
            elif "VENTA" in texto_upper or "BEAR" in texto_upper:
                clase_css = ' class="bg-rojo"'
            elif "NEUTRO" in texto_upper:
                clase_css = ' class="bg-amarillo"'
                
        html_rows += f"<td{clase_css}>{celda}</td>"
    html_rows += "</tr>"

# 4. Plantilla de Diseño Estilizada (Mapa de Calor Pastel)
html_content = f"""
<html>
<head>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; color: #1e293b; }}
        h2 {{ color: #0f172a; border-bottom: 2px solid #3b82f6; padding-bottom: 8px; }}
        table {{ border-collapse: collapse; width: 100%; margin-top: 15px; font-size: 13px; }}
        th {{ background-color: #0f172a; color: #ffffff; padding: 10px; text-align: left; font-weight: 600; }}
        td {{ padding: 10px; border-bottom: 1px solid #e2e8f0; font-weight: 500; color: #334155; }}
        tr:hover {{ background-color: #f1f5f9; }}
        
        /* Paleta Semántica de Alertas (Sin emojis) */
        .bg-verde {{ background-color: #d1fae5 !important; color: #065f46 !important; }}
        .bg-rojo {{ background-color: #fee2e2 !important; color: #991b1b !important; }}
        .bg-amarillo {{ background-color: #fef9c3 !important; color: #854d0e !important; }}
    </style>
</head>
<body>
    <h2>Reporte Cuantitativo de Mercado</h2>
    <table>
        <thead>
            <tr>
                {"".join([f"<th>{c}</th>" for c in columnas])}
            </tr>
        </thead>
        <tbody>
            {html_rows}
        </tbody>
    </table>
    <br>
    <p style="font-size: 11px; color: #94a3b8;">Reporte procesado automáticamente de madrugada vía GitHub Actions.</p>
</body>
</html>
"""

# 5. Despliegue de Envío a través del Servidor SMTP
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
EMAIL_TO = os.getenv("EMAIL_TO")

if all([SMTP_USER, SMTP_PASSWORD, EMAIL_TO]):
    msg = MIMEMultipart()
    msg['From'] = SMTP_USER
    msg['To'] = EMAIL_TO
    msg['Subject'] = "Reporte de Confluencia"
    msg.attach(MIMEText(html_content, 'html'))
    
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)
        print("✉️ Reporte optimizado enviado con éxito.")
    except Exception as e:
        print(f"❌ Error crítico en el despacho del correo: {e}")
else:
    print("⚠️ Faltan configurar variables de entorno SMTP en los Secrets de tu GitHub.")
