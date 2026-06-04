"""Backtester experimental SMA Filter + ATR stop para acciones BYMA (sin dinero real).

Genera trades por activo, métricas agregadas, CSVs de trades y gráficos de timeline.
Uso mínimo: python trader.py
"""
import os
from math import floor
import argparse
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
import json

try:
    import plotly.offline as pyo
    import plotly.graph_objs as go
    from plotly.subplots import make_subplots
except Exception:
    pyo = None
    go = None
    make_subplots = None

# Configuración
TICKERS = [
    "ALUA","BBAR","BMA","BYMA","CEPU","COME","CRES","ECOG",
    "EDN","GGAL","LOMA","METR","PAMP","SUPV","TGNO4","TGSU2",
    "TRAN","TXAR","VALO","YPFD",
]

# Parámetros de la estrategia (caso base)
INITIAL_CAPITAL = 1_000_000.0
FIXED_FRACTION = 0.05  # 5% del capital disponible por trade para backtests individuales
MAX_POSITION_FRACTION = 0.10  # 10% del capital total disponible como límite de despliegue por trade
RISK_FRACTION = 0.01  # arriesgar 1% del capital total en la distancia al stop ATR
SLOW_SMA_LENGTH = 250
FAST_SMA_INDEX = 0.25
ATR_LENGTH = 20
ATR_STOP = 6

# Comisión de broker (Cocos Capital web/app personas humanas: 0.45% por transacción)
COMMISSION_RATE = 0.0045  # 0.45% por compra/venta
SLIPPAGE_RATE = 0.001  # 0.10% de slippage estático por orden

OUT_DIR = "trader_output"
os.makedirs(OUT_DIR, exist_ok=True)

def transaction_cost(value):
    return value * (COMMISSION_RATE + SLIPPAGE_RATE)


def download_yf(symbol, period="10y"):
    df = yf.download(symbol, period=period, progress=False)
    if df.empty:
        return None
    df.dropna(inplace=True)
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df.columns = df.columns.droplevel(1)
        except Exception:
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
    return df


def download_data(ticker, period="10y", fx_series=None):
    symbol = ticker + ".BA"
    df = download_yf(symbol, period=period)
    if df is None:
        return None
    if fx_series is not None:
        df = convert_to_usd(df, fx_series)
    return df


def download_all_data(tickers, fx_series, period="10y"):
    data = {}
    for t in tickers:
        df = download_data(t, period=period, fx_series=fx_series)
        if df is not None and not df.empty:
            data[t] = df
    return data


def prepare_ticker_signal_df(df, slow_len, fast_index, atr_len):
    fast_len = max(1, int(round(fast_index * slow_len)))
    df = compute_indicators(df, slow_len, fast_len, atr_len)
    df = df.copy()
    df["signal"] = ((df["SMA_fast"].shift(1) > df["SMA_slow"].shift(1)).astype(int))
    df["entry_atr"] = df["ATR"].shift(1)
    return df


def backtest_portfolio(ticker_data, initial_capital=INITIAL_CAPITAL,
                       slow_len=SLOW_SMA_LENGTH, fast_index=FAST_SMA_INDEX,
                       atr_len=ATR_LENGTH, atr_stop=ATR_STOP,
                       max_position_fraction=MAX_POSITION_FRACTION,
                       risk_fraction=RISK_FRACTION,
                       start_date=None, end_date=None):
    prepared = {}
    for ticker, df in ticker_data.items():
        if df is None or len(df) < slow_len + atr_len:
            continue
        prepared[ticker] = prepare_ticker_signal_df(df, slow_len, fast_index, atr_len)

    if not prepared:
        return pd.DataFrame(), pd.DataFrame()

    if start_date is not None:
        start_date = pd.Timestamp(start_date)
    if end_date is not None:
        end_date = pd.Timestamp(end_date)

    dates = sorted({date for df in prepared.values() for date in df.index
                    if (start_date is None or date >= start_date) and
                       (end_date is None or date <= end_date)})
    cash = initial_capital
    positions = {}
    trades = []
    equity_ts = []
    per_ticker_equity = {t: [] for t in prepared.keys()}

    def current_equity():
        return cash + sum(pos["shares"] * pos["last_close"] for pos in positions.values())

    for date in dates:
        # Exits
        for ticker in list(positions.keys()):
            pos = positions[ticker]
            df = prepared[ticker]
            if date not in df.index:
                continue
            row = df.loc[date]
            low_p = float(row["Low"])
            open_p = float(row["Open"])
            close_p = float(row["Close"])

            if pos["stop_level"] is not None and low_p <= pos["stop_level"]:
                exit_price = pos["stop_level"]
                pnl_gross = (exit_price - pos["entry_price"]) * pos["shares"]
                commission_entry = transaction_cost(pos["entry_price"] * pos["shares"])
                commission_exit = transaction_cost(exit_price * pos["shares"])
                pnl = pnl_gross - commission_entry - commission_exit
                cash += pos["shares"] * exit_price - commission_exit
                trades.append({
                    "ticker": ticker,
                    "entry_date": pos["entry_date"],
                    "entry_price": pos["entry_price"],
                    "exit_date": date,
                    "exit_price": exit_price,
                    "side": "LONG",
                    "shares": pos["shares"],
                    "pnl_gross": pnl_gross,
                    "commission_entry": commission_entry,
                    "commission_exit": commission_exit,
                    "pnl": pnl,
                })
                del positions[ticker]
            elif row["signal"] == 0:
                exit_price = open_p
                pnl_gross = (exit_price - pos["entry_price"]) * pos["shares"]
                commission_entry = transaction_cost(pos["entry_price"] * pos["shares"])
                commission_exit = transaction_cost(exit_price * pos["shares"])
                pnl = pnl_gross - commission_entry - commission_exit
                cash += pos["shares"] * exit_price - commission_exit
                trades.append({
                    "ticker": ticker,
                    "entry_date": pos["entry_date"],
                    "entry_price": pos["entry_price"],
                    "exit_date": date,
                    "exit_price": exit_price,
                    "side": "LONG",
                    "shares": pos["shares"],
                    "pnl_gross": pnl_gross,
                    "commission_entry": commission_entry,
                    "commission_exit": commission_exit,
                    "pnl": pnl,
                })
                del positions[ticker]
            else:
                pos["last_close"] = close_p

        # Entries
        for ticker, df in prepared.items():
            if ticker in positions or date not in df.index:
                continue
            row = df.loc[date]
            if row["signal"] != 1:
                continue
            open_p = float(row["Open"])
            atr_prev = row["entry_atr"]
            if open_p <= 0:
                continue

            risk_per_share = float(atr_prev) * atr_stop if not pd.isna(atr_prev) and atr_prev > 0 else open_p * 0.02
            max_deployment = cash * max_position_fraction
            total_equity = current_equity()
            risk_budget = total_equity * risk_fraction
            trade_shares = floor(risk_budget / risk_per_share) if risk_per_share > 0 else 0
            cap_shares = floor(max_deployment / open_p)
            shares = max(0, min(trade_shares, cap_shares))
            commission_entry = transaction_cost(shares * open_p)
            if shares <= 0 or cash < (shares * open_p + commission_entry):
                continue

            positions[ticker] = {
                "shares": shares,
                "entry_price": open_p,
                "entry_date": date,
                "stop_level": open_p - risk_per_share,
                "last_close": float(row["Close"]),
            }
            cash -= shares * open_p + commission_entry

        # Record per-ticker equities at this date
        for t in prepared.keys():
            if t in positions:
                pos = positions[t]
                per_ticker_equity[t].append({"date": date, "equity": pos["shares"] * pos["last_close"]})
            else:
                per_ticker_equity[t].append({"date": date, "equity": 0.0})

        equity = current_equity()
        equity_ts.append({"date": date, "equity": equity})

    # Liquidate remaining positions at last known close
    final_date = dates[-1] if dates else None
    for ticker, pos in list(positions.items()):
        exit_price = pos["last_close"]
        pnl_gross = (exit_price - pos["entry_price"]) * pos["shares"]
        commission_entry = transaction_cost(pos["entry_price"] * pos["shares"])
        commission_exit = transaction_cost(exit_price * pos["shares"])
        pnl = pnl_gross - commission_entry - commission_exit
        cash += pos["shares"] * exit_price - commission_exit
        trades.append({
            "ticker": ticker,
            "entry_date": pos["entry_date"],
            "entry_price": pos["entry_price"],
            "exit_date": final_date,
            "exit_price": exit_price,
            "side": "LONG",
            "shares": pos["shares"],
            "pnl_gross": pnl_gross,
            "commission_entry": commission_entry,
            "commission_exit": commission_exit,
            "pnl": pnl,
        })
        del positions[ticker]

        # After liquidation, set final per-ticker equity for this ticker to 0 at final_date
        if final_date is not None:
            for t in per_ticker_equity.keys():
                if t in positions:
                    # already handled above
                    pass
                else:
                    # append zero for final_date to keep lengths consistent
                    per_ticker_equity[t].append({"date": final_date, "equity": 0.0})

    if final_date is not None:
        equity_ts.append({"date": final_date, "equity": cash})

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_ts).drop_duplicates(subset=["date"]).set_index("date")

    # convert per_ticker_equity to DataFrames
    per_ticker_equity_dfs = {}
    for t, rows in per_ticker_equity.items():
        if not rows:
            per_ticker_equity_dfs[t] = pd.DataFrame(columns=["equity"]).astype(float)
            continue
        df_t = pd.DataFrame(rows).drop_duplicates(subset=["date"]).set_index("date").sort_index()
        per_ticker_equity_dfs[t] = df_t

    return trades_df, equity_df, per_ticker_equity_dfs


def portfolio_aggregate(fx_series):
    data = download_all_data(TICKERS, fx_series)
    if not data:
        return None
    trades_df, equity_df, per_ticker_equity = backtest_portfolio(data)
    if equity_df.empty:
        return None

    metrics = metrics_from_trades(trades_df, equity_df)
    metrics.update({
        "symbols": sorted(data.keys()),
        "total_start": INITIAL_CAPITAL,
        "total_end": equity_df["equity"].iloc[-1],
        "total_net_profit": metrics["net_profit"],
        "num_symbols": len(data),
    })
    return metrics


def run_oos_160_80(fx_series):
    data = download_all_data(TICKERS, fx_series, period="10y")
    if not data:
        raise RuntimeError("No se pudo descargar datos para OoST 2024-2026.")
    oos_start = pd.Timestamp("2024-01-01")
    oos_end = pd.Timestamp("2026-12-31")
    trades_df, equity_df, per_ticker_equity = backtest_portfolio(
        data,
        slow_len=160,
        fast_index=0.5,
        atr_len=ATR_LENGTH,
        atr_stop=ATR_STOP,
        max_position_fraction=MAX_POSITION_FRACTION,
        risk_fraction=RISK_FRACTION,
        start_date=oos_start,
        end_date=oos_end,
    )
    oos_dir = os.path.join(OUT_DIR, "oos_160_80")
    os.makedirs(oos_dir, exist_ok=True)
    trades_df.to_csv(os.path.join(oos_dir, "oos_160_80_trades.csv"), index=False)
    equity_df.to_csv(os.path.join(oos_dir, "oos_160_80_equity.csv"))
    metrics = metrics_from_trades(trades_df, equity_df)
    summary = {
        "start_date": oos_start.strftime("%Y-%m-%d"),
        "end_date": oos_end.strftime("%Y-%m-%d"),
        "symbols": ",".join(sorted(data.keys())),
        "net_profit": metrics["net_profit"],
        "cagr": metrics["cagr"],
        "sharpe": metrics["sharpe"],
        "max_drawdown": metrics["max_drawdown"],
        "pct_profitable": metrics["pct_profitable"],
        "avg_win_loss": metrics["avg_win_loss"],
        "profit_factor": metrics["profit_factor"],
        "final_equity": equity_df["equity"].iloc[-1] if not equity_df.empty else np.nan,
    }
    pd.DataFrame([summary]).to_csv(os.path.join(oos_dir, "oos_160_80_summary.csv"), index=False)
    return oos_dir, summary


def download_ccl(period="10y"):
    df_ba = download_yf("GGAL.BA", period=period)
    df_nyse = download_yf("GGAL", period=period)
    if df_ba is None or df_nyse is None:
        return None

    fx = pd.DataFrame({
        "ba": df_ba["Close"],
        "nyse": df_nyse["Close"]
    }).dropna()
    fx["dolar_ccl"] = (fx["ba"] / fx["nyse"]) * 10
    fx["dolar_ccl"] = fx["dolar_ccl"].replace([np.inf, -np.inf], np.nan).ffill().bfill()
    return fx["dolar_ccl"]


def convert_to_usd(df, fx_series):
    joined = df.join(fx_series.rename("dolar_ccl"), how="left")
    joined["dolar_ccl"] = joined["dolar_ccl"].ffill().bfill()
    for col in ["Open", "High", "Low", "Close", "Adj Close"]:
        if col in joined.columns:
            joined[col] = joined[col] / joined["dolar_ccl"]
    return joined.drop(columns=["dolar_ccl"])


def compute_indicators(df, slow_len, fast_len, atr_len):
    df = df.copy()
    df["SMA_slow"] = df["Close"].rolling(window=slow_len).mean()
    df["SMA_fast"] = df["Close"].rolling(window=fast_len).mean()

    # ATR
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(window=atr_len).mean()
    return df


def backtest(df, initial_capital=INITIAL_CAPITAL, fixed_fraction=FIXED_FRACTION,
             slow_len=SLOW_SMA_LENGTH, fast_index=FAST_SMA_INDEX,
             atr_len=ATR_LENGTH, atr_stop=ATR_STOP):
    fast_len = max(1, int(round(fast_index * slow_len)))
    df = compute_indicators(df, slow_len, fast_len, atr_len)

    equity = initial_capital
    cash = initial_capital
    position = 0
    entry_price = 0.0
    stop_level = None

    trades = []
    equity_ts = []

    for i in range(1, len(df)):
        date = df.index[i]
        prev = i - 1

        sma_fast_prev = df["SMA_fast"].iloc[prev]
        sma_slow_prev = df["SMA_slow"].iloc[prev]
        atr_prev = df["ATR"].iloc[prev]

        if np.isnan(sma_fast_prev) or np.isnan(sma_slow_prev):
            signal = 0
        else:
            signal = 1 if sma_fast_prev > sma_slow_prev else 0

        open_p = float(df["Open"].iloc[i])
        high_p = float(df["High"].iloc[i])
        low_p = float(df["Low"].iloc[i])
        close_p = float(df["Close"].iloc[i])

        if position > 0:
            if stop_level is not None and low_p <= stop_level:
                exit_price = stop_level
                exit_date = date
                pnl_gross = (exit_price - entry_price) * position
                commission_entry = transaction_cost(entry_price * position)
                commission_exit = transaction_cost(exit_price * position)
                pnl = pnl_gross - commission_entry - commission_exit
                cash += position * exit_price - commission_exit
                trades.append({
                    "entry_date": entry_date,
                    "entry_price": entry_price,
                    "exit_date": exit_date,
                    "exit_price": exit_price,
                    "side": "LONG",
                    "shares": position,
                    "pnl_gross": pnl_gross,
                    "commission_entry": commission_entry,
                    "commission_exit": commission_exit,
                    "pnl": pnl,
                })
                position = 0
                stop_level = None
            elif signal == 0:
                exit_price = open_p
                exit_date = date
                pnl_gross = (exit_price - entry_price) * position
                commission_entry = transaction_cost(entry_price * position)
                commission_exit = transaction_cost(exit_price * position)
                pnl = pnl_gross - commission_entry - commission_exit
                cash += position * exit_price - commission_exit
                trades.append({
                    "entry_date": entry_date,
                    "entry_price": entry_price,
                    "exit_date": exit_date,
                    "exit_price": exit_price,
                    "side": "LONG",
                    "shares": position,
                    "pnl_gross": pnl_gross,
                    "commission_entry": commission_entry,
                    "commission_exit": commission_exit,
                    "pnl": pnl,
                })
                position = 0
                stop_level = None

        if position == 0 and signal == 1:
            alloc = cash * fixed_fraction
            shares = floor(alloc / open_p) if open_p > 0 else 0
            if shares > 0:
                commission_entry = transaction_cost(shares * open_p)
                if cash >= (shares * open_p + commission_entry):
                    position = shares
                    entry_price = open_p
                    entry_date = date
                    cash -= shares * open_p + commission_entry
                    stop_level = entry_price - atr_prev * atr_stop if not np.isnan(atr_prev) else None

        if position > 0:
            equity = cash + position * close_p
        else:
            equity = cash

        equity_ts.append({"date": date, "equity": equity})

    if position > 0:
        exit_price = float(df["Close"].iloc[-1])
        exit_date = df.index[-1]
        pnl_gross = (exit_price - entry_price) * position
        commission_entry = transaction_cost(entry_price * position)
        commission_exit = transaction_cost(exit_price * position)
        pnl = pnl_gross - commission_entry - commission_exit
        cash += position * exit_price - commission_exit
        trades.append({
            "entry_date": entry_date,
            "entry_price": entry_price,
            "exit_date": exit_date,
            "exit_price": exit_price,
            "side": "LONG",
            "shares": position,
            "pnl_gross": pnl_gross,
            "commission_entry": commission_entry,
            "commission_exit": commission_exit,
            "pnl": pnl,
        })
        position = 0
        equity = cash
        equity_ts.append({"date": exit_date, "equity": equity})

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_ts).set_index("date")
    return trades_df, equity_df


def metrics_from_trades(trades_df, equity_df, initial_capital=INITIAL_CAPITAL):
    if trades_df.empty or equity_df.empty:
        return {
            "net_profit": 0.0,
            "cagr": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "pct_profitable": np.nan,
            "avg_win_loss": np.nan,
            "profit_factor": np.nan,
        }

    net_profit = trades_df["pnl"].sum()
    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] <= 0]
    pct_profitable = len(wins) / len(trades_df) * 100
    avg_win_loss = (wins["pnl"].mean() / (-losses["pnl"].mean())) if len(wins) and len(losses) else np.nan

    start_val = initial_capital
    end_val = equity_df["equity"].iloc[-1]
    days = (equity_df.index[-1] - equity_df.index[0]).days if len(equity_df) > 1 else 1
    years = max(days / 365.25, 1 / 365.25)
    if start_val > 0 and end_val > 0:
        with np.errstate(divide='ignore', invalid='ignore'):
            cagr = (end_val / start_val) ** (1 / years) - 1
    else:
        cagr = np.nan

    equity_series = equity_df["equity"].sort_index()
    returns = equity_series.pct_change().dropna()
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if len(returns) > 1 and returns.std() != 0 else np.nan

    cum = equity_series.cummax()
    drawdown = (equity_series - cum) / cum
    maxdd = drawdown.min() if len(drawdown) else np.nan

    profit_factor = wins["pnl"].sum() / (-losses["pnl"].sum()) if len(losses) and losses["pnl"].sum() != 0 else np.nan

    return {
        "net_profit": net_profit,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": maxdd,
        "pct_profitable": pct_profitable,
        "avg_win_loss": avg_win_loss,
        "profit_factor": profit_factor,
    }


def plot_equity_and_trades(equity_df, trades_df, ticker):
    fig, ax = plt.subplots(figsize=(12, 5))
    equity_df = equity_df.sort_index()
    ax.plot(equity_df.index, equity_df["equity"], label="Equity", color="#0b69a3")

    for _, r in trades_df.iterrows():
        entry = r["entry_date"]
        exitd = r["exit_date"]
        ax.axvline(entry, color="#16a34a" if r["side"] == "LONG" else "#dc2626", linestyle="--", alpha=0.6)
        ax.axvline(exitd, color="#15803d" if r["side"] == "LONG" else "#b91c1c", linestyle=":", alpha=0.6)

    ax.set_title(f"Equity Curve & Trades: {ticker}")
    ax.set_ylabel("Equity ($)")
    ax.legend()
    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"equity_{ticker}.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def generate_html_report(data_map, trades_df_portfolio, equity_df_portfolio, per_ticker_equity, fx_series, out_file):
    if pyo is None or go is None:
        raise RuntimeError("Plotly no está disponible. Instala 'plotly' para generar el reporte HTML.")

    # Prepare plotly divs per ticker
    ticker_divs = {}
    tabs_buttons = []

    for t, df in data_map.items():
        # SOLUCIÓN TRUCO: Asegurar orden cronológico para eliminar las líneas diagonales cruzadas
        df = df.sort_index()
        if hasattr(df.index, 'tz') and df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index = pd.to_datetime(df.index).normalize()

        df_ind = compute_indicators(df.copy(), SLOW_SMA_LENGTH, max(1, int(round(FAST_SMA_INDEX * SLOW_SMA_LENGTH))), ATR_LENGTH)
        
        # --- CAMBIO: Serie de tiempo lineal simple para el precio en lugar de Candlestick ---
        price_trace = go.Scatter(
            x=df.index, 
            y=df["Close"], 
            mode='lines', 
            name="Precio (Close)", 
            line=dict(color="#1f77b4", width=2)
        )
        
        # SMA band: fill area between SMA_fast and SMA_slow
        sma_slow = go.Scatter(x=df.index, y=df_ind["SMA_slow"], name="SMA_slow", line=dict(color="rgba(0,128,0,0.8)"))
        sma_fast = go.Scatter(x=df.index, y=df_ind["SMA_fast"], name="SMA_fast", line=dict(color="rgba(255,165,0,0.8)"), fill='tonexty', fillcolor='rgba(255,165,0,0.2)')

        # markers for trades
        trades_t = trades_df_portfolio[trades_df_portfolio["ticker"] == t] if not trades_df_portfolio.empty else pd.DataFrame()
        entry_markers = []
        exit_markers = []
        if not trades_t.empty:
            for _, r in trades_t.iterrows():
                ed = pd.Timestamp(r["entry_date"]) if not pd.isna(r["entry_date"]) else None
                xd = pd.Timestamp(r["exit_date"]) if not pd.isna(r["exit_date"]) else None
                if ed is not None and ed in df.index:
                    entry_markers.append(dict(x=ed, y=df.loc[ed, "Close"], text=f"Entry {r['shares']} @ {r['entry_price']}", mode='markers', marker=dict(color='blue', symbol='triangle-up', size=10)))
                if xd is not None and xd in df.index:
                    exit_markers.append(dict(x=xd, y=df.loc[xd, "Close"], text=f"Exit @ {r['exit_price']}", mode='markers', marker=dict(color='red', symbol='triangle-down', size=10)))

        marker_traces = []
        if entry_markers:
            marker_traces.append(go.Scatter(x=[m['x'] for m in entry_markers], y=[m['y'] for m in entry_markers], mode='markers', marker=dict(color='blue', symbol='triangle-up', size=10), name='Entries'))
        if exit_markers:
            marker_traces.append(go.Scatter(x=[m['x'] for m in exit_markers], y=[m['y'] for m in exit_markers], mode='markers', marker=dict(color='red', symbol='triangle-down', size=10), name='Exits'))

        # Per-ticker equity (area)
        eq_df = per_ticker_equity.get(t)
        eq_trace = None
        if eq_df is not None and not eq_df.empty:
            eq_df = eq_df.sort_index() # También ordenamos el histórico de equity por las dudas
            if hasattr(eq_df.index, 'tz') and eq_df.index.tz is not None:
                eq_df.index = eq_df.index.tz_localize(None)
            eq_trace = go.Scatter(x=eq_df.index, y=eq_df["equity"], name=f"Equity {t}", line=dict(color='purple'), fill='tozeroy', fillcolor='rgba(128,0,128,0.2)')

        # --- Subplots estilo TradingView ---
        fig = make_subplots(
            rows=2, 
            cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.06, 
            row_heights=[0.7, 0.3]
        )
        
        # Añadir traza lineal de precio e indicadores a la Fila 1
        fig.add_trace(price_trace, row=1, col=1)
        fig.add_trace(sma_slow, row=1, col=1)
        fig.add_trace(sma_fast, row=1, col=1)
        for mt in marker_traces:
            fig.add_trace(mt, row=1, col=1)
            
        # Añadir traza de equity a la Fila 2
        if eq_trace is not None:
            fig.add_trace(eq_trace, row=2, col=1)

        # Desactivar barras deslizadoras molestas
        fig.update_xaxes(rangeslider_visible=False, type='date')

        # Actualizar diseño general con hover unificado en el eje X
        fig.update_layout(
            height=750, 
            title_text=f"{t} - Precio USD, Medias y Equity",
            hovermode="x unified",
            xaxis_rangeslider_visible=False
        )

        div = pyo.plot(fig, output_type='div', include_plotlyjs=False)
        ticker_divs[t] = div
        tabs_buttons.append(t)

    # Portfolio composition: stack per-ticker equity shares proportion over portfolio equity
    comp_traces = []
    all_dates = sorted(set(equity_df_portfolio.index))
    total_series = equity_df_portfolio.reindex(all_dates).ffill().bfill()['equity']
    per_ticker_series = {}
    for t in data_map.keys():
        eq_df = per_ticker_equity.get(t)
        if eq_df is None or eq_df.empty:
            comp = [0.0] * len(all_dates)
        else:
            series = eq_df.reindex(all_dates).ffill().fillna(0.0)['equity']
            comp = series.tolist()
        per_ticker_series[t] = comp
        comp_traces.append(go.Scatter(x=all_dates, y=comp, stackgroup='one', name=t))

    # CASH series = total_series - sum(all tickers)
    summed = [0.0] * len(all_dates)
    for i in range(len(all_dates)):
        summed[i] = sum(per_ticker_series[t][i] for t in per_ticker_series.keys())
    cash_series = (total_series.values - np.array(summed)).tolist()
    comp_traces.insert(0, go.Scatter(x=all_dates, y=cash_series, stackgroup='one', name='CASH', fillcolor='rgba(128,128,128,0.3)'))

    comp_fig = go.Figure(data=comp_traces)
    comp_fig.update_layout(title='Composición de la cartera (valor por ticker)', height=500)
    comp_div = pyo.plot(comp_fig, output_type='div', include_plotlyjs=False)

    # Build HTML
    plotly_js = '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'
    html_parts = ["<html><head><meta charset='utf-8'><title>Portfolio Report</title>", plotly_js, "<style>body{font-family:Arial,Helvetica,sans-serif} .tabs{display:flex;flex-wrap:wrap;margin-bottom:10px} .tab-btn{margin:2px;padding:6px 10px;border:1px solid #ccc;cursor:pointer;background:#f7f7f7}</style></head><body>"]
    html_parts.append(f"<h2>Portfolio report</h2>")
    html_parts.append(f"<p>Total start: {INITIAL_CAPITAL:.2f} - Final equity: {equity_df_portfolio['equity'].iloc[-1]:.2f}</p>")

    # tabs
    html_parts.append('<div class="tabs">')
    for i, t in enumerate(tabs_buttons):
        html_parts.append(f'<div class="tab-btn" onclick="showTab(\'{t}\')">{t}</div>')
    html_parts.append('</div>')

    # content divs
    for t, div in ticker_divs.items():
        html_parts.append(f'<div id="{t}" class="tab-content" style="display:none">')
        html_parts.append(div)
        
        # add trade table for ticker
        trades_t = trades_df_portfolio[trades_df_portfolio['ticker'] == t] if not trades_df_portfolio.empty else pd.DataFrame()
        if not trades_t.empty:
            html_parts.append('<h4>Trades</h4>')
            html_parts.append('<table border="1" cellpadding="4"><tr><th>entry_date</th><th>entry_price</th><th>exit_date</th><th>exit_price</th><th>shares</th><th>pnl</th><th>SMA_fast(entry)</th><th>SMA_slow(entry)</th><th>ATR(entry)</th></tr>')
            df_ind = compute_indicators(data_map[t].copy(), SLOW_SMA_LENGTH, max(1, int(round(FAST_SMA_INDEX * SLOW_SMA_LENGTH))), ATR_LENGTH)
            for _, r in trades_t.iterrows():
                ed = pd.Timestamp(r['entry_date']) if not pd.isna(r['entry_date']) else None
                exd = pd.Timestamp(r['exit_date']) if not pd.isna(r['exit_date']) else None
                sma_f = df_ind.loc[ed, 'SMA_fast'] if ed is not None and ed in df_ind.index else ''
                sma_s = df_ind.loc[ed, 'SMA_slow'] if ed is not None and ed in df_ind.index else ''
                atr_v = df_ind.loc[ed, 'ATR'] if ed is not None and ed in df_ind.index else ''
                html_parts.append(f"<tr><td>{ed}</td><td>{r['entry_price']}</td><td>{exd}</td><td>{r['exit_price']}</td><td>{r['shares']}</td><td>{r['pnl']}</td><td>{sma_f}</td><td>{sma_s}</td><td>{atr_v}</td></tr>")
            html_parts.append('</table>')

        html_parts.append('</div>')

    # portfolio composition
    html_parts.append('<h3>Composición de cartera</h3>')
    html_parts.append(comp_div)

    # JS for tabs
    html_parts.append("<script>function showTab(t){document.querySelectorAll('.tab-content').forEach(d=>d.style.display='none');document.getElementById(t).style.display='block';} if(document.querySelectorAll('.tab-content').length>0){showTab(\'" + tabs_buttons[0] + "\');}</script>")
    html_parts.append('</body></html>')

    with open(out_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(html_parts))

    return out_file

def compute_benchmark_cagr(symbol, fx_series, period="10y"):
    df = download_yf(symbol, period=period)
    if df is None or df.empty:
        return None
    joined = df.join(fx_series.rename("dolar_ccl"), how="left")
    joined["dolar_ccl"] = joined["dolar_ccl"].ffill().bfill()
    joined["Close_USD"] = joined["Close"] / joined["dolar_ccl"]
    joined = joined.dropna(subset=["Close_USD"])
    if len(joined) < 2:
        return None
    start = joined["Close_USD"].iloc[0]
    end = joined["Close_USD"].iloc[-1]
    days = (joined.index[-1] - joined.index[0]).days
    years = max(days / 365.25, 1 / 365.25)
    if start <= 0:
        return None
    return (end / start) ** (1 / years) - 1


def main():
    parser = argparse.ArgumentParser(description="Trader backtester and sensitivity runner")
    parser.add_argument("--portfolio", action="store_true", help="Ejecutar backtest con capital único agregado")
    parser.add_argument("--oos", action="store_true", help="Ejecutar OoST 2024-2026 con SMA 160/80")
    parser.add_argument("--sensitivity", action="store_true", help="Ejecutar prueba de sensibilidad")
    parser.add_argument("--full", action="store_true", help="Usar malla completa (costosa)")
    parser.add_argument("--benchmark", action="store_true", help="Comparar el portfolio contra ^MERV USD")
    args = parser.parse_args()

    fx_series = download_ccl(period="10y")
    if fx_series is None or fx_series.empty:
        raise RuntimeError("No se pudo descargar la serie Dolar CCL GGAL. Revisar la conexión o los símbolos.")

    if args.portfolio:
        data = download_all_data(TICKERS, fx_series)
        trades_df, equity_df, per_ticker_equity = backtest_portfolio(data)
        if equity_df.empty:
            raise RuntimeError("No se pudo ejecutar el backtest de portafolio.")
        trades_df.to_csv(os.path.join(OUT_DIR, "portfolio_trades.csv"), index=False)
        equity_df.to_csv(os.path.join(OUT_DIR, "portfolio_equity.csv"))
        metrics = metrics_from_trades(trades_df, equity_df)
        metrics.update({
            "total_start": INITIAL_CAPITAL,
            "total_end": equity_df["equity"].iloc[-1],
            "symbols": ",".join(sorted(data.keys())),
            "num_symbols": len(data),
        })
        out_csv = os.path.join(OUT_DIR, "portfolio_summary.csv")
        pd.DataFrame([metrics]).to_csv(out_csv, index=False)
        # Generar reporte HTML interactivo
        try:
            report_path = os.path.join(OUT_DIR, "portfolio_report.html")
            generate_html_report(data, trades_df, equity_df, per_ticker_equity, fx_series, report_path)
            print("Reporte HTML generado en:", report_path)
        except Exception as e:
            print("No se pudo generar el reporte HTML:", e)
        print("Backtest de portafolio completado. Resultados guardados en:", out_csv)
        return

    if args.oos:
        oos_dir, summary = run_oos_160_80(fx_series)
        print("OoST 2024-2026 completado. Resultados guardados en:", oos_dir)
        print(summary)
        return

    if args.benchmark:
        benchmark = compute_benchmark_cagr("^MERV", fx_series, period="10y")
        portfolio = portfolio_aggregate(fx_series)
        if benchmark is None or portfolio is None:
            raise RuntimeError("No se pudo calcular el benchmark o el portfolio.")

        result = {
            "benchmark_symbol": "^MERV",
            "benchmark_cagr": benchmark,
            "portfolio_cagr": portfolio["cagr"],
            "portfolio_net_profit": portfolio["total_net_profit"],
            "portfolio_start": portfolio["total_start"],
            "portfolio_end": portfolio["total_end"],
            "symbols": ",".join(portfolio["symbols"]),
            "num_symbols": portfolio["num_symbols"],
            "cagr_diff": portfolio["cagr"] - benchmark,
        }
        out_path = os.path.join(OUT_DIR, "benchmark_merv.csv")
        pd.DataFrame([result]).to_csv(out_path, index=False)
        print("Benchmark completado. Resultados guardados en:", out_path)
        print(result)
        return

    if args.sensitivity:
        run_sensitivity(fx_series, full_grid=args.full)
        return

    summary = []
    for t in TICKERS:
        print(f"Procesando {t}...")
        df = download_data(t, period="10y", fx_series=fx_series)
        if df is None or len(df) < SLOW_SMA_LENGTH + ATR_LENGTH:
            print(f"  Datos insuficientes para {t}, se saltea.")
            continue

        trades_df, equity_df = backtest(df)
        metrics = metrics_from_trades(trades_df, equity_df)

        trades_path = os.path.join(OUT_DIR, f"trades_{t}.csv")
        trades_df.to_csv(trades_path, index=False)
        chart_path = plot_equity_and_trades(equity_df, trades_df, t)

        summary.append({"ticker": t, **metrics, "trades_file": trades_path, "chart": chart_path})

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(OUT_DIR, "summary.csv"), index=False)
    print("Backtest completado. Resultados en:", OUT_DIR)


def run_sensitivity(fx_series, full_grid=False):
    if fx_series is None or fx_series.empty:
        raise RuntimeError("No hay serie de Dólar CCL GGAL válida para la sensibilidad.")

    data_map = download_all_data(TICKERS, fx_series)
    if not data_map:
        raise RuntimeError("No se pudo descargar ningún dato para sensibilidad.")

    if full_grid:
        slow_values = list(range(60, 1001, 20))
        fast_indices = [round(x / 100, 2) for x in range(20, 99, 2)]
    else:
        slow_values = list(range(60, 301, 20))
        fast_indices = [round(x / 100, 2) for x in range(20, 51, 5)]

    sens_dir = os.path.join(OUT_DIR, "sensitivity")
    os.makedirs(sens_dir, exist_ok=True)

    rows = []
    best = None
    best_profit = -np.inf

    for slow in slow_values:
        for fi in fast_indices:
            print(f"Sensibilidad: slow={slow}, fast_index={fi}...")
            trades_df, equity_df, _ = backtest_portfolio(data_map, slow_len=slow, fast_index=fi)
            metrics = metrics_from_trades(trades_df, equity_df)
            if equity_df.empty:
                continue
            row = {
                "slow": slow,
                "fast_index": fi,
                "net_profit": metrics["net_profit"],
                "cagr": metrics["cagr"],
                "sharpe": metrics["sharpe"],
                "max_drawdown": metrics["max_drawdown"],
                "pct_profitable": metrics["pct_profitable"],
                "avg_win_loss": metrics["avg_win_loss"],
                "profit_factor": metrics["profit_factor"],
                "trades_count": len(trades_df),
                "final_equity": equity_df["equity"].iloc[-1],
            }
            rows.append(row)
            if metrics["net_profit"] > best_profit:
                best_profit = metrics["net_profit"]
                best = {"params": (slow, fi), "metrics": metrics, "trades_df": trades_df, "equity_df": equity_df}

    df_rows = pd.DataFrame(rows)
    out_csv = os.path.join(sens_dir, "sensitivity_portfolio.csv")
    df_rows.to_csv(out_csv, index=False)

    best_chart = None
    if best is not None:
        slow_b, fi_b = best["params"]
        best_dir = os.path.join(sens_dir, "best_portfolio")
        os.makedirs(best_dir, exist_ok=True)
        best_trades_path = os.path.join(best_dir, f"trades_best_portfolio_s{slow_b}_f{fi_b}.csv")
        best["trades_df"].to_csv(best_trades_path, index=False)
        best_chart = plot_equity_and_trades(best["equity_df"], best["trades_df"], f"portfolio_best_s{slow_b}_f{fi_b}")

    summary = {
        "best_slow": best["params"][0] if best else None,
        "best_fast_index": best["params"][1] if best else None,
        "best_net_profit": best["metrics"]["net_profit"] if best else None,
        "best_cagr": best["metrics"]["cagr"] if best else None,
        "best_trades_count": len(best["trades_df"]) if best else None,
        "sensitivity_csv": out_csv,
        "best_trades_file": best_trades_path if best else None,
        "best_chart": best_chart,
    }
    pd.DataFrame([summary]).to_csv(os.path.join(sens_dir, "sensitivity_summary.csv"), index=False)
    print("Prueba de sensibilidad completada. Resultados guardados en:", sens_dir)


if __name__ == "__main__":
    main()

