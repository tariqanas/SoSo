#!/usr/bin/env python3
import os
import time
import json
import math
import ccxt
import pandas as pd
import numpy as np
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from scipy.signal import argrelextrema
from dotenv import load_dotenv
import requests

load_dotenv()

# === CONFIG (env) ===
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')

# Optional: Telegram for signals (leave empty to disable)
TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')

# Symbol/timeframe
SYMBOL = os.getenv('SYMBOL', 'BNB/USDC')
TIMEFRAME = os.getenv('TIMEFRAME', '1d')
HISTORY_LIMIT = int(os.getenv('HISTORY_LIMIT', '200'))  # how many candles to fetch

# Strategy parameters (mirror du backtest)
PORTFOLIO_SIZE = float(os.getenv('PORTFOLIO_SIZE', '10000'))
RISK_PER_TRADE = float(os.getenv('RISK_PER_TRADE', '0.07'))
BUFFER_SIZE = int(os.getenv('BUFFER_SIZE', '100'))
TRAILING_ATR_MULT = float(os.getenv('TRAILING_ATR_MULT', '2.0'))
FIXED_STOPLOSS = float(os.getenv('FIXED_STOPLOSS', '0.04'))
RSI_OVERBOUGHT = float(os.getenv('RSI_OVERBOUGHT', '70'))
RSI_OVERSOLD = float(os.getenv('RSI_OVERSOLD', '35'))
MAX_SCALE_IN = int(os.getenv('MAX_SCALE_IN', '2'))
MAX_SCALE_LOSS = float(os.getenv('MAX_SCALE_LOSS', '0.03'))
MIN_CASH = float(os.getenv('MIN_CASH', '100'))
PARTIAL_TP_RMULT = float(os.getenv('PARTIAL_TP_RMULT', '2.0'))

# CCXT exchange
exchange = ccxt.binance({
    'apiKey': BINANCE_API_KEY,
    'secret': BINANCE_API_SECRET,
    'enableRateLimit': True
})

# === HELPERS ===
def fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=HISTORY_LIMIT):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def calculate_indicators(df):
    out = {}
    out['ema_fast'] = EMAIndicator(df['close'], window=9).ema_indicator()
    out['ema_slow'] = EMAIndicator(df['close'], window=21).ema_indicator()
    out['macd'] = MACD(df['close']).macd_diff()
    out['rsi'] = RSIIndicator(df['close'], window=14).rsi()
    out['atr'] = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    bb = BollingerBands(df['close'], window=20)
    out['bb_upper'] = bb.bollinger_hband()
    out['bb_middle'] = bb.bollinger_mavg()
    out['bb_lower'] = bb.bollinger_lband()
    return out

def detect_patterns(df):
    patterns = []
    if len(df) < 3:
        return patterns
    highs = df['high'].values
    lows = df['low'].values

    peaks = argrelextrema(highs, np.greater, order=2)[0]
    if len(peaks) >= 3:
        last3 = peaks[-3:]
        if highs[last3[1]] > highs[last3[0]] and highs[last3[1]] > highs[last3[2]]:
            patterns.append("Head & Shoulders Bearish")

    troughs = argrelextrema(lows, np.less, order=2)[0]
    if len(troughs) >= 3:
        last3 = troughs[-3:]
        if lows[last3[1]] < lows[last3[0]] and lows[last3[1]] < lows[last3[2]]:
            patterns.append("Inverse Head & Shoulders Bullish")

    if len(df) >= 2:
        prev, curr = df.iloc[-2], df.iloc[-1]
        # Bullish Engulfing
        if (prev['close'] < prev['open'] and curr['close'] > curr['open'] and
            curr['close'] > prev['open'] and curr['open'] < prev['close']):
            patterns.append("Bullish Engulfing")
        # Bearish Engulfing
        elif (prev['close'] > prev['open'] and curr['close'] < curr['open'] and
              curr['close'] < prev['open'] and curr['open'] > prev['close']):
            patterns.append("Bearish Engulfing")

    return patterns

def send_telegram(text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {'chat_id': TELEGRAM_CHAT_ID, 'text': text, 'parse_mode': 'HTML'}
    try:
        r = requests.post(url, data=payload, timeout=10)
        return r.status_code == 200
    except Exception:
        return False

# === SIGNAL LOGIC (adapted from backtest) ===
def generate_signal(df, indicators, patterns, portfolio_cash, current_price,
                    trailing_stop, position_entry_price, scale_in_count, entry_stoploss):
    signal = 'Hold'
    reasons = []

    rsi_val = indicators['rsi'].iloc[-1]
    macd_val = indicators['macd'].iloc[-1]
    ema_fast_val = indicators['ema_fast'].iloc[-1]
    ema_slow_val = indicators['ema_slow'].iloc[-1]
    atr_val = indicators['atr'].iloc[-1]
    bb_upper_val = indicators['bb_upper'].iloc[-1]

    bullish_trend = (ema_fast_val > ema_slow_val) and (macd_val > 0)
    bearish_trend = (ema_fast_val < ema_slow_val) and (macd_val < 0)

    bullish_patterns = [p for p in patterns if 'Bullish' in p]
    bearish_patterns = [p for p in patterns if 'Bearish' in p]

    if bullish_patterns:
        reasons.append(f"Bullish pattern: {bullish_patterns}")
    if bearish_patterns:
        reasons.append(f"Bearish pattern: {bearish_patterns}")

    # Conflict resolution: trend > pattern
    if bullish_trend:
        bearish_patterns = []
    if bearish_trend:
        bullish_patterns = []

    peak_detected = (rsi_val > RSI_OVERBOUGHT) or (current_price > bb_upper_val)
    if peak_detected:
        reasons.append("Overbought / extension")

    # Decision rules
    if bullish_trend and portfolio_cash > MIN_CASH and not peak_detected:
        signal = 'Buy'
    elif bearish_trend:
        signal = 'Sell'

    # Stop-loss hybrid (choose stricter)
    sl = None
    if signal == 'Buy':
        atr_sl = current_price - 2.0 * atr_val
        pct_sl = current_price * (1 - FIXED_STOPLOSS)
        sl = min(atr_sl, pct_sl)

    # Breakeven if >1R
    if position_entry_price > 0 and entry_stoploss is not None:
        denom = (position_entry_price - entry_stoploss)
        if denom != 0:
            r_multiple = (current_price - position_entry_price) / denom
            if r_multiple >= 1.0:
                # move SL to breakeven at least
                sl = max(sl or entry_stoploss, position_entry_price)

    # Trailing stop (suggested)
    if position_entry_price > 0:
        ts = current_price - TRAILING_ATR_MULT * atr_val
        trailing_stop = max(trailing_stop or ts, ts)

    # Position sizing (volatility-based, confidence boost if patterns)
    position_size = 0.0
    if signal == 'Buy':
        risk_per_unit = max(abs(current_price - sl), 1e-8)
        confidence = 1.0
        if bullish_patterns: confidence += 0.3
        if (macd_val > 0) and (rsi_val > 50): confidence += 0.2
        position_size = (portfolio_cash * RISK_PER_TRADE * confidence) / risk_per_unit
        # floor/ceiling
        position_size = max(0.0, position_size)

    return {
        'signal': signal,
        'reasons': reasons,
        'price': current_price,
        'stop_loss': sl,
        'position_size': position_size,
        'trailing_stop': trailing_stop
    }

# === RUN ONCE (generate and send signal) ===
def run_signal_once(symbol=SYMBOL):
    # Note: for a live bot you'd persist position state (entry price, trailing_stop, etc.)
    # Here we simulate "no existing position" by default (position-less mode).
    df = fetch_ohlcv(symbol, timeframe=TIMEFRAME, limit=HISTORY_LIMIT)
    if df is None or df.empty:
        print("No data fetched.")
        return

    # Use last N candles (buffer)
    df = df.iloc[-BUFFER_SIZE:].reset_index(drop=True)
    indicators = calculate_indicators(df)
    patterns = detect_patterns(df)
    last_row = df.iloc[-1]
    current_price = float(last_row['close'])

    # For signal sender we assume no active position by default.
    # But we expose args in case you want to integrate with an external state (PID).
    portfolio_cash = PORTFOLIO_SIZE
    trailing_stop = None
    position_entry_price = 0.0
    scale_in_count = 0
    entry_stoploss = None

    sig = generate_signal(df, indicators, patterns, portfolio_cash,
                          current_price, trailing_stop, position_entry_price,
                          scale_in_count, entry_stoploss)

    # Normalize position_size to quantity units (if you want $ amount -> qty = $ / price)
    suggested_qty = None
    suggested_alloc = None
    if sig['position_size'] and sig['price'] > 0:
        # position_size from backtest logic represents quantity units already (cash/risk_per_unit).
        # To be safe we present both: qty and $ allocation estimate.
        suggested_qty = float(sig['position_size'])
        suggested_alloc = suggested_qty * sig['price']

    # Build readable message
    lines = []
    lines.append(f"Symbol: {symbol}")
    lines.append(f"Date: {last_row['timestamp']}")
    lines.append(f"Price: {sig['price']:.6f}")
    lines.append(f"Signal: {sig['signal']}")
    if suggested_qty is not None:
        lines.append(f"Suggested qty: {suggested_qty:.6f}  (~${suggested_alloc:.2f} allocation)")
    if sig['stop_loss'] is not None:
        lines.append(f"Suggested SL: {sig['stop_loss']:.6f}")
    if sig['trailing_stop'] is not None:
        lines.append(f"Suggested Trailing stop: {sig['trailing_stop']:.6f}")
    if sig['reasons']:
        lines.append("Reasons: " + ", ".join(sig['reasons']))
    message = "\n".join(lines)

    # Print to console
    print("=== SIGNAL ===")
    print(message)
    print("==============")

    # Optionally send to Telegram
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        tx = f"<b>SIGNAL</b>\n{symbol}\nPrice: {sig['price']:.2f}\nSignal: {sig['signal']}\n"
        if suggested_alloc: tx += f"Alloc ≈ ${suggested_alloc:.2f}\n"
        if sig['stop_loss']: tx += f"SL: {sig['stop_loss']:.2f}\n"
        if sig['trailing_stop']: tx += f"TS: {sig['trailing_stop']:.2f}\n"
        if sig['reasons']: tx += "Reasons: " + ", ".join(sig['reasons'])
        ok = send_telegram(tx)
        if ok:
            print("Telegram: sent")
        else:
            print("Telegram: not sent (check token/chat_id or connectivity)")

    # Return signal dict for integration with main bot
    return {'signal': sig, 'message': message, 'timestamp': last_row['timestamp']}

if __name__ == "__main__":
    # Run once and exit (suitable for cron). If you want periodic, wrap in loop with sleep.
    run_signal_once(SYMBOL)
