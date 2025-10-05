import os
import pandas as pd
import numpy as np
import ccxt
import backtrader as bt
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from scipy.signal import argrelextrema
from dotenv import load_dotenv
import matplotlib.pyplot as plt

load_dotenv()

# === CONFIG ===
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')
SYMBOLS = ['SOL/USDC']
TIMEFRAME = '1d'
PORTFOLIO_SIZE = 10000
RISK_PER_TRADE = 0.07
BUFFER_SIZE = 100

# Risk control
TRAILING_ATR_MULT = 2.0
FIXED_STOPLOSS = 0.04
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 35
MAX_SCALE_IN = 2
MAX_SCALE_LOSS = 0.03
MIN_CASH = 100
PARTIAL_TP_RMULT = 2.0  # Take profit partiel à 2R

exchange = ccxt.binance({
    'apiKey': BINANCE_API_KEY,
    'secret': BINANCE_API_SECRET,
    'enableRateLimit': True
})

# === FETCH DATA ===
def fetch_ohlcv(symbol, timeframe='1d', limit=365):
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

# === INDICATORS ===
def calculate_indicators(df):
    indicators = {}
    indicators['ema_fast'] = EMAIndicator(df['close'], window=9).ema_indicator()
    indicators['ema_slow'] = EMAIndicator(df['close'], window=21).ema_indicator()
    indicators['macd'] = MACD(df['close']).macd_diff()
    indicators['rsi'] = RSIIndicator(df['close'], window=14).rsi()
    indicators['atr'] = AverageTrueRange(df['high'], df['low'], df['close'], window=14).average_true_range()
    bb = BollingerBands(df['close'], window=20)
    indicators['bb_upper'] = bb.bollinger_hband()
    indicators['bb_middle'] = bb.bollinger_mavg()
    indicators['bb_lower'] = bb.bollinger_lband()
    return indicators

# === PATTERNS ===
def detect_patterns(df):
    patterns = []
    if len(df) < 3: return patterns
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
        if prev['close'] < prev['open'] and curr['close'] > curr['open'] \
           and curr['close'] > prev['open'] and curr['open'] < prev['close']:
            patterns.append("Bullish Engulfing")
        elif prev['close'] > prev['open'] and curr['close'] < curr['open'] \
             and curr['close'] < prev['open'] and curr['open'] > prev['close']:
            patterns.append("Bearish Engulfing")
    return patterns

# === SIGNAL GENERATION ===
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

    bullish_trend = ema_fast_val > ema_slow_val and macd_val > 0
    bearish_trend = ema_fast_val < ema_slow_val and macd_val < 0

    bullish_patterns = [p for p in patterns if 'Bullish' in p]
    bearish_patterns = [p for p in patterns if 'Bearish' in p]

    if bullish_patterns: reasons.append(f"Bullish pattern: {bullish_patterns}")
    if bearish_patterns: reasons.append(f"Bearish pattern: {bearish_patterns}")

    # Conflict resolution: trend > pattern
    if bullish_trend: bearish_patterns = []
    if bearish_trend: bullish_patterns = []

    peak_detected = (rsi_val > RSI_OVERBOUGHT or current_price > bb_upper_val)
    if peak_detected: reasons.append("Overbought / extension")

    if bullish_trend and portfolio_cash > MIN_CASH and not peak_detected:
        signal = 'Buy'
    elif bearish_trend:
        signal = 'Sell'

    # Stop-loss hybrid
    sl = None
    if signal == 'Buy':
        atr_sl = current_price - 2.0 * atr_val
        pct_sl = current_price * (1 - FIXED_STOPLOSS)
        sl = min(atr_sl, pct_sl)

    # Breakeven stop if profit > 1R
    if position_entry_price > 0 and entry_stoploss is not None:
        r_multiple = (current_price - position_entry_price) / (position_entry_price - entry_stoploss)
        if r_multiple >= 1.0:
            sl = max(sl or entry_stoploss, position_entry_price)  # move to breakeven

    # Trailing stop
    if position_entry_price > 0:
        ts = current_price - TRAILING_ATR_MULT * atr_val
        trailing_stop = max(trailing_stop or ts, ts)

    # Position sizing
    position_size = 0
    if signal == 'Buy':
        risk_per_unit = max(abs(current_price - sl), 1e-6)
        # More confidence = bigger size
        confidence = 1.0
        if bullish_patterns: confidence += 0.3
        if macd_val > 0 and rsi_val > 50: confidence += 0.2
        position_size = (portfolio_cash * RISK_PER_TRADE * confidence) / risk_per_unit

    return {
        'signal': signal,
        'reasons': reasons,
        'price': current_price,
        'stop_loss': sl,
        'position_size': position_size,
        'trailing_stop': trailing_stop
    }

# === STRATEGY ===
class UltraBot(bt.Strategy):
    params = dict(buffer_size=BUFFER_SIZE)

    def __init__(self):
        self.df_buffer = pd.DataFrame(columns=['open','high','low','close','volume'])
        self.cash = PORTFOLIO_SIZE
        self.position_size = 0
        self.position_entry_price = 0
        self.trailing_stop = None
        self.entry_stoploss = None
        self.partial_taken = False
        self.portfolio_history = []
        self.scale_in_count = 0

    def next(self):
        row = { 'open': self.data.open[0], 'high': self.data.high[0],
                'low': self.data.low[0], 'close': self.data.close[0],
                'volume': self.data.volume[0] }
        self.df_buffer = pd.concat([self.df_buffer, pd.DataFrame([row])], ignore_index=True)
        if len(self.df_buffer) > self.p.buffer_size:
            self.df_buffer = self.df_buffer.iloc[-self.p.buffer_size:]

        if len(self.df_buffer) < 20:
            self.portfolio_history.append(self.cash + self.position_size * row['close'])
            return

        indicators = calculate_indicators(self.df_buffer)
        patterns = detect_patterns(self.df_buffer)
        current_price = row['close']
        signal_data = generate_signal(
            self.df_buffer, indicators, patterns, self.cash,
            current_price, self.trailing_stop, self.position_entry_price,
            self.scale_in_count, self.entry_stoploss
        )

        self.trailing_stop = signal_data['trailing_stop']
        action = 'HOLD'

        # Execution
        if signal_data['signal'] == 'Buy':
            if self.position_size == 0:
                self.position_size = signal_data['position_size']
                self.position_entry_price = current_price
                self.cash -= self.position_size * current_price
                self.scale_in_count = 1
                self.entry_stoploss = signal_data['stop_loss']
                self.partial_taken = False
                action = 'BUY'
            else:
                if self.scale_in_count < MAX_SCALE_IN:
                    add_size = signal_data['position_size']
                    if self.cash >= add_size * current_price:
                        self.position_size += add_size
                        self.cash -= add_size * current_price
                        self.scale_in_count += 1
                        action = 'SCALE-IN'

        elif self.position_size > 0:
            # Partial TP
            if not self.partial_taken and self.entry_stoploss is not None:
                r_mult = (current_price - self.position_entry_price) / (self.position_entry_price - self.entry_stoploss)
                if r_mult >= PARTIAL_TP_RMULT:
                    sell_size = self.position_size * 0.5
                    self.cash += sell_size * current_price
                    self.position_size -= sell_size
                    self.partial_taken = True
                    action = 'PARTIAL TAKE PROFIT'

            # Exit
            if current_price <= self.trailing_stop or signal_data['signal'] == 'Sell':
                self.cash += self.position_size * current_price
                self.position_size = 0
                self.position_entry_price = 0
                self.trailing_stop = None
                self.entry_stoploss = None
                self.scale_in_count = 0
                self.partial_taken = False
                action = 'SELL / STOP'

        total_portfolio = self.cash + self.position_size * current_price
        self.portfolio_history.append(total_portfolio)

        print(f"{self.data.datetime.date(0)} | {action} | Price: {current_price:.2f} "
              f"| Cash: {self.cash:.2f} | Pos: {self.position_size:.4f} | Total: {total_portfolio:.2f} "
              f"| Reasons: {signal_data['reasons']}")

    def stop(self):
        if self.position_size > 0:
            last_price = self.df_buffer['close'].iloc[-1]
            self.cash += self.position_size * last_price
            self.position_size = 0
        final_portfolio = self.cash
        print("\n=== BACKTEST FINISHED ===")
        print(f"Start: {PORTFOLIO_SIZE}")
        print(f"End: {final_portfolio:.2f}")
        print(f"Perf: {((final_portfolio-PORTFOLIO_SIZE)/PORTFOLIO_SIZE)*100:.2f}%\n")
        plt.figure(figsize=(12,5))
        plt.plot(self.portfolio_history, label='Portfolio')
        plt.title('Portfolio Value')
        plt.legend()
        plt.show()

# === RUN BACKTEST ===
def run_backtest(symbol, limit=365):
    df = fetch_ohlcv(symbol, TIMEFRAME, limit=limit)
    cerebro = bt.Cerebro()
    data = bt.feeds.PandasData(dataname=df.set_index('timestamp'), name=symbol)
    cerebro.adddata(data)
    cerebro.addstrategy(UltraBot)
    cerebro.run()

if __name__ == "__main__":
    for symbol in SYMBOLS:
        run_backtest(symbol, limit=365)
