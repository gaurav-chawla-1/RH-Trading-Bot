import robin_stocks.robinhood as rh
import pandas as pd
import talib as ta
import datetime
import time
import pyotp
import logging
import json
import os
import argparse
from typing import Dict, List
import numpy as np

class RobinhoodTrader:
    def __init__(self, debug: bool = False, initial_amount: float = 100.0, 
                 strategy: str = 'basic', log_file: str = 'trading_log.txt'):
        self.logged_in = False
        self.current_position = None
        self.buy_price = None
        self.debug = debug
        self.initial_amount = initial_amount
        self.current_amount = initial_amount
        self.strategy = strategy
        self.trades_history: List[Dict] = []
        self.total_profit_loss = 0.0
        
        # Setup logging
        self.setup_logging(log_file)
        
        # Create trades directory if it doesn't exist
        if self.debug:
            os.makedirs('trades', exist_ok=True)

    def setup_logging(self, log_file: str):
        self.logger = logging.getLogger('RobinhoodTrader')
        self.logger.setLevel(logging.DEBUG if self.debug else logging.INFO)
        
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG if self.debug else logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG if self.debug else logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def save_trade(self, trade_data: Dict):
        if self.debug:
            timestamp = datetime.datetime.now().strftime('%Y%m%d')
            filename = f'trades/trades_{timestamp}.json'
            
            existing_trades = []
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    existing_trades = json.load(f)
            
            existing_trades.append(trade_data)
            
            with open(filename, 'w') as f:
                json.dump(existing_trades, f, indent=4)

    def get_advanced_signals(self, df: pd.DataFrame) -> Dict:
        """
        Calculate advanced technical indicators with proper windowing
        """
        try:
            signals = {}
            
            # Ensure we have enough data points
            if len(df) < 26:  # MACD needs at least 26 periods
                self.logger.error("Insufficient data for advanced signals")
                return None
            
            close_prices = df['close_price'].values
            
            # MACD calculation (12, 26, 9)
            exp1 = df['close_price'].ewm(span=12, adjust=False).mean()
            exp2 = df['close_price'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            
            signals['macd'] = macd.iloc[-1]
            signals['macd_signal'] = signal.iloc[-1]
            
            # Bollinger Bands (20 period, 2 standard deviations)
            rolling_mean = df['close_price'].rolling(window=20).mean()
            rolling_std = df['close_price'].rolling(window=20).std()
            signals['bb_upper'] = rolling_mean.iloc[-1] + (rolling_std.iloc[-1] * 2)
            signals['bb_lower'] = rolling_mean.iloc[-1] - (rolling_std.iloc[-1] * 2)
            
            # VWAP calculation for the current day only
            df['datetime'] = pd.to_datetime(df.index)
            today = pd.Timestamp.now(tz='UTC').date()
            df_today = df[df['datetime'].dt.date == today]
            
            if not df_today.empty:
                high = df_today['high_price'].astype(float)
                low = df_today['low_price'].astype(float)
                close = df_today['close_price'].astype(float)
                volume = df_today['volume'].astype(float)
                
                typical_price = (high + low + close) / 3
                vwap = (typical_price * volume).cumsum() / volume.cumsum()
                signals['vwap'] = vwap.iloc[-1]
            else:
                signals['vwap'] = close_prices[-1]  # Use current price if no intraday data
            
            if self.debug:
                self.logger.debug("\nAdvanced Signals Details:")
                self.logger.debug(f"MACD: {signals['macd']:.2f}")
                self.logger.debug(f"MACD Signal: {signals['macd_signal']:.2f}")
                self.logger.debug(f"BB Upper: {signals['bb_upper']:.2f}")
                self.logger.debug(f"BB Lower: {signals['bb_lower']:.2f}")
                self.logger.debug(f"VWAP: {signals['vwap']:.2f}")
                self.logger.debug(f"Data points used: {len(df)}")
                self.logger.debug(f"Today's data points: {len(df_today) if 'df_today' in locals() else 0}")
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error calculating advanced signals: {str(e)}")
            if self.debug:
                import traceback
                self.logger.debug(traceback.format_exc())
            return None

    def get_rsi_data(self, symbol, interval='5minute', lookback='week'):
        """
        Get RSI and other technical indicators for a symbol
        """
        try:
            # Get historical data with bounds parameter
            historical_data = rh.stocks.get_stock_historicals(
                symbol,
                interval=interval,
                span=lookback,
                bounds='regular'  # Use 'regular' for market hours or 'extended' for extended hours
            )
            
            if not historical_data:
                self.logger.error(f"No historical data received for {symbol}")
                return None, None
            
            # Convert to DataFrame with datetime index
            df = pd.DataFrame(historical_data)
            df['begins_at'] = pd.to_datetime(df['begins_at'])
            df.set_index('begins_at', inplace=True)
            df['close_price'] = df['close_price'].astype(float)
            
            # Get only the most recent data points
            now = pd.Timestamp.now(tz='UTC')
            cutoff = now - pd.Timedelta(days=1)  # Get last 24 hours of data
            df = df[df.index > cutoff]
            
            if len(df) < 14:  # Need at least 14 periods for RSI
                self.logger.error(f"Insufficient data points: {len(df)}")
                return None, None
            
            # Calculate RSI using talib
            close_prices = df['close_price'].values
            rsi = ta.RSI(close_prices, timeperiod=14)
            current_rsi = rsi[-1]
            
            if self.debug:
                self.logger.debug(f"Time window: {df.index[0]} to {df.index[-1]}")
                self.logger.debug(f"Data points: {len(df)}")
                self.logger.debug(f"Latest close price: {close_prices[-1]}")
                self.logger.debug(f"Calculated RSI: {current_rsi}")
                
                # Log last few prices to verify changes
                self.logger.debug("Last 5 prices:")
                for idx, price in enumerate(close_prices[-5:]):
                    self.logger.debug(f"  {df.index[-5+idx]}: ${price:.2f}")
            
            if self.strategy == 'advanced':
                advanced_signals = self.get_advanced_signals(df)
                return current_rsi, advanced_signals
            
            return current_rsi, None
            
        except Exception as e:
            self.logger.error(f"Error calculating RSI: {str(e)}")
            if self.debug:
                import traceback
                self.logger.debug(traceback.format_exc())
            return None, None

    def should_buy(self, current_price: float, current_rsi: float, 
                  advanced_signals: Dict = None) -> bool:
        if self.strategy == 'basic':
            return current_rsi <= 30
        
        elif self.strategy == 'advanced':
            if advanced_signals is None:
                return False
            
            # More sophisticated entry conditions
            macd_crossover = advanced_signals['macd'] > advanced_signals['macd_signal']
            price_below_bb = current_price < advanced_signals['bb_lower']
            price_below_vwap = current_price < advanced_signals['vwap']
            
            if self.debug:
                self.logger.debug("\nBuy Signal Analysis:")
                self.logger.debug(f"RSI Condition: {current_rsi <= 30}")
                self.logger.debug(f"MACD Crossover: {macd_crossover}")
                self.logger.debug(f"Price Below BB: {price_below_bb}")
                self.logger.debug(f"Price Below VWAP: {price_below_vwap}")
            
            return (
                current_rsi <= 30 and
                macd_crossover and
                price_below_bb and
                price_below_vwap
            )

    def should_sell(self, current_price: float, current_rsi: float, 
                   profit_percentage: float, loss_percentage: float,
                   advanced_signals: Dict = None) -> bool:
        if self.strategy == 'basic':
            return (current_rsi >= 70 or profit_percentage >= 3 or loss_percentage >= 30)
        
        elif self.strategy == 'advanced':
            if advanced_signals is None:
                return False
            
            macd_crossunder = advanced_signals['macd'] < advanced_signals['macd_signal']
            price_above_bb = current_price > advanced_signals['bb_upper']
            
            if self.debug:
                self.logger.debug("\nSell Signal Analysis:")
                self.logger.debug(f"RSI Condition: {current_rsi >= 70}")
                self.logger.debug(f"Profit Target: {profit_percentage >= 3}")
                self.logger.debug(f"Stop Loss: {loss_percentage >= 30}")
                self.logger.debug(f"MACD Crossunder: {macd_crossunder}")
                self.logger.debug(f"Price Above BB: {price_above_bb}")
            
            return (
                current_rsi >= 70 or
                profit_percentage >= 3 or
                loss_percentage >= 30 or
                (macd_crossunder and profit_percentage > 1) or
                price_above_bb
            )

    def execute_trade_strategy(self, symbol):
        if not self.logged_in:
            self.logger.error("Please login first")
            return

        last_price = None
        last_check_time = None

        while True:
            try:
                current_time = datetime.datetime.now()
                
                # Check if market is open (simple check, can be enhanced)
                if current_time.hour < 9 or current_time.hour >= 16:
                    self.logger.info("Market is closed. Waiting for market hours...")
                    time.sleep(300)  # Sleep for 5 minutes
                    continue
                    
                current_price = float(rh.stocks.get_latest_price(symbol)[0])
                
                # Skip if price hasn't changed
                if last_price == current_price and last_check_time and \
                   (current_time - last_check_time).seconds < 300:
                    time.sleep(60)
                    continue
                    
                last_price = current_price
                last_check_time = current_time
                
                current_rsi, advanced_signals = self.get_rsi_data(symbol)
                
                if current_rsi is None:
                    self.logger.error("Failed to get RSI data, retrying in 60 seconds...")
                    time.sleep(60)
                    continue
                
                self.logger.info(f"\nTime: {current_time}")
                self.logger.info(f"Current Price: ${current_price:.2f}")
                self.logger.info(f"Current RSI: {current_rsi:.2f}")
                self.logger.info(f"Total Profit/Loss: ${self.total_profit_loss:.2f}")
                
                if self.debug and advanced_signals:
                    self.logger.debug(f"Advanced Signals: {json.dumps(advanced_signals, indent=2)}")

                if self.current_position is None:
                    if self.should_buy(current_price, current_rsi, advanced_signals):
                        # Calculate position size based on risk management
                        shares = int(self.initial_amount / current_price)
                        if shares > 0:
                            self.logger.info(f"Buy Signal: RSI = {current_rsi:.2f}")
                            self.buy_price = current_price
                            self.current_position = 'LONG'
                            self.logger.info(f"Buying {shares} shares of {symbol} at ${current_price:.2f}")
                            
                            trade_data = {
                                'timestamp': datetime.datetime.now().isoformat(),
                                'action': 'BUY',
                                'symbol': symbol,
                                'price': current_price,
                                'shares': shares,
                                'rsi': current_rsi,
                                'total_profit_loss': self.total_profit_loss
                            }
                            self.save_trade(trade_data)
                
                elif self.current_position == 'LONG':
                    profit_percentage = ((current_price - self.buy_price) / self.buy_price) * 100
                    loss_percentage = ((self.buy_price - current_price) / self.buy_price) * 100

                    if self.should_sell(current_price, current_rsi, profit_percentage, 
                                      loss_percentage, advanced_signals):
                        shares = int(self.initial_amount / self.buy_price)
                        trade_profit = (current_price - self.buy_price) * shares
                        self.total_profit_loss += trade_profit

                        self.logger.info("Sell Signal:")
                        if current_rsi >= 70:
                            self.logger.info(f"RSI overbought: {current_rsi:.2f}")
                        if profit_percentage >= 3:
                            self.logger.info(f"Profit target reached: {profit_percentage:.2f}%")
                        if loss_percentage >= 30:
                            self.logger.info(f"Stop loss triggered: {loss_percentage:.2f}%")
                        
                        self.logger.info(f"Selling {symbol} at ${current_price:.2f}")
                        self.logger.info(f"Trade Profit/Loss: ${trade_profit:.2f}")
                        self.logger.info(f"Total Profit/Loss: ${self.total_profit_loss:.2f}")
                        
                        trade_data = {
                            'timestamp': datetime.datetime.now().isoformat(),
                            'action': 'SELL',
                            'symbol': symbol,
                            'price': current_price,
                            'profit_percentage': profit_percentage,
                            'trade_profit_loss': trade_profit,
                            'total_profit_loss': self.total_profit_loss,
                            'rsi': current_rsi
                        }
                        self.save_trade(trade_data)
                        
                        self.current_position = None
                        self.buy_price = None

                time.sleep(300)  # Wait for 5 minutes before next check

            except Exception as e:
                self.logger.error(f"Error occurred: {str(e)}")
                time.sleep(60)

    def login(self, username: str, password: str, mfa_key: str = None):
        """
        Two-step login to Robinhood account:
        1. First attempt with username/password
        2. Then use OTP sent to phone
        """
        try:
            # First login attempt with just username and password
            login = rh.login(username, password)
            self.logger.info("Initial login attempted. Check your phone for OTP.")
            
            # Get OTP from user
            otp_code = input("Enter the OTP sent to your phone: ")
            
            # Second login attempt with OTP
            login = rh.login(username, password, mfa_code=otp_code)
            
            self.logged_in = True
            self.logger.info("Successfully logged in to Robinhood")
            
        except Exception as e:
            self.logger.error(f"Login failed: {str(e)}")
            self.logged_in = False

def analyze_daily_trades(date_str: str = None):
    if date_str is None:
        date_str = datetime.datetime.now().strftime('%Y%m%d')
    
    filename = f'trades/trades_{date_str}.json'
    if not os.path.exists(filename):
        print(f"No trades found for date {date_str}")
        return
    
    with open(filename, 'r') as f:
        trades = json.load(f)
    
    total_profit = 0
    total_trades = len(trades) // 2  # Assuming each complete trade has a buy and sell
    
    for i in range(0, len(trades), 2):
        if i + 1 < len(trades):
            buy_trade = trades[i]
            sell_trade = trades[i + 1]
            profit = (sell_trade['price'] - buy_trade['price']) * buy_trade['shares']
            total_profit += profit
    
    print(f"\nTrading Summary for {date_str}")
    print(f"Total Trades: {total_trades}")
    print(f"Total Profit/Loss: ${total_profit:.2f}")
    print(f"Average Profit per Trade: ${(total_profit/total_trades if total_trades > 0 else 0):.2f}")

def main():
    parser = argparse.ArgumentParser(description='Robinhood Trading Bot')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--strategy', choices=['basic', 'advanced'], default='basic',
                       help='Trading strategy to use')
    parser.add_argument('--initial-amount', type=float, default=100.0,
                       help='Initial trading amount')
    parser.add_argument('--analyze-date', type=str, help='Analyze trades for specific date (YYYYMMDD)')
    
    args = parser.parse_args()
    
    if args.analyze_date:
        analyze_daily_trades(args.analyze_date)
        return
    
    trader = RobinhoodTrader(
        debug=args.debug,
        initial_amount=args.initial_amount,
        strategy=args.strategy
    )
    
    username = input("Enter Robinhood username: ")
    password = input("Enter Robinhood password: ")
    
    # Login now uses phone OTP instead of MFA key
    trader.login(username, password)
    
    if trader.logged_in:
        symbol = input("Enter stock symbol to trade: ").upper()
        trader.execute_trade_strategy(symbol)

if __name__ == "__main__":
    main() 