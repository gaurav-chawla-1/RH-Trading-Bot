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
        signals = {}
        
        close_prices = df['close_price'].values
        
        # MACD using talib
        macd, signal, _ = ta.MACD(close_prices)
        signals['macd'] = macd[-1]
        signals['macd_signal'] = signal[-1]
        
        # Bollinger Bands using talib
        upper, middle, lower = ta.BBANDS(close_prices)
        signals['bb_upper'] = upper[-1]
        signals['bb_lower'] = lower[-1]
        
        # VWAP calculation
        high = df['high_price'].astype(float).values
        low = df['low_price'].astype(float).values
        close = close_prices
        volume = df['volume'].astype(float).values
        
        typical_price = (high + low + close) / 3
        vwap = np.average(typical_price, weights=volume)
        signals['vwap'] = vwap
        
        return signals

    def get_rsi_data(self, symbol, interval='5minute', lookback='week'):
        """
        Get RSI and other technical indicators for a symbol
        Supported spans: 'day', 'week', 'month', '3month', 'year', '5year'
        Supported intervals: '5minute', '10minute', 'hour', 'day', 'week'
        """
        try:
            # Get historical data
            historical_data = rh.stocks.get_stock_historicals(
                symbol,
                interval=interval,
                span=lookback,
                bounds='regular'
            )
            
            if not historical_data:
                self.logger.error(f"No historical data received for {symbol}")
                return None, None
            
            # Convert to DataFrame
            df = pd.DataFrame(historical_data)
            df['close_price'] = df['close_price'].astype(float)
            
            # Calculate RSI using talib
            close_prices = df['close_price'].values
            rsi = ta.RSI(close_prices, timeperiod=14)
            current_rsi = rsi[-1]
            
            if self.debug:
                self.logger.debug(f"Historical data points: {len(df)}")
                self.logger.debug(f"Latest close price: {close_prices[-1]}")
                self.logger.debug(f"Calculated RSI: {current_rsi}")
            
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
            # More sophisticated entry conditions
            return (
                current_rsi <= 30 and
                advanced_signals['macd'] > advanced_signals['macd_signal'] and
                current_price < advanced_signals['bb_lower'] and
                current_price < advanced_signals['vwap']
            )

    def should_sell(self, current_price: float, current_rsi: float, 
                   profit_percentage: float, loss_percentage: float,
                   advanced_signals: Dict = None) -> bool:
        if self.strategy == 'basic':
            return (current_rsi >= 70 or profit_percentage >= 3 or loss_percentage >= 30)
        
        elif self.strategy == 'advanced':
            return (
                current_rsi >= 70 or
                profit_percentage >= 3 or
                loss_percentage >= 30 or
                (advanced_signals['macd'] < advanced_signals['macd_signal'] and profit_percentage > 1) or
                current_price > advanced_signals['bb_upper']
            )

    def execute_trade_strategy(self, symbol):
        if not self.logged_in:
            self.logger.error("Please login first")
            return

        while True:
            try:
                current_price = float(rh.stocks.get_latest_price(symbol)[0])
                current_rsi, advanced_signals = self.get_rsi_data(symbol)
                
                if current_rsi is None:
                    self.logger.error("Failed to get RSI data, retrying in 60 seconds...")
                    time.sleep(60)
                    continue
                
                self.logger.info(f"\nCurrent Price: ${current_price:.2f}")
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

                time.sleep(60)  # Wait for 1 minute before next check

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