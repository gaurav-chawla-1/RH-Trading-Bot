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
                 strategy: str = 'basic', log_file: str = 'trading_log.txt',
                 stop_loss: float = 30.0, take_profit: float = 3.0,
                 interval_minutes: int = 1, extended_hours: bool = True,
                 asset_type: str = 'stock', rsi_upper: float = 70.0,
                 rsi_lower: float = 30.0):
        self.logged_in = False
        self.current_position = None
        self.buy_price = None
        self.debug = debug
        self.initial_amount = initial_amount
        self.current_amount = initial_amount
        self.strategy = strategy
        self.trades_history: List[Dict] = []
        self.total_profit_loss = 0.0
        
        # Trading parameters
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.interval_minutes = interval_minutes
        self.extended_hours = extended_hours
        self.rsi_upper = rsi_upper
        self.rsi_lower = rsi_lower
        
        # Setup logging
        self.setup_logging(log_file)
        
        # Create trades directory if it doesn't exist
        if self.debug:
            os.makedirs('trades', exist_ok=True)
        
        self.asset_type = asset_type.lower()
        
        # Add portfolio tracking
        self.portfolio = {
            'initial_amount': initial_amount,
            'current_amount': initial_amount,
            'total_profit_loss': 0.0,
            'per_ticker_amount': 0.0,  # Will be set when tickers are known
            'positions': {}  # Track positions and profits per ticker
        }

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

    def get_advanced_signals(self, df: pd.DataFrame, price_col: str) -> Dict:
        """Calculate advanced signals for both stocks and crypto"""
        try:
            signals = {}
            
            close_prices = df[price_col].values
            
            # MACD calculation
            exp1 = df[price_col].ewm(span=12, adjust=False).mean()
            exp2 = df[price_col].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            
            signals['macd'] = macd.iloc[-1]
            signals['macd_signal'] = signal.iloc[-1]
            
            # Bollinger Bands
            rolling_mean = df[price_col].rolling(window=20).mean()
            rolling_std = df[price_col].rolling(window=20).std()
            signals['bb_upper'] = rolling_mean.iloc[-1] + (rolling_std.iloc[-1] * 2)
            signals['bb_lower'] = rolling_mean.iloc[-1] - (rolling_std.iloc[-1] * 2)
            
            # VWAP calculation with proper handling for crypto
            try:
                if self.asset_type == 'stock':
                    high = df['high_price'].astype(float)
                    low = df['low_price'].astype(float)
                    volume = df['volume'].astype(float)
                else:
                    # For crypto, use mark_price as both high and low if specific prices not available
                    high = df['high_mark_price'].astype(float) if 'high_mark_price' in df.columns else df[price_col]
                    low = df['low_mark_price'].astype(float) if 'low_mark_price' in df.columns else df[price_col]
                    volume = df['volume'].astype(float) if 'volume' in df.columns else pd.Series(1, index=df.index)

                # Calculate VWAP
                df['typical_price'] = (high + low + df[price_col]) / 3
                df['price_volume'] = df['typical_price'] * volume
                df['cumulative_volume'] = volume.cumsum()
                
                # Avoid division by zero
                df['vwap'] = np.where(
                    df['cumulative_volume'] > 0,
                    df['price_volume'].cumsum() / df['cumulative_volume'],
                    df[price_col]
                )
                
                signals['vwap'] = df['vwap'].iloc[-1]
                
                if self.debug:
                    self.logger.debug(f"VWAP calculation successful")
                    self.logger.debug(f"Last typical price: {df['typical_price'].iloc[-1]}")
                    self.logger.debug(f"Last volume: {volume.iloc[-1]}")
                    self.logger.debug(f"VWAP: {signals['vwap']}")
                    
            except Exception as e:
                self.logger.warning(f"VWAP calculation failed: {str(e)}")
                # Fallback to using current price if VWAP calculation fails
                signals['vwap'] = close_prices[-1]
                
            if self.debug:
                self.logger.debug("\nAdvanced Signals Calculation:")
                self.logger.debug(f"Using price column: {price_col}")
                self.logger.debug(f"Data points: {len(df)}")
                self.logger.debug(f"Latest price: {close_prices[-1]}")
                self.logger.debug(f"MACD: {signals['macd']}")
                self.logger.debug(f"Signal: {signals['macd_signal']}")
                self.logger.debug(f"BB Upper: {signals['bb_upper']}")
                self.logger.debug(f"BB Lower: {signals['bb_lower']}")
                self.logger.debug(f"VWAP: {signals['vwap']}")
            
            return signals
                
        except Exception as e:
            self.logger.error(f"Error calculating advanced signals: {str(e)}")
            if self.debug:
                import traceback
                self.logger.debug(traceback.format_exc())
                if 'df' in locals():
                    self.logger.debug(f"DataFrame columns: {df.columns.tolist()}")
                    self.logger.debug(f"First row: {df.iloc[0].to_dict()}")
            return None

    def get_rsi_data(self, symbol, interval=None, lookback='day'):
        """Get RSI and other technical indicators for a symbol"""
        try:
            # Set interval based on configured interval_minutes
            if interval is None:
                if self.interval_minutes <= 5:
                    interval = '5minute'
                elif self.interval_minutes <= 10:
                    interval = '10minute'
                else:
                    interval = 'hour'

            if self.asset_type == 'stock':
                historical_data = rh.stocks.get_stock_historicals(
                    symbol,
                    interval=interval,
                    span='day',
                    bounds='extended' if self.extended_hours else 'regular'
                )
                price_col = 'close_price'
            else:
                # Remove .X suffix for crypto API calls
                clean_symbol = symbol.replace('.X', '')
                # Verify crypto symbol first
                crypto_info = rh.crypto.get_crypto_info(clean_symbol)
                if not crypto_info:
                    self.logger.error(f"Invalid crypto symbol: {clean_symbol}")
                    return None, None
                
                historical_data = rh.crypto.get_crypto_historicals(
                    clean_symbol,
                    interval=interval,
                    span='day',
                    bounds='24_7'
                )
                price_col = 'close_price'  # Crypto also uses close_price

            if not historical_data:
                self.logger.error(f"No historical data received for {symbol}")
                return None, None

            # Convert to DataFrame
            df = pd.DataFrame(historical_data)

            # Process timestamps
            df['timestamp'] = pd.to_datetime([x['begins_at'] for x in historical_data])
            df.set_index('timestamp', inplace=True)
            
            # Convert price columns
            df[price_col] = df[price_col].astype(float)

            if self.debug:
                self.logger.debug(f"Available columns: {df.columns.tolist()}")
                self.logger.debug(f"First row data: {df.iloc[0].to_dict()}")

            # Calculate RSI
            close_prices = df[price_col].values
            rsi = ta.RSI(close_prices, timeperiod=14)
            current_rsi = rsi[-1]

            if self.strategy == 'advanced':
                advanced_signals = self.get_advanced_signals(df, price_col)
                return current_rsi, advanced_signals

            return current_rsi, None

        except Exception as e:
            self.logger.error(f"Error calculating RSI: {str(e)}")
            if self.debug:
                import traceback
                self.logger.debug(traceback.format_exc())
                if 'df' in locals():
                    self.logger.debug(f"DataFrame columns: {df.columns.tolist()}")
            return None, None

    def should_buy(self, current_price: float, current_rsi: float, 
                  advanced_signals: Dict = None) -> bool:
        if self.strategy == 'basic':
            return current_rsi <= self.rsi_lower
        
        elif self.strategy == 'advanced':
            if advanced_signals is None:
                return False
            
            # More sophisticated entry conditions
            macd_crossover = advanced_signals['macd'] > advanced_signals['macd_signal']
            price_below_bb = current_price < advanced_signals['bb_lower']
            price_below_vwap = current_price < advanced_signals['vwap']
            
            if self.debug:
                self.logger.debug("\n=Buy Signal Analysis:")
                self.logger.debug(f"RSI Condition: {current_rsi <= self.rsi_lower}")
                self.logger.debug(f"MACD Crossover: {macd_crossover}")
                self.logger.debug(f"Price Below BB: {price_below_bb}")
                self.logger.debug(f"Price Below VWAP: {price_below_vwap}")

            return (
                current_rsi <= self.rsi_lower and
                macd_crossover and
                price_below_bb and
                price_below_vwap
            )

    def should_sell(self, current_price: float, current_rsi: float, 
                   profit_percentage: float, loss_percentage: float,
                   advanced_signals: Dict = None) -> bool:
        if self.strategy == 'basic':
            return (current_rsi >= self.rsi_upper or 
                    profit_percentage >= self.take_profit or 
                    loss_percentage >= self.stop_loss)
        
        elif self.strategy == 'advanced':
            if advanced_signals is None:
                return False
            
            macd_crossunder = advanced_signals['macd'] < advanced_signals['macd_signal']
            price_above_bb = current_price > advanced_signals['bb_upper']
            
            if self.debug:
                self.logger.debug("\nSell Signal Analysis:")
                self.logger.debug(f"RSI Condition: {current_rsi >= self.rsi_upper}")
                self.logger.debug(f"Profit Target: {profit_percentage >= self.take_profit}")
                self.logger.debug(f"Stop Loss: {loss_percentage >= self.stop_loss}")
                self.logger.debug(f"MACD Crossunder: {macd_crossunder}")
                self.logger.debug(f"Price Above BB: {price_above_bb}")

            return (
                current_rsi >= self.rsi_upper or
                profit_percentage >= self.take_profit or
                loss_percentage >= self.stop_loss or
                (macd_crossunder and profit_percentage > 1) or
                price_above_bb
            )

    def execute_trade_strategies(self, symbols: List[str]):
        if not self.logged_in:
            self.logger.error("Please login first")
            return

        # Calculate initial amount per ticker
        self.portfolio['per_ticker_amount'] = self.portfolio['current_amount'] / len(symbols)
        
        # Initialize positions tracking for each symbol
        for symbol in symbols:
            self.portfolio['positions'][symbol] = {
                'current_position': None,
                'buy_price': None,
                'last_price': None,
                'last_check_time': None,
                'allocated_amount': self.portfolio['per_ticker_amount'],
                'profit_loss': 0.0
            }

        self.logger.info("\nStarting trading bot with parameters:")
        self.logger.info(f"Total Portfolio Amount: ${self.portfolio['current_amount']:.2f}")
        self.logger.info(f"Amount per ticker: ${self.portfolio['per_ticker_amount']:.2f}")
        self.logger.info(f"Symbols: {', '.join(symbols)}")
        self.logger.info(f"Stop Loss: {self.stop_loss}%")
        self.logger.info(f"Take Profit: {self.take_profit}%")
        self.logger.info(f"Check Interval: {self.interval_minutes} minute(s)")
        self.logger.info(f"Extended Hours: {'Yes' if self.extended_hours else 'No'}")
        self.logger.info(f"RSI Bounds: Upper={self.rsi_upper}, Lower={self.rsi_lower}")

        while True:
            try:
                current_time = datetime.datetime.now()
                
                # Check if market is open (if not using extended hours)
                if not self.extended_hours and (current_time.hour < 9 or current_time.hour >= 16):
                    self.logger.info("Market is closed. Waiting for market hours...")
                    time.sleep(60)
                    continue

                # Process each symbol
                for symbol in symbols:
                    position = self.portfolio['positions'][symbol]
                    
                    current_price = self.get_current_price(symbol)
                    if current_price is None:
                        continue
                    
                    # Skip if price hasn't changed
                    if (position['last_price'] == current_price and 
                        position['last_check_time'] and 
                        (current_time - position['last_check_time']).seconds < 60):
                        continue
                    
                    position['last_price'] = current_price
                    position['last_check_time'] = current_time
                    
                    current_rsi, advanced_signals = self.get_rsi_data(symbol)
                    
                    if current_rsi is None:
                        continue
                    
                    self.logger.info(f"\nSymbol: {symbol}")
                    self.logger.info(f"Time: {current_time}")
                    self.logger.info(f"Current Price: ${current_price:.2f}")
                    self.logger.info(f"Current RSI: {current_rsi:.2f}")
                    self.logger.info(f"Allocated Amount: ${position['allocated_amount']:.2f}")
                    self.logger.info(f"Position P/L: ${position['profit_loss']:.2f}")
                    
                    if position['current_position'] is None:
                        if self.should_buy(current_price, current_rsi, advanced_signals):
                            # Calculate shares based on allocated amount
                            shares = int(position['allocated_amount'] / current_price)
                            if shares > 0:
                                self.logger.info(f"Buy Signal for {symbol}: RSI = {current_rsi:.2f}")
                                position['buy_price'] = current_price
                                position['current_position'] = 'LONG'
                                self.logger.info("================================================")
                                self.logger.info(f"Buying {shares} shares at ${current_price:.2f}")
                                self.logger.info("================================================")

                                
                                trade_data = {
                                    'timestamp': datetime.datetime.now().isoformat(),
                                    'action': 'BUY',
                                    'symbol': symbol,
                                    'price': current_price,
                                    'shares': shares,
                                    'rsi': current_rsi,
                                    'allocated_amount': position['allocated_amount']
                                }
                                self.save_trade(trade_data)
                    
                    elif position['current_position'] == 'LONG':
                        profit_percentage = ((current_price - position['buy_price']) / position['buy_price']) * 100
                        loss_percentage = ((position['buy_price'] - current_price) / position['buy_price']) * 100

                        if self.should_sell(current_price, current_rsi, profit_percentage, 
                                          loss_percentage, advanced_signals):
                            shares = int(position['allocated_amount'] / position['buy_price'])
                            trade_profit = (current_price - position['buy_price']) * shares
                            
                            # Update position and portfolio profits
                            position['profit_loss'] += trade_profit
                            self.portfolio['total_profit_loss'] += trade_profit
                            
                            # Reinvest profits by updating allocated amounts
                            self.portfolio['current_amount'] += trade_profit
                            new_per_ticker_amount = self.portfolio['current_amount'] / len(symbols)
                            
                            # Update allocated amounts for all positions
                            for sym in symbols:
                                self.portfolio['positions'][sym]['allocated_amount'] = new_per_ticker_amount

                            self.logger.info("================================================")
                            self.logger.info(f"Sell Signal for {symbol}:")
                            self.logger.info(f"Trade Profit/Loss: ${trade_profit:.2f}")
                            self.logger.info(f"New Allocated Amount: ${new_per_ticker_amount:.2f}")
                            self.logger.info(f"Total Portfolio Value: ${self.portfolio['current_amount']:.2f}")
                            self.logger.info("================================================")
                            trade_data = {
                                'timestamp': datetime.datetime.now().isoformat(),
                                'action': 'SELL',
                                'symbol': symbol,
                                'price': current_price,
                                'profit_percentage': profit_percentage,
                                'trade_profit_loss': trade_profit,
                                'total_portfolio_value': self.portfolio['current_amount'],
                                'rsi': current_rsi
                            }
                            self.save_trade(trade_data)
                            
                            position['current_position'] = None
                            position['buy_price'] = None

                time.sleep(self.interval_minutes * 60)

            except Exception as e:
                self.logger.error(f"Error occurred: {str(e)}")
                if self.debug:
                    self.logger.debug(traceback.format_exc())
                time.sleep(10)

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

    def get_current_price(self, symbol: str) -> float:
        """Get current price for stock or crypto"""
        try:
            if self.asset_type == 'stock':
                return float(rh.stocks.get_latest_price(symbol)[0])
            else:
                # Remove .X suffix if present for API call
                clean_symbol = symbol.replace('.X', '')
                # Get crypto info first to verify symbol
                crypto_info = rh.crypto.get_crypto_info(clean_symbol)
                if not crypto_info:
                    self.logger.error(f"Invalid crypto symbol: {clean_symbol}")
                    return None
                
                quote = rh.crypto.get_crypto_quote(clean_symbol)
                if quote and 'mark_price' in quote:
                    return float(quote['mark_price'])
                else:
                    self.logger.error(f"Could not get price for {clean_symbol}")
                    return None
        except Exception as e:
            self.logger.error(f"Error getting price for {symbol}: {str(e)}")
            return None

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
    parser.add_argument('--analyze-date', type=str, 
                       help='Analyze trades for specific date (YYYYMMDD)')
    parser.add_argument('--stop-loss', type=float, default=30.0,
                       help='Stop loss percentage (default: 30%)')
    parser.add_argument('--take-profit', type=float, default=3.0,
                       help='Take profit percentage (default: 3%)')
    parser.add_argument('--interval', type=int, default=1,
                       help='Trading check interval in minutes (default: 1)')
    parser.add_argument('--no-extended-hours', action='store_true',
                       help='Disable extended hours trading')
    parser.add_argument('--asset-type', choices=['stock', 'crypto'], default='stock',
                       help='Asset type to trade (stock or crypto)')
    parser.add_argument('--rsi-upper', type=float, default=70.0,
                       help='RSI upper bound for sell signal (default: 70)')
    parser.add_argument('--rsi-lower', type=float, default=30.0,
                       help='RSI lower bound for buy signal (default: 30)')
    parser.add_argument('--tickers', type=str, nargs='+',
                       help='List of tickers to trade (space-separated)')
    
    args = parser.parse_args()
    
    if args.analyze_date:
        analyze_daily_trades(args.analyze_date)
        return
    
    trader = RobinhoodTrader(
        debug=args.debug,
        initial_amount=args.initial_amount,
        strategy=args.strategy,
        stop_loss=args.stop_loss,
        take_profit=args.take_profit,
        interval_minutes=args.interval,
        extended_hours=not args.no_extended_hours,
        asset_type=args.asset_type,
        rsi_upper=args.rsi_upper,
        rsi_lower=args.rsi_lower
    )
    
    username = input("Enter Robinhood username: ")
    password = input("Enter Robinhood password: ")
    
    trader.login(username, password)
    
    if trader.logged_in:
        tickers = []
        if args.tickers:
            tickers = args.tickers
        else:
            if args.asset_type == 'crypto':
                print("\nAvailable crypto tickers: BTC, ETH, DOGE, LTC, BCH, ETC")
                tickers_input = input("Enter crypto tickers to trade (space-separated): ")
            else:
                tickers_input = input("Enter stock tickers to trade (space-separated): ")
            tickers = tickers_input.upper().split()
        
        if tickers:
            trader.execute_trade_strategies(tickers)
        else:
            print("No tickers provided. Exiting...")

if __name__ == "__main__":
    main() 