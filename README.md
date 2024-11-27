# ROBINHOOD TRADING BOT 
====================

A Python-based automated trading bot for Robinhood that uses RSI (Relative Strength Index) and other technical indicators to make trading decisions for both stocks and cryptocurrencies.

Note: This is a work in progress and is not yet ready use and only provides the sample trades in the trades/ directory.

# FEATURES
--------
* Automated trading for both stocks and cryptocurrencies
* Two trading strategies: Basic and Advanced
* Real-time trade logging and tracking
* Trade history analysis
* Risk management with configurable initial amount
* Phone-based 2FA authentication
* JSON-based trade history storage
* Daily profit/loss tracking
* 24/7 crypto trading support
* Extended hours trading for stocks

# PREREQUISITES
------------
1. Python 3.9 or higher
2. Robinhood account
3. TA-Lib (Technical Analysis Library)
4. Active internet connection
5. Phone for 2FA authentication

# INSTALLATION INSTRUCTIONS (Use docker instructions from below if possible)
------------------------

## DOCKER SETUP AND USAGE
---------------------

#### Prerequisites:
- Docker installed on your system
- Docker Compose (optional)

#### Quick Start:
1. Setup environment and build Docker image:
```
./run.sh setup
```

2. Start trading (no actual trades are made):
```
# For stocks
./run.sh start stock --debug --tickers AAPL MSFT GOOGL

# For crypto
./run.sh start crypto --debug --tickers BTC ETH DOGE
```

3. View logs:
```
## View trade files
./run.sh logs trades

## View bot logs
./run.sh logs bot

## View docker logs
./run.sh logs
```

4. Analyze trades:
```
./run.sh analyze 20240315
```

5. Stop the bot:
```
./run.sh stop
```

#### Directory Structure:
```
project/
├── data/
│   ├── trades/    # Trade history (persisted)
│   └── logs/      # Log files (persisted)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── trading_bot.py
└── run.sh
```

#### Available Commands:
```
##### Setup environment
./run.sh setup

##### Start trading with options
./run.sh start stock --debug --tickers AAPL MSFT --strategy advanced \
                     --initial-amount 500 --stop-loss 20 --take-profit 4 \
                     --interval 2 --rsi-upper 75 --rsi-lower 25

##### View different types of logs
./run.sh logs trades  # View trade files
./run.sh logs bot    # View bot logs
./run.sh logs        # View docker logs

##### Analyze trades for a specific date
./run.sh analyze 20240315

##### Stop the bot
./run.sh stop
```

#### Environment Variables:
The following environment variables are set in the container:
- `TZ=America/Los_Angeles` (PST timezone)
- `LOG_DIR=/app/data/logs`
- `TRADES_DIR=/app/data/trades`
- `PYTHONUNBUFFERED=1`

#### Data Persistence:
Trade history and logs are stored in the `data/` directory on your host machine:
- `data/trades/`: Contains JSON files with trade history
- `data/logs/`: Contains log files

# Use docker instructions from above if possible, following instructions are for local installation
### FOR MAC USERS:
-------------
1. Install TA-Lib:
```
   $ brew install ta-lib
```

2. Create virtual environment:
```
   $ python3 -m venv venvrh
```

3. Activate virtual environment:
```
   $ source venvrh/bin/activate
```

4. Install requirements:
```
   $ pip install --upgrade pip
   $ pip install -r requirements.txt
```

### FOR LINUX USERS:
---------------
1. Install TA-Lib:
```
   $ wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
   $ tar -xzf ta-lib-0.4.0-src.tar.gz
   $ cd ta-lib/
   $ ./configure --prefix=/usr
   $ make
   $ sudo make install
```

2. Create and activate virtual environment:
```
   $ python3 -m venv venvrh
   $ source venvrh/bin/activate
   $ pip install --upgrade pip
   $ pip install -r requirements.txt
```

### FOR WINDOWS USERS:
-----------------
1. Download TA-Lib:
```
   - Download from: http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-msvc.zip
   - Extract to C:\ta-lib
```

2. Create and activate virtual environment:
```
    > python -m venv venvrh
    > venvrh\Scripts\activate
    > pip install --upgrade pip
    > pip install -r requirements.txt
```

# USAGE INSTRUCTIONS
-----------------

1. BASIC STOCK TRADING:
```
   $ python trading_bot.py --asset-type stock
   $ python trading_bot.py --asset-type stock --tickers AAPL MSFT GOOGL

```

2. BASIC CRYPTO TRADING:
```
   $ python trading_bot.py --asset-type crypto
   $ python trading_bot.py --asset-type crypto --tickers BTC ETH DOGE
```

3. ADVANCED CRYPTO TRADING:
```
   $ python trading_bot.py --asset-type crypto --debug --strategy advanced --initial-amount 500
```

4. ANALYZE TRADES:
```
   $ python trading_bot.py --analyze-date 20240315
```

# COMMAND LINE ARGUMENTS:
----------------------
```
--debug            : Enable debug mode
--strategy         : Choose 'basic' or 'advanced' (default: basic)
--initial-amount   : Set initial trading amount (default: $100)
--analyze-date     : Analyze trades for specific date (format: YYYYMMDD)
--interval         : Set trading check interval in minutes (default: 1)
--stop-loss        : Set stop loss percentage (default: 30%)
--take-profit      : Set take profit percentage (default: 3%)
--interval         : Set trading check interval in minutes (default: 1)
--no-extended-hours: Disable extended hours trading (Default: Enabled)
--asset-type       : Choose 'stock' or 'crypto' (default: stock)
--rsi-upper        : Set RSI upper bound for sell signal (default: 70)
--rsi-lower        : Set RSI lower bound for buy signal (default: 30)
--tickers          : List of tickers to trade (space-separated)
```

# TRADING STRATEGIES
-----------------

1. BASIC STRATEGY (Both Stock and Crypto):
   Buy Signals:
   - RSI <= 30

   Sell Signals:
   - RSI >= 70
   - Profit >= 3%
   - Loss >= 30%

2. ADVANCED STRATEGY:
   Buy Signals:
   - RSI <= 30
   - MACD > Signal Line
   - Price < Lower Bollinger Band
   - Price < VWAP

   Sell Signals:
   - RSI >= 70
   - Profit >= 3%
   - Loss >= 30%
   - MACD < Signal Line with profit > 1%
   - Price > Upper Bollinger Band


# CRYPTO-SPECIFIC FEATURES
-----------------------
1. 24/7 Trading:
   - No market hours restrictions
   - Continuous monitoring and trading

2. Price Handling:
   - Uses mark_price for crypto assets
   - Handles crypto-specific price formatting

3. Volume Analysis:
   - Adapted VWAP calculation for crypto markets
   - Crypto-specific volume weighting

4. Risk Management:
   - Same stop-loss and take-profit mechanics
   - Position sizing adapted for crypto volatility


# CONFIGURATION
------------
1. Initial Amount:
   - Default: $100
   -Change using: --initial-amount flag

2. Risk Management:
   - Stop Loss: 30% of purchase price
   - Take Profit: 3% gain
   - Position sizing based on initial amount

3. Trading Intervals:
   - Checks every 5 minutes
   - Uses regular market hours

# LOGGING AND MONITORING
---------------------
1. Standard Logs:
   - Basic trade information
   - Current price and RSI
   - Profit/Loss tracking

2. Debug Logs (--debug flag):
   - Detailed technical indicators
   - Trade signals
   - Error messages
   - JSON trade history

# EXAMPLE USAGE SESSIONS
--------------------
For Stocks:
```
$ python trading_bot.py --asset-type stock --debug --strategy advanced
Enter Robinhood username: your_username
Enter Robinhood password: your_password
Enter the OTP sent to your phone: 123456
Enter symbol to trade: AAPL
```

For Crypto:
```
$ python trading_bot.py --asset-type crypto --debug --strategy advanced
Enter Robinhood username: your_username
Enter Robinhood password: your_password
Enter the OTP sent to your phone: 123456
Enter symbol to trade: BTC
```

# CRYPTO TRADING CONSIDERATIONS
---------------------------
1. Volatility:
   - Crypto markets are more volatile
   - Consider using tighter stop-losses
   - Monitor positions more frequently

2. 24/7 Trading:
   - Bot runs continuously
   - No market hour restrictions
   - Consider server/system uptime

3. Price Feeds:
   - Uses Robinhood's crypto price feeds
   - Mark price for current values
   - Historical data for analysis

4. Risk Management:
   - Start with smaller positions
   - Monitor trades more frequently
   - Consider crypto-specific risk levels

# SAFETY FEATURES
--------------
1. Two-Factor Authentication
2. Stop Loss Protection
3. Trade History Tracking (Sample trades in trades/ directory)
4. Error Handling
5. Logging System

# DISCLAIMER
----------
This trading bot is for educational purposes only. Trading stocks carries risk, and you should never trade with money you cannot afford to lose. The authors are not responsible for any financial losses incurred through the use of this software.

# LICENSE
-------
MIT License

# SUPPORT
-------
For issues and questions:
1. Review debug logs
2. Verify system requirements
3. Check Robinhood API status

# VERSION INFORMATION
-----------------
Version: 1.0.0
Python: 3.9+
TA-Lib: 0.4.19
Robin Stocks: 3.0.6 