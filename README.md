# ROBINHOOD TRADING BOT 
====================

A Python-based automated trading bot for Robinhood that uses RSI (Relative Strength Index) and other technical indicators to make trading decisions.

Note: This is a work in progress and is not yet ready for production use and only provides the sample trades in the trades/ directory.

# FEATURES
--------
* Automated trading based on RSI (Relative Strength Index)
* Two trading strategies: Basic and Advanced
* Real-time trade logging and tracking
* Trade history analysis
* Risk management with configurable initial amount
* Phone-based 2FA authentication
* JSON-based trade history storage
* Daily profit/loss tracking

# PREREQUISITES
------------
1. Python 3.9 or higher
2. Robinhood account
3. TA-Lib (Technical Analysis Library)
4. Active internet connection
5. Phone for 2FA authentication

# INSTALLATION INSTRUCTIONS
------------------------

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

1. BASIC USAGE:
```
   $ python trading_bot.py
```
2. ADVANCED USAGE:
```
   $ python trading_bot.py --debug --strategy advanced --initial-amount 500
```
3. ANALYZE TRADES:
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
```

# TRADING STRATEGIES
-----------------

1. BASIC STRATEGY:
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

# FILE STRUCTURE
-------------
```
robinhood-trading-bot/
├── trading_bot.py      : Main application file
├── requirements.txt    : Python dependencies
├── README.txt         : Documentation
├── trading_log.txt    : Trading activity logs
└── trades/            : Directory containing trade history
    └── trades_YYYYMMDD.json
```

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

# EXAMPLE USAGE SESSION
--------------------
```
$ python trading_bot.py --debug --strategy advanced
Enter Robinhood username: your_username
Enter Robinhood password: your_password
Enter the OTP sent to your phone: 123456
Enter stock symbol to trade: AAPL
```

# SAFETY FEATURES
--------------
1. Two-Factor Authentication
2. Stop Loss Protection
3. Trade History Tracking
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
1. Check the troubleshooting section
2. Review debug logs
3. Verify system requirements
4. Check Robinhood API status

# VERSION INFORMATION
-----------------
Version: 1.0.0
Python: 3.9+
TA-Lib: 0.4.19
Robin Stocks: 3.0.6 