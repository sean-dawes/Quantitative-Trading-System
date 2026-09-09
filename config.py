"""
Central configuration for the trading bot.
Loads secrets from a local .env file (never commit .env).
"""
import os
from dotenv import load_dotenv

load_dotenv()

# --- Alpaca (paper trading) ---
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
PAPER_TRADING = True  # this project only ever trades paper, by design

# --- Model / signal thresholds ---
BUY_PROB_THRESHOLD = 0.60   # model P(up) above this -> BUY
SELL_PROB_THRESHOLD = 0.40  # model P(up) below this -> SELL / avoid
PREDICTION_HORIZON_DAYS = 1  # predict next-day direction
LOOKBACK_PERIOD = "5y"       # historical data window used for training
DATA_INTERVAL = "1d"

# --- Risk management ---
MAX_POSITION_PCT = 0.05      # never put more than 5% of equity in one name
BASE_RISK_PCT = 0.01         # risk 1% of equity per trade at the stop-loss
STOP_LOSS_PCT = 0.03         # 3% below entry
TAKE_PROFIT_PCT = 0.06       # 6% above entry (2:1 reward:risk)
MAX_ATR_PCT = 0.08           # skip names whose ATR% of price exceeds this (too volatile)
MIN_ATR_PCT = 0.005          # skip names that are basically flat (no signal edge)
EARNINGS_BLACKOUT_DAYS = 3   # skip trading within N days of an earnings date
MIN_SENTIMENT_SCORE = -0.2   # skip BUY signals when recent headline sentiment is below this

# --- Paths ---
MODEL_DIR = "models"
MODEL_PATH = f"{MODEL_DIR}/xgb_model.pkl"
FEATURE_COLUMNS_PATH = f"{MODEL_DIR}/feature_columns.json"
LOG_DIR = "logs"
TRADE_LOG_PATH = f"{LOG_DIR}/trade_log.csv"
