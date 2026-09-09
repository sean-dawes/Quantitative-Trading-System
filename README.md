# Quantitative Trading System

> A machine-learning trading bot that scans a 20-stock watchlist every weekday morning, predicts next-day price direction, and automatically places (paper) trades through a rule-based risk management layer — no human intervention required.

**Note:** This project only trades with fake ("paper") money through Alpaca's practice trading platform. It's a learning and portfolio project, not real investment advice or a real trading track record.

---

## Live Dashboard

**[Live Dashboard →](https://sean-dawes-quantitative-trading-system-dashboard-gbtllc.streamlit.app/)**

View the bot's live equity, per-stock buy/sell calls, and full trade log through this public dashboard.

---

## What This Bot Does

Every weekday at 9:35 AM US/Eastern, hosted 24/7 on a cloud server, the bot:

1. Pulls 2 years of daily price history for all 20 watchlist stocks
2. Computes 15 technical indicator features for each one (RSI, MACD, Bollinger Bands, ATR, a 50-day trend measure, a stochastic oscillator, on-balance volume, and momentum/volume signals)
3. Feeds those features into a trained XGBoost model, which outputs a probability that the stock closes higher tomorrow
4. Converts that probability into a BUY / SELL / HOLD signal
5. Runs every BUY signal through a risk-management gate (volatility check, earnings blackout, news sentiment) before it's allowed to trade
6. Sizes and places any surviving trades through Alpaca's API as bracket orders (entry + stop-loss + take-profit)

No manual clicking, no daily check-ins required — it runs, decides, and trades entirely on its own.

---

## The Signal — How a Prediction Becomes a Trade

**Plain English:** The model was trained on how these 20 stocks actually moved over the past two years. For each stock, it looks at where RSI, MACD, and a handful of other indicators sit *today*, and estimates the odds the stock closes higher tomorrow than it is right now. That's the "signal" — not a guess, but a probability learned from real price history.

```python
BUY_PROB_THRESHOLD  = 0.60   # model's odds of "up" above this → BUY
SELL_PROB_THRESHOLD = 0.40   # model's odds of "up" below this → SELL / avoid
PREDICTION_HORIZON_DAYS = 1  # predicting tomorrow's close vs. today's
```

The model itself is a gradient-boosted decision tree classifier (XGBoost), trained on several years of pooled daily historical data across the watchlist, with hyperparameters chosen by a small validation-based search rather than one fixed guess:

```python
FEATURE_COLUMNS = [
    "rsi",
    "macd", "macd_signal", "macd_hist",
    "bb_pct", "bb_width",
    "atr_pct",
    "returns_1d", "returns_5d", "returns_10d", "returns_20d",
    "volume_change",
    "price_vs_sma50",
    "stoch_k",
    "obv_change",
]
```

Accuracy is evaluated on a per-ticker, time-ordered hold-out slice (each stock's own most recent data, never seen during training) — never better than a modest edge over a coin flip, which is exactly why the risk-management layer below matters more than the raw prediction itself.

---

## The Risk Gate — What Has to Be True Before Any Trade

**Plain English:** A BUY signal alone isn't enough. Before the bot commits real (paper) dollars, three separate checks all have to pass:

```
Model says BUY (probability ≥ 60%)
        ↓
Is this stock's volatility (ATR%) between 0.5% and 8%?
        → No  : skip — too flat or too wild to trust the signal
        → Yes ↓
Is an earnings announcement within the next 3 days?
        → Yes : skip — earnings moves overwhelm the signal
        → No  ↓
Is recent news sentiment above -0.2?
        → No  : skip — negative headlines override a technical BUY
        → Yes : trade approved → size the position and submit the order
```

```python
MAX_ATR_PCT = 0.08            # skip names more volatile than this
MIN_ATR_PCT = 0.005           # skip names too flat to trust
EARNINGS_BLACKOUT_DAYS = 3    # skip trading within N days of earnings
MIN_SENTIMENT_SCORE = -0.2    # skip BUYs when headline sentiment is this negative
```

---

## Position Sizing — How Much to Actually Bet

**Plain English:** The bot never bets a fixed amount. It sizes every trade so that *if the stop-loss gets hit*, the loss is roughly 1% of the account — then scales that up or down based on how confident the model's prediction was. A 51% "coin-flip" signal gets a small position; an 85% high-conviction signal gets close to the full 5%-of-account cap.

```python
def calculate_position_size(account_equity, price, probability, stop_loss_pct):
    conviction = min(abs(probability - 0.5) / 0.5, 1.0)      # 0 → 1
    confidence_multiplier = 0.2 + 1.3 * conviction             # 0.2x → 1.5x

    risk_dollars = account_equity * BASE_RISK_PCT * confidence_multiplier
    position_value = risk_dollars / stop_loss_pct
    position_value = min(position_value, account_equity * MAX_POSITION_PCT)

    return int(position_value // price)
```

| Parameter | Value | Meaning |
|---|---|---|
| `BASE_RISK_PCT` | 1% | Target loss on a stopped-out trade, at minimum conviction |
| `MAX_POSITION_PCT` | 5% | Hard cap — no single stock can exceed 5% of the account |
| `STOP_LOSS_PCT` | 3% | Automatic exit if the price drops 3% from entry |
| `TAKE_PROFIT_PCT` | 6% | Automatic exit if the price rises 6% from entry (2:1 reward-to-risk) |

Every order that goes out is a bracket order — the entry, stop-loss, and take-profit are all submitted together, so exits are automatic and don't depend on the bot being awake to react.

---

## Watchlist

The bot scans these 20 large-cap U.S. stocks every trading day:

| | | | | |
|---|---|---|---|---|
| AAPL | MSFT | GOOGL | AMZN | NVDA |
| META | TSLA | AMD | NFLX | JPM |
| V | UNH | XOM | PG | HD |
| DIS | BAC | KO | PEP | CSCO |

---

## Tech Stack

- **Python** — core language for the entire pipeline
- **XGBoost / scikit-learn** — the prediction model
- **pandas / NumPy** — indicator math and feature engineering
- **yfinance** — historical price and news data
- **Alpaca Trade API** — automated paper trade execution
- **VADER (vaderSentiment)** — headline sentiment scoring
- **APScheduler** — daily cron-style scheduling
- **Railway** — 24/7 cloud hosting
- **Git / GitHub** — version control

## Project Structure

```
algo-trading-bot/
├── config.py                thresholds, risk parameters, paths
├── watchlist.py              the 20-stock universe
├── data_pipeline/              price history + news headlines (yfinance)
├── indicators/                   RSI / MACD / Bollinger / ATR, built from scratch
├── model/                          feature engineering, training, prediction
├── risk/                             position sizing, volatility/earnings filters, sentiment gate
├── execution/                          Alpaca broker wrapper + trade executor
├── run_bot.py                            one full daily run across the watchlist
├── scheduler.py                            cron-style scheduler for 24/7 deployment
├── train_model.py                          CLI to (re)train the model
└── tests/                                    unit tests for the indicators
```

---

## Running It Locally

```bash
git clone https://github.com/sean-dawes/Quantitative-Trading-System.git
cd Quantitative-Trading-System

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env
# paste in your own Alpaca PAPER trading API keys

python train_model.py           # train the model first
python run_bot.py               # run one trading pass by hand
python scheduler.py             # or run it automatically every weekday morning
```

---

## Disclaimer

This is an educational project built for learning and portfolio purposes. It only trades Alpaca **paper** (practice) accounts, never real money. Nothing in this project is financial advice, and past or simulated performance does not predict future results.
