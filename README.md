# Algorithmic Trading Bot

A daily algorithmic trading system that generates buy/sell signals on a
20-stock watchlist using an XGBoost classifier trained on technical
indicators (RSI, MACD, Bollinger Bands, ATR), then paper-trades those
signals through Alpaca's API behind a rule-based risk management layer
(confidence-weighted position sizing, stop-loss/take-profit brackets,
a volatility filter, an earnings-date blackout, and headline sentiment
gating).

This project trades **paper money only**. It is for learning/portfolio
purposes, not investment advice.

## How it works

1. `indicators/technical.py` computes RSI, MACD, Bollinger Bands, and ATR
   from OHLCV price history (pulled via `yfinance`).
2. `model/train.py` builds a labeled dataset across the whole watchlist
   (label = did the stock close higher the next day?) and trains an
   `XGBClassifier` on it.
3. `model/predict.py` loads that model and turns today's indicators into a
   probability of a next-day up move, then a BUY / SELL / HOLD signal.
4. `risk/` filters out low-quality trades (too volatile, near an earnings
   date, negative news sentiment) and sizes whatever's left based on model
   confidence and a fixed % of account equity at risk.
5. `execution/broker.py` and `execution/executor.py` send the surviving
   trades to Alpaca as bracket orders (entry + stop-loss + take-profit).
6. `run_bot.py` ties it all together for one run; `scheduler.py` runs it
   automatically every weekday morning, for a 24/7-hosted deployment.

## Setup

```bash
cd algo-trading-bot
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Get free Alpaca **paper trading** API keys at
https://app.alpaca.markets/paper/dashboard/overview, then:

```bash
cp .env.example .env
# edit .env and paste in your ALPACA_API_KEY / ALPACA_SECRET_KEY
```

## Usage

Train the model (run this first, and re-run periodically to refresh it):

```bash
python train_model.py
```

Run one trading pass by hand:

```bash
python run_bot.py
```

Run automatically every weekday morning (leave this running, e.g. in a
`screen`/`tmux` session, on a small cloud VM, or as a systemd service):

```bash
python scheduler.py
```

Run the indicator unit tests:

```bash
pytest tests/
```

## Project layout

```
config.py              thresholds, risk parameters, paths
watchlist.py            the 20-stock universe
data_pipeline/           price history + news headlines (yfinance)
indicators/               RSI / MACD / Bollinger / ATR
model/                     feature engineering, training, prediction
risk/                       position sizing, volatility/earnings filters, sentiment gate
execution/                   Alpaca broker wrapper + trade executor
run_bot.py                    one full daily run
scheduler.py                   cron-style scheduler for a 24/7 deployment
train_model.py                  CLI to (re)train the model
tests/                            unit tests for the indicators
```

## Tuning

Everything that shapes risk and signal thresholds lives in `config.py`:
buy/sell probability cutoffs, max position size, stop-loss/take-profit %,
the ATR band used for the volatility filter, the earnings blackout window,
and the minimum sentiment score to allow a BUY.

## Disclaimer

This is an educational project. It trades Alpaca **paper** accounts only.
Nothing here is financial advice, and past/simulated performance is not
indicative of future results.
