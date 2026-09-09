# Quantitative Trading System

A trading bot that predicts short-term stock price movements and automatically
places (paper) trades based on those predictions — built to combine a finance
background with real coding and data skills.

**Note:** This project only trades with fake ("paper") money through Alpaca's
practice trading platform. It's a learning and portfolio project, not real
investment advice or a real trading track record.

## What This Project Does

Every weekday morning, this bot:

1. Looks at 20 well-known stocks (Apple, Microsoft, Tesla, and others).
2. Uses a machine learning model to predict whether each stock is more likely
   to go up or down over the next day.
3. Runs those predictions through a set of risk-management rules before
   deciding whether to actually trade.
4. Automatically places any resulting trades through a brokerage account —
   with built-in stop-loss and take-profit orders — using fake money.
5. Repeats this process every trading day, on its own, running 24/7 in the
   cloud.

## Why I Built This

As a finance student on the CFA track, I wanted a hands-on way to combine
what I'm learning about markets and risk management with real technical
skills — data analysis, machine learning, and software deployment. This
project let me build every piece of that pipeline myself, from the
prediction model to the risk controls to the live deployment.

## How It Works (In Plain English)

**Step 1 — Reading the market.**
The bot pulls two years of daily price history for each stock and
calculates a handful of common "technical indicators" — measurements
traders use to spot patterns in price and momentum (things like RSI,
MACD, and Bollinger Bands).

**Step 2 — Making a prediction.**
Those indicators get fed into a machine learning model (specifically an
XGBoost classifier), which was trained on historical data to recognize
patterns that tend to come before a stock goes up or down. Each day, the
model estimates a probability that a given stock will rise the next day.

**Step 3 — Deciding whether it's actually worth trading.**
A prediction alone isn't enough to trade on. Before any trade happens, it
has to pass a series of risk checks:
- Is this stock too quiet or too wildly volatile right now?
- Is the company about to announce earnings (which can cause unpredictable
  price swings)?
- Is recent news about this stock too negative?

Only trades that clear all of these checks move forward.

**Step 4 — Sizing and placing the trade.**
The bot decides how much money to put into a trade based on how confident
the model's prediction was — more confidence, slightly larger position;
less confidence, smaller position. Every trade automatically comes with a
built-in stop-loss (to limit losses) and take-profit (to lock in gains).

**Step 5 — Running automatically.**
The entire process above repeats on its own every weekday morning, hosted
on a cloud server so it runs continuously without needing a computer to be
turned on.

## Key Features

| Feature | What it does |
|---|---|
| Machine learning predictions | Learns patterns from 2 years of historical price data instead of relying on fixed, hardcoded rules |
| Technical indicators | RSI, MACD, Bollinger Bands, and ATR (volatility) — common tools traders use to read price charts |
| Risk management | Position sizing, stop-loss/take-profit, volatility filtering, earnings blackout, news sentiment check |
| Automated execution | Places trades automatically through Alpaca's paper trading API — no manual clicking required |
| 24/7 cloud deployment | Runs on a schedule every trading day, hosted remotely rather than on a personal computer |

## Tech Stack

- **Python** — the language the entire project is built in
- **XGBoost / scikit-learn** — the machine learning model
- **Pandas / NumPy** — data processing
- **Alpaca API** — brokerage connection for placing paper trades
- **Railway** — cloud hosting for 24/7 automated execution
- **Git / GitHub** — version control

## Setup (For Anyone Who Wants to Run This)

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

Train the prediction model (run this first):

```bash
python train_model.py
```

Run one trading check by hand:

```bash
python run_bot.py
```

Run it automatically every weekday morning:

```bash
python scheduler.py
```

## Disclaimer

This is an educational project built for learning and portfolio purposes.
It only trades Alpaca **paper** (practice) accounts, never real money.
Nothing in this project is financial advice, and past or simulated
performance does not predict future results.
