"""Headline sentiment gating: blocks BUY signals when recent news sentiment is too negative."""
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import config
from data_pipeline.news_data import get_recent_headlines

_analyzer = SentimentIntensityAnalyzer()


def get_sentiment_score(headlines) -> float:
    """Average VADER compound sentiment score across headlines, in [-1, 1]. 0 if no headlines."""
    if not headlines:
        return 0.0
    scores = [_analyzer.polarity_scores(h)["compound"] for h in headlines]
    return sum(scores) / len(scores)


def passes_sentiment_gate(ticker: str) -> bool:
    """Only relevant for BUY signals -- don't buy into a name with very negative recent news."""
    headlines = get_recent_headlines(ticker)
    score = get_sentiment_score(headlines)
    return score >= config.MIN_SENTIMENT_SCORE
