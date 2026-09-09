"""Confidence-weighted position sizing based on model probability and account risk."""
import config


def calculate_position_size(account_equity: float, price: float, probability: float,
                             stop_loss_pct: float = None) -> int:
    """
    Size a position so that if the stop-loss is hit, the loss is roughly
    BASE_RISK_PCT of equity, scaled up/down by how confident the model is.

    confidence_multiplier ranges ~0.2x-1.5x based on how far probability sits
    from the 0.5 (coin-flip) midpoint, so higher-conviction signals get bigger size.
    """
    stop_loss_pct = stop_loss_pct or config.STOP_LOSS_PCT

    conviction = min(abs(probability - 0.5) / 0.5, 1.0)  # 0..1
    confidence_multiplier = 0.2 + 1.3 * conviction  # 0.2x .. 1.5x

    risk_dollars = account_equity * config.BASE_RISK_PCT * confidence_multiplier
    position_value = risk_dollars / stop_loss_pct
    position_value = min(position_value, account_equity * config.MAX_POSITION_PCT)

    shares = int(position_value // price)
    return max(shares, 0)


def stop_loss_price(entry_price: float, side: str = "buy") -> float:
    if side == "buy":
        return round(entry_price * (1 - config.STOP_LOSS_PCT), 2)
    return round(entry_price * (1 + config.STOP_LOSS_PCT), 2)


def take_profit_price(entry_price: float, side: str = "buy") -> float:
    if side == "buy":
        return round(entry_price * (1 + config.TAKE_PROFIT_PCT), 2)
    return round(entry_price * (1 - config.TAKE_PROFIT_PCT), 2)
