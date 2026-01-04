"""
Payoff calculations for common options strategies.

Each strategy function calculates the payoff at expiry for a range of underlying prices.
Payoff = value at expiry - net premium paid (if applicable).

Key concepts:
- Payoff diagrams show profit/loss at expiry
- Breakeven: underlying price where strategy breaks even
- Maximum profit/loss: bounds of strategy outcomes
- Convexity: curvature of payoff (gamma exposure)
"""

import numpy as np
from typing import Tuple, Dict, Optional


def long_call_payoff(
    spot_prices: np.ndarray,
    strike: float,
    premium: float
) -> np.ndarray:
    """
    Long Call payoff: Buy a call option.
    
    Strategy: Bullish, limited downside, unlimited upside.
    Payoff = max(spot - strike, 0) - premium
    
    Breakeven: strike + premium
    Max loss: -premium (if spot <= strike)
    Max profit: Unlimited (as spot -> infinity)
    """
    intrinsic = np.maximum(spot_prices - strike, 0)
    return intrinsic - premium


def long_put_payoff(
    spot_prices: np.ndarray,
    strike: float,
    premium: float
) -> np.ndarray:
    """
    Long Put payoff: Buy a put option.
    
    Strategy: Bearish, limited downside, large upside (bounded by spot -> 0).
    Payoff = max(strike - spot, 0) - premium
    
    Breakeven: strike - premium
    Max loss: -premium (if spot >= strike)
    Max profit: strike - premium (if spot -> 0)
    """
    intrinsic = np.maximum(strike - spot_prices, 0)
    return intrinsic - premium


def covered_call_payoff(
    spot_prices: np.ndarray,
    initial_spot: float,
    strike: float,
    call_premium: float
) -> np.ndarray:
    """
    Covered Call payoff: Own stock + sell call option.
    
    Strategy: Neutral to slightly bullish, income generation.
    Payoff = (spot - initial_spot) + min(strike - spot, 0) + call_premium
    
    Intuition:
        - Stock ownership provides upside to strike
        - Short call caps upside at strike
        - Premium collected reduces cost basis
    
    Breakeven: initial_spot - call_premium
    Max loss: Unlimited downside (stock can go to zero)
    Max profit: (strike - initial_spot) + call_premium
    """
    stock_pnl = spot_prices - initial_spot
    short_call_pnl = -np.maximum(spot_prices - strike, 0) + call_premium
    return stock_pnl + short_call_pnl


def bull_call_spread_payoff(
    spot_prices: np.ndarray,
    lower_strike: float,
    higher_strike: float,
    lower_premium: float,
    higher_premium: float
) -> np.ndarray:
    """
    Bull Call Spread: Buy lower strike call, sell higher strike call.
    
    Strategy: Moderately bullish, limited risk/reward.
    Payoff = max(spot - lower_strike, 0) - max(spot - higher_strike, 0) - net_premium
    
    Intuition:
        - Long call provides upside participation
        - Short call caps upside but reduces cost
        - Net premium = lower_premium - higher_premium (typically positive)
    
    Breakeven: lower_strike + net_premium
    Max loss: -net_premium (if spot <= lower_strike)
    Max profit: (higher_strike - lower_strike) - net_premium (if spot >= higher_strike)
    """
    net_premium = lower_premium - higher_premium
    long_call = np.maximum(spot_prices - lower_strike, 0)
    short_call = -np.maximum(spot_prices - higher_strike, 0)
    return long_call + short_call - net_premium


def bear_put_spread_payoff(
    spot_prices: np.ndarray,
    higher_strike: float,
    lower_strike: float,
    higher_premium: float,
    lower_premium: float
) -> np.ndarray:
    """
    Bear Put Spread: Buy higher strike put, sell lower strike put.
    
    Strategy: Moderately bearish, limited risk/reward.
    Payoff = max(higher_strike - spot, 0) - max(lower_strike - spot, 0) - net_premium
    
    Intuition:
        - Long put provides downside protection
        - Short put caps profit but reduces cost
        - Net premium = higher_premium - lower_premium (typically positive)
    
    Breakeven: higher_strike - net_premium
    Max loss: -net_premium (if spot >= higher_strike)
    Max profit: (higher_strike - lower_strike) - net_premium (if spot <= lower_strike)
    """
    net_premium = higher_premium - lower_premium
    long_put = np.maximum(higher_strike - spot_prices, 0)
    short_put = -np.maximum(lower_strike - spot_prices, 0)
    return long_put + short_put - net_premium


def long_straddle_payoff(
    spot_prices: np.ndarray,
    strike: float,
    call_premium: float,
    put_premium: float
) -> np.ndarray:
    """
    Long Straddle: Buy call and put at same strike.
    
    Strategy: Volatility play, direction-neutral.
    Payoff = max(spot - strike, 0) + max(strike - spot, 0) - total_premium
    
    Intuition:
        - Profits from large moves in either direction
        - Maximum loss if stock stays near strike
        - High vega exposure (benefits from volatility increase)
    
    Breakevens: strike - total_premium (downside), strike + total_premium (upside)
    Max loss: -total_premium (if spot = strike at expiry)
    Max profit: Unlimited (both directions)
    """
    total_premium = call_premium + put_premium
    call_payoff = np.maximum(spot_prices - strike, 0)
    put_payoff = np.maximum(strike - spot_prices, 0)
    return call_payoff + put_payoff - total_premium


def long_strangle_payoff(
    spot_prices: np.ndarray,
    call_strike: float,
    put_strike: float,
    call_premium: float,
    put_premium: float
) -> np.ndarray:
    """
    Long Strangle: Buy OTM call and OTM put (different strikes).
    
    Strategy: Volatility play, cheaper than straddle, wider breakeven.
    Payoff = max(spot - call_strike, 0) + max(put_strike - spot, 0) - total_premium
    
    Intuition:
        - Similar to straddle but cheaper (both options OTM)
        - Requires larger move to profit
        - Lower cost = lower max loss
    
    Breakevens: put_strike - total_premium (downside), call_strike + total_premium (upside)
    Max loss: -total_premium (if put_strike <= spot <= call_strike)
    Max profit: Unlimited (both directions)
    """
    total_premium = call_premium + put_premium
    call_payoff = np.maximum(spot_prices - call_strike, 0)
    put_payoff = np.maximum(put_strike - spot_prices, 0)
    return call_payoff + put_payoff - total_premium


def calculate_breakevens(
    payoff_func,
    spot_range: np.ndarray,
    *args
) -> list:
    """
    Find breakeven points where payoff crosses zero.
    
    Uses linear interpolation between grid points.
    """
    payoffs = payoff_func(spot_range, *args)
    breakevens = []
    
    for i in range(len(spot_range) - 1):
        if payoffs[i] * payoffs[i + 1] <= 0:  # Sign change
            # Linear interpolation
            x1, x2 = spot_range[i], spot_range[i + 1]
            y1, y2 = payoffs[i], payoffs[i + 1]
            if y1 != y2:
                breakeven = x1 - y1 * (x2 - x1) / (y2 - y1)
                breakevens.append(breakeven)
    
    return sorted(breakevens)


def calculate_max_profit_loss(
    payoffs: np.ndarray
) -> Dict[str, float]:
    """
    Calculate maximum profit and loss from payoff array.
    """
    return {
        "max_profit": float(np.max(payoffs)),
        "max_loss": float(np.min(payoffs))
    }

