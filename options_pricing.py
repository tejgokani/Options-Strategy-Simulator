"""
Black-Scholes pricing model for European options.

This module implements the Black-Scholes-Merton formula for pricing European
options and calculates the Greeks (sensitivities) that traders use to manage risk.

Key financial concepts:
- Black-Scholes assumes constant volatility and risk-free rate
- Delta: price sensitivity to underlying movement (directional risk)
- Vega: price sensitivity to volatility changes (volatility risk)
- Theta: time decay (not implemented here, but important for traders)
- Gamma: convexity (rate of change of delta)
"""

import numpy as np
from scipy.stats import norm
from typing import Optional


def black_scholes_price(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: str = "call"
) -> float:
    """
    Calculate Black-Scholes option price.
    
    Args:
        spot: Current underlying price
        strike: Strike price
        time_to_expiry: Time to expiration in years
        risk_free_rate: Annual risk-free rate (e.g., 0.05 for 5%)
        volatility: Annual volatility (e.g., 0.20 for 20%)
        option_type: "call" or "put"
    
    Returns:
        Option premium
    
    Financial intuition:
        - Higher volatility increases premium (more uncertainty = higher value)
        - Longer time to expiry increases premium (more time for price to move)
        - Deep ITM options trade near intrinsic value (spot - strike for calls)
        - Deep OTM options trade near zero
    """
    if time_to_expiry <= 0:
        # At expiry, option value is intrinsic value
        if option_type == "call":
            return max(spot - strike, 0)
        else:
            return max(strike - spot, 0)
    
    # Black-Scholes formula components
    d1 = (np.log(spot / strike) + (risk_free_rate + 0.5 * volatility ** 2) * time_to_expiry) / (
        volatility * np.sqrt(time_to_expiry)
    )
    d2 = d1 - volatility * np.sqrt(time_to_expiry)
    
    if option_type == "call":
        price = spot * norm.cdf(d1) - strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2)
    else:  # put
        price = strike * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) - spot * norm.cdf(-d1)
    
    return max(price, 0)  # Option cannot have negative value


def black_scholes_delta(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: str = "call"
) -> float:
    """
    Calculate option Delta (price sensitivity to underlying).
    
    Delta interpretation:
        - Call delta: 0 to 1 (increases as spot rises)
        - Put delta: -1 to 0 (decreases as spot rises)
        - Delta near 0.5: at-the-money option
        - Delta near 1/-1: deep in-the-money
        - Delta near 0: deep out-of-the-money
    
    For strategies:
        - Long call: positive delta (bullish)
        - Long put: negative delta (bearish)
        - Net delta determines directional exposure
    """
    if time_to_expiry <= 0:
        # At expiry, delta is step function
        if option_type == "call":
            return 1.0 if spot > strike else 0.0
        else:
            return -1.0 if spot < strike else 0.0
    
    d1 = (np.log(spot / strike) + (risk_free_rate + 0.5 * volatility ** 2) * time_to_expiry) / (
        volatility * np.sqrt(time_to_expiry)
    )
    
    if option_type == "call":
        return norm.cdf(d1)
    else:  # put
        return -norm.cdf(-d1)


def black_scholes_vega(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: str = "call"
) -> float:
    """
    Calculate option Vega (price sensitivity to volatility).
    
    Vega interpretation:
        - Always positive for long options (both calls and puts)
        - Higher for at-the-money options
        - Increases with time to expiry
        - Measured as price change per 1% volatility change
    
    For strategies:
        - Long options: positive vega (benefit from volatility increase)
        - Short options: negative vega (hurt by volatility increase)
        - Net vega determines volatility exposure
    """
    if time_to_expiry <= 0:
        return 0.0  # No time value at expiry
    
    d1 = (np.log(spot / strike) + (risk_free_rate + 0.5 * volatility ** 2) * time_to_expiry) / (
        volatility * np.sqrt(time_to_expiry)
    )
    
    # Vega is the same for calls and puts
    vega = spot * norm.pdf(d1) * np.sqrt(time_to_expiry) * 0.01  # Per 1% vol change
    return vega


def calculate_greeks(
    spot: float,
    strike: float,
    time_to_expiry: float,
    risk_free_rate: float,
    volatility: float,
    option_type: str = "call"
) -> dict:
    """
    Calculate all Greeks for a single option.
    
    Returns:
        Dictionary with 'delta' and 'vega' keys
    """
    return {
        "delta": black_scholes_delta(spot, strike, time_to_expiry, risk_free_rate, volatility, option_type),
        "vega": black_scholes_vega(spot, strike, time_to_expiry, risk_free_rate, volatility, option_type)
    }

