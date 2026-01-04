"""
Main strategy analyzer that combines pricing, payoffs, and risk metrics.

This module provides a unified interface for analyzing options strategies,
calculating payoffs, Greeks, and providing risk interpretations.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from options_pricing import (
    black_scholes_price,
    black_scholes_delta,
    black_scholes_vega
)
from strategy_payoffs import (
    long_call_payoff,
    long_put_payoff,
    covered_call_payoff,
    bull_call_spread_payoff,
    bear_put_spread_payoff,
    long_straddle_payoff,
    long_strangle_payoff,
    calculate_breakevens,
    calculate_max_profit_loss
)


class OptionsStrategy:
    """
    Base class for analyzing options strategies.
    
    Handles:
    - Option pricing (Black-Scholes)
    - Payoff calculations
    - Greeks aggregation
    - Risk interpretation
    """
    
    def __init__(
        self,
        spot: float,
        time_to_expiry: float,
        risk_free_rate: float,
        volatility: float,
        strategy_name: str
    ):
        """
        Initialize strategy parameters.
        
        Args:
            spot: Current underlying price
            time_to_expiry: Time to expiration in years
            risk_free_rate: Annual risk-free rate
            volatility: Annual volatility
            strategy_name: Name of the strategy
        """
        self.spot = spot
        self.time_to_expiry = time_to_expiry
        self.risk_free_rate = risk_free_rate
        self.volatility = volatility
        self.strategy_name = strategy_name
        
        # Will be populated by strategy-specific methods
        self.premiums = {}
        self.strikes = {}
        self.option_types = {}
        self.positions = {}  # +1 for long, -1 for short
        
    def _price_option(
        self,
        strike: float,
        option_type: str,
        premium_override: Optional[float] = None
    ) -> float:
        """Calculate or override option premium."""
        if premium_override is not None:
            return premium_override
        return black_scholes_price(
            self.spot, strike, self.time_to_expiry,
            self.risk_free_rate, self.volatility, option_type
        )
    
    def _calculate_greeks(self) -> Dict[str, float]:
        """
        Aggregate Greeks across all options in strategy.
        
        Returns:
            Dictionary with 'delta' and 'vega' (net exposure)
        """
        net_delta = 0.0
        net_vega = 0.0
        
        for key, strike in self.strikes.items():
            option_type = self.option_types[key]
            position = self.positions[key]
            
            delta = black_scholes_delta(
                self.spot, strike, self.time_to_expiry,
                self.risk_free_rate, self.volatility, option_type
            )
            vega = black_scholes_vega(
                self.spot, strike, self.time_to_expiry,
                self.risk_free_rate, self.volatility, option_type
            )
            
            net_delta += position * delta
            net_vega += position * vega
        
        return {"delta": net_delta, "vega": net_vega}
    
    def _interpret_risk(self, greeks: Dict[str, float]) -> str:
        """
        Provide qualitative risk interpretation.
        
        Helps traders understand:
        - Directional exposure (delta)
        - Volatility sensitivity (vega)
        - Strategy classification
        """
        delta = greeks["delta"]
        vega = greeks["vega"]
        
        # Directional interpretation
        if abs(delta) > 0.5:
            direction = "Strongly directional"
            bias = "bullish" if delta > 0 else "bearish"
        elif abs(delta) > 0.2:
            direction = "Moderately directional"
            bias = "bullish" if delta > 0 else "bearish"
        else:
            direction = "Direction-neutral"
            bias = ""
        
        # Volatility interpretation
        if abs(vega) > 0.1:
            vol_sensitivity = "High volatility sensitivity"
            vol_bias = "benefits from" if vega > 0 else "hurt by"
        elif abs(vega) > 0.05:
            vol_sensitivity = "Moderate volatility sensitivity"
            vol_bias = "benefits from" if vega > 0 else "hurt by"
        else:
            vol_sensitivity = "Low volatility sensitivity"
            vol_bias = ""
        
        interpretation = f"{direction}"
        if bias:
            interpretation += f" ({bias})"
        interpretation += f". {vol_sensitivity}"
        if vol_bias:
            interpretation += f" ({vol_bias} volatility increases)"
        
        return interpretation
    
    def analyze(
        self,
        spot_range: np.ndarray,
        premium_overrides: Optional[Dict[str, float]] = None
    ) -> Dict:
        """
        Complete strategy analysis.
        
        Args:
            spot_range: Array of underlying prices for payoff calculation
            premium_overrides: Optional dict to override calculated premiums
        
        Returns:
            Dictionary with payoffs, Greeks, breakevens, max profit/loss, etc.
        """
        # Calculate payoffs
        payoffs = self.calculate_payoff(spot_range, premium_overrides)
        
        # Calculate Greeks
        greeks = self._calculate_greeks()
        
        # Calculate breakevens
        def payoff_func(spot_range, *args):
            return self.calculate_payoff(spot_range, premium_overrides)
        breakevens = calculate_breakevens(payoff_func, spot_range)
        
        # Max profit/loss
        profit_loss = calculate_max_profit_loss(payoffs)
        
        # Risk interpretation
        risk_interpretation = self._interpret_risk(greeks)
        
        return {
            "strategy_name": self.strategy_name,
            "spot_range": spot_range,
            "payoffs": payoffs,
            "greeks": greeks,
            "breakevens": breakevens,
            "max_profit": profit_loss["max_profit"],
            "max_loss": profit_loss["max_loss"],
            "risk_interpretation": risk_interpretation,
            "premiums": self.premiums.copy(),
            "description": self.get_description(),
            "initial_spot": self.spot
        }
    
    def calculate_payoff(
        self,
        spot_prices: np.ndarray,
        premium_overrides: Optional[Dict[str, float]] = None
    ) -> np.ndarray:
        """
        Calculate payoff for given spot prices.
        Must be implemented by strategy-specific subclasses.
        """
        raise NotImplementedError
    
    def get_description(self) -> str:
        """
        Get plain English description of strategy.
        Must be implemented by strategy-specific subclasses.
        """
        raise NotImplementedError


class LongCall(OptionsStrategy):
    """Long Call strategy: Buy a call option."""
    
    def __init__(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        risk_free_rate: float,
        volatility: float,
        premium_override: Optional[float] = None
    ):
        super().__init__(spot, time_to_expiry, risk_free_rate, volatility, "Long Call")
        self.strike = strike
        self.premiums["call"] = self._price_option(strike, "call", premium_override)
        self.strikes["call"] = strike
        self.option_types["call"] = "call"
        self.positions["call"] = 1  # Long
    
    def calculate_payoff(self, spot_prices: np.ndarray, premium_overrides: Optional[Dict] = None) -> np.ndarray:
        premium = premium_overrides.get("call", self.premiums["call"]) if premium_overrides else self.premiums["call"]
        return long_call_payoff(spot_prices, self.strike, premium)
    
    def get_description(self) -> str:
        return (
            f"Buy a call option with strike ${self.strike:.2f}. "
            f"Bullish strategy with limited downside (max loss = premium ${self.premiums['call']:.2f}) "
            f"and unlimited upside potential. Breakeven at ${self.strike + self.premiums['call']:.2f}."
        )


class LongPut(OptionsStrategy):
    """Long Put strategy: Buy a put option."""
    
    def __init__(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        risk_free_rate: float,
        volatility: float,
        premium_override: Optional[float] = None
    ):
        super().__init__(spot, time_to_expiry, risk_free_rate, volatility, "Long Put")
        self.strike = strike
        self.premiums["put"] = self._price_option(strike, "put", premium_override)
        self.strikes["put"] = strike
        self.option_types["put"] = "put"
        self.positions["put"] = 1  # Long
    
    def calculate_payoff(self, spot_prices: np.ndarray, premium_overrides: Optional[Dict] = None) -> np.ndarray:
        premium = premium_overrides.get("put", self.premiums["put"]) if premium_overrides else self.premiums["put"]
        return long_put_payoff(spot_prices, self.strike, premium)
    
    def get_description(self) -> str:
        return (
            f"Buy a put option with strike ${self.strike:.2f}. "
            f"Bearish strategy with limited downside (max loss = premium ${self.premiums['put']:.2f}) "
            f"and large upside potential if stock falls. Breakeven at ${self.strike - self.premiums['put']:.2f}."
        )


class CoveredCall(OptionsStrategy):
    """Covered Call strategy: Own stock + sell call."""
    
    def __init__(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        risk_free_rate: float,
        volatility: float,
        call_premium_override: Optional[float] = None
    ):
        super().__init__(spot, time_to_expiry, risk_free_rate, volatility, "Covered Call")
        self.strike = strike
        self.initial_spot = spot
        self.premiums["call"] = self._price_option(strike, "call", call_premium_override)
        self.strikes["call"] = strike
        self.option_types["call"] = "call"
        self.positions["call"] = -1  # Short call
        # Stock position: +1 (long stock, not an option, so no Greeks from stock itself)
    
    def calculate_payoff(self, spot_prices: np.ndarray, premium_overrides: Optional[Dict] = None) -> np.ndarray:
        premium = premium_overrides.get("call", self.premiums["call"]) if premium_overrides else self.premiums["call"]
        return covered_call_payoff(spot_prices, self.initial_spot, self.strike, premium)
    
    def _calculate_greeks(self) -> Dict[str, float]:
        # Stock has delta = 1, vega = 0
        greeks = super()._calculate_greeks()
        greeks["delta"] += 1.0  # Long stock
        return greeks
    
    def get_description(self) -> str:
        return (
            f"Own stock at ${self.initial_spot:.2f} and sell a call with strike ${self.strike:.2f}. "
            f"Income generation strategy: collect ${self.premiums['call']:.2f} premium but cap upside at strike. "
            f"Breakeven at ${self.initial_spot - self.premiums['call']:.2f}."
        )


class BullCallSpread(OptionsStrategy):
    """Bull Call Spread: Buy lower strike call, sell higher strike call."""
    
    def __init__(
        self,
        spot: float,
        lower_strike: float,
        higher_strike: float,
        time_to_expiry: float,
        risk_free_rate: float,
        volatility: float,
        premium_overrides: Optional[Dict[str, float]] = None
    ):
        super().__init__(spot, time_to_expiry, risk_free_rate, volatility, "Bull Call Spread")
        self.lower_strike = lower_strike
        self.higher_strike = higher_strike
        
        lower_premium_override = premium_overrides.get("lower") if premium_overrides else None
        higher_premium_override = premium_overrides.get("higher") if premium_overrides else None
        
        self.premiums["lower"] = self._price_option(lower_strike, "call", lower_premium_override)
        self.premiums["higher"] = self._price_option(higher_strike, "call", higher_premium_override)
        
        self.strikes["lower"] = lower_strike
        self.strikes["higher"] = higher_strike
        self.option_types["lower"] = "call"
        self.option_types["higher"] = "call"
        self.positions["lower"] = 1  # Long
        self.positions["higher"] = -1  # Short
    
    def calculate_payoff(self, spot_prices: np.ndarray, premium_overrides: Optional[Dict] = None) -> np.ndarray:
        lower_prem = premium_overrides.get("lower", self.premiums["lower"]) if premium_overrides else self.premiums["lower"]
        higher_prem = premium_overrides.get("higher", self.premiums["higher"]) if premium_overrides else self.premiums["higher"]
        return bull_call_spread_payoff(spot_prices, self.lower_strike, self.higher_strike, lower_prem, higher_prem)
    
    def get_description(self) -> str:
        net_premium = self.premiums["lower"] - self.premiums["higher"]
        max_profit = (self.higher_strike - self.lower_strike) - net_premium
        return (
            f"Buy call at ${self.lower_strike:.2f}, sell call at ${self.higher_strike:.2f}. "
            f"Moderately bullish with limited risk. Net cost: ${net_premium:.2f}. "
            f"Max profit: ${max_profit:.2f} if stock >= ${self.higher_strike:.2f}."
        )


class BearPutSpread(OptionsStrategy):
    """Bear Put Spread: Buy higher strike put, sell lower strike put."""
    
    def __init__(
        self,
        spot: float,
        higher_strike: float,
        lower_strike: float,
        time_to_expiry: float,
        risk_free_rate: float,
        volatility: float,
        premium_overrides: Optional[Dict[str, float]] = None
    ):
        super().__init__(spot, time_to_expiry, risk_free_rate, volatility, "Bear Put Spread")
        self.higher_strike = higher_strike
        self.lower_strike = lower_strike
        
        higher_premium_override = premium_overrides.get("higher") if premium_overrides else None
        lower_premium_override = premium_overrides.get("lower") if premium_overrides else None
        
        self.premiums["higher"] = self._price_option(higher_strike, "put", higher_premium_override)
        self.premiums["lower"] = self._price_option(lower_strike, "put", lower_premium_override)
        
        self.strikes["higher"] = higher_strike
        self.strikes["lower"] = lower_strike
        self.option_types["higher"] = "put"
        self.option_types["lower"] = "put"
        self.positions["higher"] = 1  # Long
        self.positions["lower"] = -1  # Short
    
    def calculate_payoff(self, spot_prices: np.ndarray, premium_overrides: Optional[Dict] = None) -> np.ndarray:
        higher_prem = premium_overrides.get("higher", self.premiums["higher"]) if premium_overrides else self.premiums["higher"]
        lower_prem = premium_overrides.get("lower", self.premiums["lower"]) if premium_overrides else self.premiums["lower"]
        return bear_put_spread_payoff(spot_prices, self.higher_strike, self.lower_strike, higher_prem, lower_prem)
    
    def get_description(self) -> str:
        net_premium = self.premiums["higher"] - self.premiums["lower"]
        max_profit = (self.higher_strike - self.lower_strike) - net_premium
        return (
            f"Buy put at ${self.higher_strike:.2f}, sell put at ${self.lower_strike:.2f}. "
            f"Moderately bearish with limited risk. Net cost: ${net_premium:.2f}. "
            f"Max profit: ${max_profit:.2f} if stock <= ${self.lower_strike:.2f}."
        )


class LongStraddle(OptionsStrategy):
    """Long Straddle: Buy call and put at same strike."""
    
    def __init__(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        risk_free_rate: float,
        volatility: float,
        premium_overrides: Optional[Dict[str, float]] = None
    ):
        super().__init__(spot, time_to_expiry, risk_free_rate, volatility, "Long Straddle")
        self.strike = strike
        
        call_premium_override = premium_overrides.get("call") if premium_overrides else None
        put_premium_override = premium_overrides.get("put") if premium_overrides else None
        
        self.premiums["call"] = self._price_option(strike, "call", call_premium_override)
        self.premiums["put"] = self._price_option(strike, "put", put_premium_override)
        
        self.strikes["call"] = strike
        self.strikes["put"] = strike
        self.option_types["call"] = "call"
        self.option_types["put"] = "put"
        self.positions["call"] = 1  # Long
        self.positions["put"] = 1  # Long
    
    def calculate_payoff(self, spot_prices: np.ndarray, premium_overrides: Optional[Dict] = None) -> np.ndarray:
        call_prem = premium_overrides.get("call", self.premiums["call"]) if premium_overrides else self.premiums["call"]
        put_prem = premium_overrides.get("put", self.premiums["put"]) if premium_overrides else self.premiums["put"]
        return long_straddle_payoff(spot_prices, self.strike, call_prem, put_prem)
    
    def get_description(self) -> str:
        total_premium = self.premiums["call"] + self.premiums["put"]
        return (
            f"Buy call and put at strike ${self.strike:.2f}. "
            f"Volatility play: profits from large moves in either direction. "
            f"Total cost: ${total_premium:.2f}. "
            f"Breakevens: ${self.strike - total_premium:.2f} (down) and ${self.strike + total_premium:.2f} (up)."
        )


class LongStrangle(OptionsStrategy):
    """Long Strangle: Buy OTM call and OTM put."""
    
    def __init__(
        self,
        spot: float,
        call_strike: float,
        put_strike: float,
        time_to_expiry: float,
        risk_free_rate: float,
        volatility: float,
        premium_overrides: Optional[Dict[str, float]] = None
    ):
        super().__init__(spot, time_to_expiry, risk_free_rate, volatility, "Long Strangle")
        self.call_strike = call_strike
        self.put_strike = put_strike
        
        call_premium_override = premium_overrides.get("call") if premium_overrides else None
        put_premium_override = premium_overrides.get("put") if premium_overrides else None
        
        self.premiums["call"] = self._price_option(call_strike, "call", call_premium_override)
        self.premiums["put"] = self._price_option(put_strike, "put", put_premium_override)
        
        self.strikes["call"] = call_strike
        self.strikes["put"] = put_strike
        self.option_types["call"] = "call"
        self.option_types["put"] = "put"
        self.positions["call"] = 1  # Long
        self.positions["put"] = 1  # Long
    
    def calculate_payoff(self, spot_prices: np.ndarray, premium_overrides: Optional[Dict] = None) -> np.ndarray:
        call_prem = premium_overrides.get("call", self.premiums["call"]) if premium_overrides else self.premiums["call"]
        put_prem = premium_overrides.get("put", self.premiums["put"]) if premium_overrides else self.premiums["put"]
        return long_strangle_payoff(spot_prices, self.call_strike, self.put_strike, call_prem, put_prem)
    
    def get_description(self) -> str:
        total_premium = self.premiums["call"] + self.premiums["put"]
        return (
            f"Buy OTM call at ${self.call_strike:.2f} and OTM put at ${self.put_strike:.2f}. "
            f"Cheaper volatility play than straddle. Total cost: ${total_premium:.2f}. "
            f"Breakevens: ${self.put_strike - total_premium:.2f} (down) and ${self.call_strike + total_premium:.2f} (up)."
        )

