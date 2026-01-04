"""
Example usage of the Options Strategy Simulator.

This script demonstrates how to analyze various options strategies,
showing payoff diagrams, risk metrics, and breakeven points.

Run this script to see examples of:
1. Long Call (bullish directional)
2. Long Straddle (volatility play)
3. Bull Call Spread (moderate bullish)
4. Covered Call (income generation)
"""

import numpy as np
from strategy_analyzer import (
    LongCall,
    LongPut,
    CoveredCall,
    BullCallSpread,
    BearPutSpread,
    LongStraddle,
    LongStrangle
)
from visualizer import plot_strategy, plot_multiple_strategies


def main():
    """
    Main example demonstrating multiple strategies.
    """
    # Common market parameters
    spot = 100.0  # Current stock price
    time_to_expiry = 0.25  # 3 months
    risk_free_rate = 0.05  # 5% annual
    volatility = 0.20  # 20% annual volatility
    
    # Create range of underlying prices for payoff calculation
    spot_range = np.linspace(70, 130, 200)
    
    print("=" * 80)
    print("OPTIONS STRATEGY SIMULATOR - EXAMPLE USAGE")
    print("=" * 80)
    print()
    
    # Example 1: Long Call
    print("Example 1: Long Call Strategy")
    print("-" * 80)
    long_call = LongCall(
        spot=spot,
        strike=100.0,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        volatility=volatility
    )
    call_analysis = long_call.analyze(spot_range)
    print(f"Strategy: {call_analysis['strategy_name']}")
    print(f"Description: {call_analysis['description']}")
    print(f"Delta: {call_analysis['greeks']['delta']:.3f}")
    print(f"Vega: {call_analysis['greeks']['vega']:.3f}")
    print(f"Risk: {call_analysis['risk_interpretation']}")
    print(f"Max Profit: ${call_analysis['max_profit']:.2f}")
    print(f"Max Loss: ${call_analysis['max_loss']:.2f}")
    print(f"Breakevens: {[f'${be:.2f}' for be in call_analysis['breakevens']]}")
    print()
    
    plot_strategy(call_analysis, save_path="long_call_payoff.png")
    
    # Example 2: Long Straddle
    print("Example 2: Long Straddle Strategy")
    print("-" * 80)
    straddle = LongStraddle(
        spot=spot,
        strike=100.0,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        volatility=volatility
    )
    straddle_analysis = straddle.analyze(spot_range)
    print(f"Strategy: {straddle_analysis['strategy_name']}")
    print(f"Description: {straddle_analysis['description']}")
    print(f"Delta: {straddle_analysis['greeks']['delta']:.3f}")
    print(f"Vega: {straddle_analysis['greeks']['vega']:.3f}")
    print(f"Risk: {straddle_analysis['risk_interpretation']}")
    print(f"Max Profit: ${straddle_analysis['max_profit']:.2f}")
    print(f"Max Loss: ${straddle_analysis['max_loss']:.2f}")
    print(f"Breakevens: {[f'${be:.2f}' for be in straddle_analysis['breakevens']]}")
    print()
    
    plot_strategy(straddle_analysis, save_path="long_straddle_payoff.png")
    
    # Example 3: Bull Call Spread
    print("Example 3: Bull Call Spread Strategy")
    print("-" * 80)
    bull_spread = BullCallSpread(
        spot=spot,
        lower_strike=95.0,
        higher_strike=110.0,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        volatility=volatility
    )
    spread_analysis = bull_spread.analyze(spot_range)
    print(f"Strategy: {spread_analysis['strategy_name']}")
    print(f"Description: {spread_analysis['description']}")
    print(f"Delta: {spread_analysis['greeks']['delta']:.3f}")
    print(f"Vega: {spread_analysis['greeks']['vega']:.3f}")
    print(f"Risk: {spread_analysis['risk_interpretation']}")
    print(f"Max Profit: ${spread_analysis['max_profit']:.2f}")
    print(f"Max Loss: ${spread_analysis['max_loss']:.2f}")
    print(f"Breakevens: {[f'${be:.2f}' for be in spread_analysis['breakevens']]}")
    print()
    
    plot_strategy(spread_analysis, save_path="bull_call_spread_payoff.png")
    
    # Example 4: Covered Call
    print("Example 4: Covered Call Strategy")
    print("-" * 80)
    covered = CoveredCall(
        spot=spot,
        strike=105.0,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        volatility=volatility
    )
    covered_analysis = covered.analyze(spot_range)
    print(f"Strategy: {covered_analysis['strategy_name']}")
    print(f"Description: {covered_analysis['description']}")
    print(f"Delta: {covered_analysis['greeks']['delta']:.3f}")
    print(f"Vega: {covered_analysis['greeks']['vega']:.3f}")
    print(f"Risk: {covered_analysis['risk_interpretation']}")
    print(f"Max Profit: ${covered_analysis['max_profit']:.2f}")
    print(f"Max Loss: ${covered_analysis['max_loss']:.2f}")
    print(f"Breakevens: {[f'${be:.2f}' for be in covered_analysis['breakevens']]}")
    print()
    
    plot_strategy(covered_analysis, save_path="covered_call_payoff.png")
    
    # Example 5: Long Strangle
    print("Example 5: Long Strangle Strategy")
    print("-" * 80)
    strangle = LongStrangle(
        spot=spot,
        call_strike=110.0,
        put_strike=90.0,
        time_to_expiry=time_to_expiry,
        risk_free_rate=risk_free_rate,
        volatility=volatility
    )
    strangle_analysis = strangle.analyze(spot_range)
    print(f"Strategy: {strangle_analysis['strategy_name']}")
    print(f"Description: {strangle_analysis['description']}")
    print(f"Delta: {strangle_analysis['greeks']['delta']:.3f}")
    print(f"Vega: {strangle_analysis['greeks']['vega']:.3f}")
    print(f"Risk: {strangle_analysis['risk_interpretation']}")
    print(f"Max Profit: ${strangle_analysis['max_profit']:.2f}")
    print(f"Max Loss: ${strangle_analysis['max_loss']:.2f}")
    print(f"Breakevens: {[f'${be:.2f}' for be in strangle_analysis['breakevens']]}")
    print()
    
    plot_strategy(strangle_analysis, save_path="long_strangle_payoff.png")
    
    # Comparison plot
    print("Creating comparison plot of all strategies...")
    plot_multiple_strategies(
        [call_analysis, straddle_analysis, spread_analysis, covered_analysis, strangle_analysis],
        save_path="strategy_comparison.png"
    )
    
    print()
    print("=" * 80)
    print("All examples completed. Check the generated PNG files for payoff diagrams.")
    print("=" * 80)


if __name__ == "__main__":
    main()

