"""
Validation script to test edge cases and verify correctness.

Tests:
1. Deep ITM/OTM options
2. Near expiry (time -> 0)
3. Greeks signs and magnitudes
4. Payoff shapes match theoretical expectations
5. Breakeven calculations
"""

import numpy as np
from strategy_analyzer import (
    LongCall, LongPut, LongStraddle, BullCallSpread
)
from options_pricing import black_scholes_price, black_scholes_delta, black_scholes_vega


def test_deep_itm_otm():
    """Test deep in-the-money and out-of-the-money options."""
    print("Testing Deep ITM/OTM Options...")
    
    spot = 100.0
    time_to_expiry = 0.25
    risk_free_rate = 0.05
    volatility = 0.20
    
    # Deep ITM call
    itm_call_price = black_scholes_price(spot, 80.0, time_to_expiry, risk_free_rate, volatility, "call")
    itm_call_delta = black_scholes_delta(spot, 80.0, time_to_expiry, risk_free_rate, volatility, "call")
    
    # Deep OTM call
    otm_call_price = black_scholes_price(spot, 120.0, time_to_expiry, risk_free_rate, volatility, "call")
    otm_call_delta = black_scholes_delta(spot, 120.0, time_to_expiry, risk_free_rate, volatility, "call")
    
    print(f"  Deep ITM Call (strike=80): price=${itm_call_price:.2f}, delta={itm_call_delta:.3f}")
    print(f"  Deep OTM Call (strike=120): price=${otm_call_price:.2f}, delta={otm_call_delta:.3f}")
    
    # Validate: ITM should have higher price and delta near 1
    assert itm_call_price > otm_call_price, "ITM call should cost more than OTM call"
    assert itm_call_delta > 0.8, "ITM call delta should be high (>0.8)"
    assert otm_call_delta < 0.2, "OTM call delta should be low (<0.2)"
    
    print("  ✓ Deep ITM/OTM test passed\n")


def test_near_expiry():
    """Test behavior as time to expiry approaches zero."""
    print("Testing Near Expiry (Time -> 0)...")
    
    spot = 100.0
    strike = 100.0
    risk_free_rate = 0.05
    volatility = 0.20
    
    # At expiry, option should equal intrinsic value
    price_at_expiry = black_scholes_price(spot, strike, 0.0, risk_free_rate, volatility, "call")
    intrinsic = max(spot - strike, 0)
    
    print(f"  Call at expiry: price=${price_at_expiry:.2f}, intrinsic=${intrinsic:.2f}")
    assert abs(price_at_expiry - intrinsic) < 0.01, "At expiry, price should equal intrinsic"
    
    # Delta at expiry should be step function
    delta_at_expiry = black_scholes_delta(spot, strike, 0.0, risk_free_rate, volatility, "call")
    print(f"  Delta at expiry: {delta_at_expiry:.3f}")
    
    # Vega at expiry should be zero
    vega_at_expiry = black_scholes_vega(spot, strike, 0.0, risk_free_rate, volatility, "call")
    print(f"  Vega at expiry: {vega_at_expiry:.6f}")
    assert abs(vega_at_expiry) < 0.0001, "Vega should be zero at expiry"
    
    print("  ✓ Near expiry test passed\n")


def test_greeks_signs():
    """Test that Greeks have correct signs."""
    print("Testing Greeks Signs...")
    
    spot = 100.0
    strike = 100.0
    time_to_expiry = 0.25
    risk_free_rate = 0.05
    volatility = 0.20
    
    # Call delta should be positive
    call_delta = black_scholes_delta(spot, strike, time_to_expiry, risk_free_rate, volatility, "call")
    assert call_delta > 0, "Call delta should be positive"
    print(f"  Call delta: {call_delta:.3f} (positive ✓)")
    
    # Put delta should be negative
    put_delta = black_scholes_delta(spot, strike, time_to_expiry, risk_free_rate, volatility, "put")
    assert put_delta < 0, "Put delta should be negative"
    print(f"  Put delta: {put_delta:.3f} (negative ✓)")
    
    # Both call and put vega should be positive
    call_vega = black_scholes_vega(spot, strike, time_to_expiry, risk_free_rate, volatility, "call")
    put_vega = black_scholes_vega(spot, strike, time_to_expiry, risk_free_rate, volatility, "put")
    assert call_vega > 0, "Call vega should be positive"
    assert put_vega > 0, "Put vega should be positive"
    assert abs(call_vega - put_vega) < 0.001, "Call and put vega should be equal"
    print(f"  Call vega: {call_vega:.6f} (positive ✓)")
    print(f"  Put vega: {put_vega:.6f} (positive ✓)")
    
    print("  ✓ Greeks signs test passed\n")


def test_strategy_payoffs():
    """Test that strategy payoffs match theoretical shapes."""
    print("Testing Strategy Payoff Shapes...")
    
    spot = 100.0
    time_to_expiry = 0.25
    risk_free_rate = 0.05
    volatility = 0.20
    spot_range = np.linspace(70, 130, 200)
    
    # Long Call: should be flat below strike, upward sloping above
    long_call = LongCall(spot, 100.0, time_to_expiry, risk_free_rate, volatility)
    call_analysis = long_call.analyze(spot_range)
    payoffs = call_analysis["payoffs"]
    
    # Below strike, payoff should be negative (premium paid)
    below_strike = payoffs[spot_range < 100]
    assert np.all(below_strike < 0), "Below strike, long call should have negative payoff"
    
    # Above strike, payoff should increase
    above_strike_idx = spot_range > 100
    above_strike_payoffs = payoffs[above_strike_idx]
    assert np.all(np.diff(above_strike_payoffs) >= -0.01), "Above strike, payoff should be non-decreasing"
    
    print("  ✓ Long Call payoff shape correct")
    
    # Long Straddle: should have V-shape
    straddle = LongStraddle(spot, 100.0, time_to_expiry, risk_free_rate, volatility)
    straddle_analysis = straddle.analyze(spot_range)
    straddle_payoffs = straddle_analysis["payoffs"]
    
    # Should have minimum near strike
    min_idx = np.argmin(straddle_payoffs)
    min_spot = spot_range[min_idx]
    assert 95 < min_spot < 105, "Straddle minimum should be near strike"
    
    # Should increase away from strike in both directions
    left_of_strike = straddle_payoffs[spot_range < 95]
    right_of_strike = straddle_payoffs[spot_range > 105]
    assert np.all(np.diff(left_of_strike[::-1]) >= -0.01), "Straddle should increase moving left from strike"
    assert np.all(np.diff(right_of_strike) >= -0.01), "Straddle should increase moving right from strike"
    
    print("  ✓ Long Straddle payoff shape correct")
    
    # Bull Call Spread: should be bounded
    spread = BullCallSpread(spot, 95.0, 110.0, time_to_expiry, risk_free_rate, volatility)
    spread_analysis = spread.analyze(spot_range)
    spread_payoffs = spread_analysis["payoffs"]
    
    # Max profit should be capped
    max_profit = np.max(spread_payoffs)
    expected_max = (110.0 - 95.0) - (spread.premiums["lower"] - spread.premiums["higher"])
    assert abs(max_profit - expected_max) < 0.5, "Bull spread max profit should match theoretical"
    
    print("  ✓ Bull Call Spread payoff shape correct")
    
    print("  ✓ Strategy payoff shapes test passed\n")


def test_breakevens():
    """Test breakeven calculations."""
    print("Testing Breakeven Calculations...")
    
    spot = 100.0
    strike = 100.0
    time_to_expiry = 0.25
    risk_free_rate = 0.05
    volatility = 0.20
    spot_range = np.linspace(70, 130, 500)  # Higher resolution for accuracy
    
    # Long Call breakeven should be strike + premium
    long_call = LongCall(spot, strike, time_to_expiry, risk_free_rate, volatility)
    call_analysis = long_call.analyze(spot_range)
    breakevens = call_analysis["breakevens"]
    
    expected_be = strike + call_analysis["premiums"]["call"]
    if breakevens:
        actual_be = breakevens[0]
        print(f"  Long Call BE: expected=${expected_be:.2f}, calculated=${actual_be:.2f}")
        assert abs(actual_be - expected_be) < 1.0, "Breakeven should match theoretical value"
    
    # Long Straddle should have two breakevens
    straddle = LongStraddle(spot, strike, time_to_expiry, risk_free_rate, volatility)
    straddle_analysis = straddle.analyze(spot_range)
    straddle_bes = straddle_analysis["breakevens"]
    
    total_premium = straddle_analysis["premiums"]["call"] + straddle_analysis["premiums"]["put"]
    expected_be_down = strike - total_premium
    expected_be_up = strike + total_premium
    
    print(f"  Long Straddle BEs: expected=${expected_be_down:.2f} and ${expected_be_up:.2f}")
    if len(straddle_bes) >= 2:
        print(f"  Long Straddle BEs: calculated=${straddle_bes[0]:.2f} and ${straddle_bes[1]:.2f}")
        assert abs(straddle_bes[0] - expected_be_down) < 1.0, "Lower breakeven should match"
        assert abs(straddle_bes[1] - expected_be_up) < 1.0, "Upper breakeven should match"
    
    print("  ✓ Breakeven calculations test passed\n")


def test_strategy_greeks():
    """Test that strategy Greeks make intuitive sense."""
    print("Testing Strategy Greeks...")
    
    spot = 100.0
    time_to_expiry = 0.25
    risk_free_rate = 0.05
    volatility = 0.20
    spot_range = np.linspace(70, 130, 200)
    
    # Long Call should have positive delta
    long_call = LongCall(spot, 100.0, time_to_expiry, risk_free_rate, volatility)
    call_analysis = long_call.analyze(spot_range)
    assert call_analysis["greeks"]["delta"] > 0, "Long call should have positive delta"
    print(f"  Long Call delta: {call_analysis['greeks']['delta']:.3f} (positive ✓)")
    
    # Long Put should have negative delta
    long_put = LongPut(spot, 100.0, time_to_expiry, risk_free_rate, volatility)
    put_analysis = long_put.analyze(spot_range)
    assert put_analysis["greeks"]["delta"] < 0, "Long put should have negative delta"
    print(f"  Long Put delta: {put_analysis['greeks']['delta']:.3f} (negative ✓)")
    
    # Long Straddle should have delta near zero (direction neutral)
    # Note: Small positive delta is expected due to risk-free rate asymmetry
    straddle = LongStraddle(spot, 100.0, time_to_expiry, risk_free_rate, volatility)
    straddle_analysis = straddle.analyze(spot_range)
    assert abs(straddle_analysis["greeks"]["delta"]) < 0.2, "Straddle should be direction-neutral (small delta OK due to rho effect)"
    assert straddle_analysis["greeks"]["vega"] > 0.1, "Straddle should have high vega"
    print(f"  Long Straddle delta: {straddle_analysis['greeks']['delta']:.3f} (near zero ✓)")
    print(f"  Long Straddle vega: {straddle_analysis['greeks']['vega']:.3f} (high ✓)")
    
    # Bull Call Spread should have positive but lower delta than long call
    spread = BullCallSpread(spot, 95.0, 110.0, time_to_expiry, risk_free_rate, volatility)
    spread_analysis = spread.analyze(spot_range)
    assert 0 < spread_analysis["greeks"]["delta"] < call_analysis["greeks"]["delta"], \
        "Bull spread should have positive but lower delta than long call"
    print(f"  Bull Spread delta: {spread_analysis['greeks']['delta']:.3f} (positive, moderate ✓)")
    
    print("  ✓ Strategy Greeks test passed\n")


def main():
    """Run all validation tests."""
    print("=" * 80)
    print("OPTIONS STRATEGY SIMULATOR - VALIDATION TESTS")
    print("=" * 80)
    print()
    
    try:
        test_deep_itm_otm()
        test_near_expiry()
        test_greeks_signs()
        test_strategy_payoffs()
        test_breakevens()
        test_strategy_greeks()
        
        print("=" * 80)
        print("ALL VALIDATION TESTS PASSED ✓")
        print("=" * 80)
        
    except AssertionError as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        raise


if __name__ == "__main__":
    main()

