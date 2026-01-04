# Options Strategy Simulator

A Python-based options strategy simulator for analyzing payoff structures, breakeven points, and volatility exposure across common equity derivatives strategies. Built for quantitative finance professionals and trading candidates.

## Overview

This simulator implements Black-Scholes pricing for European options and provides comprehensive analysis of seven common options strategies:

1. **Long Call** - Bullish directional play with limited downside
2. **Long Put** - Bearish directional play with limited downside
3. **Covered Call** - Income generation strategy (long stock + short call)
4. **Bull Call Spread** - Moderately bullish with limited risk/reward
5. **Bear Put Spread** - Moderately bearish with limited risk/reward
6. **Long Straddle** - Volatility play (long call + long put, same strike)
7. **Long Strangle** - Volatility play (long OTM call + long OTM put)

## Features

- **Black-Scholes Pricing**: Accurate European option pricing
- **Greeks Calculation**: Delta and Vega exposure for risk management
- **Payoff Diagrams**: Visual representation of profit/loss at expiry
- **Breakeven Analysis**: Automatic calculation of breakeven points
- **Risk Interpretation**: Qualitative assessment of directional and volatility exposure
- **Modular Design**: Clean separation of pricing, payoffs, and visualization

## Installation

### Using a Virtual Environment (Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

### Direct Installation (if your system allows)

```bash
pip install -r requirements.txt
```

## Quick Start

```python
import numpy as np
from strategy_analyzer import LongCall
from visualizer import plot_strategy

# Market parameters
spot = 100.0
strike = 100.0
time_to_expiry = 0.25  # 3 months
risk_free_rate = 0.05  # 5%
volatility = 0.20  # 20%

# Create strategy
strategy = LongCall(spot, strike, time_to_expiry, risk_free_rate, volatility)

# Analyze
spot_range = np.linspace(70, 130, 200)
analysis = strategy.analyze(spot_range)

# Visualize
plot_strategy(analysis, save_path="long_call.png")
```

## Example Usage

Run the example script to see multiple strategies:

```bash
python example_usage.py
```

This will generate payoff diagrams for:
- Long Call
- Long Straddle
- Bull Call Spread
- Covered Call
- Long Strangle

## Key Financial Concepts

### Delta (Directional Risk)
- Measures price sensitivity to underlying movement
- Call delta: 0 to 1 (increases as stock rises)
- Put delta: -1 to 0 (decreases as stock rises)
- Net delta determines overall directional exposure

### Vega (Volatility Risk)
- Measures price sensitivity to volatility changes
- Always positive for long options
- Higher for at-the-money options
- Net vega determines volatility exposure

### Payoff Diagrams
- Show profit/loss at expiry for different underlying prices
- Breakeven points mark where strategy breaks even
- Maximum profit/loss bounds show risk-reward profile

## Project Structure

```
.
├── options_pricing.py      # Black-Scholes pricing and Greeks
├── strategy_payoffs.py     # Payoff calculations for each strategy
├── strategy_analyzer.py    # Main strategy classes and analysis
├── visualizer.py          # Matplotlib plotting functions
├── example_usage.py        # Example demonstrations
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Strategy Details

### Long Call
- **Use Case**: Bullish outlook, limited risk
- **Max Loss**: Premium paid
- **Max Profit**: Unlimited
- **Breakeven**: Strike + Premium

### Long Straddle
- **Use Case**: Expect large move, direction unknown
- **Max Loss**: Total premium (if stock stays at strike)
- **Max Profit**: Unlimited (both directions)
- **Breakevens**: Strike ± Total Premium

### Bull Call Spread
- **Use Case**: Moderately bullish, reduce cost
- **Max Loss**: Net premium paid
- **Max Profit**: (Higher Strike - Lower Strike) - Net Premium
- **Breakeven**: Lower Strike + Net Premium

## Technical Notes

- Uses European options (no early exercise)
- Assumes constant volatility (Black-Scholes limitation)
- Payoffs calculated at expiry only
- Greeks calculated using analytical formulas
- Vectorized NumPy operations for performance

## Resume Alignment

This project demonstrates:
- Strong understanding of derivatives pricing
- Risk management through Greeks
- Ability to translate financial concepts into code
- Clean, modular software engineering
- Visualization of complex financial relationships

## Future Enhancements

Potential additions (not implemented):
- Theta (time decay) and Gamma (convexity)
- American options pricing
- Implied volatility calculation
- Portfolio-level analysis
- Interactive web interface

## License

This project is for educational and demonstration purposes.

