"""
Visualization module for options strategy payoffs and risk metrics.

Creates publication-quality plots showing:
- Payoff diagrams at expiry
- Breakeven points
- Maximum profit/loss annotations
- Risk metrics display
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Optional, Tuple


def plot_strategy(
    analysis: Dict,
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 6)
) -> None:
    """
    Plot complete strategy analysis.
    
    Args:
        analysis: Dictionary from strategy.analyze() method
        save_path: Optional path to save figure
        figsize: Figure size (width, height)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    spot_range = analysis["spot_range"]
    payoffs = analysis["payoffs"]
    strategy_name = analysis["strategy_name"]
    breakevens = analysis["breakevens"]
    max_profit = analysis["max_profit"]
    max_loss = analysis["max_loss"]
    greeks = analysis["greeks"]
    risk_interpretation = analysis["risk_interpretation"]
    
    # Plot payoff curve
    ax.plot(spot_range, payoffs, linewidth=2, label="Payoff at Expiry", color="steelblue")
    
    # Zero line
    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    
    # Current spot price line
    initial_spot = analysis.get("initial_spot", spot_range[len(spot_range) // 2])
    ax.axvline(x=initial_spot, color="red", linestyle=":", linewidth=1, alpha=0.7, label="Current Spot")
    
    # Breakeven points
    for be in breakevens:
        if be >= spot_range[0] and be <= spot_range[-1]:
            be_payoff = np.interp(be, spot_range, payoffs)
            ax.plot(be, be_payoff, "go", markersize=8, zorder=5)
            ax.annotate(
                f"BE: ${be:.2f}",
                xy=(be, be_payoff),
                xytext=(10, 10),
                textcoords="offset points",
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7)
            )
    
    # Max profit/loss annotations
    max_profit_idx = np.argmax(payoffs)
    max_loss_idx = np.argmin(payoffs)
    
    ax.plot(spot_range[max_profit_idx], max_profit, "g^", markersize=10, zorder=5)
    ax.annotate(
        f"Max Profit: ${max_profit:.2f}",
        xy=(spot_range[max_profit_idx], max_profit),
        xytext=(10, 20),
        textcoords="offset points",
        fontsize=9,
        color="green",
        weight="bold"
    )
    
    if max_loss < 0:
        ax.plot(spot_range[max_loss_idx], max_loss, "rv", markersize=10, zorder=5)
        ax.annotate(
            f"Max Loss: ${max_loss:.2f}",
            xy=(spot_range[max_loss_idx], max_loss),
            xytext=(10, -30),
            textcoords="offset points",
            fontsize=9,
            color="red",
            weight="bold"
        )
    
    # Formatting
    ax.set_xlabel("Underlying Price at Expiry ($)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Profit / Loss ($)", fontsize=11, fontweight="bold")
    ax.set_title(f"{strategy_name} - Payoff Diagram", fontsize=13, fontweight="bold", pad=15)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="best", fontsize=9)
    
    # Add risk metrics as text box
    risk_text = (
        f"Delta: {greeks['delta']:.3f}\n"
        f"Vega: {greeks['vega']:.3f}\n"
        f"\n{risk_interpretation}"
    )
    ax.text(
        0.02, 0.98, risk_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.8),
        family="monospace"
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    
    plt.show()


def plot_multiple_strategies(
    analyses: List[Dict],
    save_path: Optional[str] = None,
    figsize: Tuple[float, float] = (14, 8)
) -> None:
    """
    Plot multiple strategies on same axes for comparison.
    
    Args:
        analyses: List of analysis dictionaries
        save_path: Optional path to save figure
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ["steelblue", "crimson", "forestgreen", "darkorange", "purple", "teal", "brown"]
    
    for i, analysis in enumerate(analyses):
        spot_range = analysis["spot_range"]
        payoffs = analysis["payoffs"]
        strategy_name = analysis["strategy_name"]
        color = colors[i % len(colors)]
        
        ax.plot(spot_range, payoffs, linewidth=2, label=strategy_name, color=color)
    
    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Underlying Price at Expiry ($)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Profit / Loss ($)", fontsize=11, fontweight="bold")
    ax.set_title("Options Strategy Comparison", fontsize=13, fontweight="bold", pad=15)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="best", fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Comparison plot saved to {save_path}")
    
    plt.show()

