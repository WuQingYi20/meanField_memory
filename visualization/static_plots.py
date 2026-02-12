"""
Static visualization plots for ABM simulation analysis.

Generates publication-quality figures for simulation results.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import List, Dict, Any, Optional
from pathlib import Path


def plot_strategy_evolution(
    df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    title: str = "Strategy Distribution Over Time",
) -> plt.Axes:
    """Plot strategy distribution evolution over time."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    ax.fill_between(
        df["tick"],
        0,
        df["strategy_0_fraction"],
        alpha=0.7,
        label="Strategy A",
        color="steelblue",
    )
    ax.fill_between(
        df["tick"],
        df["strategy_0_fraction"],
        1,
        alpha=0.7,
        label="Strategy B",
        color="coral",
    )

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Tick")
    ax.set_ylabel("Fraction of Population")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    return ax


def plot_tau_evolution(
    df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    title: str = "Temperature Evolution",
) -> plt.Axes:
    """Plot mean temperature (tau) over time, with trust fallback."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    if "mean_tau" in df.columns:
        series = df["mean_tau"]
        ylabel = "Mean Temperature (tau)"
        line_color = "green"
    else:
        # Fallback for trust-based decision modes without tau.
        series = df["mean_trust"]
        ylabel = "Mean Trust"
        line_color = "purple"

    ax.plot(df["tick"], series, color=line_color, linewidth=2)
    ax.fill_between(df["tick"], series, alpha=0.3, color=line_color)
    ax.set_xlabel("Tick")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return ax


def plot_trust_distribution(
    trust_values: List[float],
    ax: Optional[plt.Axes] = None,
    title: str = "Trust Level Distribution",
    bins: int = 20,
) -> plt.Axes:
    """Plot histogram of trust levels."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    ax.hist(trust_values, bins=bins, edgecolor="black", alpha=0.7, color="purple")
    ax.axvline(
        x=np.mean(trust_values),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(trust_values):.2f}",
    )

    ax.set_xlabel("Trust Level")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_coordination_rate(
    df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    title: str = "Coordination Success Rate",
    window: int = 10,
) -> plt.Axes:
    """Plot coordination rate with rolling average."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    ax.plot(df["tick"], df["coordination_rate"], alpha=0.3, color="orange", linewidth=1)
    rolling_avg = df["coordination_rate"].rolling(window=window, min_periods=1).mean()
    ax.plot(df["tick"], rolling_avg, color="orange", linewidth=2, label=f"Rolling avg (w={window})")

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random baseline")
    ax.set_xlabel("Tick")
    ax.set_ylabel("Coordination Rate")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_memory_window_distribution(
    window_sizes: List[int],
    ax: Optional[plt.Axes] = None,
    title: str = "Memory Window Size Distribution",
) -> plt.Axes:
    """Plot distribution of dynamic memory window sizes."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    unique_sizes = sorted(set(window_sizes))
    counts = [window_sizes.count(s) for s in unique_sizes]

    ax.bar(unique_sizes, counts, edgecolor="black", alpha=0.7, color="teal")
    ax.set_xlabel("Window Size")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.set_xticks(unique_sizes)
    ax.grid(True, alpha=0.3, axis="y")
    return ax


def plot_convergence_comparison(
    results_list: List[Dict[str, Any]],
    labels: Optional[List[str]] = None,
    ax: Optional[plt.Axes] = None,
    title: str = "Convergence Time Comparison",
) -> plt.Axes:
    """Compare convergence times across different configurations."""
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))

    if labels is None:
        labels = [f"Config {i+1}" for i in range(len(results_list))]

    convergence_times = []
    for result in results_list:
        if result.get("convergence_tick") is not None:
            convergence_times.append(result["convergence_tick"])
        else:
            convergence_times.append(result.get("final_tick", 1000))

    colors = plt.cm.viridis(np.linspace(0, 0.8, len(labels)))
    bars = ax.bar(labels, convergence_times, color=colors, edgecolor="black")

    for i, result in enumerate(results_list):
        if result.get("convergence_tick") is None:
            bars[i].set_hatch("//")
            bars[i].set_alpha(0.5)

    ax.set_xlabel("Configuration")
    ax.set_ylabel("Convergence Time (ticks)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis="y")

    if len(labels) > 5:
        plt.xticks(rotation=45, ha="right")

    return ax


def create_summary_figure(
    df: pd.DataFrame,
    agent_states: Optional[List[Dict[str, Any]]] = None,
    title: str = "Simulation Summary",
    figsize: tuple = (16, 12),
) -> Figure:
    """Create a comprehensive summary figure with multiple subplots."""
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(title, fontsize=16, fontweight="bold")

    plot_strategy_evolution(df, ax=axes[0, 0])
    plot_tau_evolution(df, ax=axes[0, 1])
    plot_coordination_rate(df, ax=axes[0, 2])

    # Trust and tau together (or trust-only for non-tau modes)
    ax = axes[1, 0]
    if "mean_tau" in df.columns:
        ax.plot(df["tick"], df["mean_tau"], color="green", label="Mean tau", linewidth=2)
        ax.set_xlabel("Tick")
        ax.set_ylabel("Mean tau", color="green")
        ax2 = ax.twinx()
        ax2.plot(df["tick"], df["mean_trust"], color="purple", linestyle="--", label="Mean Trust", linewidth=2)
        ax2.set_ylabel("Mean Trust", color="purple")
        ax.set_title("Temperature vs Trust")
        ax.legend(loc="upper left")
        ax2.legend(loc="upper right")
    else:
        ax.plot(df["tick"], df["mean_trust"], color="purple", label="Mean Trust", linewidth=2)
        ax.set_xlabel("Tick")
        ax.set_ylabel("Mean Trust")
        ax.set_title("Trust Evolution")
        ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(df["tick"], df["mean_memory_window"], color="teal", linewidth=2)
    ax.set_xlabel("Tick")
    ax.set_ylabel("Mean Memory Window")
    ax.set_title("Memory Window Evolution")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.plot(df["tick"], df["strategy_switches"], color="crimson", linewidth=1, alpha=0.7)
    rolling = df["strategy_switches"].rolling(window=20, min_periods=1).mean()
    ax.plot(df["tick"], rolling, color="crimson", linewidth=2, label="Rolling avg")
    ax.set_xlabel("Tick")
    ax.set_ylabel("Strategy Switches")
    ax.set_title("Strategy Switches per Tick")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_phase_diagram(
    df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    title: str = "Phase Diagram",
) -> plt.Axes:
    """Plot phase diagram of strategy fraction vs trust."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    scatter = ax.scatter(
        df["strategy_0_fraction"],
        df["mean_trust"],
        c=df["tick"],
        cmap="viridis",
        alpha=0.6,
        s=10,
    )

    ax.plot(
        df["strategy_0_fraction"],
        df["mean_trust"],
        color="gray",
        alpha=0.3,
        linewidth=0.5,
    )

    ax.scatter(
        [df["strategy_0_fraction"].iloc[0]],
        [df["mean_trust"].iloc[0]],
        color="green",
        s=100,
        marker="o",
        label="Start",
        zorder=5,
    )
    ax.scatter(
        [df["strategy_0_fraction"].iloc[-1]],
        [df["mean_trust"].iloc[-1]],
        color="red",
        s=100,
        marker="s",
        label="End",
        zorder=5,
    )

    plt.colorbar(scatter, ax=ax, label="Tick")

    ax.set_xlabel("Strategy A Fraction")
    ax.set_ylabel("Mean Trust")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def save_figure(
    fig: Figure,
    filename: str,
    output_dir: str = "data",
    formats: List[str] = ["png", "pdf"],
    dpi: int = 150,
) -> List[str]:
    """Save figure in multiple formats."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    saved = []
    for fmt in formats:
        filepath = output_path / f"{filename}.{fmt}"
        fig.savefig(filepath, format=fmt, dpi=dpi, bbox_inches="tight")
        saved.append(str(filepath))

    return saved
