"""
Real-time visualization for ABM simulation.

Uses matplotlib animation to show strategy distribution evolution.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Optional, List, Callable
import matplotlib.patches as mpatches


class RealtimeVisualizer:
    """
    Real-time visualization of simulation progress.

    Shows animated plots of:
    - Strategy distribution over time
    - Temperature/trust evolution
    - Coordination rate
    """

    def __init__(
        self,
        num_agents: int,
        max_ticks: int = 1000,
        update_interval: int = 50,  # milliseconds
        figsize: tuple = (14, 8),
    ):
        """
        Initialize visualizer.

        Args:
            num_agents: Number of agents in simulation
            max_ticks: Maximum number of ticks (for x-axis scaling)
            update_interval: Animation update interval in ms
            figsize: Figure size
        """
        self._num_agents = num_agents
        self._max_ticks = max_ticks
        self._update_interval = update_interval

        # Data storage
        self._ticks: List[int] = []
        self._strategy_0: List[float] = []
        self._strategy_1: List[float] = []
        self._mean_tau: List[float] = []
        self._mean_trust: List[float] = []
        self._coord_rate: List[float] = []

        # Setup figure
        self._fig, self._axes = plt.subplots(2, 2, figsize=figsize)
        self._fig.suptitle("ABM Coordination Game Simulation", fontsize=14)

        self._setup_plots()

        # Animation state
        self._animation: Optional[FuncAnimation] = None
        self._running = False

    def _setup_plots(self) -> None:
        """Initialize all subplot elements."""
        # Strategy distribution (top-left)
        ax = self._axes[0, 0]
        ax.set_xlim(0, self._max_ticks)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Tick")
        ax.set_ylabel("Fraction")
        ax.set_title("Strategy Distribution")
        (self._line_s0,) = ax.plot([], [], "b-", label="Strategy A", linewidth=2)
        (self._line_s1,) = ax.plot([], [], "r-", label="Strategy B", linewidth=2)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        # Temperature/Trust (top-right)
        ax = self._axes[0, 1]
        ax.set_xlim(0, self._max_ticks)
        ax.set_ylim(0, 2.5)
        ax.set_xlabel("Tick")
        ax.set_ylabel("Value")
        ax.set_title("Temperature & Trust")
        (self._line_tau,) = ax.plot([], [], "g-", label="Mean τ", linewidth=2)
        ax2 = ax.twinx()
        ax2.set_ylim(0, 1)
        ax2.set_ylabel("Trust", color="purple")
        (self._line_trust,) = ax2.plot(
            [], [], "purple", linestyle="--", label="Mean Trust", linewidth=2
        )
        ax.legend(loc="upper left")
        ax2.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        self._ax_trust = ax2

        # Coordination rate (bottom-left)
        ax = self._axes[1, 0]
        ax.set_xlim(0, self._max_ticks)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Tick")
        ax.set_ylabel("Rate")
        ax.set_title("Coordination Success Rate")
        (self._line_coord,) = ax.plot([], [], "orange", linewidth=2)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random baseline")
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)

        # Status panel (bottom-right)
        ax = self._axes[1, 1]
        ax.axis("off")
        self._status_text = ax.text(
            0.1,
            0.5,
            "",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="center",
            fontfamily="monospace",
        )

        plt.tight_layout()

    def update_data(
        self,
        tick: int,
        strategy_distribution: List[float],
        mean_tau: float,
        mean_trust: float,
        coordination_rate: float,
    ) -> None:
        """
        Add new data point.

        Args:
            tick: Current tick number
            strategy_distribution: [fraction_0, fraction_1]
            mean_tau: Mean temperature across agents
            mean_trust: Mean trust level
            coordination_rate: This tick's coordination success rate
        """
        self._ticks.append(tick)
        self._strategy_0.append(strategy_distribution[0])
        self._strategy_1.append(strategy_distribution[1])
        self._mean_tau.append(mean_tau)
        self._mean_trust.append(mean_trust)
        self._coord_rate.append(coordination_rate)

    def _update_frame(self, frame: int) -> tuple:
        """Update animation frame."""
        if not self._ticks:
            return (
                self._line_s0,
                self._line_s1,
                self._line_tau,
                self._line_trust,
                self._line_coord,
            )

        # Update strategy plot
        self._line_s0.set_data(self._ticks, self._strategy_0)
        self._line_s1.set_data(self._ticks, self._strategy_1)

        # Update tau/trust plot
        self._line_tau.set_data(self._ticks, self._mean_tau)
        self._line_trust.set_data(self._ticks, self._mean_trust)

        # Update coordination rate
        self._line_coord.set_data(self._ticks, self._coord_rate)

        # Update status text
        if self._ticks:
            status = (
                f"Tick: {self._ticks[-1]}\n"
                f"Strategy A: {self._strategy_0[-1]:.2%}\n"
                f"Strategy B: {self._strategy_1[-1]:.2%}\n"
                f"Mean τ: {self._mean_tau[-1]:.3f}\n"
                f"Mean Trust: {self._mean_trust[-1]:.2%}\n"
                f"Coord Rate: {self._coord_rate[-1]:.2%}"
            )
            self._status_text.set_text(status)

        return (
            self._line_s0,
            self._line_s1,
            self._line_tau,
            self._line_trust,
            self._line_coord,
        )

    def start_animation(self) -> None:
        """Start the animation loop."""
        self._running = True
        self._animation = FuncAnimation(
            self._fig,
            self._update_frame,
            interval=self._update_interval,
            blit=False,
            cache_frame_data=False,
        )
        plt.show(block=False)

    def stop_animation(self) -> None:
        """Stop the animation."""
        self._running = False
        if self._animation:
            self._animation.event_source.stop()

    def show(self) -> None:
        """Display the figure (blocking)."""
        plt.show()

    def save_animation(self, filename: str, fps: int = 20) -> None:
        """
        Save animation to file.

        Args:
            filename: Output filename (e.g., 'animation.gif' or 'animation.mp4')
            fps: Frames per second
        """
        if self._animation:
            self._animation.save(filename, fps=fps)

    def clear_data(self) -> None:
        """Clear all stored data."""
        self._ticks.clear()
        self._strategy_0.clear()
        self._strategy_1.clear()
        self._mean_tau.clear()
        self._mean_trust.clear()
        self._coord_rate.clear()

    def close(self) -> None:
        """Close the figure."""
        plt.close(self._fig)


def create_callback_visualizer(
    num_agents: int,
    max_ticks: int = 1000,
) -> tuple:
    """
    Create a visualizer and callback function for use with simulation.

    Usage:
        viz, callback = create_callback_visualizer(100, 1000)
        viz.start_animation()
        env.run(max_ticks=1000, progress_callback=callback)
        viz.show()

    Args:
        num_agents: Number of agents
        max_ticks: Maximum ticks

    Returns:
        Tuple of (RealtimeVisualizer, callback_function)
    """
    viz = RealtimeVisualizer(num_agents=num_agents, max_ticks=max_ticks)

    def callback(tick, metrics):
        viz.update_data(
            tick=tick,
            strategy_distribution=metrics.strategy_distribution,
            mean_tau=metrics.mean_tau,
            mean_trust=metrics.mean_trust,
            coordination_rate=metrics.coordination_rate,
        )

    return viz, callback
