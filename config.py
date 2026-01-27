"""
Configuration module for ABM Memory & Norm Formation Simulation.

This module defines all configurable parameters for the simulation,
including agent settings, memory mechanisms, decision parameters,
and simulation controls.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class SimulationConfig:
    """Configuration for the ABM simulation."""

    # Agent settings
    num_agents: int = 100
    initial_belief: Literal["uniform", "random", "biased"] = "uniform"
    initial_bias: float = 0.5  # Only used when initial_belief == "biased"

    # Memory settings
    memory_type: Literal["fixed", "decay", "dynamic"] = "fixed"
    fixed_memory_size: int = 5
    decay_rate: float = 0.9  # Lambda parameter for decay memory
    dynamic_max_size: int = 6  # Human cognitive limit
    dynamic_base_size: int = 2  # Minimum memory window

    # Trust-based decision mechanism
    initial_trust: float = 0.5  # Starting trust level
    alpha: float = 0.1  # Trust increase rate on correct prediction
    beta: float = 0.3  # Trust decay rate on wrong prediction

    # Game settings
    num_strategies: int = 2
    coordination_payoff: float = 1.0
    miscoordination_payoff: float = 0.0

    # Network and matching
    network_type: Literal["fully_connected"] = "fully_connected"
    matching_type: Literal["random"] = "random"

    # Simulation settings
    max_ticks: int = 1000
    random_seed: Optional[int] = 42
    convergence_threshold: float = 0.95  # Fraction for consensus
    convergence_window: int = 50  # Ticks to maintain threshold

    # Output settings
    output_dir: str = "data"
    save_history: bool = True
    save_interval: int = 10  # Save metrics every N ticks

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.num_agents < 2:
            raise ValueError("num_agents must be at least 2")
        if self.num_agents > 200:
            raise ValueError("num_agents must be at most 200")
        if not 0 < self.decay_rate <= 1:
            raise ValueError("decay_rate must be in (0, 1]")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        if not 0 < self.beta < 1:
            raise ValueError("beta must be in (0, 1)")
        if not 0 < self.initial_trust < 1:
            raise ValueError("initial_trust must be in (0, 1)")
        if self.dynamic_base_size > self.dynamic_max_size:
            raise ValueError("dynamic_base_size must not exceed dynamic_max_size")
        if self.max_ticks < 1:
            raise ValueError("max_ticks must be at least 1")


@dataclass
class ExperimentConfig:
    """Configuration for batch experiments."""

    # Parameter sweep settings
    num_agents_range: list = field(default_factory=lambda: [10, 50, 100, 200])
    memory_types: list = field(
        default_factory=lambda: ["fixed", "decay", "dynamic"]
    )
    num_trials: int = 10  # Repetitions per configuration
    parallel_workers: int = 4

    # Analysis settings
    compute_convergence_time: bool = True
    compute_consensus_strength: bool = True
    compute_strategy_switches: bool = True
    compute_trust_distribution: bool = True


# Default configurations
DEFAULT_CONFIG = SimulationConfig()

QUICK_TEST_CONFIG = SimulationConfig(
    num_agents=10,
    max_ticks=100,
    save_history=False,
)

FULL_EXPERIMENT_CONFIG = SimulationConfig(
    num_agents=100,
    max_ticks=1000,
    save_history=True,
    save_interval=1,
)
