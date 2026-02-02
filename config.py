"""
Configuration module for ABM Memory & Norm Formation Simulation.

This module defines all configurable parameters for the simulation,
including agent settings, memory mechanisms, decision parameters,
and simulation controls.

Supports multiple decision modes:
- DUAL_FEEDBACK: Original τ-based softmax (two feedback loops)
- COGNITIVE_LOCKIN: Probability matching with Trust (cognitive only)
- EPSILON_GREEDY: Best response with exploration (robustness check)
"""

from dataclasses import dataclass, field
from typing import Literal, Optional

from src.decision import DecisionMode


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

    # Decision mode
    decision_mode: DecisionMode = DecisionMode.COGNITIVE_LOCKIN

    # DUAL_FEEDBACK mode parameters (τ-based softmax)
    initial_tau: float = 1.0  # Starting temperature (mid-randomness)
    tau_min: float = 0.1  # Minimum temperature (most deterministic)
    tau_max: float = 2.0  # Maximum temperature (most random)
    cooling_rate: float = 0.1  # τ decrease rate on correct prediction
    heating_penalty: float = 0.3  # τ increase on wrong prediction

    # COGNITIVE_LOCKIN and EPSILON_GREEDY mode parameters
    initial_trust: float = 0.5  # Starting trust level
    alpha: float = 0.1  # Trust increase rate on correct prediction
    beta: float = 0.3  # Trust decay rate on wrong prediction

    # EPSILON_GREEDY specific
    exploration_mode: Literal["random", "opposite"] = "random"

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

        # Mode-specific validation
        if self.decision_mode == DecisionMode.DUAL_FEEDBACK:
            if not 0 < self.tau_min < self.tau_max:
                raise ValueError("tau_min must be positive and less than tau_max")
            if not 0 < self.cooling_rate < 1:
                raise ValueError("cooling_rate must be in (0, 1)")
            if not 0 < self.heating_penalty:
                raise ValueError("heating_penalty must be positive")
        else:
            # COGNITIVE_LOCKIN or EPSILON_GREEDY
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
    decision_modes: list = field(
        default_factory=lambda: [
            DecisionMode.COGNITIVE_LOCKIN,
            DecisionMode.DUAL_FEEDBACK,
            DecisionMode.EPSILON_GREEDY,
        ]
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

# =============================================================================
# Preset Configurations for Different Experimental Scenarios
# =============================================================================

# RECOMMENDED: Cognitive Lock-in with asymmetric trust (Slovic 1993)
# - Trust is hard to build, easy to break (negativity bias)
# - Steady state: T* = p*alpha / (p*alpha + (1-p)*beta)
# - At p=0.5: T*=0.25, At p=0.8: T*=0.57, At p=0.9: T*=0.75
COGNITIVE_LOCKIN_CONFIG = SimulationConfig(
    decision_mode=DecisionMode.COGNITIVE_LOCKIN,
    memory_type="dynamic",
    initial_trust=0.5,
    alpha=0.1,   # Slow trust building
    beta=0.3,    # Fast trust breaking (3x faster than building)
)

# Fast convergence variant: symmetric trust updates
# - Equal weight to success and failure
# - Steady state: T* = p (directly matches prediction accuracy)
FAST_CONVERGENCE_CONFIG = SimulationConfig(
    decision_mode=DecisionMode.COGNITIVE_LOCKIN,
    memory_type="dynamic",
    initial_trust=0.5,
    alpha=0.2,   # Symmetric
    beta=0.2,    # Symmetric
)

# Sensitive trust variant: high responsiveness
# - Trust changes rapidly with each prediction
# - Good for studying transient dynamics
SENSITIVE_TRUST_CONFIG = SimulationConfig(
    decision_mode=DecisionMode.COGNITIVE_LOCKIN,
    memory_type="dynamic",
    initial_trust=0.5,
    alpha=0.3,   # Fast increase
    beta=0.4,    # Faster decrease
)

# Original dual feedback model (for comparison)
# NOTE: Requires ~75% prediction accuracy to maintain tau=1.0
# May not converge well from 50-50 initial distribution
DUAL_FEEDBACK_CONFIG = SimulationConfig(
    decision_mode=DecisionMode.DUAL_FEEDBACK,
    memory_type="dynamic",
    initial_tau=1.0,
    tau_min=0.1,
    tau_max=2.0,
    cooling_rate=0.1,
    heating_penalty=0.3,
)

# Epsilon-greedy for robustness check
EPSILON_GREEDY_CONFIG = SimulationConfig(
    decision_mode=DecisionMode.EPSILON_GREEDY,
    memory_type="dynamic",
    initial_trust=0.5,
    alpha=0.1,
    beta=0.3,
    exploration_mode="random",
)
