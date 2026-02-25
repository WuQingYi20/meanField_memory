"""
Decision mechanism module.

Provides different decision modes for agent behavior:

1. DUAL_FEEDBACK (Original):
   - τ-based softmax action selection
   - τ affects both behavior (softmax) and cognition (memory window)
   - Two interacting feedback loops

2. COGNITIVE_LOCKIN (Simplified):
   - Probability matching action selection (behaviorally neutral)
   - Trust only affects memory window (cognitive only)
   - Clean attribution to memory mechanism

3. EPSILON_GREEDY (Robustness Check):
   - Best response with trust-based exploration
   - Behavioral amplification of majority
   - Faster convergence than probability matching

Usage:
    from src.decision import create_decision, DecisionMode

    # Create specific mechanism
    decision = create_decision(
        mode=DecisionMode.COGNITIVE_LOCKIN,
        initial_trust=0.5,
        alpha=0.1,
        beta=0.3
    )

    # Use in agent
    action, prediction = decision.choose_action(belief)
    decision.update(prediction, observed)
    trust = decision.get_trust()
"""

from .base import BaseDecision, DecisionMode
from .dual_feedback import DualFeedbackDecision
from .cognitive_lockin import CognitiveLockInDecision
from .epsilon_greedy import EpsilonGreedyDecision

from typing import Union


def create_decision(
    mode: Union[DecisionMode, str] = DecisionMode.COGNITIVE_LOCKIN,
    **kwargs
) -> BaseDecision:
    """
    Factory function to create a decision mechanism.

    Args:
        mode: Decision mode (enum or string)
        **kwargs: Mode-specific parameters

    For DUAL_FEEDBACK:
        initial_tau: Starting temperature (default: 1.0)
        tau_min: Minimum temperature (default: 0.1)
        tau_max: Maximum temperature (default: 2.0)
        cooling_rate: τ decrease rate on success (default: 0.1)
        heating_penalty: τ increase on failure (default: 0.3)

    For COGNITIVE_LOCKIN and EPSILON_GREEDY:
        initial_trust: Starting trust (default: 0.5)
        alpha: Trust increase rate (default: 0.1)
        beta: Trust decay rate (default: 0.3)

    For EPSILON_GREEDY only:
        exploration_mode: "random" or "opposite" (default: "random")

    Returns:
        Configured decision mechanism instance
    """
    # Convert string to enum if needed
    if isinstance(mode, str):
        mode = DecisionMode(mode)

    if mode == DecisionMode.DUAL_FEEDBACK:
        return DualFeedbackDecision(**kwargs)
    elif mode == DecisionMode.COGNITIVE_LOCKIN:
        return CognitiveLockInDecision(**kwargs)
    elif mode == DecisionMode.EPSILON_GREEDY:
        return EpsilonGreedyDecision(**kwargs)
    else:
        raise ValueError(f"Unknown decision mode: {mode}")


# Backward compatibility aliases
TrustBasedDecision = CognitiveLockInDecision
PredictionErrorDecision = DualFeedbackDecision


__all__ = [
    "BaseDecision",
    "DecisionMode",
    "DualFeedbackDecision",
    "CognitiveLockInDecision",
    "EpsilonGreedyDecision",
    "create_decision",
    # Backward compatibility
    "TrustBasedDecision",
    "PredictionErrorDecision",
]
