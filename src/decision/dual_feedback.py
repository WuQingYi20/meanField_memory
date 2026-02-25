"""
Dual Feedback decision mechanism.

Original model where τ (temperature) controls BOTH:
1. Action selection randomness (via softmax)
2. Memory window size (via trust derived from τ)

This creates two interacting feedback loops:
- Positive loop: Success → τ↓ → Trust↑ → Window↑ + Deterministic behavior → More success
- Negative loop: Failure → τ↑ → Trust↓ → Window↓ + Random behavior → Exploration
"""

import numpy as np
from typing import Optional, Dict, Any

from .base import BaseDecision, DecisionMode


class DualFeedbackDecision(BaseDecision):
    """
    Temperature-based softmax decision with dual feedback loops.

    Key characteristics:
    - τ is the primary state variable (anxiety/uncertainty)
    - Trust is derived from τ: Trust = 1 - (τ - τ_min) / (τ_max - τ_min)
    - Action selection: softmax(belief, τ)
    - Both behavior and cognition are affected by τ

    Temperature update rules:
    - Correct prediction: τ *= (1 - cooling_rate)  [multiplicative, slow]
    - Wrong prediction: τ += heating_penalty       [additive, fast]
    """

    def __init__(
        self,
        initial_tau: float = 1.0,
        tau_min: float = 0.1,
        tau_max: float = 2.0,
        cooling_rate: float = 0.1,
        heating_penalty: float = 0.3,
        **kwargs
    ):
        """
        Initialize dual feedback mechanism.

        Args:
            initial_tau: Starting temperature
            tau_min: Minimum temperature (maximum commitment)
            tau_max: Maximum temperature (maximum randomness)
            cooling_rate: τ decrease rate on correct prediction
            heating_penalty: τ increase on wrong prediction
        """
        super().__init__(**kwargs)

        if tau_min <= 0:
            raise ValueError("tau_min must be positive")
        if tau_max <= tau_min:
            raise ValueError("tau_max must be greater than tau_min")
        if not tau_min <= initial_tau <= tau_max:
            raise ValueError("initial_tau must be between tau_min and tau_max")
        if not 0 <= cooling_rate <= 1:
            raise ValueError("cooling_rate must be in [0, 1]")
        if heating_penalty < 0:
            raise ValueError("heating_penalty must be non-negative")

        self._initial_tau = initial_tau
        self._tau = initial_tau
        self._tau_min = tau_min
        self._tau_max = tau_max
        self._cooling_rate = cooling_rate
        self._heating_penalty = heating_penalty

    @property
    def mode(self) -> DecisionMode:
        return DecisionMode.DUAL_FEEDBACK

    @property
    def tau(self) -> float:
        """Current temperature value."""
        return self._tau

    def get_trust(self) -> float:
        """
        Convert temperature to trust level.

        Linear mapping: τ_min → 1.0, τ_max → 0.0
        """
        return 1.0 - (self._tau - self._tau_min) / (self._tau_max - self._tau_min)

    def predict(self, belief: np.ndarray) -> int:
        """Predict partner's strategy as argmax of belief."""
        return int(np.argmax(belief))

    def select_action(self, belief: np.ndarray) -> int:
        """
        Select action using temperature-modulated softmax.

        P(action=i) = exp(belief[i] / τ) / Σ exp(belief[j] / τ)

        Low τ → deterministic (argmax)
        High τ → uniform random
        """
        belief = np.asarray(belief, dtype=np.float64)
        scaled = belief / self._tau

        # Numerically stable softmax
        exp_scaled = np.exp(scaled - np.max(scaled))
        probs = exp_scaled / exp_scaled.sum()

        return np.random.choice(len(probs), p=probs)

    def update(self, predicted: int, observed: int) -> bool:
        """
        Update temperature based on prediction accuracy.

        Asymmetric update:
        - Correct: multiplicative cooling (slow trust building)
        - Wrong: additive heating (fast trust breaking)
        """
        success = (predicted == observed)
        self._total_predictions += 1

        if success:
            self._tau = max(self._tau_min, self._tau * (1 - self._cooling_rate))
            self._correct_predictions += 1
        else:
            self._tau = min(self._tau_max, self._tau + self._heating_penalty)

        return success

    def reset(self) -> None:
        """Reset to initial state."""
        super().reset()
        self._tau = self._initial_tau

    def get_state(self) -> Dict[str, Any]:
        """Get current internal state."""
        return {
            "mode": self.mode.value,
            "tau": self._tau,
            "trust": self.get_trust(),
            "tau_min": self._tau_min,
            "tau_max": self._tau_max,
            "cooling_rate": self._cooling_rate,
            "heating_penalty": self._heating_penalty,
        }

    def __repr__(self) -> str:
        return (
            f"DualFeedbackDecision(τ={self._tau:.3f}, "
            f"trust={self.get_trust():.2f})"
        )
