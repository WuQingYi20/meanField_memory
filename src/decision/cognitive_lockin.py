"""
Cognitive Lock-in decision mechanism.

Simplified model where Trust is the primary state variable and
only affects cognition (memory window), not behavior.

Action selection uses probability matching (behaviorally neutral):
- P(A) = belief_A

This creates clean attribution: any norm emergence can only
be attributed to the memory mechanism, not to behavioral biases.

The feedback loop operates through "cognitive lock-in":
- Majority forms → Prediction accuracy ↑ → Trust ↑ → Window ↑ → Beliefs stabilize
"""

import numpy as np
from typing import Optional, Dict, Any

from .base import BaseDecision, DecisionMode


class CognitiveLockInDecision(BaseDecision):
    """
    Trust-based decision with probability matching.

    Key characteristics:
    - Trust is the primary state variable
    - Action selection is behaviorally neutral (probability matching)
    - Trust only affects memory window (cognitive, not behavioral)

    Trust update rules (asymmetric, Slovic 1993):
    - Correct prediction: Trust += α(1 - Trust)  [slow build, saturates at 1]
    - Wrong prediction: Trust *= (1 - β)         [fast break, proportional]

    Higher trust means more "to lose" — a failure causes greater absolute
    loss when trust is high.
    """

    def __init__(
        self,
        initial_trust: float = 0.5,
        alpha: float = 0.1,
        beta: float = 0.3,
        trust_min: float = 0.01,
        trust_max: float = 0.99,
        **kwargs
    ):
        """
        Initialize cognitive lock-in mechanism.

        Args:
            initial_trust: Starting trust level
            alpha: Trust increase rate on correct prediction
            beta: Trust decay rate on wrong prediction
            trust_min: Minimum trust (avoid edge cases)
            trust_max: Maximum trust (avoid edge cases)
        """
        super().__init__(**kwargs)

        if not 0 < trust_min < trust_max < 1:
            raise ValueError("trust bounds must satisfy 0 < trust_min < trust_max < 1")
        if not trust_min <= initial_trust <= trust_max:
            raise ValueError("initial_trust must be between trust_min and trust_max")
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        if not 0 < beta < 1:
            raise ValueError("beta must be in (0, 1)")

        self._initial_trust = initial_trust
        self._trust = initial_trust
        self._alpha = alpha
        self._beta = beta
        self._trust_min = trust_min
        self._trust_max = trust_max

    @property
    def mode(self) -> DecisionMode:
        return DecisionMode.COGNITIVE_LOCKIN

    @property
    def trust(self) -> float:
        """Current trust level."""
        return self._trust

    def get_trust(self) -> float:
        """Get current trust level."""
        return self._trust

    def predict(self, belief: np.ndarray) -> int:
        """
        Predict partner's strategy as argmax of belief.

        When belief is tied (e.g., [0.5, 0.5]), randomize to avoid
        systematic bias toward strategy 0.
        """
        if abs(belief[0] - belief[1]) < 0.001:
            # Tie: randomize to preserve symmetry
            return np.random.choice([0, 1])
        return int(np.argmax(belief))

    def select_action(self, belief: np.ndarray) -> int:
        """
        Select action using probability matching.

        P(action=i) = belief[i]

        This is behaviorally neutral: no amplification of majority.
        """
        belief = np.asarray(belief, dtype=np.float64)
        belief = np.clip(belief, 0, 1)
        belief = belief / belief.sum()

        return np.random.choice(len(belief), p=belief)

    def update(self, predicted: int, observed: int) -> bool:
        """
        Update trust based on prediction accuracy.

        Asymmetric update (Slovic 1993):
        - Correct: additive increase with saturation
        - Wrong: multiplicative decay (proportional to current level)
        """
        success = (predicted == observed)
        self._total_predictions += 1

        if success:
            # Slow build: Trust += α(1 - Trust)
            self._trust = self._trust + self._alpha * (1 - self._trust)
            self._correct_predictions += 1
        else:
            # Fast break: Trust *= (1 - β)
            self._trust = self._trust * (1 - self._beta)

        # Clamp to valid range
        self._trust = np.clip(self._trust, self._trust_min, self._trust_max)

        return success

    def get_trust_steady_state(self, prediction_accuracy: float) -> float:
        """
        Calculate theoretical trust steady state.

        At steady state: E[ΔTrust] = 0
        p × α(1-T*) = (1-p) × βT*
        T* = pα / (pα + (1-p)β)

        Args:
            prediction_accuracy: Probability of correct prediction (p)

        Returns:
            Steady state trust level
        """
        p = prediction_accuracy
        denominator = p * self._alpha + (1 - p) * self._beta
        if denominator == 0:
            return 0.5
        return (p * self._alpha) / denominator

    def reset(self) -> None:
        """Reset to initial state."""
        super().reset()
        self._trust = self._initial_trust

    def get_state(self) -> Dict[str, Any]:
        """Get current internal state."""
        return {
            "mode": self.mode.value,
            "trust": self._trust,
            "alpha": self._alpha,
            "beta": self._beta,
            "trust_min": self._trust_min,
            "trust_max": self._trust_max,
        }

    def __repr__(self) -> str:
        return f"CognitiveLockInDecision(trust={self._trust:.3f})"
