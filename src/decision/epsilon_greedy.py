"""
Epsilon-Greedy decision mechanism.

Best response with trust-based exploration rate.
Useful as a robustness check to compare with probability matching.

Action selection:
- With probability (1-ε): play best response (argmax of belief)
- With probability ε: explore (random or opposite of best response)

Where ε = 1 - trust (high trust → low exploration)
"""

import numpy as np
from typing import Optional, Dict, Any, Literal

from .base import BaseDecision, DecisionMode


class EpsilonGreedyDecision(BaseDecision):
    """
    Trust-based ε-greedy decision mechanism.

    Key characteristics:
    - Trust is the primary state variable
    - Exploration rate ε = 1 - trust
    - Best response with probability (1 - ε)

    This creates behavioral amplification: even small majorities
    are exploited, leading to faster convergence than probability matching.

    Trust update rules (same as CognitiveLockIn):
    - Correct prediction: Trust += α(1 - Trust)
    - Wrong prediction: Trust *= (1 - β)
    """

    def __init__(
        self,
        initial_trust: float = 0.5,
        alpha: float = 0.1,
        beta: float = 0.3,
        trust_min: float = 0.01,
        trust_max: float = 0.99,
        exploration_mode: Literal["random", "opposite"] = "random",
        **kwargs
    ):
        """
        Initialize ε-greedy mechanism.

        Args:
            initial_trust: Starting trust level
            alpha: Trust increase rate on correct prediction
            beta: Trust decay rate on wrong prediction
            trust_min: Minimum trust (avoid edge cases)
            trust_max: Maximum trust (avoid edge cases)
            exploration_mode: "random" for uniform, "opposite" for anti-best-response
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
        self._exploration_mode = exploration_mode

    @property
    def mode(self) -> DecisionMode:
        return DecisionMode.EPSILON_GREEDY

    @property
    def trust(self) -> float:
        """Current trust level."""
        return self._trust

    @property
    def epsilon(self) -> float:
        """Current exploration rate."""
        return 1.0 - self._trust

    def get_trust(self) -> float:
        """Get current trust level."""
        return self._trust

    def predict(self, belief: np.ndarray) -> int:
        """Predict partner's strategy as argmax of belief."""
        return int(np.argmax(belief))

    def select_action(self, belief: np.ndarray) -> int:
        """
        Select action using ε-greedy.

        With probability (1-ε): best response
        With probability ε: explore
        """
        best_response = self.predict(belief)
        epsilon = self.epsilon

        if np.random.random() < epsilon:
            # Explore
            if self._exploration_mode == "opposite":
                # Try the other option
                return 1 - best_response
            else:
                # Random
                return np.random.choice(len(belief))
        else:
            # Exploit
            return best_response

    def update(self, predicted: int, observed: int) -> bool:
        """
        Update trust based on prediction accuracy.

        Same asymmetric update as CognitiveLockIn.
        """
        success = (predicted == observed)
        self._total_predictions += 1

        if success:
            self._trust = self._trust + self._alpha * (1 - self._trust)
            self._correct_predictions += 1
        else:
            self._trust = self._trust * (1 - self._beta)

        self._trust = np.clip(self._trust, self._trust_min, self._trust_max)

        return success

    def reset(self) -> None:
        """Reset to initial state."""
        super().reset()
        self._trust = self._initial_trust

    def get_state(self) -> Dict[str, Any]:
        """Get current internal state."""
        return {
            "mode": self.mode.value,
            "trust": self._trust,
            "epsilon": self.epsilon,
            "alpha": self._alpha,
            "beta": self._beta,
            "exploration_mode": self._exploration_mode,
        }

    def __repr__(self) -> str:
        return f"EpsilonGreedyDecision(trust={self._trust:.3f}, ε={self.epsilon:.3f})"
