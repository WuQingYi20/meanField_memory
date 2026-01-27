"""
Decision mechanism based on probability matching and trust dynamics.

Implements trust-based learning where prediction accuracy directly updates trust,
and action selection uses probability matching (P(A) = belief_A).
"""

import numpy as np
from typing import Optional, Tuple


class TrustBasedDecision:
    """
    Trust-based decision mechanism with probability matching.

    Key simplification from the original model:
    - Trust is the primary state variable (no intermediate τ)
    - Action selection uses probability matching: P(A) = b_A
    - Trust only affects memory window, not action selection

    Trust update rules (asymmetric):
    - Correct prediction: Trust += α(1 - Trust)  [slow build, saturates at 1]
    - Wrong prediction: Trust *= (1 - β)         [fast break, proportional loss]

    This creates "cognitive lock-in": higher trust → larger memory window →
    more stable beliefs → harder to reverse established norms.
    """

    def __init__(
        self,
        initial_trust: float = 0.5,
        alpha: float = 0.1,
        beta: float = 0.3,
        trust_min: float = 0.01,
        trust_max: float = 0.99,
    ):
        """
        Initialize decision mechanism.

        Args:
            initial_trust: Starting trust level
            alpha: Trust increase rate on correct prediction
            beta: Trust decay rate on wrong prediction
            trust_min: Minimum trust (avoid edge cases)
            trust_max: Maximum trust (avoid edge cases)
        """
        self._validate_params(initial_trust, alpha, beta, trust_min, trust_max)

        self._initial_trust = initial_trust
        self._trust = initial_trust
        self._alpha = alpha
        self._beta = beta
        self._trust_min = trust_min
        self._trust_max = trust_max

        # Statistics tracking
        self._total_predictions = 0
        self._correct_predictions = 0

    def _validate_params(
        self,
        initial_trust: float,
        alpha: float,
        beta: float,
        trust_min: float,
        trust_max: float,
    ) -> None:
        """Validate initialization parameters."""
        if not 0 < trust_min < trust_max < 1:
            raise ValueError("trust bounds must satisfy 0 < trust_min < trust_max < 1")
        if not trust_min <= initial_trust <= trust_max:
            raise ValueError("initial_trust must be between trust_min and trust_max")
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        if not 0 < beta < 1:
            raise ValueError("beta must be in (0, 1)")

    @property
    def trust(self) -> float:
        """Current trust level."""
        return self._trust

    @property
    def alpha(self) -> float:
        """Trust increase rate."""
        return self._alpha

    @property
    def beta(self) -> float:
        """Trust decay rate."""
        return self._beta

    def predict(self, belief: np.ndarray) -> int:
        """
        Make a prediction based on current beliefs.

        The prediction is the strategy with highest belief.

        Args:
            belief: Array [P(strategy=0), P(strategy=1)]

        Returns:
            Predicted strategy (0 or 1)
        """
        return int(np.argmax(belief))

    def probability_matching_choice(self, belief: np.ndarray) -> int:
        """
        Make a probabilistic choice using probability matching.

        P(choose strategy i) = belief[i]

        This is behaviorally neutral: if population is 70% A,
        agent chooses A with 70% probability.

        Args:
            belief: Array [P(strategy=0), P(strategy=1)]

        Returns:
            Chosen strategy (0 or 1)
        """
        belief = np.asarray(belief, dtype=np.float64)
        # Ensure valid probability distribution
        belief = np.clip(belief, 0, 1)
        belief = belief / belief.sum()

        return np.random.choice(len(belief), p=belief)

    def choose_action(self, belief: np.ndarray) -> Tuple[int, int]:
        """
        Choose an action and make a prediction.

        Args:
            belief: Strategy distribution estimate [P(0), P(1)]

        Returns:
            Tuple of (chosen_action, predicted_partner_action)
        """
        prediction = self.predict(belief)
        action = self.probability_matching_choice(belief)
        return action, prediction

    def update_trust(self, predicted: int, observed: int) -> bool:
        """
        Update trust based on prediction accuracy.

        Asymmetric update (Slovic 1993 - "trust is fragile"):
        - Correct: Trust builds slowly, saturating at 1
        - Wrong: Trust breaks quickly, proportional to current level

        Args:
            predicted: What we predicted the partner would do
            observed: What the partner actually did

        Returns:
            True if prediction was correct, False otherwise
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

    def get_prediction_accuracy(self) -> Optional[float]:
        """
        Get historical prediction accuracy.

        Returns:
            Fraction of correct predictions, or None if no predictions made
        """
        if self._total_predictions == 0:
            return None
        return self._correct_predictions / self._total_predictions

    def get_trust_steady_state(self, prediction_accuracy: float) -> float:
        """
        Calculate theoretical trust steady state for given prediction accuracy.

        At steady state: E[ΔTrust] = 0
        p × α(1-T*) = (1-p) × βT*
        T* = pα / (pα + (1-p)β)

        Args:
            prediction_accuracy: Probability of correct prediction (p)

        Returns:
            Steady state trust level
        """
        p = prediction_accuracy
        return (p * self._alpha) / (p * self._alpha + (1 - p) * self._beta)

    def reset(self, initial_trust: Optional[float] = None) -> None:
        """
        Reset decision state.

        Args:
            initial_trust: New starting trust (uses original if None)
        """
        if initial_trust is not None:
            if not self._trust_min <= initial_trust <= self._trust_max:
                raise ValueError("initial_trust must be between trust_min and trust_max")
            self._trust = initial_trust
        else:
            self._trust = self._initial_trust

        self._total_predictions = 0
        self._correct_predictions = 0

    def __repr__(self) -> str:
        return (
            f"TrustBasedDecision(trust={self._trust:.3f}, "
            f"α={self._alpha}, β={self._beta})"
        )


# Backward compatibility alias
PredictionErrorDecision = TrustBasedDecision
