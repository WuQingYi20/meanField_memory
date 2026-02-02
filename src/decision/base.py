"""
Abstract base class for decision mechanisms.

Provides a common interface for different decision modes:
- DualFeedback: Original model with τ-based softmax (behavioral + cognitive)
- CognitiveLockIn: Probability matching with Trust (cognitive only)
- EpsilonGreedy: Best response with exploration (for robustness check)
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Tuple, Dict, Any
import numpy as np


class DecisionMode(Enum):
    """Available decision mechanism modes."""
    DUAL_FEEDBACK = "dual_feedback"        # Original: softmax + τ affects both behavior and memory
    COGNITIVE_LOCKIN = "cognitive_lockin"  # New: probability matching, trust only affects memory
    EPSILON_GREEDY = "epsilon_greedy"      # Robustness check: best response + exploration


class BaseDecision(ABC):
    """
    Abstract base class for all decision mechanisms.

    All decision mechanisms must implement:
    - choose_action(): Select action and make prediction
    - update(): Update internal state based on prediction accuracy
    - get_trust(): Return current trust level (for memory window)

    This allows different mechanisms to be swapped while maintaining
    compatibility with the rest of the system.
    """

    def __init__(self, **kwargs):
        """Initialize with mechanism-specific parameters."""
        self._total_predictions = 0
        self._correct_predictions = 0

    @property
    @abstractmethod
    def mode(self) -> DecisionMode:
        """Return the decision mode."""
        pass

    @abstractmethod
    def get_trust(self) -> float:
        """
        Get current trust level.

        Returns:
            Trust level in [0, 1], used for memory window calculation
        """
        pass

    @abstractmethod
    def predict(self, belief: np.ndarray) -> int:
        """
        Make a prediction about partner's strategy.

        Args:
            belief: [P(strategy=0), P(strategy=1)]

        Returns:
            Predicted strategy (0 or 1)
        """
        pass

    @abstractmethod
    def select_action(self, belief: np.ndarray) -> int:
        """
        Select an action based on belief.

        Args:
            belief: [P(strategy=0), P(strategy=1)]

        Returns:
            Chosen action (0 or 1)
        """
        pass

    def choose_action(self, belief: np.ndarray) -> Tuple[int, int]:
        """
        Choose action and make prediction.

        This is the main interface used by Agent.

        Args:
            belief: Strategy distribution estimate

        Returns:
            Tuple of (action, prediction)
        """
        prediction = self.predict(belief)
        action = self.select_action(belief)
        return action, prediction

    @abstractmethod
    def update(self, predicted: int, observed: int) -> bool:
        """
        Update internal state based on prediction accuracy.

        Args:
            predicted: What we predicted
            observed: What actually happened

        Returns:
            True if prediction was correct
        """
        pass

    def get_prediction_accuracy(self) -> Optional[float]:
        """Get historical prediction accuracy."""
        if self._total_predictions == 0:
            return None
        return self._correct_predictions / self._total_predictions

    @abstractmethod
    def reset(self) -> None:
        """Reset to initial state."""
        self._total_predictions = 0
        self._correct_predictions = 0

    @abstractmethod
    def get_state(self) -> Dict[str, Any]:
        """
        Get current internal state for logging/analysis.

        Returns:
            Dict with mechanism-specific state variables
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mode={self.mode.value})"
