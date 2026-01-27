"""
Base memory class for ABM simulation.

Defines the interface and common functionality for all memory mechanisms.
"""

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class Interaction:
    """
    Record of a single interaction between agents.

    Agents are anonymous: only the partner's strategy is observed, not their identity.

    Attributes:
        tick: Time step when interaction occurred
        partner_strategy: Strategy chosen by partner (0 or 1)
        own_strategy: Strategy chosen by self (0 or 1)
        success: Whether coordination was successful
        payoff: Payoff received from this interaction
    """

    tick: int
    partner_strategy: int
    own_strategy: int
    success: bool
    payoff: float


class BaseMemory(ABC):
    """
    Abstract base class for memory mechanisms.

    Memory systems store interaction history and provide
    weighted estimates of the strategy distribution in the population.
    """

    def __init__(self, max_size: int = 5):
        """
        Initialize memory.

        Args:
            max_size: Maximum number of interactions to store
        """
        self._history: deque = deque(maxlen=max_size)
        self._max_size = max_size

    @property
    def max_size(self) -> int:
        """Maximum memory capacity."""
        return self._max_size

    @property
    def current_size(self) -> int:
        """Current number of stored interactions."""
        return len(self._history)

    @property
    def history(self) -> List[Interaction]:
        """Get list of stored interactions (oldest first)."""
        return list(self._history)

    def add(self, interaction: Interaction) -> None:
        """
        Add a new interaction to memory.

        Args:
            interaction: The interaction to record
        """
        self._history.append(interaction)

    def clear(self) -> None:
        """Clear all stored interactions."""
        self._history.clear()

    def is_empty(self) -> bool:
        """Check if memory is empty."""
        return len(self._history) == 0

    @abstractmethod
    def get_weighted_history(self) -> List[Tuple[Interaction, float]]:
        """
        Get interactions with their weights.

        Returns:
            List of (interaction, weight) tuples, weights sum to 1
        """
        pass

    def get_strategy_distribution(self) -> np.ndarray:
        """
        Compute weighted estimate of partner strategy distribution.

        Returns:
            Array [P(strategy=0), P(strategy=1)] based on memory
            Returns [0.5, 0.5] if memory is empty
        """
        if self.is_empty():
            return np.array([0.5, 0.5])

        weighted_history = self.get_weighted_history()
        counts = np.zeros(2)

        for interaction, weight in weighted_history:
            counts[interaction.partner_strategy] += weight

        # Normalize to get distribution
        total = counts.sum()
        if total > 0:
            return counts / total
        return np.array([0.5, 0.5])

    def get_success_rate(self) -> float:
        """
        Compute weighted success rate from memory.

        Returns:
            Weighted proportion of successful coordinations
        """
        if self.is_empty():
            return 0.5

        weighted_history = self.get_weighted_history()
        success_sum = sum(
            weight for interaction, weight in weighted_history if interaction.success
        )
        return success_sum

    def get_most_frequent_partner_strategy(self) -> Optional[int]:
        """
        Get the most frequently observed partner strategy.

        Returns:
            0 or 1 (the more frequent strategy), or None if empty
        """
        if self.is_empty():
            return None

        dist = self.get_strategy_distribution()
        return int(np.argmax(dist))

    def __len__(self) -> int:
        """Return current memory size."""
        return len(self._history)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(max_size={self._max_size}, current={len(self)})"
