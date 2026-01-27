"""
Decay memory mechanism.

Weights interactions exponentially based on recency.
"""

from typing import List, Tuple
import numpy as np

from .base import BaseMemory, Interaction


class DecayMemory(BaseMemory):
    """
    Exponentially decaying memory.

    Recent interactions have higher weights than older ones,
    following: weight(age) = lambda^age

    where age is the number of interactions since this one occurred.
    The newest interaction has age 0.

    This models how memory naturally fades over time,
    with recent experiences having more influence on decisions.
    """

    def __init__(self, max_size: int = 10, decay_rate: float = 0.9):
        """
        Initialize decay memory.

        Args:
            max_size: Maximum number of interactions to store
            decay_rate: Lambda parameter for exponential decay (0 < lambda <= 1)
                       Higher values mean slower decay (more weight on old memories)
        """
        super().__init__(max_size=max_size)

        if not 0 < decay_rate <= 1:
            raise ValueError("decay_rate must be in (0, 1]")

        self._decay_rate = decay_rate

    @property
    def decay_rate(self) -> float:
        """The decay rate parameter (lambda)."""
        return self._decay_rate

    def get_weighted_history(self) -> List[Tuple[Interaction, float]]:
        """
        Get interactions with exponentially decaying weights.

        The most recent interaction has weight lambda^0 = 1,
        the second most recent has weight lambda^1, etc.
        All weights are then normalized to sum to 1.

        Returns:
            List of (interaction, weight) tuples, newest last
        """
        if self.is_empty():
            return []

        n = len(self._history)

        # Calculate raw weights: newest (index n-1) gets highest weight
        # age = (n-1) - i for interaction at index i
        raw_weights = np.array(
            [self._decay_rate ** (n - 1 - i) for i in range(n)]
        )

        # Normalize weights to sum to 1
        total = raw_weights.sum()
        weights = raw_weights / total

        return list(zip(self._history, weights))

    def get_effective_window(self, threshold: float = 0.01) -> int:
        """
        Calculate the effective memory window size.

        This is the number of recent interactions that contribute
        at least `threshold` fraction of total weight.

        Args:
            threshold: Minimum relative weight to count

        Returns:
            Number of interactions with significant weight
        """
        if self._decay_rate >= 1:
            return self._max_size

        # Weight at age k relative to newest: lambda^k
        # Find k where lambda^k >= threshold
        # k <= log(threshold) / log(lambda)
        import math

        if self._decay_rate <= 0:
            return 1

        k_max = math.log(threshold) / math.log(self._decay_rate)
        return min(int(k_max) + 1, self._max_size)

    def __repr__(self) -> str:
        return (
            f"DecayMemory(max_size={self._max_size}, "
            f"decay_rate={self._decay_rate:.2f}, stored={len(self)})"
        )
