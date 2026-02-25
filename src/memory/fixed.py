"""
Fixed-length memory mechanism.

Stores the k most recent interactions with equal weights.
"""

from typing import List, Tuple

from .base import BaseMemory, Interaction


class FixedMemory(BaseMemory):
    """
    Fixed-length sliding window memory.

    Maintains exactly the k most recent interactions,
    weighting them all equally when computing statistics.

    This represents a simple, bounded memory model where
    only recent experiences matter and all are equally important.
    """

    def __init__(self, size: int = 5):
        """
        Initialize fixed-length memory.

        Args:
            size: Number of recent interactions to keep (default: 5)
        """
        super().__init__(max_size=size)

    def get_weighted_history(self) -> List[Tuple[Interaction, float]]:
        """
        Get interactions with equal weights.

        All stored interactions receive weight 1/n where n is
        the current number of stored interactions.

        Returns:
            List of (interaction, weight) tuples
        """
        if self.is_empty():
            return []

        n = len(self._history)
        weight = 1.0 / n

        return [(interaction, weight) for interaction in self._history]

    def __repr__(self) -> str:
        return f"FixedMemory(size={self._max_size}, stored={len(self)})"
