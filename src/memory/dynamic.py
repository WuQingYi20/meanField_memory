"""
Dynamic memory mechanism.

Memory window size adapts based on trust level,
implementing the feedback loop between coordination success and memory.
"""

from typing import List, Tuple, Optional, Callable

from .base import BaseMemory, Interaction


class DynamicMemory(BaseMemory):
    """
    Trust-linked dynamic memory.

    The effective memory window expands or contracts based on trust:
    - High trust (from successful coordination) -> longer window -> more stable
    - Low trust (from failed coordination) -> shorter window -> more adaptive

    This models how confidence in the environment affects
    how much past experience we rely on.

    Window size formula:
        window = base_size + int(trust * (max_size - base_size))

    Where trust is in [0, 1], giving window in [base_size, max_size].
    """

    def __init__(
        self,
        max_size: int = 6,
        base_size: int = 2,
        trust_getter: Optional[Callable[[], float]] = None,
    ):
        """
        Initialize dynamic memory.

        Args:
            max_size: Maximum memory window (cognitive limit, default: 6)
            base_size: Minimum memory window (default: 2)
            trust_getter: Callable that returns current trust level [0, 1]
                         If None, uses internal trust tracking
        """
        # Store all interactions up to max, but use subset based on window
        super().__init__(max_size=max_size)

        if base_size > max_size:
            raise ValueError("base_size cannot exceed max_size")
        if base_size < 1:
            raise ValueError("base_size must be at least 1")

        self._base_size = base_size
        self._trust_getter = trust_getter
        self._internal_trust = 0.5  # Default trust level

    @property
    def base_size(self) -> int:
        """Minimum memory window size."""
        return self._base_size

    def set_trust_getter(self, trust_getter: Callable[[], float]) -> None:
        """
        Set the trust getter function.

        Args:
            trust_getter: Callable returning current trust [0, 1]
        """
        self._trust_getter = trust_getter

    def set_internal_trust(self, trust: float) -> None:
        """
        Set internal trust level (used when no trust_getter is set).

        Args:
            trust: Trust level in [0, 1]
        """
        self._internal_trust = max(0.0, min(1.0, trust))

    def get_current_trust(self) -> float:
        """
        Get current trust level.

        Returns:
            Trust level from getter or internal value
        """
        if self._trust_getter is not None:
            return self._trust_getter()
        return self._internal_trust

    def get_effective_window(self) -> int:
        """
        Calculate current effective window size based on trust.

        Returns:
            Window size in [base_size, max_size]
        """
        trust = self.get_current_trust()

        # Linear interpolation between base and max
        extra = int(trust * (self._max_size - self._base_size))
        return self._base_size + extra

    def get_weighted_history(self) -> List[Tuple[Interaction, float]]:
        """
        Get interactions within effective window, equally weighted.

        Only the most recent `effective_window` interactions are used,
        and they receive equal weights.

        Returns:
            List of (interaction, weight) tuples
        """
        if self.is_empty():
            return []

        window = self.get_effective_window()
        # Take only the most recent `window` interactions
        recent = list(self._history)[-window:]

        n = len(recent)
        weight = 1.0 / n

        return [(interaction, weight) for interaction in recent]

    @property
    def current_effective_size(self) -> int:
        """
        Current number of interactions being used.

        This is min(stored interactions, effective window).
        """
        return min(len(self._history), self.get_effective_window())

    def __repr__(self) -> str:
        trust = self.get_current_trust()
        window = self.get_effective_window()
        return (
            f"DynamicMemory(max={self._max_size}, base={self._base_size}, "
            f"trust={trust:.2f}, window={window}, stored={len(self)})"
        )
