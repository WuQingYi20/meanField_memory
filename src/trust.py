"""
Trust management system.

Simplified model where Trust is the primary state variable,
directly controlling memory window size.
"""

from typing import Callable


class TrustManager:
    """
    Manages trust-memory window linkage.

    In the simplified model:
    - Trust is updated directly (no intermediate τ)
    - Trust only affects memory window size
    - Action selection uses probability matching (independent of Trust)

    The feedback loop:
    - Higher majority → Higher prediction accuracy → Higher Trust steady state
    - Higher Trust → Larger memory window → More stable beliefs
    - More stable beliefs → Harder to reverse established norms

    This creates "cognitive lock-in" rather than behavioral amplification.
    """

    def __init__(
        self,
        memory_base: int = 2,
        memory_max: int = 6,
    ):
        """
        Initialize trust manager.

        Args:
            memory_base: Minimum memory window size (when trust=0)
            memory_max: Maximum memory window size (when trust=1)
        """
        if memory_base < 1:
            raise ValueError("memory_base must be at least 1")
        if memory_max < memory_base:
            raise ValueError("memory_max must be >= memory_base")

        self._memory_base = memory_base
        self._memory_max = memory_max

    @property
    def memory_base(self) -> int:
        """Minimum memory window size."""
        return self._memory_base

    @property
    def memory_max(self) -> int:
        """Maximum memory window size."""
        return self._memory_max

    def trust_to_memory_window(self, trust: float) -> int:
        """
        Calculate memory window size from trust level.

        Higher trust → Longer window (more reliance on history)
        Lower trust → Shorter window (more adaptive)

        Args:
            trust: Trust level in [0, 1]

        Returns:
            Memory window size in [memory_base, memory_max]
        """
        # Clamp trust
        trust = max(0.0, min(1.0, trust))

        # Linear interpolation
        extra = int(trust * (self._memory_max - self._memory_base))
        return self._memory_base + extra

    def create_window_getter(self, trust_getter: Callable[[], float]) -> Callable[[], int]:
        """
        Create a window size getter function from a trust getter.

        Useful for linking dynamic memory to decision mechanism.

        Args:
            trust_getter: Function that returns current trust level

        Returns:
            Function that returns current memory window size
        """
        return lambda: self.trust_to_memory_window(trust_getter())

    def get_trust_steady_state(
        self,
        prediction_accuracy: float,
        alpha: float = 0.1,
        beta: float = 0.3
    ) -> float:
        """
        Calculate theoretical trust steady state for given prediction accuracy.

        At steady state: E[ΔTrust] = 0
        T* = pα / (pα + (1-p)β)

        Args:
            prediction_accuracy: Probability of correct prediction (p)
            alpha: Trust increase rate
            beta: Trust decay rate

        Returns:
            Steady state trust level
        """
        p = prediction_accuracy
        if p * alpha + (1 - p) * beta == 0:
            return 0.5
        return (p * alpha) / (p * alpha + (1 - p) * beta)

    def describe_state(self, trust: float) -> dict:
        """
        Get a complete description of the trust state.

        Args:
            trust: Current trust level

        Returns:
            Dict with trust and memory window info
        """
        window = self.trust_to_memory_window(trust)

        return {
            "trust": trust,
            "memory_window": window,
            "window_range": (self._memory_base, self._memory_max),
        }

    def __repr__(self) -> str:
        return (
            f"TrustManager(window=[{self._memory_base}, {self._memory_max}])"
        )


def calculate_memory_window(
    trust: float,
    base: int = 2,
    max_limit: int = 6,
) -> int:
    """
    Standalone function to calculate memory window from trust.

    Args:
        trust: Trust level in [0, 1]
        base: Minimum window size
        max_limit: Maximum window size

    Returns:
        Window size in [base, max_limit]
    """
    trust = max(0.0, min(1.0, trust))
    extra = int(trust * (max_limit - base))
    return base + extra
