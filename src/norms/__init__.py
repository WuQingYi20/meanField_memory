"""
Norms module (Future Extension).

This module will handle norm detection, representation, and dynamics:
- Detecting when a norm has emerged
- Representing norms as first-class objects
- Tracking norm strength and stability
- Handling norm transitions and conflicts

Planned components:
- NormDetector: Detect norm emergence from population state
- NormRepresentation: How agents represent norms internally
- NormStrength: Measure how established a norm is
- NormTransition: Track changes between norms

Key questions to address:
1. What counts as a "norm"? (threshold-based, stability-based, belief-based)
2. How do agents represent norms? (implicit in memory, explicit beliefs)
3. When is a norm "internalized"? (see internalization module)
4. How do norm conflicts get resolved?

Example future usage:
    from src.norms import NormDetector, NormCriteria

    detector = NormDetector(
        criteria=NormCriteria.STABILITY_BASED,
        threshold=0.9,
        stability_window=50
    )

    # Check population state
    norm_status = detector.detect(population_strategies, tick)

    if norm_status.has_norm:
        print(f"Norm detected: {norm_status.dominant_strategy}")
        print(f"Strength: {norm_status.strength}")
        print(f"Established at tick: {norm_status.emergence_tick}")
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, List
from abc import ABC, abstractmethod


class NormCriteria(Enum):
    """Criteria for determining when a norm exists."""
    THRESHOLD_BASED = "threshold"      # Simple majority threshold
    STABILITY_BASED = "stability"      # Stable for N ticks
    BELIEF_BASED = "belief"            # Agents believe it's a norm
    CONVERGENCE_BASED = "convergence"  # Rate of strategy changes low


@dataclass
class NormStatus:
    """Status of norm detection."""
    has_norm: bool
    dominant_strategy: Optional[int]
    strength: float  # 0 to 1
    emergence_tick: Optional[int]
    stability: float  # How stable the norm has been


class BaseNormDetector(ABC):
    """Abstract base class for norm detection."""

    @property
    @abstractmethod
    def criteria(self) -> NormCriteria:
        """Return detection criteria."""
        pass

    @abstractmethod
    def detect(
        self,
        strategies: List[int],
        tick: int,
        history: Optional[List[List[int]]] = None
    ) -> NormStatus:
        """
        Detect norm from current state.

        Args:
            strategies: Current strategy of each agent
            tick: Current time step
            history: Optional history of strategies

        Returns:
            NormStatus with detection results
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset detector state."""
        pass


class ThresholdNormDetector(BaseNormDetector):
    """Simple threshold-based norm detection."""

    def __init__(self, threshold: float = 0.95):
        self._threshold = threshold
        self._emergence_tick: Optional[int] = None

    @property
    def criteria(self) -> NormCriteria:
        return NormCriteria.THRESHOLD_BASED

    def detect(
        self,
        strategies: List[int],
        tick: int,
        history: Optional[List[List[int]]] = None
    ) -> NormStatus:
        n = len(strategies)
        count_0 = strategies.count(0)
        count_1 = n - count_0

        if count_0 > count_1:
            dominant = 0
            strength = count_0 / n
        else:
            dominant = 1
            strength = count_1 / n

        has_norm = strength >= self._threshold

        if has_norm and self._emergence_tick is None:
            self._emergence_tick = tick

        return NormStatus(
            has_norm=has_norm,
            dominant_strategy=dominant if has_norm else None,
            strength=strength,
            emergence_tick=self._emergence_tick,
            stability=1.0 if has_norm else 0.0  # Simplified
        )

    def reset(self) -> None:
        self._emergence_tick = None


def create_norm_detector(
    criteria: NormCriteria = NormCriteria.THRESHOLD_BASED,
    **kwargs
) -> BaseNormDetector:
    """Factory function for norm detectors."""
    if criteria == NormCriteria.THRESHOLD_BASED:
        return ThresholdNormDetector(**kwargs)
    else:
        raise NotImplementedError(f"Norm criteria {criteria} not yet implemented")


from .detector import NormDetector, NormLevel, NormState

__all__ = [
    # Legacy (simple threshold-based)
    "NormCriteria",
    "NormStatus",
    "BaseNormDetector",
    "ThresholdNormDetector",
    "create_norm_detector",
    # New multi-level detector
    "NormDetector",
    "NormLevel",
    "NormState",
]
