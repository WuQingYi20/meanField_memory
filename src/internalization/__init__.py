"""
Internalization module (DEPRECATED).

.. deprecated:: V5
    This module is superseded by ``src.memory.normative.NormativeMemory``,
    which implements norm internalization via DDM crystallisation,
    compliance (sigma^k), anomaly-driven crisis, and enforcement.

This module will handle norm internalization by agents:
- When do agents "internalize" a norm vs merely comply?
- How does internalization affect behavior differently from compliance?
- What triggers internalization?

Key distinctions:
1. COMPLIANCE: Follow the norm because of external incentives
   - Behavior depends on observation/enforcement
   - Would deviate if not observed

2. INTERNALIZATION: Adopt the norm as personal value
   - Behavior consistent even without observation
   - Intrinsic motivation to follow
   - May enforce norm on others

Planned components:
- InternalizationTracker: Track each agent's internalization level
- InternalizationTrigger: What causes internalization
- InternalizedBehavior: How behavior differs after internalization

Possible internalization mechanisms:
1. REPETITION_BASED: Repeated compliance leads to internalization
2. SUCCESS_BASED: Successful coordination leads to internalization
3. TRUST_BASED: High trust leads to internalization
4. SOCIAL_PROOF: Observing others internalize leads to internalization

Example future usage:
    from src.internalization import InternalizationTracker, InternalizationMode

    tracker = InternalizationTracker(
        mode=InternalizationMode.TRUST_BASED,
        threshold=0.8,
        decay_rate=0.01
    )

    # Update internalization based on experience
    tracker.update(
        agent_id=0,
        trust=agent.trust,
        strategy_history=agent.memory
    )

    # Check if internalized
    if tracker.is_internalized(agent_id=0, strategy=1):
        # Agent has internalized strategy 1 as a norm
        # Behavior may differ (e.g., enforce on others, resist change)
        pass
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, List
from abc import ABC, abstractmethod


class InternalizationMode(Enum):
    """Mechanisms for norm internalization."""
    NONE = "none"                    # No internalization tracking
    REPETITION_BASED = "repetition"  # Internalize through repeated behavior
    SUCCESS_BASED = "success"        # Internalize through positive outcomes
    TRUST_BASED = "trust"            # Internalize when trust is high
    THRESHOLD_BASED = "threshold"    # Internalize when commitment crosses threshold


@dataclass
class InternalizationState:
    """State of an agent's norm internalization."""
    level: float  # 0 to 1
    internalized_strategy: Optional[int]
    is_internalized: bool
    since_tick: Optional[int]


class BaseInternalizationTracker(ABC):
    """Abstract base class for internalization tracking."""

    @property
    @abstractmethod
    def mode(self) -> InternalizationMode:
        """Return internalization mode."""
        pass

    @abstractmethod
    def update(
        self,
        agent_id: int,
        current_strategy: int,
        trust: float,
        success: bool,
        tick: int
    ) -> None:
        """Update internalization state for an agent."""
        pass

    @abstractmethod
    def get_state(self, agent_id: int) -> InternalizationState:
        """Get internalization state for an agent."""
        pass

    @abstractmethod
    def is_internalized(self, agent_id: int, strategy: int) -> bool:
        """Check if agent has internalized a specific strategy as norm."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset all internalization states."""
        pass


class NoInternalization(BaseInternalizationTracker):
    """Default: no internalization tracking."""

    @property
    def mode(self) -> InternalizationMode:
        return InternalizationMode.NONE

    def update(self, agent_id: int, current_strategy: int,
               trust: float, success: bool, tick: int) -> None:
        pass

    def get_state(self, agent_id: int) -> InternalizationState:
        return InternalizationState(
            level=0.0,
            internalized_strategy=None,
            is_internalized=False,
            since_tick=None
        )

    def is_internalized(self, agent_id: int, strategy: int) -> bool:
        return False

    def reset(self) -> None:
        pass


class TrustBasedInternalization(BaseInternalizationTracker):
    """
    Trust-based internalization.

    When trust exceeds threshold for sustained period,
    the current strategy becomes internalized.
    """

    def __init__(
        self,
        trust_threshold: float = 0.8,
        duration_threshold: int = 50
    ):
        self._trust_threshold = trust_threshold
        self._duration_threshold = duration_threshold
        self._agent_states: Dict[int, Dict] = {}

    @property
    def mode(self) -> InternalizationMode:
        return InternalizationMode.TRUST_BASED

    def update(self, agent_id: int, current_strategy: int,
               trust: float, success: bool, tick: int) -> None:
        if agent_id not in self._agent_states:
            self._agent_states[agent_id] = {
                "high_trust_since": None,
                "strategy_during_high_trust": None,
                "internalized": False,
                "internalized_strategy": None,
                "internalized_tick": None
            }

        state = self._agent_states[agent_id]

        # Already internalized
        if state["internalized"]:
            return

        if trust >= self._trust_threshold:
            if state["high_trust_since"] is None:
                state["high_trust_since"] = tick
                state["strategy_during_high_trust"] = current_strategy
            elif state["strategy_during_high_trust"] == current_strategy:
                # Check if duration threshold met
                if tick - state["high_trust_since"] >= self._duration_threshold:
                    state["internalized"] = True
                    state["internalized_strategy"] = current_strategy
                    state["internalized_tick"] = tick
            else:
                # Strategy changed during high trust, reset
                state["high_trust_since"] = tick
                state["strategy_during_high_trust"] = current_strategy
        else:
            # Trust dropped, reset counter
            state["high_trust_since"] = None
            state["strategy_during_high_trust"] = None

    def get_state(self, agent_id: int) -> InternalizationState:
        if agent_id not in self._agent_states:
            return InternalizationState(0.0, None, False, None)

        state = self._agent_states[agent_id]

        # Calculate level as progress toward internalization
        if state["internalized"]:
            level = 1.0
        elif state["high_trust_since"] is not None:
            # Would need current tick to calculate, return estimate
            level = 0.5
        else:
            level = 0.0

        return InternalizationState(
            level=level,
            internalized_strategy=state["internalized_strategy"],
            is_internalized=state["internalized"],
            since_tick=state["internalized_tick"]
        )

    def is_internalized(self, agent_id: int, strategy: int) -> bool:
        if agent_id not in self._agent_states:
            return False
        state = self._agent_states[agent_id]
        return state["internalized"] and state["internalized_strategy"] == strategy

    def reset(self) -> None:
        self._agent_states.clear()


def create_internalization_tracker(
    mode: InternalizationMode = InternalizationMode.NONE,
    **kwargs
) -> BaseInternalizationTracker:
    """Factory function for internalization trackers."""
    if mode == InternalizationMode.NONE:
        return NoInternalization()
    elif mode == InternalizationMode.TRUST_BASED:
        return TrustBasedInternalization(**kwargs)
    else:
        raise NotImplementedError(f"Internalization mode {mode} not yet implemented")


__all__ = [
    "InternalizationMode",
    "InternalizationState",
    "BaseInternalizationTracker",
    "NoInternalization",
    "TrustBasedInternalization",
    "create_internalization_tracker",
]
