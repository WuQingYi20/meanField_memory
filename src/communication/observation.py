"""
Observation-based communication mechanism.

Agents can observe interactions beyond their own, enabling
social learning and conformist transmission.

Theoretical basis:
- Henrich & Boyd (1998): Conformist transmission
- Bandura (1977): Social learning theory
- Boyd & Richerson (1985): Frequency-dependent bias
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod

from ..memory.base import Interaction


@dataclass
class Observation:
    """A single observation of another's behavior."""
    tick: int
    observed_strategy: int
    source: str  # "direct", "observed", "gossip", "broadcast"
    weight: float = 1.0  # reliability weight


class ObservationCollector:
    """
    Collects observations for an agent from multiple sources.

    Combines:
    - Direct interactions (weight = 1.0)
    - Observed interactions (weight = observation_weight)
    - Gossip (weight = gossip_weight)
    - Broadcast (weight = broadcast_weight)
    """

    def __init__(
        self,
        observation_weight: float = 0.5,
        gossip_weight: float = 0.3,
        broadcast_weight: float = 0.8,
    ):
        self._observation_weight = observation_weight
        self._gossip_weight = gossip_weight
        self._broadcast_weight = broadcast_weight
        self._observations: List[Observation] = []

    def add_direct(self, tick: int, strategy: int) -> None:
        """Add observation from direct interaction."""
        self._observations.append(Observation(
            tick=tick,
            observed_strategy=strategy,
            source="direct",
            weight=1.0
        ))

    def add_observed(self, tick: int, strategy: int) -> None:
        """Add observation from watching others."""
        self._observations.append(Observation(
            tick=tick,
            observed_strategy=strategy,
            source="observed",
            weight=self._observation_weight
        ))

    def add_gossip(self, tick: int, strategy: int) -> None:
        """Add observation from gossip/hearsay."""
        self._observations.append(Observation(
            tick=tick,
            observed_strategy=strategy,
            source="gossip",
            weight=self._gossip_weight
        ))

    def add_broadcast(self, tick: int, strategy: int) -> None:
        """Add observation from public broadcast."""
        self._observations.append(Observation(
            tick=tick,
            observed_strategy=strategy,
            source="broadcast",
            weight=self._broadcast_weight
        ))

    def get_weighted_distribution(self, recent_n: Optional[int] = None) -> np.ndarray:
        """
        Calculate weighted strategy distribution from observations.

        Args:
            recent_n: Only use most recent n observations (None = all)

        Returns:
            [P(strategy=0), P(strategy=1)]
        """
        if not self._observations:
            return np.array([0.5, 0.5])

        obs = self._observations
        if recent_n is not None:
            obs = obs[-recent_n:]

        counts = np.zeros(2)
        for o in obs:
            counts[o.observed_strategy] += o.weight

        total = counts.sum()
        if total > 0:
            return counts / total
        return np.array([0.5, 0.5])

    def clear(self) -> None:
        """Clear all observations."""
        self._observations.clear()

    def __len__(self) -> int:
        return len(self._observations)


class BaseObservationNetwork(ABC):
    """
    Abstract base for observation network structures.

    Determines who can observe whom.
    """

    @abstractmethod
    def get_observable_interactions(
        self,
        agent_id: int,
        all_interactions: List[Tuple[int, int, int, int]],  # (id1, id2, strat1, strat2)
    ) -> List[Tuple[int, int]]:
        """
        Get interactions observable by this agent.

        Returns:
            List of (strategy1, strategy2) that agent can observe
        """
        pass


class FullObservationNetwork(BaseObservationNetwork):
    """
    All agents can observe all interactions.

    Models complete information / transparent society.
    """

    def __init__(self, observation_probability: float = 1.0):
        self._prob = observation_probability

    def get_observable_interactions(
        self,
        agent_id: int,
        all_interactions: List[Tuple[int, int, int, int]],
    ) -> List[Tuple[int, int]]:
        result = []
        for id1, id2, s1, s2 in all_interactions:
            if id1 != agent_id and id2 != agent_id:  # exclude own interaction
                if np.random.random() < self._prob:
                    result.append((s1, s2))
        return result


class LocalObservationNetwork(BaseObservationNetwork):
    """
    Agents observe only k random interactions per tick.

    Models limited attention / local social networks.
    """

    def __init__(self, k: int = 3):
        self._k = k

    def get_observable_interactions(
        self,
        agent_id: int,
        all_interactions: List[Tuple[int, int, int, int]],
    ) -> List[Tuple[int, int]]:
        # Exclude own interaction
        others = [
            (s1, s2) for id1, id2, s1, s2 in all_interactions
            if id1 != agent_id and id2 != agent_id
        ]

        if len(others) <= self._k:
            return others

        indices = np.random.choice(len(others), size=self._k, replace=False)
        return [others[i] for i in indices]


class NoObservationNetwork(BaseObservationNetwork):
    """No observation of others (baseline)."""

    def get_observable_interactions(
        self,
        agent_id: int,
        all_interactions: List[Tuple[int, int, int, int]],
    ) -> List[Tuple[int, int]]:
        return []
