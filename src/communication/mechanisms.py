"""
Communication mechanisms for norm emergence.

Three theoretically-grounded mechanisms:

1. NormativeSignaling (Bicchieri 2006):
   - Agents express what they think SHOULD be done
   - Enables normative expectations, not just empirical

2. PrePlaySignaling (Skyrms 2010):
   - Agents signal intended action BEFORE interaction
   - Cheap talk or costly signaling

3. ThresholdContagion (Centola 2018):
   - Agents need multiple sources to update beliefs
   - Complex contagion vs simple contagion
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod


# =============================================================================
# 1. Normative Signaling (Bicchieri 2006)
# =============================================================================

@dataclass
class NormativeMessage:
    """A message expressing normative expectation."""
    sender_id: int
    expressed_norm: int  # What sender thinks SHOULD be done (0 or 1)
    confidence: float    # How strongly they believe this (0 to 1)


class NormativeSignaling:
    """
    Normative signaling mechanism (Bicchieri 2006).

    Key distinction from observation:
    - Observation: "What do others DO?" (empirical expectation)
    - Normative: "What do others think I SHOULD do?" (normative expectation)

    According to Bicchieri, a social norm requires BOTH:
    1. Empirical expectations (handled by observation)
    2. Normative expectations (handled by this mechanism)

    Implementation:
    - Each agent has a "normative belief" about what should be done
    - Agents broadcast this belief (weighted by confidence)
    - Receiving agents update their normative expectations
    """

    def __init__(
        self,
        broadcast_probability: float = 0.3,  # Probability of broadcasting each tick
        normative_weight: float = 0.5,       # Weight of normative vs empirical
    ):
        self._broadcast_prob = broadcast_probability
        self._normative_weight = normative_weight
        self._messages: List[NormativeMessage] = []

    def generate_message(
        self,
        agent_id: int,
        belief: np.ndarray,
        current_strategy: int,
    ) -> Optional[NormativeMessage]:
        """
        Agent decides whether to broadcast normative message.

        Normative belief is influenced by:
        - Current strategy (what I do -> what I think is right)
        - Belief strength (confidence in the norm)
        """
        if np.random.random() > self._broadcast_prob:
            return None

        # Express current strategy as norm (cognitive dissonance reduction)
        expressed_norm = current_strategy

        # Confidence based on belief strength
        confidence = max(belief[0], belief[1])

        return NormativeMessage(
            sender_id=agent_id,
            expressed_norm=expressed_norm,
            confidence=confidence,
        )

    def broadcast_all(
        self,
        agent_data: List[Tuple[int, np.ndarray, int]],  # [(id, belief, strategy), ...]
    ) -> List[NormativeMessage]:
        """Generate messages from all agents."""
        self._messages.clear()

        for agent_id, belief, strategy in agent_data:
            msg = self.generate_message(agent_id, belief, strategy)
            if msg is not None:
                self._messages.append(msg)

        return self._messages

    def receive_messages(
        self,
        receiver_id: int,
        exclude_self: bool = True,
    ) -> List[NormativeMessage]:
        """Get messages received by an agent."""
        if exclude_self:
            return [m for m in self._messages if m.sender_id != receiver_id]
        return self._messages.copy()

    def compute_normative_expectation(
        self,
        received_messages: List[NormativeMessage],
    ) -> np.ndarray:
        """
        Compute normative expectation from received messages.

        Returns:
            [P(norm=0), P(norm=1)] weighted by confidence
        """
        if not received_messages:
            return np.array([0.5, 0.5])

        counts = np.zeros(2)
        for msg in received_messages:
            counts[msg.expressed_norm] += msg.confidence

        total = counts.sum()
        if total > 0:
            return counts / total
        return np.array([0.5, 0.5])

    @property
    def normative_weight(self) -> float:
        return self._normative_weight


# =============================================================================
# 2. Pre-play Signaling (Skyrms 2010)
# =============================================================================

class SignalType(Enum):
    """Types of pre-play signals."""
    CHEAP_TALK = "cheap_talk"      # Costless, non-binding
    COSTLY = "costly"              # Has a cost, more credible


@dataclass
class PrePlaySignal:
    """A signal sent before interaction."""
    sender_id: int
    signal: int           # Signaled intention (0 or 1)
    signal_type: SignalType
    honesty: float        # Probability signal matches actual action


class PrePlaySignaling:
    """
    Pre-play signaling mechanism (Skyrms 2010).

    Before interaction, agents exchange signals about intended play.
    This can help coordinate on equilibria.

    Two modes:
    1. Cheap talk: Signal is costless, may not be honest
    2. Costly signaling: Signal has cost, more credible

    The key insight (Skyrms):
    - Even cheap talk can evolve to be informative
    - Costly signals are more reliable but less frequent
    """

    def __init__(
        self,
        signal_type: SignalType = SignalType.CHEAP_TALK,
        honesty_rate: float = 0.8,    # Base probability of honest signaling
        signal_cost: float = 0.1,     # Cost of costly signal
        trust_bonus: float = 0.2,     # How much trust affects honesty
    ):
        self._signal_type = signal_type
        self._base_honesty = honesty_rate
        self._signal_cost = signal_cost
        self._trust_bonus = trust_bonus

    def generate_signal(
        self,
        agent_id: int,
        intended_action: int,
        trust: float,
    ) -> PrePlaySignal:
        """
        Generate pre-play signal.

        Honesty increases with trust (high trust -> more honest signaling).
        """
        # Honesty depends on trust
        honesty = min(1.0, self._base_honesty + trust * self._trust_bonus)

        # Decide whether to signal honestly
        if np.random.random() < honesty:
            signal = intended_action
        else:
            signal = 1 - intended_action  # Deceptive signal

        return PrePlaySignal(
            sender_id=agent_id,
            signal=signal,
            signal_type=self._signal_type,
            honesty=honesty,
        )

    def interpret_signal(
        self,
        signal: PrePlaySignal,
        receiver_trust: float,
    ) -> Tuple[int, float]:
        """
        Interpret received signal.

        In pure coordination games, rational agents should follow signals
        since coordination is the goal. Confidence reflects how much
        to trust the signal vs. prior belief.

        Returns:
            (predicted_action, confidence)
        """
        # Confidence in signal depends on signal type and receiver's trust
        if signal.signal_type == SignalType.COSTLY:
            confidence = 0.95  # Costly signals are highly credible
        else:
            # Cheap talk: confidence starts high and increases with trust
            # In coordination games, following signals is rational
            confidence = 0.7 + 0.25 * receiver_trust

        return signal.signal, confidence

    def get_signal_cost(self, signal: PrePlaySignal) -> float:
        """Get cost of sending the signal."""
        if signal.signal_type == SignalType.COSTLY:
            return self._signal_cost
        return 0.0


# =============================================================================
# 3. Threshold Contagion (Centola 2018)
# =============================================================================

class ContagionType(Enum):
    """Types of social contagion."""
    SIMPLE = "simple"        # Any exposure can cause adoption
    COMPLEX = "complex"      # Need multiple exposures


class ThresholdContagion:
    """
    Threshold-based contagion mechanism (Centola 2018).

    Key distinction:
    - Simple contagion: One exposure is enough (like disease)
    - Complex contagion: Need multiple independent sources

    For norms/behaviors:
    - Adopting a new norm is risky
    - Need "social proof" from multiple sources
    - This explains why some behaviors spread slowly

    Implementation:
    - Agent tracks exposures to each strategy
    - Only updates belief if exposure count >= threshold
    - Threshold can be absolute (n sources) or relative (fraction)
    """

    def __init__(
        self,
        contagion_type: ContagionType = ContagionType.COMPLEX,
        absolute_threshold: int = 2,       # Need this many sources
        relative_threshold: float = 0.3,   # Or this fraction of observations
        use_relative: bool = False,        # Which threshold to use
        decay_rate: float = 0.5,           # Exposure decay per tick
    ):
        self._contagion_type = contagion_type
        self._absolute_threshold = absolute_threshold
        self._relative_threshold = relative_threshold
        self._use_relative = use_relative
        self._decay_rate = decay_rate

        # Track exposures per agent: {agent_id: [count_0, count_1]}
        self._exposures: Dict[int, np.ndarray] = {}

    def reset_agent(self, agent_id: int) -> None:
        """Reset exposure counts for an agent."""
        self._exposures[agent_id] = np.zeros(2)

    def add_exposure(
        self,
        agent_id: int,
        observed_strategy: int,
        source_id: int,  # For tracking unique sources
    ) -> None:
        """Add an exposure to a strategy."""
        if agent_id not in self._exposures:
            self._exposures[agent_id] = np.zeros(2)

        self._exposures[agent_id][observed_strategy] += 1

    def decay_exposures(self) -> None:
        """Apply decay to all exposure counts (called each tick)."""
        for agent_id in self._exposures:
            self._exposures[agent_id] *= (1 - self._decay_rate)

    def get_exposure_counts(self, agent_id: int) -> np.ndarray:
        """Get current exposure counts for an agent."""
        if agent_id not in self._exposures:
            return np.zeros(2)
        return self._exposures[agent_id].copy()

    def check_threshold(
        self,
        agent_id: int,
        strategy: int,
    ) -> bool:
        """
        Check if exposure to strategy meets threshold.

        Returns:
            True if threshold is met (belief update should occur)
        """
        if self._contagion_type == ContagionType.SIMPLE:
            return True  # Any exposure triggers update

        # Complex contagion: need threshold
        exposures = self.get_exposure_counts(agent_id)

        if self._use_relative:
            total = exposures.sum()
            if total == 0:
                return False
            fraction = exposures[strategy] / total
            return fraction >= self._relative_threshold
        else:
            return exposures[strategy] >= self._absolute_threshold

    def filter_observations(
        self,
        agent_id: int,
        observations: List[Tuple[int, float]],  # [(strategy, weight), ...]
    ) -> List[Tuple[int, float]]:
        """
        Filter observations based on threshold.

        Only return observations for strategies that meet threshold.
        """
        if self._contagion_type == ContagionType.SIMPLE:
            return observations

        # Add to exposure counts first
        for strategy, _ in observations:
            self.add_exposure(agent_id, strategy, source_id=-1)

        # Filter based on threshold
        filtered = []
        for strategy, weight in observations:
            if self.check_threshold(agent_id, strategy):
                filtered.append((strategy, weight))

        return filtered


# =============================================================================
# Combined Communication Manager
# =============================================================================

class CommunicationManager:
    """
    Unified manager for all communication mechanisms.

    Combines:
    - Observation (basic, already in environment)
    - Normative signaling (Bicchieri)
    - Pre-play signaling (Skyrms)
    - Threshold contagion (Centola)
    """

    def __init__(
        self,
        # Enable/disable mechanisms
        enable_normative: bool = False,
        enable_preplay: bool = False,
        enable_threshold: bool = False,
        # Normative parameters
        normative_broadcast_prob: float = 0.3,
        normative_weight: float = 0.3,
        # Pre-play parameters
        preplay_signal_type: SignalType = SignalType.CHEAP_TALK,
        preplay_honesty: float = 0.8,
        # Threshold parameters
        contagion_type: ContagionType = ContagionType.COMPLEX,
        contagion_threshold: int = 2,
    ):
        self._enable_normative = enable_normative
        self._enable_preplay = enable_preplay
        self._enable_threshold = enable_threshold

        # Initialize mechanisms
        if enable_normative:
            self._normative = NormativeSignaling(
                broadcast_probability=normative_broadcast_prob,
                normative_weight=normative_weight,
            )
        else:
            self._normative = None

        if enable_preplay:
            self._preplay = PrePlaySignaling(
                signal_type=preplay_signal_type,
                honesty_rate=preplay_honesty,
            )
        else:
            self._preplay = None

        if enable_threshold:
            self._threshold = ThresholdContagion(
                contagion_type=contagion_type,
                absolute_threshold=contagion_threshold,
            )
        else:
            self._threshold = None

    @property
    def normative(self) -> Optional[NormativeSignaling]:
        return self._normative

    @property
    def preplay(self) -> Optional[PrePlaySignaling]:
        return self._preplay

    @property
    def threshold(self) -> Optional[ThresholdContagion]:
        return self._threshold

    def tick_start(self) -> None:
        """Called at start of each tick."""
        if self._threshold:
            self._threshold.decay_exposures()

    def get_config(self) -> Dict[str, Any]:
        """Get configuration for serialization."""
        return {
            "enable_normative": self._enable_normative,
            "enable_preplay": self._enable_preplay,
            "enable_threshold": self._enable_threshold,
        }
