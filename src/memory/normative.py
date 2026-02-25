"""
Normative memory: rule-based memory for internalised norms.

Unlike experience memory (BaseMemory), normative memory stores a discrete rule,
not a FIFO queue. It is maintained through anomaly accumulation and crisis dynamics,
not through sliding windows or decay.

Based on:
- Germar et al. (2014): DDM-based norm formation
- Germar & Mojzisch (2019): persistent norm internalisation
- Kuhn (1962): anomaly accumulation and crisis-triggered paradigm shifts
- Fehr & Gaechter (2002): violation-triggered enforcement
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class NormativeState:
    """Snapshot of normative memory state for metrics and logging."""

    norm: Optional[int]        # r_i: the norm strategy, or None if no norm
    strength: float            # sigma_i: how constraining the rule is
    anomaly_count: int         # a_i: accumulated violations observed
    evidence: float            # e_i: DDM evidence accumulator
    has_norm: bool             # whether a norm has crystallised
    compliance: float          # sigma^k: current compliance level
    enforcement_count: int     # total enforcement signals sent


class NormativeMemory:
    """
    Rule-based normative memory. NOT a BaseMemory subclass.

    Implements the normative memory system from the dual-memory model (V5):
    - DDM norm formation (drift-diffusion crystallisation)
    - Anomaly-driven maintenance
    - Crisis and dissolution
    - Compliance via sigma^k
    - Violation-triggered enforcement

    State variables:
        r_i (norm): which strategy is "the rule", or None
        sigma_i (strength): how constraining the norm is [0, 1]
        a_i (anomaly_count): accumulated violations observed
        e_i (evidence): DDM evidence accumulator (pre-crystallisation)
    """

    def __init__(
        self,
        ddm_noise: float = 0.1,
        crystal_threshold: float = 3.0,
        initial_strength: float = 0.8,
        crisis_threshold: int = 10,
        crisis_decay: float = 0.3,
        min_strength: float = 0.1,
        compliance_exponent: float = 2.0,
        rng: Optional[np.random.RandomState] = None,
    ):
        """
        Initialise normative memory.

        Args:
            ddm_noise: Standard deviation of DDM noise (sigma_noise)
            crystal_threshold: Evidence threshold for crystallisation (theta_crystal)
            initial_strength: Norm strength upon crystallisation (sigma_0)
            crisis_threshold: Anomaly count triggering crisis (theta_crisis)
            crisis_decay: Multiplicative decay on crisis (lambda_crisis)
            min_strength: Below this, norm dissolves (sigma_min)
            compliance_exponent: Exponent k in compliance = sigma^k
            rng: Per-agent random state for reproducibility
        """
        self._ddm_noise = ddm_noise
        self._crystal_threshold = crystal_threshold
        self._initial_strength = initial_strength
        self._crisis_threshold = crisis_threshold
        self._crisis_decay = crisis_decay
        self._min_strength = min_strength
        self._compliance_exponent = compliance_exponent
        self._rng = rng or np.random.RandomState()

        # State variables
        self._norm: Optional[int] = None       # r_i
        self._strength: float = 0.0            # sigma_i (0 until crystallised)
        self._anomaly_count: int = 0           # a_i
        self._evidence: float = 0.0            # e_i

        # Signal boost: applied once then reset
        self._signal_boost: float = 1.0

        # Statistics
        self._enforcement_count: int = 0

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def norm(self) -> Optional[int]:
        """The internalised norm strategy, or None if no norm."""
        return self._norm

    @property
    def strength(self) -> float:
        """Norm strength sigma_i."""
        return self._strength

    @property
    def anomaly_count(self) -> int:
        """Accumulated anomaly count a_i."""
        return self._anomaly_count

    @property
    def evidence(self) -> float:
        """DDM evidence accumulator e_i."""
        return self._evidence

    def has_norm(self) -> bool:
        """Whether a norm has crystallised."""
        return self._norm is not None

    # =========================================================================
    # DDM Norm Formation (Eq. 5-7)
    # =========================================================================

    def update_evidence(
        self,
        confidence: float,
        consistency: float,
        dominant_strategy: int,
    ) -> bool:
        """
        Update DDM evidence accumulator. Only active when no norm exists.

        drift = (1 - C) * consistency * signal_boost
        e(t+1) = max(0, e(t) + drift + N(0, sigma_noise^2))

        If evidence crosses threshold, norm crystallises.

        Args:
            confidence: Agent's predictive confidence C_i in [0, 1]
            consistency: max(f_A, f_B) over observed strategies
            dominant_strategy: The most frequent observed strategy

        Returns:
            True if norm crystallised this tick, False otherwise
        """
        if self._norm is not None:
            return False  # Already have a norm

        # Eq. 5: drift = (1 - C) * consistency * signal_boost
        drift = (1.0 - confidence) * consistency * self._signal_boost

        # Reset signal boost after use
        self._signal_boost = 1.0

        # Eq. 6: e(t+1) = max(0, e(t) + drift + noise)
        noise = self._rng.normal(0, self._ddm_noise)
        self._evidence = max(0.0, self._evidence + drift + noise)

        # Eq. 7: crystallisation check
        if self._evidence >= self._crystal_threshold:
            self._norm = dominant_strategy
            self._strength = self._initial_strength
            self._anomaly_count = 0
            return True

        return False

    # =========================================================================
    # Anomaly Tracking (Eq. 10)
    # =========================================================================

    def record_anomaly(self, observed_strategy: int) -> None:
        """
        Record an observation against the norm.

        If the observed strategy violates the norm, increment anomaly count.
        Conforming observations cause nothing (non-statistical property).

        Args:
            observed_strategy: Strategy observed in an interaction
        """
        if self._norm is None:
            return

        if observed_strategy != self._norm:
            self._anomaly_count += 1

    # =========================================================================
    # Crisis and Dissolution (Eq. 11-12)
    # =========================================================================

    def check_crisis(self) -> bool:
        """
        Check if anomalies have triggered a crisis.

        If anomalies >= threshold: strength *= crisis_decay, anomalies reset.
        If strength < min_strength: norm dissolves.

        Returns:
            True if norm was dissolved, False otherwise
        """
        if self._norm is None:
            return False

        if self._anomaly_count >= self._crisis_threshold:
            # Eq. 11: crisis
            self._strength *= self._crisis_decay
            self._anomaly_count = 0

            # Eq. 12: dissolution check
            if self._strength < self._min_strength:
                self._norm = None
                self._evidence = 0.0
                self._strength = 0.0
                return True

        return False

    # =========================================================================
    # Compliance (Eq. 13)
    # =========================================================================

    def get_compliance(self) -> float:
        """
        Get current compliance level: sigma^k.

        Returns:
            Compliance in [0, 1]. Returns 0.0 if no norm.
        """
        if self._norm is None:
            return 0.0
        return self._strength ** self._compliance_exponent

    # =========================================================================
    # Norm Belief (Eq. 14)
    # =========================================================================

    def get_norm_belief(self) -> Optional[np.ndarray]:
        """
        Get the norm as a one-hot belief vector.

        Returns:
            one_hot(r_i) as [P(A), P(B)], or None if no norm
        """
        if self._norm is None:
            return None
        belief = np.zeros(2)
        belief[self._norm] = 1.0
        return belief

    # =========================================================================
    # Enforcement (Eq. 16)
    # =========================================================================

    def should_enforce(
        self,
        observed_strategy: int,
        enforce_threshold: float,
    ) -> bool:
        """
        Check whether this agent should enforce the norm.

        Three conditions must hold simultaneously:
        1. Agent has a norm (r_i != None)
        2. High norm strength (sigma_i > theta_enforce)
        3. Violation observed (observed_strategy != r_i)

        Args:
            observed_strategy: The strategy that was observed
            enforce_threshold: Minimum sigma for enforcement (theta_enforce)

        Returns:
            True if agent should broadcast enforcement signal
        """
        if self._norm is None:
            return False
        if self._strength <= enforce_threshold:
            return False
        if observed_strategy == self._norm:
            return False
        return True

    def record_enforcement(self) -> None:
        """Record that an enforcement signal was sent."""
        self._enforcement_count += 1

    # =========================================================================
    # Signal Reception (Eq. 17)
    # =========================================================================

    def receive_signal(self, signal_amplification: float) -> None:
        """
        Receive a normative enforcement signal.

        Boosts DDM drift rate for the next evidence update.
        Only effective if no norm has crystallised yet.

        Args:
            signal_amplification: Multiplier gamma_signal (> 1)
        """
        if self._norm is None:
            self._signal_boost = signal_amplification

    # =========================================================================
    # State and Reset
    # =========================================================================

    def get_state(self) -> NormativeState:
        """Get a snapshot of current normative memory state."""
        return NormativeState(
            norm=self._norm,
            strength=self._strength,
            anomaly_count=self._anomaly_count,
            evidence=self._evidence,
            has_norm=self._norm is not None,
            compliance=self.get_compliance(),
            enforcement_count=self._enforcement_count,
        )

    def reset(self) -> None:
        """Reset normative memory to initial state."""
        self._norm = None
        self._strength = 0.0
        self._anomaly_count = 0
        self._evidence = 0.0
        self._signal_boost = 1.0
        self._enforcement_count = 0
