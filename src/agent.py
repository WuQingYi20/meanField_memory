"""
Agent class integrating experience memory, decision, trust, and normative memory.

V5 dual-memory architecture:
- Experience memory (FIFO): statistical belief formation
- Normative memory (rule-based): DDM crystallisation, anomaly tracking, enforcement
- Predictive confidence (trust): bridges both systems

Supports multiple decision modes:
- DUAL_FEEDBACK: Original tau-based softmax (two feedback loops)
- COGNITIVE_LOCKIN: Probability matching with Trust (cognitive only)
- EPSILON_GREEDY: Best response with exploration (robustness check)

Supports observation-based communication:
- Agents can observe other interactions beyond their own
- Observations are weighted and combined with direct experience
- Observations also feed normative memory (DDM or anomaly tracking)
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, Union, List

from .memory import BaseMemory, Interaction, create_memory, NormativeMemory, NormativeState
from .decision import (
    BaseDecision, DecisionMode, create_decision,
    DualFeedbackDecision, CognitiveLockInDecision, EpsilonGreedyDecision
)
from .trust import TrustManager


class Agent:
    """
    An agent in the ABM simulation.

    Integrates:
    - Experience memory: Stores interaction history (FIFO)
    - Decision mechanism: Action selection and trust dynamics
    - Trust -> Memory linkage: Trust affects memory window (for dynamic memory)
    - Normative memory (V5): DDM norm formation, compliance, enforcement

    The agent's behaviour emerges from three feedback loops:
    1. Cognitive lock-in: prediction -> confidence -> window -> belief stability
    2. Social amplification: observations -> belief shift -> more coordination
    3. Normative pressure: consistency -> DDM -> norm -> compliance -> enforcement
    """

    def __init__(
        self,
        agent_id: int,
        memory: BaseMemory,
        decision: BaseDecision,
        trust_manager: Optional[TrustManager] = None,
        initial_strategy: Optional[int] = None,
        normative_memory: Optional[NormativeMemory] = None,
        enforce_threshold: float = 0.7,
        signal_amplification: float = 2.0,
    ):
        """
        Initialize an agent.

        Args:
            agent_id: Unique identifier
            memory: Experience memory system instance
            decision: Decision mechanism instance
            trust_manager: Trust manager (optional, creates default if None)
            initial_strategy: Starting strategy (random if None)
            normative_memory: Normative memory instance (None = V1 experience-only)
            enforce_threshold: Min sigma for enforcement (theta_enforce)
            signal_amplification: DDM drift multiplier for enforcement signals
        """
        self._id = agent_id
        self._memory = memory
        self._decision = decision
        self._trust_manager = trust_manager or TrustManager()
        self._normative_memory = normative_memory
        self._enforce_threshold = enforce_threshold
        self._signal_amplification = signal_amplification

        # Initialize strategy (random if not specified)
        if initial_strategy is not None:
            self._current_strategy = initial_strategy
        else:
            self._current_strategy = np.random.choice([0, 1])

        # Link dynamic memory to decision mechanism's trust
        self._link_memory_to_trust()

        # Statistics
        self._total_interactions = 0
        self._successful_coordinations = 0
        self._strategy_switches = 0
        self._last_prediction: Optional[int] = None

        # Observation-based communication
        self._observations: List[Tuple[int, float]] = []  # [(strategy, weight), ...]
        self._observation_weight: float = 0.5  # Weight of observed vs direct

    def _link_memory_to_trust(self) -> None:
        """Connect dynamic memory's window to decision's trust."""
        from .memory import DynamicMemory

        if isinstance(self._memory, DynamicMemory):
            # Create trust getter from decision mechanism
            trust_getter = lambda: self._decision.get_trust()
            self._memory.set_trust_getter(trust_getter)

    @property
    def id(self) -> int:
        """Agent's unique identifier."""
        return self._id

    @property
    def current_strategy(self) -> int:
        """Agent's current strategy (0 or 1)."""
        return self._current_strategy

    @property
    def decision_mode(self) -> DecisionMode:
        """Current decision mode."""
        return self._decision.mode

    @property
    def trust(self) -> float:
        """Current trust (predictive confidence) level."""
        return self._decision.get_trust()

    @property
    def belief(self) -> np.ndarray:
        """
        Current belief about partner strategy distribution.

        Combines:
        - Direct experience from memory (weight = 1.0)
        - Observations from communication (weight = observation_weight)
        """
        memory_belief = self._memory.get_strategy_distribution()

        if not self._observations:
            return memory_belief

        # Calculate observation-based belief
        obs_counts = np.zeros(2)
        obs_total_weight = 0.0
        for strategy, weight in self._observations:
            obs_counts[strategy] += weight
            obs_total_weight += weight

        if obs_total_weight > 0:
            obs_belief = obs_counts / obs_total_weight
        else:
            obs_belief = np.array([0.5, 0.5])

        # Combine: memory has implicit weight based on size
        memory_weight = len(self._memory) if len(self._memory) > 0 else 1
        combined = (memory_belief * memory_weight + obs_belief * obs_total_weight)
        combined = combined / (memory_weight + obs_total_weight)

        return combined

    @property
    def normative_state(self) -> Optional[NormativeState]:
        """Snapshot of normative memory state, or None if not enabled."""
        if self._normative_memory is None:
            return None
        return self._normative_memory.get_state()

    def receive_observation(self, strategy: int, weight: float = 0.5) -> None:
        """
        Receive an observation from communication.

        Args:
            strategy: Observed strategy (0 or 1)
            weight: Reliability weight of this observation
        """
        self._observations.append((strategy, weight))

    def clear_observations(self) -> None:
        """Clear accumulated observations (called each tick)."""
        self._observations.clear()

    @property
    def memory_window(self) -> int:
        """Current effective memory window size."""
        from .memory import DynamicMemory

        if isinstance(self._memory, DynamicMemory):
            return self._memory.get_effective_window()
        return self._memory.max_size

    def choose_action(self) -> Tuple[int, int]:
        """
        Choose an action based on current beliefs, optionally constrained by norm.

        V5 effective belief integration (Eq. 13-15):
        - b_exp: experience-based belief from memory
        - If norm exists: b_eff = compliance * b_norm + (1 - compliance) * b_exp
        - If no norm: b_eff = b_exp

        Returns:
            Tuple of (chosen_action, predicted_partner_action)
        """
        b_exp = self._memory.get_strategy_distribution()

        # V5: normative constraint
        if self._normative_memory is not None and self._normative_memory.has_norm():
            compliance = self._normative_memory.get_compliance()
            b_norm = self._normative_memory.get_norm_belief()
            b_eff = compliance * b_norm + (1.0 - compliance) * b_exp
        else:
            b_eff = b_exp

        action, prediction = self._decision.choose_action(b_eff)

        # Track for later update
        self._last_prediction = prediction

        # Track strategy switches
        if action != self._current_strategy:
            self._strategy_switches += 1

        self._current_strategy = action
        return action, prediction

    def update(
        self,
        tick: int,
        partner_strategy: int,
        success: bool,
        payoff: float,
    ) -> None:
        """
        Update agent state after an interaction.

        Args:
            tick: Current simulation time
            partner_strategy: Strategy partner chose
            success: Whether coordination was successful
            payoff: Payoff received
        """
        self.record_experience(
            tick=tick,
            partner_strategy=partner_strategy,
            success=success,
            payoff=payoff,
        )
        self.update_predictive_confidence(partner_strategy=partner_strategy)

    def record_experience(
        self,
        tick: int,
        partner_strategy: int,
        success: bool,
        payoff: float,
    ) -> None:
        """
        Update only experience-related state (M^E and interaction counters).

        This method is used by synchronous tick pipelines where experience and
        confidence are updated in separate global phases.
        """
        # Create interaction record
        interaction = Interaction(
            tick=tick,
            partner_strategy=partner_strategy,
            own_strategy=self._current_strategy,
            success=success,
            payoff=payoff,
        )

        # Update experience memory
        self._memory.add(interaction)

        # Update statistics
        self._total_interactions += 1
        if success:
            self._successful_coordinations += 1

    def update_predictive_confidence(self, partner_strategy: int) -> None:
        """
        Update only confidence/decision state from prediction correctness.

        This corresponds to the standalone C-update phase in the tick pipeline.
        """
        if self._last_prediction is not None:
            self._decision.update(self._last_prediction, partner_strategy)

    # =========================================================================
    # V5 Normative Memory Processing
    # =========================================================================

    def process_normative_observations(
        self,
        observed_strategies: List[int],
        tick: Optional[int] = None,
    ) -> Tuple[bool, bool, int]:
        """
        Process observations through normative memory.

        Each observed strategy is routed to the appropriate normative process:
        - If no norm: feeds DDM evidence accumulator
        - If norm exists: checks for anomalies

        After processing all observations, checks for crisis/dissolution.

        Args:
            observed_strategies: List of strategies observed this tick
            tick: Current tick for crystallisation timing

        Returns:
            Tuple of (crystallised, dissolved, enforcement_count):
            - crystallised: True if a new norm formed this tick
            - dissolved: True if an existing norm was overthrown
            - enforcement_count: Number of enforcement signals triggered
        """
        if self._normative_memory is None:
            return (False, False, 0)

        crystallised = False
        dissolved = False
        enforcement_count = 0

        if not self._normative_memory.has_norm():
            # Pre-crystallisation: feed DDM
            if observed_strategies:
                counts = np.zeros(2)
                for s in observed_strategies:
                    counts[s] += 1
                total = len(observed_strategies)
                consistency = max(counts) / total
                dominant = int(np.argmax(counts))

                crystallised = self._normative_memory.update_evidence(
                    confidence=self.trust,
                    consistency=consistency,
                    dominant_strategy=dominant,
                    tick=tick,
                )
        else:
            # Post-crystallisation: anomaly tracking + enforcement
            for s in observed_strategies:
                if self._normative_memory.should_enforce(s, self._enforce_threshold):
                    self._normative_memory.record_enforcement()
                    enforcement_count += 1
                else:
                    self._normative_memory.record_anomaly(s)

            dissolved = self._normative_memory.check_crisis()

        return (crystallised, dissolved, enforcement_count)

    def get_enforcement_signal(
        self,
        observed_strategies: List[int],
    ) -> Optional[int]:
        """
        Check if this agent should broadcast an enforcement signal.

        Returns the norm strategy to broadcast if enforcement conditions are met
        for any observed violation, or None.

        Args:
            observed_strategies: Strategies observed this tick

        Returns:
            The norm strategy to broadcast, or None if no enforcement
        """
        if self._normative_memory is None or not self._normative_memory.has_norm():
            return None

        for s in observed_strategies:
            if self._normative_memory.should_enforce(s, self._enforce_threshold):
                return self._normative_memory.norm

        return None

    def receive_normative_signal(self, signaled_strategy: int) -> None:
        """
        Receive a normative enforcement signal from another agent.

        Boosts DDM drift rate for the next evidence update.
        Only effective if this agent has not yet crystallised a norm.

        Args:
            signaled_strategy: The norm strategy being enforced
        """
        if self._normative_memory is not None:
            self._normative_memory.receive_signal(self._signal_amplification)

    # =========================================================================
    # Metrics and Reset
    # =========================================================================

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current agent metrics.

        Returns:
            Dict with agent statistics
        """
        metrics = {
            "id": self._id,
            "strategy": self._current_strategy,
            "decision_mode": self.decision_mode.value,
            "trust": self.trust,
            "belief": self.belief.tolist(),
            "memory_window": self.memory_window,
            "memory_size": len(self._memory),
            "total_interactions": self._total_interactions,
            "successful_coordinations": self._successful_coordinations,
            "coordination_rate": (
                self._successful_coordinations / self._total_interactions
                if self._total_interactions > 0
                else 0.0
            ),
            "strategy_switches": self._strategy_switches,
            "prediction_accuracy": self._decision.get_prediction_accuracy(),
        }

        # Add decision-specific state
        metrics.update(self._decision.get_state())

        # Add normative state if enabled
        if self._normative_memory is not None:
            ns = self._normative_memory.get_state()
            metrics["norm"] = ns.norm
            metrics["norm_strength"] = ns.strength
            metrics["norm_anomalies"] = ns.anomaly_count
            metrics["norm_evidence"] = ns.evidence
            metrics["has_norm"] = ns.has_norm
            metrics["compliance"] = ns.compliance
            metrics["enforcement_count"] = ns.enforcement_count
            metrics["first_crystallisation_tick"] = ns.first_crystallisation_tick

        return metrics

    def reset(self, initial_strategy: Optional[int] = None) -> None:
        """
        Reset agent to initial state.

        Args:
            initial_strategy: Starting strategy (random if None)
        """
        self._memory.clear()
        self._decision.reset()
        self._observations.clear()

        if initial_strategy is not None:
            self._current_strategy = initial_strategy
        else:
            self._current_strategy = np.random.choice([0, 1])

        self._total_interactions = 0
        self._successful_coordinations = 0
        self._strategy_switches = 0
        self._last_prediction = None

        # Reset normative memory if present
        if self._normative_memory is not None:
            self._normative_memory.reset()

    def __repr__(self) -> str:
        norm_str = ""
        if self._normative_memory is not None and self._normative_memory.has_norm():
            norm_str = f", norm={self._normative_memory.norm}"
        return (
            f"Agent(id={self._id}, strategy={self._current_strategy}, "
            f"mode={self.decision_mode.value}, trust={self.trust:.3f}{norm_str})"
        )


def create_agent(
    agent_id: int,
    # Memory settings
    memory_type: str = "fixed",
    memory_size: int = 5,
    decay_rate: float = 0.9,
    dynamic_base: int = 2,
    dynamic_max: int = 6,
    # Decision settings
    decision_mode: Union[DecisionMode, str] = DecisionMode.COGNITIVE_LOCKIN,
    # For DUAL_FEEDBACK mode
    initial_tau: float = 1.0,
    tau_min: float = 0.1,
    tau_max: float = 2.0,
    cooling_rate: float = 0.1,
    heating_penalty: float = 0.3,
    # For COGNITIVE_LOCKIN and EPSILON_GREEDY modes
    initial_trust: float = 0.5,
    alpha: float = 0.1,
    beta: float = 0.3,
    # For EPSILON_GREEDY mode
    exploration_mode: str = "random",
    # Normative memory (V5)
    enable_normative: bool = False,
    ddm_noise: float = 0.1,
    crystal_threshold: float = 3.0,
    normative_initial_strength: float = 0.8,
    crisis_threshold: int = 10,
    crisis_decay: float = 0.3,
    min_strength: float = 0.1,
    enforce_threshold: float = 0.7,
    compliance_exponent: float = 2.0,
    signal_amplification: float = 2.0,
    normative_rng: Optional[np.random.RandomState] = None,
    # Other
    initial_strategy: Optional[int] = None,
) -> Agent:
    """
    Factory function to create an agent with specified configuration.

    Args:
        agent_id: Unique identifier
        memory_type: 'fixed', 'decay', or 'dynamic'
        memory_size: Size for fixed memory
        decay_rate: Lambda for decay memory
        dynamic_base: Base window for dynamic memory
        dynamic_max: Max window for dynamic memory
        decision_mode: Decision mechanism mode
        initial_tau: Starting temperature (DUAL_FEEDBACK)
        tau_min: Minimum temperature (DUAL_FEEDBACK)
        tau_max: Maximum temperature (DUAL_FEEDBACK)
        cooling_rate: Temperature decrease rate (DUAL_FEEDBACK)
        heating_penalty: Temperature increase (DUAL_FEEDBACK)
        initial_trust: Starting trust (COGNITIVE_LOCKIN, EPSILON_GREEDY)
        alpha: Trust increase rate (COGNITIVE_LOCKIN, EPSILON_GREEDY)
        beta: Trust decay rate (COGNITIVE_LOCKIN, EPSILON_GREEDY)
        exploration_mode: 'random' or 'opposite' (EPSILON_GREEDY)
        enable_normative: Whether to create normative memory (V5)
        ddm_noise: DDM noise sigma_noise
        crystal_threshold: Evidence threshold theta_crystal
        normative_initial_strength: Norm strength on crystallisation
        crisis_threshold: Anomaly count for crisis
        crisis_decay: Strength decay on crisis
        min_strength: Below this, norm dissolves
        enforce_threshold: Min sigma for enforcement
        compliance_exponent: Exponent k in sigma^k
        signal_amplification: DDM drift multiplier gamma_signal
        normative_rng: Per-agent RNG for normative memory
        initial_strategy: Starting strategy

    Returns:
        Configured Agent instance
    """
    # Convert string to enum if needed
    if isinstance(decision_mode, str):
        decision_mode = DecisionMode(decision_mode)

    # Create memory based on type
    if memory_type == "fixed":
        memory = create_memory("fixed", size=memory_size)
    elif memory_type == "decay":
        memory = create_memory("decay", max_size=memory_size * 2, decay_rate=decay_rate)
    elif memory_type == "dynamic":
        memory = create_memory(
            "dynamic",
            max_size=dynamic_max,
            base_size=dynamic_base,
        )
    else:
        raise ValueError(f"Unknown memory type: {memory_type}")

    # Create decision mechanism based on mode
    if decision_mode == DecisionMode.DUAL_FEEDBACK:
        decision = create_decision(
            mode=decision_mode,
            initial_tau=initial_tau,
            tau_min=tau_min,
            tau_max=tau_max,
            cooling_rate=cooling_rate,
            heating_penalty=heating_penalty,
        )
    elif decision_mode == DecisionMode.COGNITIVE_LOCKIN:
        decision = create_decision(
            mode=decision_mode,
            initial_trust=initial_trust,
            alpha=alpha,
            beta=beta,
        )
    elif decision_mode == DecisionMode.EPSILON_GREEDY:
        decision = create_decision(
            mode=decision_mode,
            initial_trust=initial_trust,
            alpha=alpha,
            beta=beta,
            exploration_mode=exploration_mode,
        )
    else:
        raise ValueError(f"Unknown decision mode: {decision_mode}")

    # Create trust manager
    trust_manager = TrustManager(
        memory_base=dynamic_base,
        memory_max=dynamic_max,
    )

    # Create normative memory if enabled (V5)
    normative_memory = None
    if enable_normative:
        normative_memory = NormativeMemory(
            ddm_noise=ddm_noise,
            crystal_threshold=crystal_threshold,
            initial_strength=normative_initial_strength,
            crisis_threshold=crisis_threshold,
            crisis_decay=crisis_decay,
            min_strength=min_strength,
            compliance_exponent=compliance_exponent,
            rng=normative_rng or np.random.RandomState(),
        )

    return Agent(
        agent_id=agent_id,
        memory=memory,
        decision=decision,
        trust_manager=trust_manager,
        initial_strategy=initial_strategy,
        normative_memory=normative_memory,
        enforce_threshold=enforce_threshold,
        signal_amplification=signal_amplification,
    )
