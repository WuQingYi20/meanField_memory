"""
Agent class integrating memory, decision, and trust systems.

Supports multiple decision modes:
- DUAL_FEEDBACK: Original τ-based softmax (two feedback loops)
- COGNITIVE_LOCKIN: Probability matching with Trust (cognitive only)
- EPSILON_GREEDY: Best response with exploration (robustness check)

Supports observation-based communication:
- Agents can observe other interactions beyond their own
- Observations are weighted and combined with direct experience
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, Union, List

from .memory import BaseMemory, Interaction, create_memory
from .decision import (
    BaseDecision, DecisionMode, create_decision,
    DualFeedbackDecision, CognitiveLockInDecision, EpsilonGreedyDecision
)
from .trust import TrustManager


class Agent:
    """
    An agent in the ABM simulation.

    Integrates:
    - Memory system: Stores interaction history
    - Decision mechanism: Action selection and trust dynamics
    - Trust → Memory linkage: Trust affects memory window (for dynamic memory)

    The agent's behavior emerges from the feedback loop:
    - Memory informs beliefs about partner strategies
    - Decision mechanism selects action based on beliefs
    - Prediction accuracy updates internal state (τ or Trust)
    - Internal state affects memory window (for dynamic memory)
    """

    def __init__(
        self,
        agent_id: int,
        memory: BaseMemory,
        decision: BaseDecision,
        trust_manager: Optional[TrustManager] = None,
        initial_strategy: Optional[int] = None,
    ):
        """
        Initialize an agent.

        Args:
            agent_id: Unique identifier
            memory: Memory system instance
            decision: Decision mechanism instance
            trust_manager: Trust manager (optional, creates default if None)
            initial_strategy: Starting strategy (random if None)
        """
        self._id = agent_id
        self._memory = memory
        self._decision = decision
        self._trust_manager = trust_manager or TrustManager()

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

        # Normative expectations (Bicchieri 2006)
        self._normative_expectation: np.ndarray = np.array([0.5, 0.5])
        self._normative_weight: float = 0.3  # Weight of normative vs empirical

        # Pre-play signaling (Skyrms 2010)
        self._last_received_signal: Optional[int] = None
        self._signal_confidence: float = 0.5

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
        """Current trust level."""
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

    # =========================================================================
    # Normative Expectations (Bicchieri 2006)
    # =========================================================================

    @property
    def normative_expectation(self) -> np.ndarray:
        """What agent thinks others expect them to do."""
        return self._normative_expectation.copy()

    def update_normative_expectation(
        self,
        received_norms: np.ndarray,
        learning_rate: float = 0.2,
    ) -> None:
        """
        Update normative expectation based on received messages.

        Args:
            received_norms: [P(norm=0), P(norm=1)] from messages
            learning_rate: How fast to update expectations
        """
        self._normative_expectation = (
            (1 - learning_rate) * self._normative_expectation +
            learning_rate * received_norms
        )

    def get_combined_expectation(self) -> np.ndarray:
        """
        Get combined expectation (empirical + normative).

        Following Bicchieri (2006):
        - Empirical: What do others DO?
        - Normative: What do others EXPECT me to do?

        Returns:
            Weighted combination of empirical belief and normative expectation
        """
        empirical = self.belief  # What others do
        normative = self._normative_expectation  # What others expect

        combined = (
            (1 - self._normative_weight) * empirical +
            self._normative_weight * normative
        )
        return combined

    def express_normative_belief(self) -> Tuple[int, float]:
        """
        Express what this agent thinks SHOULD be done.

        Returns:
            (expressed_norm, confidence)
        """
        # Tend to express current strategy as the norm (cognitive consistency)
        expressed = self._current_strategy
        confidence = max(self._normative_expectation)
        return expressed, confidence

    # =========================================================================
    # Pre-play Signaling (Skyrms 2010)
    # =========================================================================

    def receive_signal(self, signal: int, confidence: float) -> None:
        """
        Receive pre-play signal from interaction partner.

        Args:
            signal: Partner's signaled intention
            confidence: Confidence in signal honesty
        """
        self._last_received_signal = signal
        self._signal_confidence = confidence

    def get_signal_adjusted_belief(self) -> np.ndarray:
        """
        Get belief adjusted for received signal.

        If a signal was received, incorporate it into belief.
        The adjustment follows a convention: agents with lower trust
        defer more strongly to signals (creates coordination focal point).
        """
        base_belief = self.belief

        if self._last_received_signal is None:
            return base_belief

        # Signal shifts belief toward signaled strategy
        signal_belief = np.zeros(2)
        signal_belief[self._last_received_signal] = 1.0

        # Effective signal weight increases with lower trust
        # Low trust agents defer more to signals (convention for breaking ties)
        trust = self._decision.get_trust()
        defer_factor = 1.0 - trust * 0.5  # Range: [0.5, 1.0] as trust goes [1, 0]
        effective_confidence = self._signal_confidence * defer_factor

        adjusted = (
            (1 - effective_confidence) * base_belief +
            effective_confidence * signal_belief
        )
        return adjusted

    def clear_signal(self) -> None:
        """Clear received signal (after interaction)."""
        self._last_received_signal = None
        self._signal_confidence = 0.5

    @property
    def memory_window(self) -> int:
        """Current effective memory window size."""
        from .memory import DynamicMemory

        if isinstance(self._memory, DynamicMemory):
            return self._memory.get_effective_window()
        return self._memory.max_size

    def choose_action(self, use_signal: bool = True) -> Tuple[int, int]:
        """
        Choose an action based on current beliefs and received signals.

        In coordination games, signals communicate intended actions.
        The rational response is to match the signal to coordinate.

        Args:
            use_signal: If True and a signal was received, consider following it

        Returns:
            Tuple of (chosen_action, predicted_partner_action)
        """
        belief = self._memory.get_strategy_distribution()

        # If we received a signal, consider following it to coordinate
        if use_signal and self._last_received_signal is not None:
            # Probability of following signal increases with confidence
            # In coordination games, following signals is rational
            follow_prob = self._signal_confidence

            if np.random.random() < follow_prob:
                # Follow the signal: play what partner signaled they'll play
                action = self._last_received_signal
                prediction = self._last_received_signal
            else:
                # Don't follow signal: use normal decision mechanism
                action, prediction = self._decision.choose_action(belief)
        else:
            # No signal: use normal decision mechanism
            action, prediction = self._decision.choose_action(belief)

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
        # Create interaction record
        interaction = Interaction(
            tick=tick,
            partner_strategy=partner_strategy,
            own_strategy=self._current_strategy,
            success=success,
            payoff=payoff,
        )

        # Update memory
        self._memory.add(interaction)

        # Update decision mechanism based on prediction accuracy
        if self._last_prediction is not None:
            self._decision.update(self._last_prediction, partner_strategy)

        # Update statistics
        self._total_interactions += 1
        if success:
            self._successful_coordinations += 1

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

        # Reset communication state
        self._normative_expectation = np.array([0.5, 0.5])
        self._last_received_signal = None
        self._signal_confidence = 0.5

    def __repr__(self) -> str:
        return (
            f"Agent(id={self._id}, strategy={self._current_strategy}, "
            f"mode={self.decision_mode.value}, trust={self.trust:.3f})"
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

    return Agent(
        agent_id=agent_id,
        memory=memory,
        decision=decision,
        trust_manager=trust_manager,
        initial_strategy=initial_strategy,
    )
