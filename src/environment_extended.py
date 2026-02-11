"""
Extended simulation environment with full communication support.

.. deprecated:: V5
    V5 normative features (enforcement, DDM crystallisation, anomaly tracking)
    are now integrated directly into ``src.environment.SimulationEnvironment``
    via the ``enable_normative`` flag.  This extended environment remains for
    the Bicchieri broadcasting, Skyrms pre-play signaling, and Centola
    threshold contagion mechanisms which are orthogonal to V5.

Integrates three theoretically-grounded communication mechanisms:
1. Normative Signaling (Bicchieri 2006)
2. Pre-play Signaling (Skyrms 2010)
3. Threshold Contagion (Centola 2018)
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass

from .environment import SimulationEnvironment, TickMetrics, SimulationResult
from .decision import DecisionMode
from .communication import (
    CommunicationManager,
    NormativeSignaling,
    PrePlaySignaling,
    ThresholdContagion,
    SignalType,
    ContagionType,
)


@dataclass
class ExtendedTickMetrics(TickMetrics):
    """Extended metrics including communication effects."""
    # Normative expectations (Bicchieri)
    mean_normative_alignment: float  # How aligned are normative expectations?
    normative_empirical_gap: float   # Gap between normative and empirical

    # Signaling (Skyrms)
    signal_accuracy: float  # How often signals matched actions?

    # Contagion (Centola)
    mean_exposure_count: float  # Average exposures per agent


class ExtendedEnvironment(SimulationEnvironment):
    """
    Extended environment with full communication support.

    Adds three mechanisms to the base environment:
    1. Normative Signaling: Agents broadcast what they think SHOULD be done
    2. Pre-play Signaling: Agents signal before each interaction
    3. Threshold Contagion: Agents need multiple sources to update

    These can be enabled/disabled independently for experiments.
    """

    def __init__(
        self,
        # Base parameters (inherited)
        num_agents: int = 100,
        memory_type: str = "dynamic",
        memory_size: int = 5,
        decision_mode: Union[DecisionMode, str] = DecisionMode.COGNITIVE_LOCKIN,
        initial_trust: float = 0.5,
        alpha: float = 0.1,
        beta: float = 0.3,
        random_seed: Optional[int] = None,
        # Observation (basic)
        observation_k: int = 0,
        observation_weight: float = 0.5,
        # Normative Signaling (Bicchieri 2006)
        enable_normative: bool = False,
        normative_broadcast_prob: float = 0.3,
        normative_weight: float = 0.3,
        # Pre-play Signaling (Skyrms 2010)
        enable_preplay: bool = False,
        preplay_signal_type: str = "cheap_talk",  # or "costly"
        preplay_honesty: float = 0.8,
        # Threshold Contagion (Centola 2018)
        enable_threshold: bool = False,
        contagion_type: str = "complex",  # or "simple"
        contagion_threshold: int = 2,
        **kwargs,
    ):
        # Initialize base environment
        super().__init__(
            num_agents=num_agents,
            memory_type=memory_type,
            memory_size=memory_size,
            decision_mode=decision_mode,
            initial_trust=initial_trust,
            alpha=alpha,
            beta=beta,
            random_seed=random_seed,
            observation_k=observation_k,
            observation_weight=observation_weight,
            **kwargs,
        )

        # Store extended config
        self._config.update({
            "enable_normative": enable_normative,
            "normative_broadcast_prob": normative_broadcast_prob,
            "normative_weight": normative_weight,
            "enable_preplay": enable_preplay,
            "preplay_signal_type": preplay_signal_type,
            "preplay_honesty": preplay_honesty,
            "enable_threshold": enable_threshold,
            "contagion_type": contagion_type,
            "contagion_threshold": contagion_threshold,
        })

        # Initialize communication manager
        self._comm_manager = CommunicationManager(
            enable_normative=enable_normative,
            enable_preplay=enable_preplay,
            enable_threshold=enable_threshold,
            normative_broadcast_prob=normative_broadcast_prob,
            normative_weight=normative_weight,
            preplay_signal_type=SignalType(preplay_signal_type),
            preplay_honesty=preplay_honesty,
            contagion_type=ContagionType(contagion_type),
            contagion_threshold=contagion_threshold,
        )

        # Set normative weight on agents
        if enable_normative:
            for agent in self._agents:
                agent._normative_weight = normative_weight

        # Extended metrics tracking
        self._signal_matches = 0
        self._signal_total = 0

    def step(self) -> TickMetrics:
        """
        Execute one simulation tick with extended communication.

        Extended flow:
        1. Communication manager tick start
        2. Normative broadcasting (if enabled)
        3. Random matching
        4. Pre-play signaling (if enabled)
        5. Interaction execution
        6. Observation distribution (with threshold filtering if enabled)
        7. Normative expectation updates
        8. Metrics collection
        """
        # Clear agent observations
        for agent in self._agents:
            agent.clear_observations()
            agent.clear_signal()

        # Communication manager tick start
        self._comm_manager.tick_start()

        # === Normative Broadcasting (Bicchieri 2006) ===
        if self._comm_manager.normative:
            self._broadcast_normative_messages()

        # Get random pairs
        pairs = self.random_matching()

        # Track statistics
        tick_switches = 0
        tick_successes = 0
        tick_interactions = len(pairs)
        strategies_before = {a.id: a.current_strategy for a in self._agents}

        # Store interactions for observation
        tick_interactions_list = []

        # Execute interactions with pre-play signaling
        for idx1, idx2 in pairs:
            agent1 = self._agents[idx1]
            agent2 = self._agents[idx2]

            # === Pre-play Signaling (Skyrms 2010) ===
            if self._comm_manager.preplay:
                self._exchange_signals(agent1, agent2)

            # Execute interaction
            outcome, pred1, pred2 = self._execute_interaction(agent1, agent2)
            tick_interactions_list.append((idx1, idx2, outcome.strategy1, outcome.strategy2))

            # Track signal accuracy
            if self._comm_manager.preplay:
                self._track_signal_accuracy(agent1, agent2, outcome.strategy1, outcome.strategy2)

            # Update agents
            agent1.update(
                tick=self._current_tick,
                partner_strategy=outcome.strategy2,
                success=outcome.success,
                payoff=outcome.payoff1,
            )
            agent2.update(
                tick=self._current_tick,
                partner_strategy=outcome.strategy1,
                success=outcome.success,
                payoff=outcome.payoff2,
            )

            if outcome.success:
                tick_successes += 1

        # === Observation Distribution (with threshold filtering) ===
        if self._observation_k > 0:
            self._distribute_observations_extended(tick_interactions_list)

        self._last_interactions = tick_interactions_list

        # Count strategy switches
        for agent in self._agents:
            if agent.current_strategy != strategies_before[agent.id]:
                tick_switches += 1

        # Collect extended metrics
        metrics = self._collect_extended_metrics(
            tick_interactions=tick_interactions,
            tick_successes=tick_successes,
            tick_switches=tick_switches,
        )

        self._tick_history.append(metrics)
        self._current_tick += 1
        self._check_convergence(metrics)

        return metrics

    def _broadcast_normative_messages(self) -> None:
        """Broadcast normative messages and update agents."""
        normative = self._comm_manager.normative

        # Collect agent data
        agent_data = [
            (a.id, a.belief, a.current_strategy)
            for a in self._agents
        ]

        # Generate messages
        messages = normative.broadcast_all(agent_data)

        # Each agent receives and updates
        for agent in self._agents:
            received = normative.receive_messages(agent.id)
            if received:
                norm_expectation = normative.compute_normative_expectation(received)
                agent.update_normative_expectation(norm_expectation)

    def _exchange_signals(self, agent1, agent2) -> Tuple[int, int]:
        """
        Exchange pre-play signals between two agents.

        Uses asymmetric protocol (Skyrms 2010) to avoid coordination failure:
        - The agent with LOWER trust takes the "follower" role
        - The agent with HIGHER trust takes the "leader" role
        - Only the follower receives and uses the signal

        This creates a focal point convention that enables coordination.

        Returns:
            (intended_action1, intended_action2) - not used, kept for compatibility
        """
        preplay = self._comm_manager.preplay

        # Determine roles based on trust (higher trust = leader)
        # In case of tie, use agent ID
        if agent1.trust > agent2.trust or (agent1.trust == agent2.trust and agent1.id < agent2.id):
            leader, follower = agent1, agent2
        else:
            leader, follower = agent2, agent1

        # Leader signals their intended action
        leader_belief = leader._memory.get_strategy_distribution()
        leader_intended = 1 if leader_belief[1] > leader_belief[0] else 0

        signal = preplay.generate_signal(leader.id, leader_intended, leader.trust)
        pred_action, conf = preplay.interpret_signal(signal, follower.trust)

        # Only follower receives the signal (and should follow it)
        follower.receive_signal(pred_action, conf)

        # Leader doesn't receive a signal - they lead based on their belief
        # This prevents the symmetric swap problem

        return leader_intended, leader_intended  # Both should converge to same

    def _track_signal_accuracy(self, agent1, agent2, actual1: int, actual2: int) -> None:
        """Track how accurately signals predicted actions."""
        if agent1._last_received_signal is not None:
            self._signal_total += 1
            if agent1._last_received_signal == actual2:
                self._signal_matches += 1

        if agent2._last_received_signal is not None:
            self._signal_total += 1
            if agent2._last_received_signal == actual1:
                self._signal_matches += 1

    def _distribute_observations_extended(
        self,
        interactions: List[Tuple[int, int, int, int]],
    ) -> None:
        """
        Distribute observations with threshold filtering.
        """
        threshold = self._comm_manager.threshold

        for agent in self._agents:
            # Get other interactions
            other_interactions = [
                (s1, s2) for id1, id2, s1, s2 in interactions
                if id1 != agent.id and id2 != agent.id
            ]

            if not other_interactions:
                continue

            # Sample k interactions
            k = min(self._observation_k, len(other_interactions))
            if k > 0:
                indices = np.random.choice(len(other_interactions), size=k, replace=False)
                observations = []
                for i in indices:
                    s1, s2 = other_interactions[i]
                    observations.append((s1, self._observation_weight))
                    observations.append((s2, self._observation_weight))

                # Apply threshold filtering if enabled
                if threshold:
                    observations = threshold.filter_observations(agent.id, observations)

                # Add to agent
                for strategy, weight in observations:
                    agent.receive_observation(strategy, weight)

    def _collect_extended_metrics(
        self,
        tick_interactions: int,
        tick_successes: int,
        tick_switches: int,
    ) -> TickMetrics:
        """Collect extended metrics including communication effects."""
        # Get base metrics
        base_metrics = self._collect_metrics(
            tick_interactions=tick_interactions,
            tick_successes=tick_successes,
            tick_switches=tick_switches,
        )

        # Would extend with additional metrics here
        # For now, return base metrics
        return base_metrics

    def get_signal_accuracy(self) -> float:
        """Get overall signal accuracy."""
        if self._signal_total == 0:
            return 0.0
        return self._signal_matches / self._signal_total

    def get_normative_alignment(self) -> float:
        """Get how aligned normative expectations are across agents."""
        if not self._comm_manager.normative:
            return 0.0

        norms = [a.normative_expectation for a in self._agents]
        # Use variance of dominant strategy probability
        dominant_probs = [max(n) for n in norms]
        return 1 - np.var(dominant_probs)

    def reset(self, random_seed: Optional[int] = None) -> None:
        """Reset extended environment."""
        super().reset(random_seed)
        self._signal_matches = 0
        self._signal_total = 0

        # Reset threshold exposures
        if self._comm_manager.threshold:
            for agent in self._agents:
                self._comm_manager.threshold.reset_agent(agent.id)

    def __repr__(self) -> str:
        mechanisms = []
        if self._comm_manager.normative:
            mechanisms.append("normative")
        if self._comm_manager.preplay:
            mechanisms.append("preplay")
        if self._comm_manager.threshold:
            mechanisms.append("threshold")

        mech_str = "+".join(mechanisms) if mechanisms else "none"
        return (
            f"ExtendedEnvironment(agents={self._num_agents}, "
            f"mechanisms={mech_str})"
        )
