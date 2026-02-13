"""
Simulation environment for ABM coordination game.

Manages agent interactions, random matching, and data collection.
Supports multiple decision modes for comparative experiments.
Supports observation-based communication for studying norm emergence.

V5: Supports dual-memory architecture with normative memory,
enforcement signalling, and normative metrics collection.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
import json
from pathlib import Path

from .agent import Agent, create_agent
from .game import CoordinationGame, GameOutcome
from .decision import DecisionMode
from .norms import NormDetector, NormLevel, NormState


@dataclass
class TickMetrics:
    """Metrics collected at each tick."""

    tick: int
    strategy_distribution: List[float]  # [fraction_0, fraction_1]
    mean_trust: float
    coordination_rate: float  # This tick's coordination success rate
    num_interactions: int
    strategy_switches: int
    converged: bool
    majority_strategy: int
    majority_fraction: float
    mean_memory_window: float
    # Norm detection (Bicchieri 2006)
    norm_level: int  # 0=NONE, 1=BEHAVIORAL, 2=COGNITIVE/EMPIRICAL, 3=SHARED, 4=NORMATIVE, 5=INSTITUTIONAL
    belief_error: float  # Mean |agent_belief - true_distribution|
    belief_variance: float  # Variance across agent beliefs
    # V5 normative memory metrics (defaults maintain backward compat)
    norm_adoption_rate: float = 0.0       # Fraction of agents with crystallised norm
    mean_norm_strength: float = 0.0       # Mean sigma across agents with norms
    mean_ddm_evidence: float = 0.0        # Mean DDM evidence across agents without norms
    total_anomalies: int = 0              # Total anomalies accumulated this tick
    norm_crises: int = 0                  # Number of norm dissolutions this tick
    enforcement_events: int = 0           # Number of enforcement signals this tick
    norm_crystallisations: int = 0        # Number of new norms formed this tick
    correct_norm_rate: float = 0.0        # Fraction of norms matching majority strategy
    mean_compliance: float = 0.0          # Mean compliance across all agents


@dataclass
class SimulationResult:
    """Complete simulation results."""

    config: Dict[str, Any]
    tick_history: List[TickMetrics]
    final_state: Dict[str, Any]
    convergence_tick: Optional[int]
    converged: bool
    agent_final_states: List[Dict[str, Any]]


class SimulationEnvironment:
    """
    Environment for running ABM coordination game simulations.

    Manages:
    - Agent population
    - Random matching each tick
    - Game execution
    - Metrics collection
    - Convergence detection
    - V5: Normative memory processing and enforcement
    """
    TICK_UPDATE_ORDER = (
        "pair_and_action",
        "observe_and_memory_update",
        "confidence_update",
        "normative_ddm_or_anomaly",
        "enforcement_signal_broadcast",
        "metrics_and_convergence",
    )

    def __init__(
        self,
        num_agents: int = 100,
        game: Optional[CoordinationGame] = None,
        # Memory settings
        memory_type: str = "fixed",
        memory_size: int = 5,
        decay_rate: float = 0.9,
        dynamic_base: int = 2,
        dynamic_max: int = 6,
        # Decision mode
        decision_mode: Union[DecisionMode, str] = DecisionMode.COGNITIVE_LOCKIN,
        # DUAL_FEEDBACK parameters
        initial_tau: float = 1.0,
        tau_min: float = 0.1,
        tau_max: float = 2.0,
        cooling_rate: float = 0.1,
        heating_penalty: float = 0.3,
        # COGNITIVE_LOCKIN and EPSILON_GREEDY parameters
        initial_trust: float = 0.5,
        alpha: float = 0.1,
        beta: float = 0.3,
        # EPSILON_GREEDY specific
        exploration_mode: str = "random",
        # Convergence settings
        convergence_threshold: float = 0.95,
        convergence_window: int = 50,
        random_seed: Optional[int] = None,
        # Observation/communication settings
        observation_k: int = 0,  # Number of interactions to observe (0 = no observation)
        observation_weight: float = 0.5,  # Weight of observed vs direct experience
        # V5: Normative memory settings
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
        strengthen_rate: float = 0.005,
        # Shock injection (robustness experiments)
        shock_tick: Optional[int] = None,
        shock_violator_fraction: float = 0.0,
        shock_violate_strategy: Optional[int] = None,
        # Performance setting
        collect_history: bool = True,
    ):
        """
        Initialize simulation environment.

        Args:
            num_agents: Number of agents (must be even for pairing)
            game: Coordination game instance
            memory_type: Type of memory ('fixed', 'decay', 'dynamic')
            memory_size: Base memory size
            decay_rate: Decay rate for decay memory
            dynamic_base: Base window for dynamic memory
            dynamic_max: Max window for dynamic memory
            decision_mode: Decision mechanism mode
            initial_tau: Starting temperature (DUAL_FEEDBACK)
            tau_min: Minimum temperature (DUAL_FEEDBACK)
            tau_max: Maximum temperature (DUAL_FEEDBACK)
            cooling_rate: Temperature decrease on success (DUAL_FEEDBACK)
            heating_penalty: Temperature increase on failure (DUAL_FEEDBACK)
            initial_trust: Starting trust level (COGNITIVE_LOCKIN, EPSILON_GREEDY)
            alpha: Trust increase rate (COGNITIVE_LOCKIN, EPSILON_GREEDY)
            beta: Trust decay rate (COGNITIVE_LOCKIN, EPSILON_GREEDY)
            exploration_mode: 'random' or 'opposite' (EPSILON_GREEDY)
            convergence_threshold: Fraction needed for consensus
            convergence_window: Ticks threshold must be maintained
            random_seed: Random seed for reproducibility
            observation_k: Number of extra interactions to observe
            observation_weight: Weight of observed vs direct experience
            enable_normative: Whether to enable normative memory (V5)
            ddm_noise: DDM noise sigma_noise
            crystal_threshold: Evidence threshold for crystallisation
            normative_initial_strength: Norm strength on crystallisation
            crisis_threshold: Anomaly count for crisis
            crisis_decay: Strength decay on crisis
            min_strength: Below this, norm dissolves
            enforce_threshold: Min sigma for enforcement
            compliance_exponent: Exponent k in sigma^k
            signal_amplification: DDM drift multiplier gamma_signal
            strengthen_rate: Sigma increase per conforming observation (V5.1)
            shock_tick: Tick T at which persistent violators are injected
            shock_violator_fraction: Injected violator ratio epsilon in [0, 1]
            shock_violate_strategy: Forced strategy of violators (None=opposite majority at T)
            collect_history: Whether to store per-tick history in memory
        """
        # Ensure even number of agents for pairing
        if num_agents % 2 != 0:
            num_agents += 1

        self._num_agents = num_agents
        self._game = game or CoordinationGame()
        self._convergence_threshold = convergence_threshold
        self._convergence_window = convergence_window
        self._enable_normative = enable_normative
        self._shock_tick = shock_tick
        self._shock_violator_fraction = max(0.0, min(1.0, shock_violator_fraction))
        self._shock_violate_strategy = shock_violate_strategy
        self._shock_active = False
        self._persistent_violator_strategy_by_id: Dict[int, int] = {}

        # Convert string to enum if needed
        if isinstance(decision_mode, str):
            decision_mode = DecisionMode(decision_mode)
        self._decision_mode = decision_mode
        self._collect_history = collect_history

        # Observation settings
        self._observation_k = observation_k
        self._observation_weight = observation_weight

        # Store config for reproducibility
        self._config = {
            "num_agents": num_agents,
            "memory_type": memory_type,
            "memory_size": memory_size,
            "decay_rate": decay_rate,
            "dynamic_base": dynamic_base,
            "dynamic_max": dynamic_max,
            "decision_mode": decision_mode.value,
            # DUAL_FEEDBACK params
            "initial_tau": initial_tau,
            "tau_min": tau_min,
            "tau_max": tau_max,
            "cooling_rate": cooling_rate,
            "heating_penalty": heating_penalty,
            # COGNITIVE_LOCKIN/EPSILON_GREEDY params
            "initial_trust": initial_trust,
            "alpha": alpha,
            "beta": beta,
            # EPSILON_GREEDY specific
            "exploration_mode": exploration_mode,
            # Convergence
            "convergence_threshold": convergence_threshold,
            "convergence_window": convergence_window,
            "random_seed": random_seed,
            # Observation/communication
            "observation_k": observation_k,
            "observation_weight": observation_weight,
            # V5 normative
            "enable_normative": enable_normative,
            "ddm_noise": ddm_noise,
            "crystal_threshold": crystal_threshold,
            "normative_initial_strength": normative_initial_strength,
            "crisis_threshold": crisis_threshold,
            "crisis_decay": crisis_decay,
            "min_strength": min_strength,
            "enforce_threshold": enforce_threshold,
            "compliance_exponent": compliance_exponent,
            "signal_amplification": signal_amplification,
            "strengthen_rate": strengthen_rate,
            "shock_tick": shock_tick,
            "shock_violator_fraction": shock_violator_fraction,
            "shock_violate_strategy": shock_violate_strategy,
        }

        # Set random seed
        if random_seed is not None:
            np.random.seed(random_seed)

        # Create agents with appropriate decision mode parameters
        self._agents: List[Agent] = []
        for i in range(num_agents):
            agent = create_agent(
                agent_id=i,
                memory_type=memory_type,
                memory_size=memory_size,
                decay_rate=decay_rate,
                dynamic_base=dynamic_base,
                dynamic_max=dynamic_max,
                decision_mode=decision_mode,
                # DUAL_FEEDBACK params
                initial_tau=initial_tau,
                tau_min=tau_min,
                tau_max=tau_max,
                cooling_rate=cooling_rate,
                heating_penalty=heating_penalty,
                # COGNITIVE_LOCKIN/EPSILON_GREEDY params
                initial_trust=initial_trust,
                alpha=alpha,
                beta=beta,
                # EPSILON_GREEDY specific
                exploration_mode=exploration_mode,
                # V5 normative
                enable_normative=enable_normative,
                ddm_noise=ddm_noise,
                crystal_threshold=crystal_threshold,
                normative_initial_strength=normative_initial_strength,
                crisis_threshold=crisis_threshold,
                crisis_decay=crisis_decay,
                min_strength=min_strength,
                enforce_threshold=enforce_threshold,
                compliance_exponent=compliance_exponent,
                signal_amplification=signal_amplification,
                strengthen_rate=strengthen_rate,
                normative_rng=np.random.RandomState(random_seed + i if random_seed is not None else None),
            )
            self._agents.append(agent)

        # Simulation state
        self._current_tick = 0
        self._tick_history: List[TickMetrics] = []
        self._last_metrics: Optional[TickMetrics] = None
        self._convergence_tick: Optional[int] = None
        self._convergence_counter = 0
        # Key event timing (kept even when history collection is off)
        self._first_norm_tick: Optional[int] = None
        self._normative_level_tick: Optional[int] = None

        # Norm detection (Bicchieri 2006)
        self._norm_detector = NormDetector(
            behavioral_threshold=convergence_threshold,
            stability_window=convergence_window // 2,  # Half of convergence window
        )
        self._last_interactions: List[Tuple[int, int, int, int]] = []  # For observation

    @property
    def agents(self) -> List[Agent]:
        """List of all agents."""
        return self._agents

    @property
    def num_agents(self) -> int:
        """Number of agents."""
        return self._num_agents

    @property
    def current_tick(self) -> int:
        """Current simulation tick."""
        return self._current_tick

    @property
    def config(self) -> Dict[str, Any]:
        """Simulation configuration."""
        return self._config.copy()

    def random_matching(self) -> List[Tuple[int, int]]:
        """
        Create random pairs of agents.

        Returns:
            List of (agent_id_1, agent_id_2) tuples
        """
        indices = np.arange(self._num_agents)
        np.random.shuffle(indices)

        pairs = []
        for i in range(0, self._num_agents, 2):
            pairs.append((indices[i], indices[i + 1]))

        return pairs

    def _execute_interaction(
        self,
        agent1: Agent,
        agent2: Agent,
    ) -> Tuple[GameOutcome, int, int]:
        """
        Execute one interaction between two agents.

        Args:
            agent1: First agent
            agent2: Second agent

        Returns:
            Tuple of (game_outcome, prediction1, prediction2)
        """
        # Both agents choose actions and make predictions
        action1, pred1 = agent1.choose_action()
        action2, pred2 = agent2.choose_action()
        action1 = self._persistent_violator_strategy_by_id.get(agent1.id, action1)
        action2 = self._persistent_violator_strategy_by_id.get(agent2.id, action2)

        # Play the game
        outcome = self._game.play(action1, action2)

        return outcome, pred1, pred2

    def get_tick_update_order(self) -> List[str]:
        """Return the canonical per-tick update sequence."""
        return list(self.TICK_UPDATE_ORDER)

    def _activate_shock_if_needed(self) -> None:
        """Activate persistent violators at the configured shock tick."""
        if self._shock_active:
            return
        if self._shock_tick is None or self._current_tick != self._shock_tick:
            return
        if self._shock_violator_fraction <= 0.0:
            self._shock_active = True
            return

        n_violators = int(round(self._num_agents * self._shock_violator_fraction))
        if n_violators <= 0:
            self._shock_active = True
            return

        agent_ids = np.arange(self._num_agents)
        chosen_ids = np.random.choice(agent_ids, size=n_violators, replace=False)

        if self._shock_violate_strategy in (0, 1):
            forced_strategy = int(self._shock_violate_strategy)
        else:
            majority_strategy = self._last_metrics.majority_strategy if self._last_metrics is not None else 0
            forced_strategy = 1 - majority_strategy

        self._persistent_violator_strategy_by_id = {
            int(agent_id): forced_strategy for agent_id in chosen_ids
        }
        self._shock_active = True

    def _distribute_observations(
        self,
        interactions: List[Tuple[int, int, int, int]],
    ) -> None:
        """
        Distribute observations to agents based on observation_k.

        Each agent observes k random interactions (not including their own).
        This models social learning / observation of others.

        Args:
            interactions: List of (id1, id2, strategy1, strategy2)
        """
        for agent in self._agents:
            # Get interactions not involving this agent
            other_interactions = [
                (s1, s2) for id1, id2, s1, s2 in interactions
                if id1 != agent.id and id2 != agent.id
            ]

            if not other_interactions:
                continue

            # Sample k interactions to observe
            k = min(self._observation_k, len(other_interactions))
            if k > 0:
                indices = np.random.choice(len(other_interactions), size=k, replace=False)
                for i in indices:
                    s1, s2 = other_interactions[i]
                    # Observe both strategies from the interaction
                    agent.receive_observation(s1, self._observation_weight)
                    agent.receive_observation(s2, self._observation_weight)

    def _collect_observed_strategies(
        self,
        agent: Agent,
        interactions: Optional[List[Tuple[int, int, int, int]]] = None,
        partner_strategy_by_agent: Optional[Dict[int, int]] = None,
    ) -> List[int]:
        """
        Collect all strategies observed by an agent this tick.

        Includes partner's strategy from direct interaction plus
        any observations from observation_k.

        Args:
            agent: The agent
            interactions: All interactions this tick (fallback path)
            partner_strategy_by_agent: Direct mapping {agent_id: partner_strategy}

        Returns:
            List of observed strategies
        """
        observed = []

        # Fast path: direct partner strategy lookup built during step().
        if partner_strategy_by_agent is not None:
            partner_strategy = partner_strategy_by_agent.get(agent.id)
            if partner_strategy is not None:
                observed.append(partner_strategy)
        elif interactions is not None:
            # Fallback path for backward compatibility.
            for id1, id2, s1, s2 in interactions:
                if id1 == agent.id:
                    observed.append(s2)
                elif id2 == agent.id:
                    observed.append(s1)

        # Observed strategies (from observation_k, stored in agent._observations)
        for strategy, weight in agent._observations:
            observed.append(strategy)

        return observed

    def _process_normative_step(
        self,
        interactions: List[Tuple[int, int, int, int]],
        partner_strategy_by_agent: Optional[Dict[int, int]] = None,
        tick: Optional[int] = None,
    ) -> Tuple[int, int, int]:
        """
        Process normative memory updates for all agents.

        Steps:
        1. Each agent processes observed strategies through normative memory
        2. Agents with enforcement signals broadcast to all others

        Args:
            interactions: All interactions this tick

        Returns:
            Tuple of (total_crystallisations, total_dissolutions, total_enforcements)
        """
        total_crystallisations = 0
        total_dissolutions = 0
        total_enforcements = 0

        # Step 1: Collect observed strategies and process normative observations
        # We only need to know if there is at least one enforcement signal this tick.
        any_enforcement_signal: Optional[int] = None

        for agent in self._agents:
            observed = self._collect_observed_strategies(
                agent,
                interactions=interactions,
                partner_strategy_by_agent=partner_strategy_by_agent,
            )

            # Check for enforcement signals BEFORE processing (agent state still pristine)
            signal = agent.get_enforcement_signal(observed)
            if signal is not None:
                any_enforcement_signal = signal
                total_enforcements += 1

            # Process observations through normative memory
            crystallised, dissolved, _ = agent.process_normative_observations(
                observed,
                tick=tick,
            )

            if crystallised:
                total_crystallisations += 1
            if dissolved:
                total_dissolutions += 1

        # Step 2: Broadcast enforcement signal once per tick.
        # V5.1: receive_normative_signal() delivers a directed push gated by
        # (1 - C_receiver), so the signal correctly targets the enforced strategy.
        if any_enforcement_signal is not None:
            for agent in self._agents:
                agent.receive_normative_signal(any_enforcement_signal)

        return total_crystallisations, total_dissolutions, total_enforcements

    def step(self) -> TickMetrics:
        """
        Execute one simulation tick.

        Each tick follows a fixed update order:
        1. Pair and action
        2. Observe and memory update
        3. Confidence update
        4. Normative update (DDM / anomaly)
        5. Enforcement signal broadcast
        6. Metrics and convergence

        Returns:
            Metrics for this tick
        """
        self._activate_shock_if_needed()

        # Stage 1: clear observations from previous tick
        for agent in self._agents:
            agent.clear_observations()

        # Stage 2: random matching
        pairs = self.random_matching()

        # Track tick statistics
        tick_switches = 0
        tick_successes = 0
        tick_interactions = len(pairs)

        # Record strategies before interactions (for switch counting)
        strategies_before = {a.id: a.current_strategy for a in self._agents}

        # Store all interactions for observation
        tick_interactions_list: List[Tuple[int, int, int, int]] = []
        # Fast lookup of each agent's direct partner strategy in this tick.
        partner_strategy_by_agent: Dict[int, int] = {}

        # Stage 3: execute pair interactions (no state update yet; synchronous pipeline)
        pending_updates: List[Tuple[Agent, int, bool, float]] = []
        for idx1, idx2 in pairs:
            agent1 = self._agents[idx1]
            agent2 = self._agents[idx2]

            # Execute interaction
            outcome, pred1, pred2 = self._execute_interaction(agent1, agent2)

            # Store for observation
            tick_interactions_list.append((idx1, idx2, outcome.strategy1, outcome.strategy2))
            partner_strategy_by_agent[idx1] = outcome.strategy2
            partner_strategy_by_agent[idx2] = outcome.strategy1

            pending_updates.append((agent1, outcome.strategy2, outcome.success, outcome.payoff1))
            pending_updates.append((agent2, outcome.strategy1, outcome.success, outcome.payoff2))

            if outcome.success:
                tick_successes += 1

        # Stage 4: distribute observation samples + write M^E experience updates
        if self._observation_k > 0:
            self._distribute_observations(tick_interactions_list)
        for agent, partner_strategy, success, payoff in pending_updates:
            agent.record_experience(
                tick=self._current_tick,
                partner_strategy=partner_strategy,
                success=success,
                payoff=payoff,
            )

        self._last_interactions = tick_interactions_list

        # Stage 5: confidence C update from prediction correctness
        for agent, partner_strategy, success, payoff in pending_updates:
            agent.update_predictive_confidence(partner_strategy=partner_strategy)

        # Stage 6: process normative memory updates (DDM/anomaly/enforcement)
        tick_crystallisations = 0
        tick_dissolutions = 0
        tick_enforcements = 0
        if self._enable_normative:
            tick_crystallisations, tick_dissolutions, tick_enforcements = \
                self._process_normative_step(
                    tick_interactions_list,
                    partner_strategy_by_agent=partner_strategy_by_agent,
                    tick=self._current_tick,
                )

        # Count strategy switches
        for agent in self._agents:
            if agent.current_strategy != strategies_before[agent.id]:
                tick_switches += 1

        # Stage 7: collect metrics
        metrics = self._collect_metrics(
            tick_interactions=tick_interactions,
            tick_successes=tick_successes,
            tick_switches=tick_switches,
            tick_crystallisations=tick_crystallisations,
            tick_dissolutions=tick_dissolutions,
            tick_enforcements=tick_enforcements,
        )

        self._last_metrics = metrics
        if self._collect_history:
            self._tick_history.append(metrics)

        if self._first_norm_tick is None and metrics.norm_adoption_rate > 0:
            self._first_norm_tick = self._current_tick
        if self._normative_level_tick is None and metrics.norm_level >= NormLevel.NORMATIVE.value:
            self._normative_level_tick = self._current_tick

        self._current_tick += 1

        # Check convergence
        self._check_convergence(metrics)

        return metrics

    def _collect_metrics(
        self,
        tick_interactions: int,
        tick_successes: int,
        tick_switches: int,
        tick_crystallisations: int = 0,
        tick_dissolutions: int = 0,
        tick_enforcements: int = 0,
    ) -> TickMetrics:
        """
        Collect metrics for current tick.

        Args:
            tick_interactions: Number of interactions this tick
            tick_successes: Number of successful coordinations
            tick_switches: Number of strategy switches
            tick_crystallisations: Number of new norms formed
            tick_dissolutions: Number of norms dissolved
            tick_enforcements: Number of enforcement signals

        Returns:
            TickMetrics instance
        """
        # Strategy distribution
        strategies = [a.current_strategy for a in self._agents]
        count_0 = strategies.count(0)
        count_1 = strategies.count(1)
        total = len(strategies)

        strategy_dist = [count_0 / total, count_1 / total]

        # Majority
        if count_0 >= count_1:
            majority_strategy = 0
            majority_fraction = count_0 / total
        else:
            majority_strategy = 1
            majority_fraction = count_1 / total

        # Agent statistics
        trusts = [a.trust for a in self._agents]
        windows = [a.memory_window for a in self._agents]
        beliefs = [a.belief for a in self._agents]

        # Convergence check
        converged = majority_fraction >= self._convergence_threshold

        # Norm detection (Bicchieri 2006)
        # Compute norm adoption rate for V5 level detection
        norm_adoption_rate = 0.0
        mean_norm_strength = 0.0
        mean_ddm_evidence = 0.0
        total_anomalies = 0
        correct_norm_rate = 0.0
        mean_compliance = 0.0

        if self._enable_normative:
            agents_with_norm = 0
            norm_strength_sum = 0.0
            ddm_evidence_sum = 0.0
            agents_without_norm = 0
            anomaly_sum = 0
            correct_norms = 0
            compliance_sum = 0.0

            for agent in self._agents:
                ns = agent.normative_state
                if ns is not None:
                    compliance_sum += ns.compliance
                    if ns.has_norm:
                        agents_with_norm += 1
                        norm_strength_sum += ns.strength
                        anomaly_sum += ns.anomaly_count
                        if ns.norm == majority_strategy:
                            correct_norms += 1
                    else:
                        agents_without_norm += 1
                        ddm_evidence_sum += ns.evidence

            norm_adoption_rate = agents_with_norm / total if total > 0 else 0.0
            mean_norm_strength = norm_strength_sum / agents_with_norm if agents_with_norm > 0 else 0.0
            mean_ddm_evidence = ddm_evidence_sum / agents_without_norm if agents_without_norm > 0 else 0.0
            total_anomalies = anomaly_sum
            correct_norm_rate = correct_norms / agents_with_norm if agents_with_norm > 0 else 0.0
            mean_compliance = compliance_sum / total if total > 0 else 0.0

        norm_state = self._norm_detector.detect(
            strategies, beliefs, self._current_tick,
            norm_adoption_rate=norm_adoption_rate,
        )

        return TickMetrics(
            tick=self._current_tick,
            strategy_distribution=strategy_dist,
            mean_trust=np.mean(trusts),
            coordination_rate=tick_successes / tick_interactions if tick_interactions > 0 else 0,
            num_interactions=tick_interactions,
            strategy_switches=tick_switches,
            converged=converged,
            majority_strategy=majority_strategy,
            majority_fraction=majority_fraction,
            mean_memory_window=np.mean(windows),
            norm_level=norm_state.level.value,
            belief_error=norm_state.mean_belief_error,
            belief_variance=norm_state.belief_variance,
            # V5 normative metrics
            norm_adoption_rate=norm_adoption_rate,
            mean_norm_strength=mean_norm_strength,
            mean_ddm_evidence=mean_ddm_evidence,
            total_anomalies=total_anomalies,
            norm_crises=tick_dissolutions,
            enforcement_events=tick_enforcements,
            norm_crystallisations=tick_crystallisations,
            correct_norm_rate=correct_norm_rate,
            mean_compliance=mean_compliance,
        )

    def _check_convergence(self, metrics: TickMetrics) -> None:
        """
        Check and track convergence.

        Convergence is reached when the threshold is maintained
        for the required number of consecutive ticks.
        """
        if metrics.converged:
            self._convergence_counter += 1
            if (
                self._convergence_counter >= self._convergence_window
                and self._convergence_tick is None
            ):
                # Mark convergence at the tick where threshold was first reached
                self._convergence_tick = self._current_tick - self._convergence_window + 1
        else:
            self._convergence_counter = 0

    def run(
        self,
        max_ticks: int = 1000,
        early_stop: bool = True,
        verbose: bool = False,
        progress_callback: Optional[callable] = None,
    ) -> SimulationResult:
        """
        Run the simulation.

        Args:
            max_ticks: Maximum number of ticks to run
            early_stop: Stop when convergence is reached
            verbose: Print progress updates
            progress_callback: Called each tick with (tick, metrics)

        Returns:
            SimulationResult with complete simulation data
        """
        for tick in range(max_ticks):
            metrics = self.step()

            if verbose and tick % 100 == 0:
                norm_info = ""
                if self._enable_normative:
                    norm_info = f", norm_adopt={metrics.norm_adoption_rate:.0%}"
                print(
                    f"Tick {tick}: "
                    f"dist={metrics.strategy_distribution}, "
                    f"coord_rate={metrics.coordination_rate:.2f}, "
                    f"mean_trust={metrics.mean_trust:.3f}"
                    f"{norm_info}"
                )

            if progress_callback:
                progress_callback(tick, metrics)

            # Early stopping
            if early_stop and self._convergence_tick is not None:
                if verbose:
                    print(f"Converged at tick {self._convergence_tick}")
                break

        return self._compile_results()

    def _compile_results(self) -> SimulationResult:
        """Compile simulation results."""
        # Final state
        final_metrics = self._last_metrics

        final_state = {
            "final_tick": self._current_tick,
            "converged": self._convergence_tick is not None,
            "convergence_tick": self._convergence_tick,
        }

        if final_metrics:
            final_state.update({
                "final_distribution": final_metrics.strategy_distribution,
                "final_majority_strategy": final_metrics.majority_strategy,
                "final_majority_fraction": final_metrics.majority_fraction,
                "final_mean_trust": final_metrics.mean_trust,
                "final_mean_memory_window": final_metrics.mean_memory_window,
                "final_norm_level": final_metrics.norm_level,
                "final_belief_error": final_metrics.belief_error,
                "final_belief_variance": final_metrics.belief_variance,
                # V5
                "final_norm_adoption_rate": final_metrics.norm_adoption_rate,
                "final_mean_norm_strength": final_metrics.mean_norm_strength,
                "final_mean_compliance": final_metrics.mean_compliance,
                "first_norm_tick": self._first_norm_tick,
                "normative_level_tick": self._normative_level_tick,
            })

        # Agent final states
        agent_states = [agent.get_metrics() for agent in self._agents]

        return SimulationResult(
            config=self._config,
            tick_history=self._tick_history,
            final_state=final_state,
            convergence_tick=self._convergence_tick,
            converged=self._convergence_tick is not None,
            agent_final_states=agent_states,
        )

    def get_history_dataframe(self) -> pd.DataFrame:
        """
        Convert tick history to pandas DataFrame.

        Returns:
            DataFrame with one row per tick
        """
        data = []
        for m in self._tick_history:
            row = {
                "tick": m.tick,
                "strategy_0_fraction": m.strategy_distribution[0],
                "strategy_1_fraction": m.strategy_distribution[1],
                "mean_trust": m.mean_trust,
                "coordination_rate": m.coordination_rate,
                "num_interactions": m.num_interactions,
                "strategy_switches": m.strategy_switches,
                "converged": m.converged,
                "majority_strategy": m.majority_strategy,
                "majority_fraction": m.majority_fraction,
                "mean_memory_window": m.mean_memory_window,
                # Norm detection (Bicchieri 2006)
                "norm_level": m.norm_level,
                "belief_error": m.belief_error,
                "belief_variance": m.belief_variance,
                # V5 normative metrics
                "norm_adoption_rate": m.norm_adoption_rate,
                "mean_norm_strength": m.mean_norm_strength,
                "mean_ddm_evidence": m.mean_ddm_evidence,
                "total_anomalies": m.total_anomalies,
                "norm_crises": m.norm_crises,
                "enforcement_events": m.enforcement_events,
                "norm_crystallisations": m.norm_crystallisations,
                "correct_norm_rate": m.correct_norm_rate,
                "mean_compliance": m.mean_compliance,
            }
            data.append(row)

        return pd.DataFrame(data)

    def save_results(
        self,
        output_dir: str = "data",
        prefix: str = "simulation",
    ) -> Dict[str, str]:
        """
        Save simulation results to files.

        Args:
            output_dir: Directory for output files
            prefix: Filename prefix

        Returns:
            Dict mapping file type to path
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        paths = {}

        # Save tick history as CSV
        df = self.get_history_dataframe()
        csv_path = output_path / f"{prefix}_history.csv"
        df.to_csv(csv_path, index=False)
        paths["history_csv"] = str(csv_path)

        # Save config and final state as JSON
        results = self._compile_results()
        json_data = {
            "config": results.config,
            "final_state": results.final_state,
            "converged": results.converged,
            "convergence_tick": results.convergence_tick,
        }
        json_path = output_path / f"{prefix}_results.json"
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)
        paths["results_json"] = str(json_path)

        return paths

    def reset(self, random_seed: Optional[int] = None) -> None:
        """
        Reset simulation to initial state.

        Args:
            random_seed: New random seed (uses original if None)
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        elif self._config.get("random_seed") is not None:
            np.random.seed(self._config["random_seed"])

        for agent in self._agents:
            agent.reset()

        self._current_tick = 0
        self._tick_history = []
        self._last_metrics = None
        self._convergence_tick = None
        self._convergence_counter = 0
        self._first_norm_tick = None
        self._normative_level_tick = None
        self._shock_active = False
        self._persistent_violator_strategy_by_id = {}
        self._norm_detector.reset()
        self._last_interactions = []

    @property
    def decision_mode(self) -> DecisionMode:
        """Current decision mode."""
        return self._decision_mode

    def __repr__(self) -> str:
        return (
            f"SimulationEnvironment(agents={self._num_agents}, "
            f"tick={self._current_tick}, "
            f"memory={self._config['memory_type']}, "
            f"decision={self._decision_mode.value}, "
            f"normative={self._enable_normative})"
        )
