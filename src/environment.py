"""
Simulation environment for ABM coordination game.

Manages agent interactions, random matching, and data collection.
Supports multiple decision modes for comparative experiments.
Supports observation-based communication for studying norm emergence.
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
    norm_level: int  # 0=NONE, 1=BEHAVIORAL, 2=COGNITIVE, 3=SHARED, 4=INSTITUTIONAL
    belief_error: float  # Mean |agent_belief - true_distribution|
    belief_variance: float  # Variance across agent beliefs


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
    """

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
        """
        # Ensure even number of agents for pairing
        if num_agents % 2 != 0:
            num_agents += 1

        self._num_agents = num_agents
        self._game = game or CoordinationGame()
        self._convergence_threshold = convergence_threshold
        self._convergence_window = convergence_window

        # Convert string to enum if needed
        if isinstance(decision_mode, str):
            decision_mode = DecisionMode(decision_mode)
        self._decision_mode = decision_mode

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
            )
            self._agents.append(agent)

        # Simulation state
        self._current_tick = 0
        self._tick_history: List[TickMetrics] = []
        self._convergence_tick: Optional[int] = None
        self._convergence_counter = 0

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

        # Play the game
        outcome = self._game.play(action1, action2)

        return outcome, pred1, pred2

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

    def step(self) -> TickMetrics:
        """
        Execute one simulation tick.

        Each tick:
        1. Clear observations from previous tick
        2. Random matching of all agents
        3. Each pair plays the coordination game
        4. Distribute observations (if observation_k > 0)
        5. Agents update based on outcomes
        6. Collect metrics including norm level

        Returns:
            Metrics for this tick
        """
        # Clear observations from previous tick
        for agent in self._agents:
            agent.clear_observations()

        # Get random pairs
        pairs = self.random_matching()

        # Track tick statistics
        tick_switches = 0
        tick_successes = 0
        tick_interactions = len(pairs)

        # Record strategies before interactions (for switch counting)
        strategies_before = {a.id: a.current_strategy for a in self._agents}

        # Store all interactions for observation
        tick_interactions_list: List[Tuple[int, int, int, int]] = []

        # Execute all interactions
        for idx1, idx2 in pairs:
            agent1 = self._agents[idx1]
            agent2 = self._agents[idx2]

            # Execute interaction
            outcome, pred1, pred2 = self._execute_interaction(agent1, agent2)

            # Store for observation
            tick_interactions_list.append((idx1, idx2, outcome.strategy1, outcome.strategy2))

            # Update both agents (anonymous: only observe partner's strategy)
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

        # Distribute observations (if enabled)
        if self._observation_k > 0:
            self._distribute_observations(tick_interactions_list)

        self._last_interactions = tick_interactions_list

        # Count strategy switches
        for agent in self._agents:
            if agent.current_strategy != strategies_before[agent.id]:
                tick_switches += 1

        # Collect metrics
        metrics = self._collect_metrics(
            tick_interactions=tick_interactions,
            tick_successes=tick_successes,
            tick_switches=tick_switches,
        )

        self._tick_history.append(metrics)
        self._current_tick += 1

        # Check convergence
        self._check_convergence(metrics)

        return metrics

    def _collect_metrics(
        self,
        tick_interactions: int,
        tick_successes: int,
        tick_switches: int,
    ) -> TickMetrics:
        """
        Collect metrics for current tick.

        Args:
            tick_interactions: Number of interactions this tick
            tick_successes: Number of successful coordinations
            tick_switches: Number of strategy switches

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
        norm_state = self._norm_detector.detect(strategies, beliefs, self._current_tick)

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
                print(
                    f"Tick {tick}: "
                    f"dist={metrics.strategy_distribution}, "
                    f"coord_rate={metrics.coordination_rate:.2f}, "
                    f"mean_trust={metrics.mean_trust:.3f}"
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
        final_metrics = self._tick_history[-1] if self._tick_history else None

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
        self._convergence_tick = None
        self._convergence_counter = 0
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
            f"decision={self._decision_mode.value})"
        )
