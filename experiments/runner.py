"""
Experiment runner for batch simulations and parameter sweeps.

Supports parallel execution, parameter sweeps, and result aggregation.
Supports multiple decision modes: cognitive_lockin, dual_feedback, epsilon_greedy.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, asdict
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import time

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment import SimulationEnvironment, SimulationResult
from src.decision import DecisionMode


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""

    name: str
    num_agents: int = 100
    memory_type: str = "fixed"
    memory_size: int = 5
    decay_rate: float = 0.9
    dynamic_base: int = 2
    dynamic_max: int = 6
    # Decision mode
    decision_mode: str = "cognitive_lockin"  # Use string for serialization
    # DUAL_FEEDBACK params
    initial_tau: float = 1.0
    tau_min: float = 0.1
    tau_max: float = 2.0
    cooling_rate: float = 0.1
    heating_penalty: float = 0.3
    # COGNITIVE_LOCKIN/EPSILON_GREEDY params
    initial_trust: float = 0.5
    alpha: float = 0.1
    beta: float = 0.3
    # EPSILON_GREEDY specific
    exploration_mode: str = "random"
    # Simulation settings
    max_ticks: int = 1000
    random_seed: Optional[int] = None
    convergence_threshold: float = 0.95
    convergence_window: int = 50
    # Optional initial condition override
    initial_strategy_0_fraction: Optional[float] = None
    # V5: Normative memory settings
    enable_normative: bool = False
    observation_k: int = 0
    ddm_noise: float = 0.1
    crystal_threshold: float = 3.0
    normative_initial_strength: float = 0.8
    crisis_threshold: int = 10
    crisis_decay: float = 0.3
    min_strength: float = 0.1
    enforce_threshold: float = 0.7
    compliance_exponent: float = 2.0
    signal_amplification: float = 2.0


@dataclass
class ExperimentResult:
    """Result from a single experiment run."""

    config: ExperimentConfig
    converged: bool
    convergence_tick: Optional[int]
    final_tick: int
    final_majority_strategy: int
    final_majority_fraction: float
    final_mean_trust: float
    final_mean_memory_window: float
    run_time_seconds: float
    # V5: Normative results
    final_norm_level: int = 0
    final_norm_adoption_rate: float = 0.0
    final_mean_norm_strength: float = 0.0
    # Norm emergence timing
    first_norm_tick: Optional[int] = None
    normative_level_tick: Optional[int] = None


def run_single_experiment(config: ExperimentConfig) -> ExperimentResult:
    """
    Run a single experiment with given configuration.

    Args:
        config: Experiment configuration

    Returns:
        ExperimentResult with outcomes
    """
    start_time = time.time()

    # Convert string to DecisionMode enum
    decision_mode = DecisionMode(config.decision_mode)

    env = SimulationEnvironment(
        num_agents=config.num_agents,
        memory_type=config.memory_type,
        memory_size=config.memory_size,
        decay_rate=config.decay_rate,
        dynamic_base=config.dynamic_base,
        dynamic_max=config.dynamic_max,
        decision_mode=decision_mode,
        # DUAL_FEEDBACK params
        initial_tau=config.initial_tau,
        tau_min=config.tau_min,
        tau_max=config.tau_max,
        cooling_rate=config.cooling_rate,
        heating_penalty=config.heating_penalty,
        # COGNITIVE_LOCKIN/EPSILON_GREEDY params
        initial_trust=config.initial_trust,
        alpha=config.alpha,
        beta=config.beta,
        exploration_mode=config.exploration_mode,
        # Other settings
        convergence_threshold=config.convergence_threshold,
        convergence_window=config.convergence_window,
        random_seed=config.random_seed,
        # V5: Normative memory
        observation_k=config.observation_k,
        enable_normative=config.enable_normative,
        ddm_noise=config.ddm_noise,
        crystal_threshold=config.crystal_threshold,
        normative_initial_strength=config.normative_initial_strength,
        crisis_threshold=config.crisis_threshold,
        crisis_decay=config.crisis_decay,
        min_strength=config.min_strength,
        enforce_threshold=config.enforce_threshold,
        compliance_exponent=config.compliance_exponent,
        signal_amplification=config.signal_amplification,
        collect_history=False,
    )

    # Optional initial condition override: set exact strategy-0 fraction at tick 0.
    if config.initial_strategy_0_fraction is not None:
        if not 0.0 <= config.initial_strategy_0_fraction <= 1.0:
            raise ValueError("initial_strategy_0_fraction must be in [0, 1]")

        n = env.num_agents
        n_strategy_0 = int(round(config.initial_strategy_0_fraction * n))
        rng = np.random.RandomState(config.random_seed)
        perm = np.arange(n)
        rng.shuffle(perm)
        strategy_0_ids = set(perm[:n_strategy_0].tolist())

        for agent in env.agents:
            initial_strategy = 0 if agent.id in strategy_0_ids else 1
            agent.reset(initial_strategy=initial_strategy)

    result = env.run(max_ticks=config.max_ticks, early_stop=True, verbose=False)

    run_time = time.time() - start_time

    first_norm_tick = result.final_state.get("first_norm_tick")
    normative_level_tick = result.final_state.get("normative_level_tick")

    return ExperimentResult(
        config=config,
        converged=result.converged,
        convergence_tick=result.convergence_tick,
        final_tick=result.final_state.get("final_tick", config.max_ticks),
        final_majority_strategy=result.final_state.get("final_majority_strategy", 0),
        final_majority_fraction=result.final_state.get("final_majority_fraction", 0.5),
        final_mean_trust=result.final_state.get("final_mean_trust", 0.5),
        final_mean_memory_window=result.final_state.get("final_mean_memory_window", 5.0),
        run_time_seconds=run_time,
        # V5
        final_norm_level=result.final_state.get("final_norm_level", 0),
        final_norm_adoption_rate=result.final_state.get("final_norm_adoption_rate", 0.0),
        final_mean_norm_strength=result.final_state.get("final_mean_norm_strength", 0.0),
        first_norm_tick=first_norm_tick,
        normative_level_tick=normative_level_tick,
    )


def run_experiment_batch(configs: List[ExperimentConfig]) -> List[ExperimentResult]:
    """Run a batch of experiments in a single worker process."""
    batch_results: List[ExperimentResult] = []
    for config in configs:
        try:
            batch_results.append(run_single_experiment(config))
        except Exception as e:
            print(f"Error in experiment {config.name}: {e}")
    return batch_results


class ExperimentRunner:
    """
    Batch experiment runner with parallel execution support.

    Handles:
    - Running multiple configurations
    - Parallel execution
    - Result aggregation
    - Output saving
    """

    def __init__(
        self,
        output_dir: str = "data/experiments",
        n_workers: int = 4,
        batch_size: int = 8,
    ):
        """
        Initialize experiment runner.

        Args:
            output_dir: Directory for output files
            n_workers: Number of parallel workers
            batch_size: Number of experiment configs per submitted process task
        """
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._n_workers = n_workers
        self._batch_size = max(1, batch_size)
        self._results: List[ExperimentResult] = []

    def run_experiments(
        self,
        configs: List[ExperimentConfig],
        parallel: bool = True,
        progress: bool = True,
    ) -> List[ExperimentResult]:
        """
        Run multiple experiments.

        Args:
            configs: List of experiment configurations
            parallel: Use parallel execution
            progress: Show progress bar

        Returns:
            List of experiment results
        """
        results = []

        if parallel and self._n_workers > 1:
            config_batches = [
                configs[i:i + self._batch_size]
                for i in range(0, len(configs), self._batch_size)
            ]
            with ProcessPoolExecutor(max_workers=self._n_workers) as executor:
                futures = {
                    executor.submit(run_experiment_batch, batch): batch
                    for batch in config_batches
                }

                iterator = as_completed(futures)
                pbar = tqdm(total=len(configs), desc="Running experiments") if progress else None

                for future in iterator:
                    try:
                        batch_results = future.result()
                        results.extend(batch_results)
                        if pbar is not None:
                            pbar.update(len(batch_results))
                    except Exception as e:
                        batch = futures[future]
                        names = ", ".join(c.name for c in batch[:3])
                        suffix = "..." if len(batch) > 3 else ""
                        print(f"Error in experiment batch ({names}{suffix}): {e}")
                if pbar is not None:
                    pbar.close()
        else:
            iterator = configs
            if progress:
                iterator = tqdm(configs, desc="Running experiments")

            for config in iterator:
                try:
                    result = run_single_experiment(config)
                    results.append(result)
                except Exception as e:
                    print(f"Error in experiment {config.name}: {e}")

        self._results.extend(results)
        return results

    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Convert results to pandas DataFrame.

        Returns:
            DataFrame with one row per experiment
        """
        rows = []
        for result in self._results:
            row = {
                "name": result.config.name,
                "num_agents": result.config.num_agents,
                "memory_type": result.config.memory_type,
                "memory_size": result.config.memory_size,
                "decision_mode": result.config.decision_mode,
                "initial_trust": result.config.initial_trust,
                "alpha": result.config.alpha,
                "beta": result.config.beta,
                "observation_k": result.config.observation_k,
                "enable_normative": result.config.enable_normative,
                "initial_strategy_0_fraction": result.config.initial_strategy_0_fraction,
                "random_seed": result.config.random_seed,
                "converged": result.converged,
                "convergence_tick": result.convergence_tick,
                "final_tick": result.final_tick,
                "final_majority_strategy": result.final_majority_strategy,
                "final_majority_fraction": result.final_majority_fraction,
                "final_mean_trust": result.final_mean_trust,
                "final_mean_memory_window": result.final_mean_memory_window,
                "final_norm_level": result.final_norm_level,
                "final_norm_adoption_rate": result.final_norm_adoption_rate,
                "final_mean_norm_strength": result.final_mean_norm_strength,
                "first_norm_tick": result.first_norm_tick,
                "normative_level_tick": result.normative_level_tick,
                "run_time_seconds": result.run_time_seconds,
            }
            rows.append(row)

        return pd.DataFrame(rows)

    def save_results(self, filename: str = "experiment_results") -> Dict[str, str]:
        """
        Save results to files.

        Args:
            filename: Base filename

        Returns:
            Dict of output file paths
        """
        paths = {}

        # Save as CSV
        df = self.get_results_dataframe()
        csv_path = self._output_dir / f"{filename}.csv"
        df.to_csv(csv_path, index=False)
        paths["csv"] = str(csv_path)

        # Save as JSON
        json_data = []
        for result in self._results:
            item = {
                "config": asdict(result.config),
                "converged": result.converged,
                "convergence_tick": result.convergence_tick,
                "final_tick": result.final_tick,
                "final_majority_strategy": result.final_majority_strategy,
                "final_majority_fraction": result.final_majority_fraction,
                "final_mean_trust": result.final_mean_trust,
                "final_mean_memory_window": result.final_mean_memory_window,
                "final_norm_level": result.final_norm_level,
                "final_norm_adoption_rate": result.final_norm_adoption_rate,
                "final_mean_norm_strength": result.final_mean_norm_strength,
                "first_norm_tick": result.first_norm_tick,
                "normative_level_tick": result.normative_level_tick,
                "run_time_seconds": result.run_time_seconds,
            }
            json_data.append(item)

        json_path = self._output_dir / f"{filename}.json"
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)
        paths["json"] = str(json_path)

        return paths

    def clear_results(self) -> None:
        """Clear stored results."""
        self._results.clear()

    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Compute summary statistics across all results.

        Returns:
            Dict with summary statistics
        """
        if not self._results:
            return {}

        df = self.get_results_dataframe()

        return {
            "total_experiments": len(df),
            "convergence_rate": df["converged"].mean(),
            "mean_convergence_tick": df[df["converged"]]["convergence_tick"].mean(),
            "std_convergence_tick": df[df["converged"]]["convergence_tick"].std(),
            "mean_final_majority_fraction": df["final_majority_fraction"].mean(),
            "mean_final_trust": df["final_mean_trust"].mean(),
            "total_run_time": df["run_time_seconds"].sum(),
        }


def run_parameter_sweep(
    param_name: str,
    param_values: List[Any],
    base_config: Optional[Dict[str, Any]] = None,
    n_trials: int = 10,
    output_dir: str = "data/experiments",
    n_workers: int = 4,
) -> pd.DataFrame:
    """
    Run parameter sweep experiment.

    Args:
        param_name: Parameter to sweep (e.g., 'num_agents', 'memory_type')
        param_values: Values to test
        base_config: Base configuration dict
        n_trials: Number of trials per value
        output_dir: Output directory
        n_workers: Parallel workers

    Returns:
        DataFrame with results
    """
    base_config = base_config or {}

    configs = []
    for value in param_values:
        for trial in range(n_trials):
            config_dict = {
                "name": f"{param_name}={value}_trial{trial}",
                param_name: value,
                "random_seed": trial * 1000 + hash(str(value)) % 1000,
                **base_config,
            }
            configs.append(ExperimentConfig(**config_dict))

    runner = ExperimentRunner(output_dir=output_dir, n_workers=n_workers)
    runner.run_experiments(configs)

    df = runner.get_results_dataframe()
    runner.save_results(f"sweep_{param_name}")

    return df


def run_memory_comparison(
    num_agents: int = 100,
    max_ticks: int = 1000,
    n_trials: int = 10,
    output_dir: str = "data/experiments",
    n_workers: int = 4,
) -> pd.DataFrame:
    """
    Compare different memory types.

    Args:
        num_agents: Number of agents
        max_ticks: Maximum ticks
        n_trials: Trials per memory type
        output_dir: Output directory
        n_workers: Parallel workers

    Returns:
        DataFrame with comparison results
    """
    memory_types = ["fixed", "decay", "dynamic"]

    configs = []
    for memory_type in memory_types:
        for trial in range(n_trials):
            config = ExperimentConfig(
                name=f"{memory_type}_trial{trial}",
                num_agents=num_agents,
                memory_type=memory_type,
                max_ticks=max_ticks,
                random_seed=trial,
            )
            configs.append(config)

    runner = ExperimentRunner(output_dir=output_dir, n_workers=n_workers)
    runner.run_experiments(configs)

    df = runner.get_results_dataframe()
    runner.save_results("memory_comparison")

    # Print summary
    print("\n=== Memory Type Comparison ===")
    summary = df.groupby("memory_type").agg({
        "converged": "mean",
        "convergence_tick": "mean",
        "final_majority_fraction": "mean",
        "final_mean_trust": "mean",
    }).round(3)
    print(summary)

    return df


def run_agent_count_sweep(
    agent_counts: List[int] = [10, 50, 100, 200],
    memory_type: str = "fixed",
    max_ticks: int = 1000,
    n_trials: int = 10,
    output_dir: str = "data/experiments",
    n_workers: int = 4,
) -> pd.DataFrame:
    """
    Sweep agent counts to analyze scaling behavior.

    Args:
        agent_counts: List of agent counts to test
        memory_type: Memory type to use
        max_ticks: Maximum ticks
        n_trials: Trials per count
        output_dir: Output directory
        n_workers: Parallel workers

    Returns:
        DataFrame with results
    """
    configs = []
    for count in agent_counts:
        for trial in range(n_trials):
            config = ExperimentConfig(
                name=f"agents{count}_trial{trial}",
                num_agents=count,
                memory_type=memory_type,
                max_ticks=max_ticks,
                random_seed=trial,
            )
            configs.append(config)

    runner = ExperimentRunner(output_dir=output_dir, n_workers=n_workers)
    runner.run_experiments(configs)

    df = runner.get_results_dataframe()
    runner.save_results(f"agent_count_sweep_{memory_type}")

    # Print summary
    print(f"\n=== Agent Count Sweep ({memory_type}) ===")
    summary = df.groupby("num_agents").agg({
        "converged": "mean",
        "convergence_tick": ["mean", "std"],
        "final_majority_fraction": "mean",
    }).round(3)
    print(summary)

    return df


def run_decision_mode_comparison(
    num_agents: int = 100,
    memory_type: str = "dynamic",
    max_ticks: int = 1000,
    n_trials: int = 10,
    output_dir: str = "data/experiments",
    n_workers: int = 4,
) -> pd.DataFrame:
    """
    Compare different decision modes.

    Args:
        num_agents: Number of agents
        memory_type: Memory type to use
        max_ticks: Maximum ticks
        n_trials: Trials per decision mode
        output_dir: Output directory
        n_workers: Parallel workers

    Returns:
        DataFrame with comparison results
    """
    decision_modes = ["cognitive_lockin", "dual_feedback", "epsilon_greedy"]

    configs = []
    for mode in decision_modes:
        for trial in range(n_trials):
            config = ExperimentConfig(
                name=f"{mode}_trial{trial}",
                num_agents=num_agents,
                memory_type=memory_type,
                decision_mode=mode,
                max_ticks=max_ticks,
                random_seed=trial,
            )
            configs.append(config)

    runner = ExperimentRunner(output_dir=output_dir, n_workers=n_workers)
    runner.run_experiments(configs)

    df = runner.get_results_dataframe()
    runner.save_results("decision_mode_comparison")

    # Print summary
    print("\n=== Decision Mode Comparison ===")
    summary = df.groupby("decision_mode").agg({
        "converged": "mean",
        "convergence_tick": "mean",
        "final_majority_fraction": "mean",
        "final_mean_trust": "mean",
    }).round(3)
    print(summary)

    return df


def run_full_comparison(
    num_agents: int = 100,
    max_ticks: int = 1000,
    n_trials: int = 10,
    output_dir: str = "data/experiments",
    n_workers: int = 4,
) -> pd.DataFrame:
    """
    Compare all combinations of memory types and decision modes.

    Args:
        num_agents: Number of agents
        max_ticks: Maximum ticks
        n_trials: Trials per configuration
        output_dir: Output directory
        n_workers: Parallel workers

    Returns:
        DataFrame with comparison results
    """
    memory_types = ["fixed", "decay", "dynamic"]
    decision_modes = ["cognitive_lockin", "dual_feedback", "epsilon_greedy"]

    configs = []
    for memory_type in memory_types:
        for mode in decision_modes:
            for trial in range(n_trials):
                config = ExperimentConfig(
                    name=f"{memory_type}_{mode}_trial{trial}",
                    num_agents=num_agents,
                    memory_type=memory_type,
                    decision_mode=mode,
                    max_ticks=max_ticks,
                    random_seed=trial,
                )
                configs.append(config)

    runner = ExperimentRunner(output_dir=output_dir, n_workers=n_workers)
    runner.run_experiments(configs)

    df = runner.get_results_dataframe()
    runner.save_results("full_comparison")

    # Print summary
    print("\n=== Full Comparison (Memory x Decision Mode) ===")
    summary = df.groupby(["memory_type", "decision_mode"]).agg({
        "converged": "mean",
        "convergence_tick": "mean",
        "final_majority_fraction": "mean",
        "final_mean_trust": "mean",
    }).round(3)
    print(summary)

    return df


def run_trust_parameter_comparison(
    num_agents: int = 100,
    memory_type: str = "dynamic",
    max_ticks: int = 1000,
    n_trials: int = 10,
    output_dir: str = "data/experiments",
    n_workers: int = 4,
) -> pd.DataFrame:
    """
    Compare different alpha/beta parameter settings for CognitiveLockIn mode.

    Tests:
    - Asymmetric (default): alpha=0.1, beta=0.3 (Slovic 1993)
    - Symmetric: alpha=0.2, beta=0.2 (faster convergence)
    - Sensitive: alpha=0.3, beta=0.4 (high responsiveness)

    Args:
        num_agents: Number of agents
        memory_type: Memory type to use
        max_ticks: Maximum ticks
        n_trials: Trials per configuration
        output_dir: Output directory
        n_workers: Parallel workers

    Returns:
        DataFrame with comparison results
    """
    parameter_sets = [
        ("asymmetric", 0.1, 0.3),   # Default (Slovic 1993)
        ("symmetric", 0.2, 0.2),    # Fast convergence
        ("sensitive", 0.3, 0.4),    # High responsiveness
    ]

    configs = []
    for name, alpha, beta in parameter_sets:
        for trial in range(n_trials):
            config = ExperimentConfig(
                name=f"{name}_trial{trial}",
                num_agents=num_agents,
                memory_type=memory_type,
                decision_mode="cognitive_lockin",
                alpha=alpha,
                beta=beta,
                max_ticks=max_ticks,
                random_seed=trial,
            )
            configs.append(config)

    runner = ExperimentRunner(output_dir=output_dir, n_workers=n_workers)
    runner.run_experiments(configs)

    df = runner.get_results_dataframe()
    runner.save_results("trust_parameter_comparison")

    # Print summary
    print("\n=== Trust Parameter Comparison ===")

    # Extract parameter type from name
    df["param_type"] = df["name"].apply(lambda x: x.rsplit("_trial", 1)[0])

    summary = df.groupby("param_type").agg({
        "converged": "mean",
        "convergence_tick": "mean",
        "final_majority_fraction": "mean",
        "final_mean_trust": "mean",
    }).round(3)
    print(summary)

    return df


def run_v5_factorial(
    num_agents: int = 100,
    max_ticks: int = 1000,
    n_trials: int = 10,
    observation_k: int = 3,
    output_dir: str = "data/experiments",
    n_workers: int = 4,
) -> pd.DataFrame:
    """
    Run 2x2 factorial experiment: memory_type × normative_enabled.

    Conditions:
    1. fixed memory, normative off  (experience-only baseline)
    2. fixed memory, normative on   (normative-only)
    3. dynamic memory, normative off (adaptive memory only)
    4. dynamic memory, normative on  (full dual-memory V5)

    Args:
        num_agents: Number of agents
        max_ticks: Maximum ticks
        n_trials: Trials per condition
        observation_k: Number of observed interactions per tick
        output_dir: Output directory
        n_workers: Parallel workers

    Returns:
        DataFrame with comparison results
    """
    conditions = [
        ("fixed", False),
        ("fixed", True),
        ("dynamic", False),
        ("dynamic", True),
    ]

    configs = []
    for memory_type, normative in conditions:
        label = f"{memory_type}_norm{'On' if normative else 'Off'}"
        for trial in range(n_trials):
            config = ExperimentConfig(
                name=f"{label}_trial{trial}",
                num_agents=num_agents,
                memory_type=memory_type,
                decision_mode="cognitive_lockin",
                enable_normative=normative,
                observation_k=observation_k if normative else 0,
                max_ticks=max_ticks,
                random_seed=trial,
            )
            configs.append(config)

    runner = ExperimentRunner(output_dir=output_dir, n_workers=n_workers)
    runner.run_experiments(configs)

    df = runner.get_results_dataframe()
    runner.save_results("v5_factorial")

    # Print summary
    print("\n=== V5 Factorial: Memory Type x Normative ===")
    df["condition"] = df["name"].apply(lambda x: x.rsplit("_trial", 1)[0])
    summary = df.groupby("condition").agg({
        "converged": "mean",
        "convergence_tick": "mean",
        "final_majority_fraction": "mean",
        "final_mean_trust": "mean",
        "final_norm_adoption_rate": "mean",
        "final_mean_norm_strength": "mean",
    }).round(3)
    print(summary)

    return df


def run_cognitive_lockin_scan(
    num_agents: int = 100,
    max_ticks: int = 10000,
    n_trials: int = 3,
    output_dir: str = "data/experiments",
    n_workers: int = 4,
    initial_fractions: Optional[List[float]] = None,
    alpha_beta_pairs: Optional[List[Tuple[float, float]]] = None,
    observation_k_values: Optional[List[int]] = None,
    crystal_threshold_values: Optional[List[float]] = None,
    memory_types: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Comprehensive scan for cognitive_lockin mode with normative memory enabled.

    Scans initial strategy ratio and key cognitive/normative parameters, and
    records norm emergence timing:
    - first_norm_tick: first tick with norm_adoption_rate > 0
    - normative_level_tick: first tick reaching norm level >= 4
    """
    initial_fractions = initial_fractions or [0.5, 0.6, 0.7, 0.8]
    alpha_beta_pairs = alpha_beta_pairs or [(0.1, 0.3), (0.2, 0.2), (0.05, 0.3)]
    observation_k_values = observation_k_values or [1, 3, 5]
    crystal_threshold_values = crystal_threshold_values or [2.0, 3.0]
    memory_types = memory_types or ["dynamic"]

    configs: List[ExperimentConfig] = []
    for memory_type, frac, (alpha, beta), obs_k, crystal in product(
        memory_types,
        initial_fractions,
        alpha_beta_pairs,
        observation_k_values,
        crystal_threshold_values,
    ):
        for trial in range(n_trials):
            seed = (
                trial * 10000
                + int(round(frac * 1000))
                + int(round(alpha * 1000))
                + int(round(beta * 1000))
                + obs_k * 17
                + int(round(crystal * 100))
            )
            name = (
                f"cogscan_{memory_type}_f{frac:.2f}"
                f"_a{alpha:.2f}_b{beta:.2f}_k{obs_k}_c{crystal:.1f}_t{trial}"
            )
            configs.append(
                ExperimentConfig(
                    name=name,
                    num_agents=num_agents,
                    memory_type=memory_type,
                    decision_mode="cognitive_lockin",
                    initial_trust=0.5,
                    alpha=alpha,
                    beta=beta,
                    enable_normative=True,
                    observation_k=obs_k,
                    crystal_threshold=crystal,
                    max_ticks=max_ticks,
                    initial_strategy_0_fraction=frac,
                    random_seed=seed,
                )
            )

    runner = ExperimentRunner(output_dir=output_dir, n_workers=n_workers)
    runner.run_experiments(configs)

    df = runner.get_results_dataframe()
    runner.save_results("cognitive_lockin_scan")

    print("\n=== Cognitive-Lockin Scan Summary ===")
    summary = df.groupby(["memory_type", "initial_strategy_0_fraction"]).agg({
        "converged": "mean",
        "convergence_tick": "mean",
        "first_norm_tick": "mean",
        "normative_level_tick": "mean",
        "final_norm_adoption_rate": "mean",
        "final_majority_fraction": "mean",
    }).round(3)
    print(summary)

    return df


def run_cognitive_lockin_pure_scan(
    num_agents: int = 100,
    max_ticks: int = 10000,
    n_trials: int = 5,
    output_dir: str = "data/experiments",
    n_workers: int = 4,
    initial_fractions: Optional[List[float]] = None,
    alpha_beta_pairs: Optional[List[Tuple[float, float]]] = None,
    initial_trust_values: Optional[List[float]] = None,
    memory_types: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Comprehensive pure cognitive-lockin scan (normative mechanism OFF).

    Focuses on:
    - initial strategy ratio effects
    - trust dynamics parameters (alpha, beta, initial_trust)
    - memory mechanism effects under the same decision mode
    """
    initial_fractions = initial_fractions or [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    alpha_beta_pairs = alpha_beta_pairs or [
        (0.05, 0.20),
        (0.10, 0.30),
        (0.20, 0.20),
        (0.05, 0.40),
    ]
    initial_trust_values = initial_trust_values or [0.30, 0.50]
    memory_types = memory_types or ["fixed", "dynamic", "decay"]

    configs: List[ExperimentConfig] = []
    for memory_type, frac, (alpha, beta), initial_trust in product(
        memory_types,
        initial_fractions,
        alpha_beta_pairs,
        initial_trust_values,
    ):
        for trial in range(n_trials):
            seed = (
                trial * 100000
                + int(round(frac * 1000)) * 11
                + int(round(alpha * 1000)) * 13
                + int(round(beta * 1000)) * 17
                + int(round(initial_trust * 1000)) * 19
                + {"fixed": 1, "dynamic": 2, "decay": 3}[memory_type]
            )
            name = (
                f"purecog_{memory_type}_f{frac:.2f}"
                f"_a{alpha:.2f}_b{beta:.2f}_t0{initial_trust:.2f}_trial{trial}"
            )
            configs.append(
                ExperimentConfig(
                    name=name,
                    num_agents=num_agents,
                    memory_type=memory_type,
                    memory_size=5,
                    decay_rate=0.9,
                    dynamic_base=2,
                    dynamic_max=6,
                    decision_mode="cognitive_lockin",
                    initial_trust=initial_trust,
                    alpha=alpha,
                    beta=beta,
                    enable_normative=False,
                    observation_k=0,
                    max_ticks=max_ticks,
                    initial_strategy_0_fraction=frac,
                    random_seed=seed,
                )
            )

    runner = ExperimentRunner(output_dir=output_dir, n_workers=n_workers)
    runner.run_experiments(configs)

    df = runner.get_results_dataframe()
    runner.save_results("cognitive_lockin_pure_scan_full")

    print("\n=== Pure Cognitive-Lockin Full Scan Summary ===")
    summary = df.groupby(["memory_type", "initial_strategy_0_fraction"]).agg({
        "converged": "mean",
        "convergence_tick": "mean",
        "final_majority_fraction": "mean",
        "final_mean_trust": "mean",
        "run_time_seconds": "mean",
    }).round(3)
    print(summary)

    return df


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--trust":
        print("Running trust parameter comparison...")
        df = run_trust_parameter_comparison(num_agents=50, n_trials=5, n_workers=2)
    elif len(sys.argv) > 1 and sys.argv[1] == "--v5":
        print("Running V5 factorial experiment...")
        df = run_v5_factorial(num_agents=50, n_trials=5, n_workers=2)
    elif len(sys.argv) > 1 and sys.argv[1] == "--cogscan":
        print("Running cognitive-lockin comprehensive scan...")
        df = run_cognitive_lockin_scan(num_agents=100, max_ticks=10000, n_trials=2, n_workers=4)
    elif len(sys.argv) > 1 and sys.argv[1] == "--pure-scan":
        print("Running pure cognitive-lockin full scan...")
        df = run_cognitive_lockin_pure_scan(num_agents=100, max_ticks=10000, n_trials=5, n_workers=4)
    else:
        print("Running decision mode comparison experiment...")
        df = run_decision_mode_comparison(num_agents=50, n_trials=5, n_workers=2)

    print(f"\nResults saved. Total experiments: {len(df)}")
