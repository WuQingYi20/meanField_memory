"""
Experiment runner for batch simulations and parameter sweeps.

Supports parallel execution, parameter sweeps, and result aggregation.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, asdict
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import time

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment import SimulationEnvironment, SimulationResult


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
    initial_tau: float = 1.0
    tau_min: float = 0.1
    tau_max: float = 2.0
    cooling_rate: float = 0.1
    heating_penalty: float = 0.3
    max_ticks: int = 1000
    random_seed: Optional[int] = None
    convergence_threshold: float = 0.95
    convergence_window: int = 50


@dataclass
class ExperimentResult:
    """Result from a single experiment run."""

    config: ExperimentConfig
    converged: bool
    convergence_tick: Optional[int]
    final_tick: int
    final_majority_strategy: int
    final_majority_fraction: float
    final_mean_tau: float
    final_mean_trust: float
    run_time_seconds: float


def run_single_experiment(config: ExperimentConfig) -> ExperimentResult:
    """
    Run a single experiment with given configuration.

    Args:
        config: Experiment configuration

    Returns:
        ExperimentResult with outcomes
    """
    start_time = time.time()

    env = SimulationEnvironment(
        num_agents=config.num_agents,
        memory_type=config.memory_type,
        memory_size=config.memory_size,
        decay_rate=config.decay_rate,
        dynamic_base=config.dynamic_base,
        dynamic_max=config.dynamic_max,
        initial_tau=config.initial_tau,
        tau_min=config.tau_min,
        tau_max=config.tau_max,
        cooling_rate=config.cooling_rate,
        heating_penalty=config.heating_penalty,
        convergence_threshold=config.convergence_threshold,
        convergence_window=config.convergence_window,
        random_seed=config.random_seed,
    )

    result = env.run(max_ticks=config.max_ticks, early_stop=True, verbose=False)

    run_time = time.time() - start_time

    return ExperimentResult(
        config=config,
        converged=result.converged,
        convergence_tick=result.convergence_tick,
        final_tick=result.final_state.get("final_tick", config.max_ticks),
        final_majority_strategy=result.final_state.get("final_majority_strategy", 0),
        final_majority_fraction=result.final_state.get("final_majority_fraction", 0.5),
        final_mean_tau=result.final_state.get("final_mean_tau", 1.0),
        final_mean_trust=result.final_state.get("final_mean_trust", 0.5),
        run_time_seconds=run_time,
    )


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
    ):
        """
        Initialize experiment runner.

        Args:
            output_dir: Directory for output files
            n_workers: Number of parallel workers
        """
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._n_workers = n_workers
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
            with ProcessPoolExecutor(max_workers=self._n_workers) as executor:
                futures = {
                    executor.submit(run_single_experiment, config): config
                    for config in configs
                }

                iterator = as_completed(futures)
                if progress:
                    iterator = tqdm(iterator, total=len(configs), desc="Running experiments")

                for future in iterator:
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        config = futures[future]
                        print(f"Error in experiment {config.name}: {e}")
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
                "decay_rate": result.config.decay_rate,
                "dynamic_base": result.config.dynamic_base,
                "dynamic_max": result.config.dynamic_max,
                "initial_tau": result.config.initial_tau,
                "cooling_rate": result.config.cooling_rate,
                "heating_penalty": result.config.heating_penalty,
                "random_seed": result.config.random_seed,
                "converged": result.converged,
                "convergence_tick": result.convergence_tick,
                "final_tick": result.final_tick,
                "final_majority_strategy": result.final_majority_strategy,
                "final_majority_fraction": result.final_majority_fraction,
                "final_mean_tau": result.final_mean_tau,
                "final_mean_trust": result.final_mean_trust,
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
                "final_mean_tau": result.final_mean_tau,
                "final_mean_trust": result.final_mean_trust,
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


if __name__ == "__main__":
    # Example usage
    print("Running memory comparison experiment...")
    df = run_memory_comparison(num_agents=50, n_trials=5, n_workers=2)
    print(f"\nResults saved. Total experiments: {len(df)}")
