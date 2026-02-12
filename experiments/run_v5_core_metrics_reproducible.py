"""
Reproducible V5 core metrics pipeline.

Outputs four paper-figure prototypes:
1. Norm coverage over time
2. Coordination rate over time
3. First crystallisation time distribution (histogram + boxplot)
4. Shock robustness (collapse probability + recovery time)
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.environment import SimulationEnvironment


@dataclass
class CoreRunConfig:
    num_agents: int = 100
    max_ticks: int = 500
    n_trials: int = 100
    master_seed: int = 20260212
    shock_tick: int = 250
    shock_epsilons: Tuple[float, ...] = (0.0, 0.05, 0.10, 0.20)
    observation_k: int = 3
    crystal_threshold: float = 3.0
    recovery_ratio: float = 0.9
    collapse_ratio: float = 0.5
    recovery_window: int = 10


def make_trial_seeds(master_seed: int, n_trials: int) -> List[int]:
    rng = np.random.RandomState(master_seed)
    return rng.randint(1, 2**31 - 1, size=n_trials).tolist()


def run_single_trial(
    seed: int,
    cfg: CoreRunConfig,
    shock_epsilon: float = 0.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    env = SimulationEnvironment(
        num_agents=cfg.num_agents,
        memory_type="dynamic",
        decision_mode="cognitive_lockin",
        initial_trust=0.5,
        alpha=0.1,
        beta=0.3,
        enable_normative=True,
        observation_k=cfg.observation_k,
        crystal_threshold=cfg.crystal_threshold,
        random_seed=seed,
        shock_tick=cfg.shock_tick,
        shock_violator_fraction=shock_epsilon,
        collect_history=True,
    )
    result = env.run(max_ticks=cfg.max_ticks, early_stop=False, verbose=False)
    history = env.get_history_dataframe()
    history["trial_seed"] = seed
    history["shock_epsilon"] = shock_epsilon

    agent_states = pd.DataFrame(result.agent_final_states)
    if "first_crystallisation_tick" not in agent_states.columns:
        agent_states["first_crystallisation_tick"] = np.nan
    if "has_norm" not in agent_states.columns:
        agent_states["has_norm"] = False
    agent_states = agent_states[["id", "has_norm", "first_crystallisation_tick"]].copy()
    agent_states["trial_seed"] = seed
    agent_states["shock_epsilon"] = shock_epsilon
    return history, agent_states


def aggregate_over_time(histories: List[pd.DataFrame], value_col: str) -> pd.DataFrame:
    df = pd.concat(histories, ignore_index=True)
    grouped = df.groupby("tick")[value_col]
    out = pd.DataFrame({
        "tick": grouped.mean().index,
        "mean": grouped.mean().values,
        "p10": grouped.quantile(0.10).values,
        "p90": grouped.quantile(0.90).values,
    })
    return out


def compute_shock_outcome(
    history: pd.DataFrame,
    shock_tick: int,
    collapse_ratio: float,
    recovery_ratio: float,
    recovery_window: int,
) -> Dict[str, Optional[float]]:
    if shock_tick <= 0 or shock_tick >= len(history):
        return {"collapsed": False, "collapse_tick": None, "recovery_time": None}

    pre_shock = float(history.loc[history["tick"] == shock_tick - 1, "norm_adoption_rate"].iloc[0])
    collapse_threshold = pre_shock * collapse_ratio
    recovery_threshold = pre_shock * recovery_ratio
    post = history[history["tick"] >= shock_tick].reset_index(drop=True)

    collapse_points = post[post["norm_adoption_rate"] <= collapse_threshold]
    if collapse_points.empty:
        return {"collapsed": False, "collapse_tick": None, "recovery_time": 0.0}

    collapse_tick = int(collapse_points["tick"].iloc[0])
    for start in range(collapse_tick, int(history["tick"].max()) - recovery_window + 2):
        window = history[(history["tick"] >= start) & (history["tick"] < start + recovery_window)]
        if len(window) == recovery_window and bool((window["norm_adoption_rate"] >= recovery_threshold).all()):
            return {
                "collapsed": True,
                "collapse_tick": collapse_tick,
                "recovery_time": float(start - shock_tick),
            }

    return {"collapsed": True, "collapse_tick": collapse_tick, "recovery_time": None}


def fingerprint_dataframe(df: pd.DataFrame, sort_cols: List[str]) -> str:
    ordered = df.sort_values(sort_cols).reset_index(drop=True)
    payload = ordered.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def verify_reproducibility(cfg: CoreRunConfig) -> Dict[str, str]:
    seeds = make_trial_seeds(cfg.master_seed, cfg.n_trials)

    def run_once() -> pd.DataFrame:
        rows = []
        for seed in seeds:
            history, _ = run_single_trial(seed, cfg, shock_epsilon=0.0)
            rows.append({
                "trial_seed": seed,
                "final_norm_adoption_rate": float(history["norm_adoption_rate"].iloc[-1]),
                "mean_coordination_rate": float(history["coordination_rate"].mean()),
                "final_coordination_rate": float(history["coordination_rate"].iloc[-1]),
            })
        return pd.DataFrame(rows)

    first = run_once()
    second = run_once()
    fp1 = fingerprint_dataframe(first, ["trial_seed"])
    fp2 = fingerprint_dataframe(second, ["trial_seed"])
    if fp1 != fp2:
        raise RuntimeError("Reproducibility check failed: same seeds produced different trial statistics.")
    return {"fingerprint_run_1": fp1, "fingerprint_run_2": fp2}


def plot_time_series(df: pd.DataFrame, y_label: str, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df["tick"], df["mean"], linewidth=2)
    ax.fill_between(df["tick"], df["p10"], df["p90"], alpha=0.25)
    ax.set_xlabel("Tick")
    ax.set_ylabel(y_label)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_crystallisation_distribution(crystal_df: pd.DataFrame, out_path: Path) -> None:
    valid = crystal_df["first_crystallisation_tick"].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(valid, bins=30, color="steelblue", edgecolor="black", alpha=0.8)
    axes[0].set_xlabel("First Crystallisation Tick")
    axes[0].set_ylabel("Agent Count")
    axes[0].set_title("Crystallisation Tick Histogram")
    axes[0].grid(alpha=0.3)

    axes[1].boxplot(valid, vert=True, patch_artist=True)
    axes[1].set_ylabel("First Crystallisation Tick")
    axes[1].set_title("Crystallisation Tick Boxplot")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_shock_robustness(summary_df: pd.DataFrame, trial_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(summary_df["shock_epsilon"], summary_df["collapse_probability"], marker="o", linewidth=2)
    axes[0].set_xlabel("Injected Violator Ratio (epsilon)")
    axes[0].set_ylabel("Collapse Probability")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Norm Collapse Probability")
    axes[0].grid(alpha=0.3)

    box_data = []
    labels = []
    for epsilon in sorted(trial_df["shock_epsilon"].unique()):
        series = trial_df[
            (trial_df["shock_epsilon"] == epsilon) & trial_df["recovery_time"].notna()
        ]["recovery_time"]
        box_data.append(series.values if len(series) > 0 else np.array([np.nan]))
        labels.append(f"{epsilon:.2f}")
    try:
        axes[1].boxplot(box_data, tick_labels=labels, patch_artist=True)
    except TypeError:
        axes[1].boxplot(box_data, labels=labels, patch_artist=True)
    axes[1].set_xlabel("Injected Violator Ratio (epsilon)")
    axes[1].set_ylabel("Recovery Time (ticks)")
    axes[1].set_title("Recovery Time Distribution")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def run_pipeline(cfg: CoreRunConfig, output_dir: Path, verify: bool = True) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = make_trial_seeds(cfg.master_seed, cfg.n_trials)

    baseline_histories: List[pd.DataFrame] = []
    baseline_agents: List[pd.DataFrame] = []
    for seed in seeds:
        history, agents = run_single_trial(seed, cfg, shock_epsilon=0.0)
        baseline_histories.append(history)
        baseline_agents.append(agents)

    norm_ts = aggregate_over_time(baseline_histories, "norm_adoption_rate")
    coord_ts = aggregate_over_time(baseline_histories, "coordination_rate")
    crystal_df = pd.concat(baseline_agents, ignore_index=True)

    shock_rows = []
    for epsilon in cfg.shock_epsilons:
        for seed in seeds:
            history, _ = run_single_trial(seed, cfg, shock_epsilon=epsilon)
            outcome = compute_shock_outcome(
                history=history,
                shock_tick=cfg.shock_tick,
                collapse_ratio=cfg.collapse_ratio,
                recovery_ratio=cfg.recovery_ratio,
                recovery_window=cfg.recovery_window,
            )
            shock_rows.append({
                "shock_epsilon": epsilon,
                "trial_seed": seed,
                "collapsed": bool(outcome["collapsed"]),
                "collapse_tick": outcome["collapse_tick"],
                "recovery_time": outcome["recovery_time"],
            })
    shock_trial_df = pd.DataFrame(shock_rows)
    shock_summary = shock_trial_df.groupby("shock_epsilon").agg(
        collapse_probability=("collapsed", "mean"),
        mean_recovery_time=("recovery_time", "mean"),
        median_recovery_time=("recovery_time", "median"),
    ).reset_index()

    norm_ts.to_csv(output_dir / "norm_coverage_over_time.csv", index=False)
    coord_ts.to_csv(output_dir / "coordination_over_time.csv", index=False)
    crystal_df.to_csv(output_dir / "first_crystallisation_ticks.csv", index=False)
    shock_trial_df.to_csv(output_dir / "shock_robustness_trials.csv", index=False)
    shock_summary.to_csv(output_dir / "shock_robustness_summary.csv", index=False)

    plot_time_series(
        norm_ts,
        y_label="Norm Coverage (r_i != None)",
        title="Norm Coverage Over Time",
        out_path=output_dir / "fig_1_norm_coverage.png",
    )
    plot_time_series(
        coord_ts,
        y_label="Match Rate",
        title="Coordination Rate Over Time",
        out_path=output_dir / "fig_2_coordination_rate.png",
    )
    plot_crystallisation_distribution(
        crystal_df,
        out_path=output_dir / "fig_3_crystallisation_distribution.png",
    )
    plot_shock_robustness(
        shock_summary,
        shock_trial_df,
        out_path=output_dir / "fig_4_shock_robustness.png",
    )

    repro_info = None
    if verify:
        repro_info = verify_reproducibility(cfg)

    report = {
        "tick_update_order": list(SimulationEnvironment.TICK_UPDATE_ORDER),
        "n_trials": cfg.n_trials,
        "master_seed": cfg.master_seed,
        "shock_tick": cfg.shock_tick,
        "shock_epsilons": list(cfg.shock_epsilons),
        "reproducibility": repro_info,
    }
    with open(output_dir / "run_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return {
        "output_dir": str(output_dir),
        "report": str(output_dir / "run_report.json"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run reproducible V5 core metrics pipeline.")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--ticks", type=int, default=500)
    parser.add_argument("--agents", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260212)
    parser.add_argument("--shock-tick", type=int, default=250)
    parser.add_argument("--out", type=str, default="data/v5_core_metrics")
    parser.add_argument("--no-verify", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = CoreRunConfig(
        num_agents=args.agents,
        max_ticks=args.ticks,
        n_trials=args.trials,
        master_seed=args.seed,
        shock_tick=args.shock_tick,
    )
    outputs = run_pipeline(cfg, output_dir=Path(args.out), verify=not args.no_verify)
    print(f"Done. Outputs in: {outputs['output_dir']}")
    print(f"Report: {outputs['report']}")


if __name__ == "__main__":
    main()
