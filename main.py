"""
Main entry point for ABM Memory & Norm Formation Simulation.

This script provides a command-line interface for running simulations
and experiments.

Supports multiple decision modes:
- cognitive_lockin: Probability matching with Trust (cognitive only)
- dual_feedback: Original τ-based softmax (two feedback loops)
- epsilon_greedy: Best response with exploration

Usage:
    python main.py                             # Run default simulation
    python main.py --mode cognitive_lockin     # Use cognitive lock-in
    python main.py --mode dual_feedback        # Use dual feedback
    python main.py --memory dynamic            # Run with dynamic memory
    python main.py --experiment                # Run comparison experiment
    python main.py --dashboard                 # Launch interactive dashboard
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from config import SimulationConfig, DEFAULT_CONFIG, QUICK_TEST_CONFIG
from src.environment import SimulationEnvironment
from src.decision import DecisionMode
from visualization.static_plots import create_summary_figure, save_figure


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="ABM Memory & Norm Formation Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                                # Default (cognitive_lockin)
    python main.py --mode cognitive_lockin        # Probability matching
    python main.py --mode dual_feedback           # τ-based softmax
    python main.py --mode epsilon_greedy          # Best response + exploration
    python main.py --agents 50 --ticks 500
    python main.py --memory decay --decay-rate 0.8
    python main.py --memory dynamic --dynamic-max 6
    python main.py --experiment                   # Compare memory types
    python main.py --dashboard                    # Interactive dashboard
        """,
    )

    # Mode selection
    parser.add_argument(
        "--experiment",
        action="store_true",
        help="Run comparison experiment across memory types",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Launch interactive Streamlit dashboard",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick test (10 agents, 100 ticks)",
    )

    # Agent settings
    parser.add_argument(
        "--agents", "-n",
        type=int,
        default=100,
        help="Number of agents (default: 100)",
    )

    # Memory settings
    parser.add_argument(
        "--memory", "-m",
        choices=["fixed", "decay", "dynamic"],
        default="fixed",
        help="Memory type (default: fixed)",
    )
    parser.add_argument(
        "--memory-size",
        type=int,
        default=5,
        help="Memory size for fixed memory (default: 5)",
    )
    parser.add_argument(
        "--decay-rate",
        type=float,
        default=0.9,
        help="Decay rate for decay memory (default: 0.9)",
    )
    parser.add_argument(
        "--dynamic-base",
        type=int,
        default=2,
        help="Base window for dynamic memory (default: 2)",
    )
    parser.add_argument(
        "--dynamic-max",
        type=int,
        default=6,
        help="Max window for dynamic memory (default: 6)",
    )

    # Decision mode
    parser.add_argument(
        "--mode",
        choices=["cognitive_lockin", "dual_feedback", "epsilon_greedy"],
        default="cognitive_lockin",
        help="Decision mode (default: cognitive_lockin)",
    )

    # DUAL_FEEDBACK mode parameters
    parser.add_argument(
        "--initial-tau",
        type=float,
        default=1.0,
        help="Initial temperature for dual_feedback mode (default: 1.0)",
    )
    parser.add_argument(
        "--tau-min",
        type=float,
        default=0.1,
        help="Minimum temperature for dual_feedback mode (default: 0.1)",
    )
    parser.add_argument(
        "--tau-max",
        type=float,
        default=2.0,
        help="Maximum temperature for dual_feedback mode (default: 2.0)",
    )
    parser.add_argument(
        "--cooling-rate",
        type=float,
        default=0.1,
        help="Cooling rate on success for dual_feedback mode (default: 0.1)",
    )
    parser.add_argument(
        "--heating-penalty",
        type=float,
        default=0.3,
        help="Heating penalty on failure for dual_feedback mode (default: 0.3)",
    )

    # COGNITIVE_LOCKIN and EPSILON_GREEDY parameters
    parser.add_argument(
        "--initial-trust",
        type=float,
        default=0.5,
        help="Initial trust level (default: 0.5)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.1,
        help="Trust increase rate on correct prediction (default: 0.1)",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.3,
        help="Trust decay rate on wrong prediction (default: 0.3)",
    )

    # EPSILON_GREEDY specific
    parser.add_argument(
        "--exploration-mode",
        choices=["random", "opposite"],
        default="random",
        help="Exploration mode for epsilon_greedy (default: random)",
    )

    # V5: Normative memory settings
    parser.add_argument(
        "--normative",
        action="store_true",
        help="Enable normative memory (V5 dual-memory model)",
    )
    parser.add_argument(
        "--observation-k",
        type=int,
        default=3,
        help="Number of interactions to observe per tick (default: 3)",
    )
    parser.add_argument(
        "--ddm-noise",
        type=float,
        default=0.1,
        help="DDM noise sigma_noise (default: 0.1)",
    )
    parser.add_argument(
        "--crystal-threshold",
        type=float,
        default=3.0,
        help="DDM evidence threshold for crystallisation (default: 3.0)",
    )
    parser.add_argument(
        "--normative-strength",
        type=float,
        default=0.8,
        help="Initial norm strength on crystallisation (default: 0.8)",
    )
    parser.add_argument(
        "--crisis-threshold",
        type=int,
        default=10,
        help="Anomaly count triggering crisis (default: 10)",
    )
    parser.add_argument(
        "--crisis-decay",
        type=float,
        default=0.3,
        help="Strength decay on crisis (default: 0.3)",
    )
    parser.add_argument(
        "--enforce-threshold",
        type=float,
        default=0.7,
        help="Min norm strength for enforcement (default: 0.7)",
    )
    parser.add_argument(
        "--compliance-exponent",
        type=float,
        default=2.0,
        help="Exponent k in compliance = sigma^k (default: 2.0)",
    )
    parser.add_argument(
        "--signal-amplification",
        type=float,
        default=2.0,
        help="DDM drift multiplier from enforcement signals (default: 2.0)",
    )

    # Simulation settings
    parser.add_argument(
        "--ticks", "-t",
        type=int,
        default=1000,
        help="Maximum ticks (default: 1000)",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--no-early-stop",
        action="store_true",
        help="Disable early stopping on convergence",
    )

    # Output settings
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="data",
        help="Output directory (default: data)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to files",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Don't generate plots",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )

    return parser.parse_args()


def run_simulation(args):
    """Run a single simulation with given arguments."""
    print("=" * 60)
    print("ABM Memory & Norm Formation Simulation")
    print("=" * 60)

    # Create environment with decision mode
    decision_mode = DecisionMode(args.mode)
    env = SimulationEnvironment(
        num_agents=args.agents,
        memory_type=args.memory,
        memory_size=args.memory_size,
        decay_rate=args.decay_rate,
        dynamic_base=args.dynamic_base,
        dynamic_max=args.dynamic_max,
        decision_mode=decision_mode,
        # DUAL_FEEDBACK params
        initial_tau=args.initial_tau,
        tau_min=args.tau_min,
        tau_max=args.tau_max,
        cooling_rate=args.cooling_rate,
        heating_penalty=args.heating_penalty,
        # COGNITIVE_LOCKIN/EPSILON_GREEDY params
        initial_trust=args.initial_trust,
        alpha=args.alpha,
        beta=args.beta,
        exploration_mode=args.exploration_mode,
        random_seed=args.seed,
        # V5: Normative memory
        observation_k=args.observation_k if args.normative else 0,
        enable_normative=args.normative,
        ddm_noise=args.ddm_noise,
        crystal_threshold=args.crystal_threshold,
        normative_initial_strength=args.normative_strength,
        crisis_threshold=args.crisis_threshold,
        crisis_decay=args.crisis_decay,
        enforce_threshold=args.enforce_threshold,
        compliance_exponent=args.compliance_exponent,
        signal_amplification=args.signal_amplification,
    )

    print(f"\nConfiguration:")
    print(f"  Agents: {args.agents}")
    print(f"  Decision mode: {args.mode}")
    print(f"  Memory: {args.memory}")
    if args.normative:
        print(f"  Normative memory: ENABLED")
        print(f"  Observation k: {args.observation_k}")
    print(f"  Max ticks: {args.ticks}")
    print(f"  Seed: {args.seed}")
    print()

    # Run simulation
    print("Running simulation...")
    result = env.run(
        max_ticks=args.ticks,
        early_stop=not args.no_early_stop,
        verbose=args.verbose,
    )

    # Print results
    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    print(f"  Converged: {'Yes' if result.converged else 'No'}")
    if result.converged:
        print(f"  Convergence tick: {result.convergence_tick}")
    print(f"  Final tick: {result.final_state['final_tick']}")
    print(f"  Majority strategy: {'A' if result.final_state['final_majority_strategy'] == 0 else 'B'}")
    print(f"  Majority fraction: {result.final_state['final_majority_fraction']:.1%}")
    print(f"  Final mean trust: {result.final_state['final_mean_trust']:.1%}")
    print(f"  Final mean memory window: {result.final_state['final_mean_memory_window']:.2f}")
    if args.normative:
        print(f"  Norm level: {result.final_state.get('final_norm_level', 0)}")
        print(f"  Norm adoption rate: {result.final_state.get('final_norm_adoption_rate', 0.0):.1%}")
        print(f"  Mean norm strength: {result.final_state.get('final_mean_norm_strength', 0.0):.3f}")
        print(f"  Mean compliance: {result.final_state.get('final_mean_compliance', 0.0):.3f}")

    # Save results
    if not args.no_save:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = env.save_results(output_dir=str(output_dir), prefix="simulation")
        print(f"\n  Results saved to:")
        for key, path in paths.items():
            print(f"    {key}: {path}")

    # Generate plots
    if not args.no_plot:
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt

            df = env.get_history_dataframe()
            fig = create_summary_figure(df, title=f"Simulation Summary ({args.memory} memory)")

            if not args.no_save:
                output_dir = Path(args.output)
                saved = save_figure(fig, "simulation_summary", str(output_dir))
                print(f"\n  Plots saved to:")
                for path in saved:
                    print(f"    {path}")

            plt.close(fig)

        except ImportError:
            print("\n  Note: matplotlib not available, skipping plots")

    return result


def run_experiment(args):
    """Run comparison experiment."""
    print("=" * 60)
    print("Running Memory Type Comparison Experiment")
    print("=" * 60)

    from experiments.runner import run_memory_comparison

    df = run_memory_comparison(
        num_agents=args.agents,
        max_ticks=args.ticks,
        n_trials=10,
        output_dir=args.output,
        n_workers=4,
    )

    print(f"\nExperiment complete. {len(df)} runs performed.")
    print(f"Results saved to {args.output}/")


def launch_dashboard(args):
    """Launch interactive dashboard."""
    print("Launching interactive dashboard...")
    print("Note: This requires streamlit to be installed.")
    print("If not installed, run: pip install streamlit")
    print()

    import subprocess
    dashboard_path = Path(__file__).parent / "visualization" / "dashboard.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_path)])


def main():
    """Main entry point."""
    args = parse_args()

    # Quick test mode
    if args.quick:
        args.agents = 10
        args.ticks = 100

    # Mode selection
    if args.dashboard:
        launch_dashboard(args)
    elif args.experiment:
        run_experiment(args)
    else:
        run_simulation(args)


if __name__ == "__main__":
    main()
