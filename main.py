"""
Main entry point for ABM Memory & Norm Formation Simulation.

This script provides a command-line interface for running simulations
and experiments.

Usage:
    python main.py                    # Run default simulation
    python main.py --memory dynamic   # Run with dynamic memory
    python main.py --experiment       # Run comparison experiment
    python main.py --dashboard        # Launch interactive dashboard
"""

import argparse
import sys
from pathlib import Path

import numpy as np

from config import SimulationConfig, DEFAULT_CONFIG, QUICK_TEST_CONFIG
from src.environment import SimulationEnvironment
from visualization.static_plots import create_summary_figure, save_figure


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="ABM Memory & Norm Formation Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py                        # Default simulation
    python main.py --agents 50 --ticks 500
    python main.py --memory decay --decay-rate 0.8
    python main.py --memory dynamic --dynamic-max 6
    python main.py --experiment           # Compare memory types
    python main.py --dashboard            # Interactive dashboard
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

    # Decision settings
    parser.add_argument(
        "--initial-tau",
        type=float,
        default=1.0,
        help="Initial temperature (default: 1.0)",
    )
    parser.add_argument(
        "--cooling-rate",
        type=float,
        default=0.1,
        help="Cooling rate on success (default: 0.1)",
    )
    parser.add_argument(
        "--heating-penalty",
        type=float,
        default=0.3,
        help="Heating penalty on failure (default: 0.3)",
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

    # Create environment
    env = SimulationEnvironment(
        num_agents=args.agents,
        memory_type=args.memory,
        memory_size=args.memory_size,
        decay_rate=args.decay_rate,
        dynamic_base=args.dynamic_base,
        dynamic_max=args.dynamic_max,
        initial_tau=args.initial_tau,
        cooling_rate=args.cooling_rate,
        heating_penalty=args.heating_penalty,
        random_seed=args.seed,
    )

    print(f"\nConfiguration:")
    print(f"  Agents: {args.agents}")
    print(f"  Memory: {args.memory}")
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
    print(f"  Final mean tau: {result.final_state['final_mean_tau']:.3f}")
    print(f"  Final mean trust: {result.final_state['final_mean_trust']:.1%}")

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
