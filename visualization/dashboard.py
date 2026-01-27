"""
Interactive dashboard for ABM simulation.

Provides a Streamlit-based interface for running and analyzing simulations.
"""

import sys
from pathlib import Path

# Check for streamlit availability
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
except ImportError:
    STREAMLIT_AVAILABLE = False

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any


def check_streamlit():
    """Check if streamlit is available."""
    if not STREAMLIT_AVAILABLE:
        raise ImportError(
            "Streamlit is required for the dashboard. "
            "Install it with: pip install streamlit"
        )


def create_dashboard():
    """
    Create and run the Streamlit dashboard.

    Run with: streamlit run visualization/dashboard.py
    """
    check_streamlit()

    # Add parent directory to path for imports
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from src.environment import SimulationEnvironment
    from visualization.static_plots import (
        plot_strategy_evolution,
        plot_tau_evolution,
        plot_coordination_rate,
        create_summary_figure,
    )

    st.set_page_config(
        page_title="ABM Memory Simulation",
        page_icon="🧠",
        layout="wide",
    )

    st.title("🧠 ABM Memory & Norm Formation Simulation")
    st.markdown("---")

    # Sidebar for configuration
    st.sidebar.header("Configuration")

    # Agent settings
    st.sidebar.subheader("Agent Settings")
    num_agents = st.sidebar.slider("Number of Agents", 2, 200, 100, step=2)

    # Memory settings
    st.sidebar.subheader("Memory Settings")
    memory_type = st.sidebar.selectbox(
        "Memory Type",
        ["fixed", "decay", "dynamic"],
        help="fixed: sliding window, decay: exponential weights, dynamic: trust-linked",
    )

    memory_size = st.sidebar.slider("Memory Size", 2, 10, 5)

    if memory_type == "decay":
        decay_rate = st.sidebar.slider("Decay Rate (λ)", 0.5, 1.0, 0.9, 0.05)
    else:
        decay_rate = 0.9

    if memory_type == "dynamic":
        dynamic_base = st.sidebar.slider("Dynamic Base Size", 1, 4, 2)
        dynamic_max = st.sidebar.slider("Dynamic Max Size", 4, 10, 6)
    else:
        dynamic_base = 2
        dynamic_max = 6

    # Decision settings
    st.sidebar.subheader("Decision Settings")
    initial_tau = st.sidebar.slider("Initial Temperature", 0.5, 2.0, 1.0, 0.1)
    tau_min = st.sidebar.slider("Min Temperature", 0.05, 0.5, 0.1, 0.05)
    tau_max = st.sidebar.slider("Max Temperature", 1.0, 3.0, 2.0, 0.1)
    cooling_rate = st.sidebar.slider("Cooling Rate", 0.01, 0.3, 0.1, 0.01)
    heating_penalty = st.sidebar.slider("Heating Penalty", 0.1, 0.5, 0.3, 0.05)

    # Simulation settings
    st.sidebar.subheader("Simulation Settings")
    max_ticks = st.sidebar.slider("Max Ticks", 100, 2000, 1000, 100)
    random_seed = st.sidebar.number_input("Random Seed", 0, 10000, 42)
    early_stop = st.sidebar.checkbox("Early Stop on Convergence", value=True)

    # Run button
    st.sidebar.markdown("---")
    run_button = st.sidebar.button("🚀 Run Simulation", use_container_width=True)

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Simulation Results")
        results_placeholder = st.empty()

    with col2:
        st.header("Statistics")
        stats_placeholder = st.empty()

    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()

    if run_button:
        # Create environment
        env = SimulationEnvironment(
            num_agents=num_agents,
            memory_type=memory_type,
            memory_size=memory_size,
            decay_rate=decay_rate,
            dynamic_base=dynamic_base,
            dynamic_max=dynamic_max,
            initial_tau=initial_tau,
            tau_min=tau_min,
            tau_max=tau_max,
            cooling_rate=cooling_rate,
            heating_penalty=heating_penalty,
            random_seed=random_seed,
        )

        # Run with progress updates
        status_text.text("Running simulation...")

        def progress_callback(tick, metrics):
            progress = (tick + 1) / max_ticks
            progress_bar.progress(progress)
            if tick % 50 == 0:
                status_text.text(
                    f"Tick {tick}: "
                    f"Strategy A = {metrics.strategy_distribution[0]:.1%}, "
                    f"Coord Rate = {metrics.coordination_rate:.1%}"
                )

        results = env.run(
            max_ticks=max_ticks,
            early_stop=early_stop,
            progress_callback=progress_callback,
        )

        progress_bar.progress(1.0)
        status_text.text("Simulation complete!")

        # Get data
        df = env.get_history_dataframe()

        # Display results
        with results_placeholder.container():
            # Create tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs([
                "📈 Strategy Evolution",
                "🌡️ Temperature",
                "🤝 Coordination",
                "📊 Summary",
            ])

            with tab1:
                fig, ax = plt.subplots(figsize=(10, 6))
                plot_strategy_evolution(df, ax=ax)
                st.pyplot(fig)
                plt.close()

            with tab2:
                fig, ax = plt.subplots(figsize=(10, 6))
                plot_tau_evolution(df, ax=ax)
                st.pyplot(fig)
                plt.close()

            with tab3:
                fig, ax = plt.subplots(figsize=(10, 6))
                plot_coordination_rate(df, ax=ax)
                st.pyplot(fig)
                plt.close()

            with tab4:
                fig = create_summary_figure(df)
                st.pyplot(fig)
                plt.close()

        # Display statistics
        with stats_placeholder.container():
            st.metric("Final Tick", results.final_state.get("final_tick", "N/A"))
            st.metric(
                "Converged",
                "✅ Yes" if results.converged else "❌ No",
            )
            if results.converged:
                st.metric("Convergence Tick", results.convergence_tick)

            st.metric(
                "Final Majority",
                f"Strategy {'A' if results.final_state.get('final_majority_strategy', 0) == 0 else 'B'}",
            )
            st.metric(
                "Majority Fraction",
                f"{results.final_state.get('final_majority_fraction', 0):.1%}",
            )
            st.metric(
                "Final Mean τ",
                f"{results.final_state.get('final_mean_tau', 0):.3f}",
            )
            st.metric(
                "Final Mean Trust",
                f"{results.final_state.get('final_mean_trust', 0):.1%}",
            )

            # Download button for data
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download Data (CSV)",
                csv,
                "simulation_results.csv",
                "text/csv",
                use_container_width=True,
            )

    else:
        # Show instructions when not running
        with results_placeholder.container():
            st.info(
                "👈 Configure simulation parameters in the sidebar and click "
                "'Run Simulation' to start."
            )

            st.markdown("""
            ### About this Simulation

            This agent-based model explores how different memory mechanisms
            affect norm formation in coordination games.

            **Memory Types:**
            - **Fixed**: Simple sliding window of recent interactions
            - **Decay**: Exponentially weighted history (recent > old)
            - **Dynamic**: Trust-linked adaptive window size

            **Key Feedback Loop:**
            - Coordination success → Lower temperature → Higher trust
            - Higher trust → Longer memory → More stable consensus
            - Failed coordination → Higher temperature → More exploration
            """)


# Run dashboard when executed directly
if __name__ == "__main__":
    create_dashboard()
