"""
V5 Dual-Memory Comparison Dashboard.

Interactive Streamlit dashboard for comparing simulation conditions.
Focus: overlaid curves showing how key metrics evolve differently
across the 2x2 factorial (memory_type x normative_enabled).

Run with:
    streamlit run visualization/v5_dashboard.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import streamlit as st
from typing import Dict, List, Optional, Tuple

from src.environment import SimulationEnvironment

# ---------------------------------------------------------------------------
# Colour palette (consistent across all panels)
# ---------------------------------------------------------------------------
CONDITION_COLORS = {
    "fixed / norm OFF":   "#4C72B0",  # blue
    "fixed / norm ON":    "#DD8452",  # orange
    "dynamic / norm OFF": "#55A868",  # green
    "dynamic / norm ON":  "#C44E52",  # red
}

CONDITION_LINESTYLES = {
    "fixed / norm OFF":   "-",
    "fixed / norm ON":    "--",
    "dynamic / norm OFF": "-",
    "dynamic / norm ON":  "--",
}


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Running simulations...")
def run_comparison(
    num_agents: int,
    max_ticks: int,
    seed: int,
    observation_k: int,
    alpha: float,
    beta: float,
    crystal_threshold: float,
    ddm_noise: float,
    crisis_threshold: int,
    enforce_threshold: float,
    signal_amplification: float,
    compliance_exponent: float,
    conditions: Tuple[str, ...],
) -> Dict[str, pd.DataFrame]:
    """Run selected conditions and return {label: tick_dataframe}."""
    results = {}

    condition_params = {
        "fixed / norm OFF": ("fixed", False),
        "fixed / norm ON": ("fixed", True),
        "dynamic / norm OFF": ("dynamic", False),
        "dynamic / norm ON": ("dynamic", True),
    }

    for label in conditions:
        mem_type, norm_on = condition_params[label]
        env = SimulationEnvironment(
            num_agents=num_agents,
            memory_type=mem_type,
            memory_size=5,
            dynamic_base=2,
            dynamic_max=6,
            decision_mode="cognitive_lockin",
            initial_trust=0.5,
            alpha=alpha,
            beta=beta,
            random_seed=seed,
            observation_k=observation_k if norm_on else 0,
            enable_normative=norm_on,
            ddm_noise=ddm_noise,
            crystal_threshold=crystal_threshold,
            crisis_threshold=crisis_threshold,
            enforce_threshold=enforce_threshold,
            signal_amplification=signal_amplification,
            compliance_exponent=compliance_exponent,
        )
        env.run(max_ticks=max_ticks, early_stop=False, verbose=False)
        results[label] = env.get_history_dataframe()

    return results


@st.cache_data(show_spinner="Running seed ensemble...")
def run_ensemble(
    num_agents: int,
    max_ticks: int,
    seeds: Tuple[int, ...],
    observation_k: int,
    alpha: float,
    beta: float,
    crystal_threshold: float,
    ddm_noise: float,
    crisis_threshold: int,
    enforce_threshold: float,
    signal_amplification: float,
    compliance_exponent: float,
    conditions: Tuple[str, ...],
) -> Dict[str, pd.DataFrame]:
    """Run multiple seeds and return mean + std per condition."""
    condition_params = {
        "fixed / norm OFF": ("fixed", False),
        "fixed / norm ON": ("fixed", True),
        "dynamic / norm OFF": ("dynamic", False),
        "dynamic / norm ON": ("dynamic", True),
    }

    all_dfs: Dict[str, List[pd.DataFrame]] = {label: [] for label in conditions}

    for seed in seeds:
        for label in conditions:
            mem_type, norm_on = condition_params[label]
            env = SimulationEnvironment(
                num_agents=num_agents,
                memory_type=mem_type,
                memory_size=5,
                dynamic_base=2,
                dynamic_max=6,
                decision_mode="cognitive_lockin",
                initial_trust=0.5,
                alpha=alpha,
                beta=beta,
                random_seed=seed,
                observation_k=observation_k if norm_on else 0,
                enable_normative=norm_on,
                ddm_noise=ddm_noise,
                crystal_threshold=crystal_threshold,
                crisis_threshold=crisis_threshold,
                enforce_threshold=enforce_threshold,
                signal_amplification=signal_amplification,
                compliance_exponent=compliance_exponent,
            )
            env.run(max_ticks=max_ticks, early_stop=False, verbose=False)
            all_dfs[label].append(env.get_history_dataframe())

    # Compute mean across seeds
    results = {}
    for label in conditions:
        frames = all_dfs[label]
        concat = pd.concat(frames)
        grouped = concat.groupby("tick")
        mean_df = grouped.mean().reset_index()
        std_df = grouped.std().reset_index()
        # Store std columns with _std suffix
        for col in std_df.columns:
            if col != "tick":
                mean_df[f"{col}_std"] = std_df[col]
        results[label] = mean_df

    return results


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def _overlay_metric(
    dfs: Dict[str, pd.DataFrame],
    col: str,
    ax: plt.Axes,
    ylabel: str,
    title: str,
    show_band: bool = False,
    rolling: int = 1,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    """Plot one metric overlaid across conditions."""
    for label, df in dfs.items():
        y = df[col]
        if rolling > 1:
            y = y.rolling(window=rolling, min_periods=1).mean()
        ax.plot(
            df["tick"], y,
            color=CONDITION_COLORS.get(label, "black"),
            linestyle=CONDITION_LINESTYLES.get(label, "-"),
            linewidth=1.8,
            label=label,
        )
        if show_band and f"{col}_std" in df.columns:
            std = df[f"{col}_std"]
            if rolling > 1:
                std = std.rolling(window=rolling, min_periods=1).mean()
            ax.fill_between(
                df["tick"], y - std, y + std,
                color=CONDITION_COLORS.get(label, "black"),
                alpha=0.12,
            )
    ax.set_xlabel("Tick")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, fontweight="bold")
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.grid(True, alpha=0.25)


def plot_comparison_grid(
    dfs: Dict[str, pd.DataFrame],
    show_band: bool = False,
    rolling: int = 5,
) -> plt.Figure:
    """Create 3x3 grid comparing all key metrics."""
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))

    # --- Row 1: Core dynamics ---
    _overlay_metric(dfs, "majority_fraction", axes[0, 0],
                    "Majority fraction", "Strategy Convergence",
                    show_band=show_band, rolling=rolling, ylim=(0.4, 1.02))

    _overlay_metric(dfs, "mean_trust", axes[0, 1],
                    "Mean confidence C", "Predictive Confidence",
                    show_band=show_band, rolling=rolling, ylim=(0, 1))

    _overlay_metric(dfs, "coordination_rate", axes[0, 2],
                    "Coordination rate", "Coordination Rate",
                    show_band=show_band, rolling=rolling, ylim=(0, 1.02))

    # --- Row 2: Memory & belief ---
    _overlay_metric(dfs, "mean_memory_window", axes[1, 0],
                    "Mean window w", "Memory Window",
                    show_band=show_band, rolling=rolling)

    _overlay_metric(dfs, "belief_error", axes[1, 1],
                    "Mean |b - true|", "Belief Error",
                    show_band=show_band, rolling=rolling, ylim=(0, 0.5))

    _overlay_metric(dfs, "belief_variance", axes[1, 2],
                    "Var(b)", "Belief Variance",
                    show_band=show_band, rolling=rolling, ylim=(0, None))

    # --- Row 3: Normative dynamics ---
    _overlay_metric(dfs, "norm_adoption_rate", axes[2, 0],
                    "Fraction with norm", "Norm Adoption",
                    show_band=show_band, rolling=1, ylim=(0, 1.02))

    _overlay_metric(dfs, "mean_norm_strength", axes[2, 1],
                    "Mean sigma", "Norm Strength",
                    show_band=show_band, rolling=1, ylim=(0, 1))

    _overlay_metric(dfs, "mean_compliance", axes[2, 2],
                    "Mean sigma^k", "Compliance",
                    show_band=show_band, rolling=1, ylim=(0, 1))

    # Legend only once
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(dfs),
               fontsize=10, frameon=True, bbox_to_anchor=(0.5, 1.0))

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def plot_normative_detail(
    dfs: Dict[str, pd.DataFrame],
    show_band: bool = False,
    rolling: int = 3,
) -> plt.Figure:
    """Focused normative dynamics: DDM, anomaly, enforcement, norm level."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    _overlay_metric(dfs, "norm_adoption_rate", axes[0, 0],
                    "Fraction with norm", "Norm Adoption Rate",
                    show_band=show_band, rolling=1, ylim=(0, 1.02))

    _overlay_metric(dfs, "mean_ddm_evidence", axes[0, 1],
                    "Mean evidence e", "DDM Evidence (pre-norm agents)",
                    show_band=show_band, rolling=rolling)

    _overlay_metric(dfs, "norm_crystallisations", axes[0, 2],
                    "New norms / tick", "Crystallisation Events",
                    show_band=show_band, rolling=rolling)

    _overlay_metric(dfs, "enforcement_events", axes[1, 0],
                    "Signals / tick", "Enforcement Signals",
                    show_band=show_band, rolling=rolling)

    _overlay_metric(dfs, "total_anomalies", axes[1, 1],
                    "Anomalies (cumul.)", "Anomaly Count",
                    show_band=show_band, rolling=1)

    _overlay_metric(dfs, "norm_level", axes[1, 2],
                    "Norm level (0-5)", "Norm Detection Level",
                    show_band=show_band, rolling=1, ylim=(-0.2, 5.5))
    axes[1, 2].yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    # Add level labels
    level_names = {0: "NONE", 1: "BEHAV", 2: "EMPIR", 3: "SHARED", 4: "NORM", 5: "INSTIT"}
    for lv, name in level_names.items():
        axes[1, 2].axhline(y=lv, color="grey", alpha=0.15, linewidth=0.8)
        axes[1, 2].text(0, lv + 0.1, name, fontsize=7, color="grey", alpha=0.6)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(dfs),
               fontsize=10, frameon=True, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig


def plot_feedback_loops(
    dfs: Dict[str, pd.DataFrame],
    rolling: int = 10,
) -> plt.Figure:
    """Phase-plane and feedback loop diagnostics."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Phase plane: majority fraction vs trust
    for label, df in dfs.items():
        axes[0].scatter(
            df["majority_fraction"], df["mean_trust"],
            c=df["tick"], cmap="viridis", alpha=0.4, s=4,
            label=label,
        )
        # Start/end markers
        axes[0].scatter(
            df["majority_fraction"].iloc[0], df["mean_trust"].iloc[0],
            marker="o", s=60, edgecolors=CONDITION_COLORS.get(label, "black"),
            facecolors="none", linewidths=2,
        )
        axes[0].scatter(
            df["majority_fraction"].iloc[-1], df["mean_trust"].iloc[-1],
            marker="s", s=60, color=CONDITION_COLORS.get(label, "black"),
        )
    axes[0].set_xlabel("Majority fraction")
    axes[0].set_ylabel("Mean confidence C")
    axes[0].set_title("Phase Plane", fontsize=11, fontweight="bold")
    axes[0].set_xlim(0.4, 1.02)
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.25)

    # Trust vs memory window
    for label, df in dfs.items():
        trust = df["mean_trust"].rolling(rolling, min_periods=1).mean()
        window = df["mean_memory_window"].rolling(rolling, min_periods=1).mean()
        axes[1].plot(trust, window,
                     color=CONDITION_COLORS.get(label, "black"),
                     linestyle=CONDITION_LINESTYLES.get(label, "-"),
                     linewidth=1.5, alpha=0.8, label=label)
    axes[1].set_xlabel("Mean confidence C")
    axes[1].set_ylabel("Mean window w")
    axes[1].set_title("Trust-Window Coupling", fontsize=11, fontweight="bold")
    axes[1].grid(True, alpha=0.25)

    # Norm adoption vs compliance
    for label, df in dfs.items():
        adopt = df["norm_adoption_rate"]
        comp = df["mean_compliance"]
        axes[2].plot(adopt, comp,
                     color=CONDITION_COLORS.get(label, "black"),
                     linestyle=CONDITION_LINESTYLES.get(label, "-"),
                     linewidth=1.5, alpha=0.8, label=label)
    axes[2].set_xlabel("Norm adoption rate")
    axes[2].set_ylabel("Mean compliance")
    axes[2].set_title("Adoption-Compliance", fontsize=11, fontweight="bold")
    axes[2].set_xlim(0, 1.02)
    axes[2].set_ylim(0, 1)
    axes[2].grid(True, alpha=0.25)

    handles, labels_list = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels_list, loc="upper center", ncol=len(dfs),
               fontsize=10, frameon=True, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


def make_summary_table(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Summary statistics per condition at final tick."""
    rows = []
    for label, df in dfs.items():
        last = df.iloc[-1]
        rows.append({
            "Condition": label,
            "Majority %": f"{last['majority_fraction']:.1%}",
            "Mean C": f"{last['mean_trust']:.3f}",
            "Coord rate": f"{last['coordination_rate']:.1%}",
            "Mean window": f"{last['mean_memory_window']:.1f}",
            "Norm adopt": f"{last['norm_adoption_rate']:.0%}",
            "Mean sigma": f"{last['mean_norm_strength']:.3f}",
            "Compliance": f"{last['mean_compliance']:.3f}",
            "Norm level": int(last["norm_level"]),
            "Belief err": f"{last['belief_error']:.4f}",
        })
    return pd.DataFrame(rows).set_index("Condition")


# ---------------------------------------------------------------------------
# Dashboard layout
# ---------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="V5 Dual-Memory Dashboard", layout="wide")
    st.title("V5 Dual-Memory Comparison Dashboard")
    st.caption("2x2 factorial: memory type (fixed / dynamic) x normative memory (OFF / ON)")

    # --- Sidebar ---
    with st.sidebar:
        st.header("Parameters")

        st.subheader("Simulation")
        num_agents = st.slider("Agents", 10, 200, 50, step=10)
        max_ticks = st.slider("Ticks", 100, 2000, 500, step=100)
        seed = st.number_input("Seed", 0, 99999, 42)

        st.subheader("Trust (Slovic 1993)")
        alpha = st.slider("alpha (build)", 0.01, 0.5, 0.1, 0.01)
        beta = st.slider("beta (break)", 0.01, 0.5, 0.3, 0.01)

        st.subheader("Normative Memory")
        observation_k = st.slider("observation_k", 0, 10, 3)
        crystal_threshold = st.slider("crystal_threshold", 0.5, 10.0, 3.0, 0.5)
        ddm_noise = st.slider("ddm_noise", 0.0, 0.5, 0.1, 0.05)
        crisis_threshold = st.slider("crisis_threshold", 3, 30, 10)
        enforce_threshold = st.slider("enforce_threshold", 0.1, 0.95, 0.7, 0.05)
        signal_amplification = st.slider("signal_amplification", 1.0, 5.0, 2.0, 0.5)
        compliance_exponent = st.slider("compliance_exponent k", 1.0, 4.0, 2.0, 0.5)

        st.subheader("Display")
        rolling = st.slider("Smoothing window", 1, 30, 5)

        st.subheader("Conditions")
        all_conditions = list(CONDITION_COLORS.keys())
        selected = st.multiselect(
            "Select conditions",
            all_conditions,
            default=all_conditions,
        )

        st.subheader("Ensemble (optional)")
        n_seeds = st.selectbox("Seeds to average", [1, 3, 5, 10], index=0)

        st.markdown("---")
        run_btn = st.button("Run Comparison", use_container_width=True, type="primary")

    # --- Main ---
    if not run_btn and "dfs" not in st.session_state:
        st.info("Configure parameters in the sidebar and click **Run Comparison**.")
        return

    if run_btn:
        if not selected:
            st.warning("Select at least one condition.")
            return

        conditions = tuple(selected)

        if n_seeds == 1:
            dfs = run_comparison(
                num_agents=num_agents,
                max_ticks=max_ticks,
                seed=seed,
                observation_k=observation_k,
                alpha=alpha,
                beta=beta,
                crystal_threshold=crystal_threshold,
                ddm_noise=ddm_noise,
                crisis_threshold=crisis_threshold,
                enforce_threshold=enforce_threshold,
                signal_amplification=signal_amplification,
                compliance_exponent=compliance_exponent,
                conditions=conditions,
            )
            show_band = False
        else:
            seeds = tuple(seed + i * 1000 for i in range(n_seeds))
            dfs = run_ensemble(
                num_agents=num_agents,
                max_ticks=max_ticks,
                seeds=seeds,
                observation_k=observation_k,
                alpha=alpha,
                beta=beta,
                crystal_threshold=crystal_threshold,
                ddm_noise=ddm_noise,
                crisis_threshold=crisis_threshold,
                enforce_threshold=enforce_threshold,
                signal_amplification=signal_amplification,
                compliance_exponent=compliance_exponent,
                conditions=conditions,
            )
            show_band = True

        st.session_state["dfs"] = dfs
        st.session_state["show_band"] = show_band
        st.session_state["rolling"] = rolling
    else:
        dfs = st.session_state["dfs"]
        show_band = st.session_state.get("show_band", False)
        rolling = st.session_state.get("rolling", rolling)

    # --- Summary table ---
    st.subheader("Final-tick summary")
    st.dataframe(make_summary_table(dfs), use_container_width=True)

    # --- Tabs ---
    tab_overview, tab_normative, tab_loops = st.tabs([
        "Overview (3x3)", "Normative Detail", "Feedback Loops"
    ])

    with tab_overview:
        fig = plot_comparison_grid(dfs, show_band=show_band, rolling=rolling)
        st.pyplot(fig)
        plt.close(fig)

    with tab_normative:
        normative_dfs = {k: v for k, v in dfs.items() if "norm ON" in k}
        if normative_dfs:
            fig = plot_normative_detail(normative_dfs, show_band=show_band, rolling=rolling)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Select at least one **norm ON** condition to see normative details.")

    with tab_loops:
        fig = plot_feedback_loops(dfs, rolling=rolling)
        st.pyplot(fig)
        plt.close(fig)


if __name__ == "__main__":
    main()
