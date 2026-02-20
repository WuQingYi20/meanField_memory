module DualMemoryABM

using Random
using StatsBase
using Statistics
using CSV
using JLD2
using Tables

# ── Core types and constants ──
include("types.jl")

# ── Parameters ──
include("params.jl")

# ── Research probes (must be before simulation.jl which references ProbeSet) ──
include("probes.jl")

# ── Initialization ──
include("init.jl")

# ── Pipeline stages ──
include("stages.jl")

# ── Simulation driver ──
include("simulation.jl")

# ── Parameter sweep ──
include("sweep.jl")

# ── EWA Baseline ──
include("ewa_types.jl")
include("ewa_init.jl")
include("ewa_stages.jl")
include("ewa_simulation.jl")

# ── Lifecycle Benchmark ──
include("lifecycle.jl")

# ── Exports ──
export
    # Constants
    STRATEGY_A, STRATEGY_B, NO_NORM, NO_SIGNAL,

    # Types
    RingBuffer, AgentState, TickWorkspace, TickMetrics,
    SimulationParams, SimulationResult, TrialSummary,

    # Probes
    NormEvent, EventLog, DDMTracker, QuantileTracker, ProbeSet,

    # Sweep
    SweepConfig, SweepResult,

    # Functions
    validate, initialize, create_workspace,
    compute_effective_belief!, map_predict,
    stage_1_pair_and_act!, stage_2_observe_and_memory!,
    stage_3_confidence!, stage_4_normative!,
    stage_5_enforce!, stage_6_metrics,
    ddm_update!, post_crystal_update!,
    run_tick!, run!, check_convergence,
    all_layers_met, count_consecutive_ticks,
    sweep, summarize, first_tick_per_layer, save_sweep_csv, save_sweep_jld2,
    to_namedtuple,
    record_event!, record_probes!, init_probes!,

    # RingBuffer helpers
    capacity, recent, count_strategy_A,
    compute_b_exp_A,

    # EWA Baseline
    EWAParams, EWAAgent, EWAWorkspace,
    ewa_initialize, ewa_create_workspace,
    ewa_pair_and_act!, ewa_update_attractions!, _ewa_update_prob!, ewa_metrics,
    ewa_run_tick!, ewa_run!, ewa_check_convergence, ewa_first_tick_per_layer,

    # Lifecycle Benchmark
    SYMMETRY_BREAK_THRESHOLD, STEADY_MAJORITY_THRESHOLD, STEADY_WINDOW, BURST_THRESHOLD,
    LifecycleMetrics, LifecycleTracker,
    override_actions!, snapshot_crystallised, compute_norm_metrics,
    update_tracker!, finalize_tracker

end # module
