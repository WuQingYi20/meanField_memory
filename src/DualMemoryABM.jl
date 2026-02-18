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
    detect_norm_level, count_consecutive_ticks,
    sweep, summarize, save_sweep_csv, save_sweep_jld2,
    to_namedtuple,
    record_event!, record_probes!, init_probes!,

    # RingBuffer helpers
    capacity, recent, count_strategy_A,
    compute_b_exp_A

end # module
