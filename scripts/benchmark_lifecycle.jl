#!/usr/bin/env julia
#
# Lifecycle Benchmark — Model-Agnostic Behavioral Perturbation
#
# Protocol (T=1600, no early termination):
#   Phase 1: Emergence → Steady     (ticks 1–600)
#   Phase 2: 20% forcing            (ticks 601–610, K=10)
#   Phase 3: Recovery from 20%      (ticks 611–1100)
#   Phase 4: 50% forcing            (ticks 1101–1110, K=10)
#   Phase 5: Recovery from 50%      (ticks 1111–1600)
#
# 5 conditions × 50 trials = 250 runs
#

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using DualMemoryABM
using CSV
using Statistics
using Random
using StatsBase

# ══════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════

const T_TOTAL = 1600
const PHASE1_END = 600
const FORCE_20_START = 601
const FORCE_20_END = 610
const PHASE3_END = 1100
const FORCE_50_START = 1101
const FORCE_50_END = 1110

const FRAC_20 = 0.20
const FRAC_50 = 0.50

const N_TRIALS = 50
const BASE_SEED = 9000
const BEST_LAMBDA = 10.0

# ══════════════════════════════════════════════════════════════
# Condition definitions
# ══════════════════════════════════════════════════════════════

struct Condition
    label::String
    model::Symbol          # :ewa or :dm
    is_normative::Bool     # true only for DM_full
    params_fn::Function    # seed::Int -> params object
end

function make_conditions()
    return [
        Condition("EWA_RL", :ewa, false, seed -> EWAParams(
            N=100, T=T_TOTAL, seed=seed,
            delta=0.0, phi=0.9, rho=0.9, lambda=BEST_LAMBDA,
        )),
        Condition("EWA_MIX", :ewa, false, seed -> EWAParams(
            N=100, T=T_TOTAL, seed=seed,
            delta=0.5, phi=0.9, rho=0.9, lambda=BEST_LAMBDA,
        )),
        Condition("EWA_BL", :ewa, false, seed -> EWAParams(
            N=100, T=T_TOTAL, seed=seed,
            delta=1.0, phi=0.9, rho=0.9, lambda=BEST_LAMBDA,
        )),
        Condition("DM_base", :dm, false, seed -> SimulationParams(
            N=100, T=T_TOTAL, seed=seed,
            w_base=2, w_max=6,
            enable_normative=false,
        )),
        Condition("DM_full", :dm, true, seed -> SimulationParams(
            N=100, T=T_TOTAL, seed=seed,
            w_base=2, w_max=6,
            enable_normative=true,
            V=3, Phi=1.0,
        )),
    ]
end

# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════

"""Determine the minority strategy from current actions."""
function minority_strategy(ws, N::Int)
    n_A = 0
    for i in 1:N
        if ws.action[i] == STRATEGY_A
            n_A += 1
        end
    end
    return n_A > N ÷ 2 ? STRATEGY_B : STRATEGY_A
end

"""Select a random subset of agent IDs."""
function select_forced_ids(rng::AbstractRNG, N::Int, fraction::Float64)
    n_forced = round(Int, N * fraction)
    return sort(sample(rng, 1:N, n_forced; replace=false))
end

"""Compute majority fraction from tick metrics."""
@inline majority_frac(m::TickMetrics) = max(m.fraction_A, 1.0 - m.fraction_A)

"""Determine dominant strategy from fraction_A."""
@inline dominant_strat(m::TickMetrics) = m.fraction_A >= 0.5 ? STRATEGY_A : STRATEGY_B

# ══════════════════════════════════════════════════════════════
# Custom tick loop with override injection
# ══════════════════════════════════════════════════════════════

"""
Run a DM tick with optional action override between stage 1 and stage 2.
"""
function dm_tick_with_override!(t::Int, agents::Vector{AgentState}, ws::TickWorkspace,
                                 history::Vector{TickMetrics}, tick_count::Int,
                                 params::SimulationParams, rng::AbstractRNG,
                                 probes, forced_ids::Vector{Int}, forced_strategy::Int8)
    stage_1_pair_and_act!(agents, ws, params, rng)

    # Inject override BEFORE stage 2
    if !isempty(forced_ids)
        override_actions!(ws, params.N, forced_ids, forced_strategy)
    end

    stage_2_observe_and_memory!(agents, ws, params, rng)
    stage_3_confidence!(agents, ws, params)

    if params.enable_normative
        stage_4_normative!(agents, ws, params, rng)
        stage_5_enforce!(agents, ws, params)
    else
        ws.num_enforcements = 0
    end

    metrics = stage_6_metrics(agents, ws, t, history, tick_count, params)
    history[tick_count + 1] = metrics
    record_probes!(probes, t, agents, ws, metrics)

    return metrics
end

"""
Run an EWA tick with optional action override between pair_and_act and update_attractions.
"""
function ewa_tick_with_override!(t::Int, agents::Vector{EWAAgent}, ws::EWAWorkspace,
                                  history::Vector{TickMetrics}, tick_count::Int,
                                  params::EWAParams, rng::AbstractRNG,
                                  forced_ids::Vector{Int}, forced_strategy::Int8)
    ewa_pair_and_act!(agents, ws, params, rng)

    # Inject override BEFORE update_attractions
    if !isempty(forced_ids)
        override_actions!(ws, params.N, forced_ids, forced_strategy)
    end

    ewa_update_attractions!(agents, ws, params)
    metrics = ewa_metrics(agents, ws, t, history, tick_count, params)
    history[tick_count + 1] = metrics

    return metrics
end

# ══════════════════════════════════════════════════════════════
# Single trial runner
# ══════════════════════════════════════════════════════════════

struct TrialResult
    metrics::LifecycleMetrics
    trajectory::Union{Nothing, Vector{NamedTuple}}  # non-nothing only for trial=1
end

function run_trial(cond::Condition, seed::Int; record_trajectory::Bool=false)
    params = cond.params_fn(seed)
    N = params.N

    # Initialize
    if cond.model == :ewa
        agents, ws, history, rng = ewa_initialize(params)
    else
        agents, ws, history, rng = initialize(params)
        probes = ProbeSet()
        init_probes!(probes, params, rng)
    end

    tick_count = 0
    tracker = LifecycleTracker()
    trajectory = record_trajectory ? NamedTuple[] : nothing

    # Tracking variables for perturbation phases
    forced_ids_20 = Int[]
    forced_ids_50 = Int[]
    forced_strategy_20 = STRATEGY_A  # will be set at forcing start
    forced_strategy_50 = STRATEGY_A

    # Recovery tracking (20% perturbation)
    min_majority_20 = 1.0
    recovery_tick_20 = 0
    pre_crystal_snapshot_20 = Dict{Int,Int8}()
    mid_crystal_snapshot_20 = Dict{Int,Int8}()

    # Recovery tracking (50% perturbation)
    min_majority_50 = 1.0
    dominant_before_50 = NO_NORM
    dominant_after_50 = NO_NORM

    # State for determining when forcing starts
    empty_forced = Int[]
    no_force_strategy = STRATEGY_A  # dummy, not used when forced_ids is empty

    for t in 1:T_TOTAL
        # Determine current forcing
        local current_forced::Vector{Int}
        local current_strategy::Int8

        if FORCE_20_START <= t <= FORCE_20_END
            # 20% forcing phase
            if t == FORCE_20_START
                # Need to run one tick first to know minority. Use previous tick's data.
                # Actually, we determine minority from previous tick's actions.
                # But at tick 601, actions haven't been computed yet.
                # Solution: compute minority from LAST tick's metrics stored in history.
                if tick_count > 0
                    last_m = history[tick_count]
                    forced_strategy_20 = last_m.fraction_A >= 0.5 ? STRATEGY_B : STRATEGY_A
                else
                    forced_strategy_20 = STRATEGY_B  # fallback
                end
                forced_ids_20 = select_forced_ids(rng, N, FRAC_20)

                # Snapshot crystallised agents before forcing (DM_full only)
                if cond.model == :dm && cond.is_normative
                    pre_crystal_snapshot_20 = snapshot_crystallised(agents)
                end
            end
            current_forced = forced_ids_20
            current_strategy = forced_strategy_20
        elseif FORCE_50_START <= t <= FORCE_50_END
            # 50% forcing phase
            if t == FORCE_50_START
                if tick_count > 0
                    last_m = history[tick_count]
                    forced_strategy_50 = last_m.fraction_A >= 0.5 ? STRATEGY_B : STRATEGY_A
                    dominant_before_50 = last_m.fraction_A >= 0.5 ? STRATEGY_A : STRATEGY_B
                else
                    forced_strategy_50 = STRATEGY_B
                    dominant_before_50 = STRATEGY_A
                end
                forced_ids_50 = select_forced_ids(rng, N, FRAC_50)
            end
            current_forced = forced_ids_50
            current_strategy = forced_strategy_50
        else
            current_forced = empty_forced
            current_strategy = no_force_strategy
        end

        # Execute tick
        local metrics::TickMetrics
        if cond.model == :ewa
            metrics = ewa_tick_with_override!(t, agents, ws, history, tick_count,
                                              params, rng, current_forced, current_strategy)
        else
            metrics = dm_tick_with_override!(t, agents, ws, history, tick_count,
                                             params, rng, probes, current_forced, current_strategy)
        end
        tick_count += 1

        # ── Phase 1: Emergence tracking ──
        if t <= PHASE1_END
            update_tracker!(tracker, metrics, agents, ws, cond.model, cond.is_normative)
        end

        # ── Phase 2+3: 20% perturbation recovery ──
        if FORCE_20_START <= t
            maj = majority_frac(metrics)
            if t <= PHASE3_END
                min_majority_20 = min(min_majority_20, maj)
            end
            if t > FORCE_20_END && recovery_tick_20 == 0 && maj >= STEADY_MAJORITY_THRESHOLD
                recovery_tick_20 = t - FORCE_20_END
            end
        end

        # Take mid-forcing snapshot at end of forcing (DM_full only)
        if t == FORCE_20_END && cond.model == :dm && cond.is_normative
            mid_crystal_snapshot_20 = snapshot_crystallised(agents)
        end

        # ── Phase 4+5: 50% perturbation ──
        if FORCE_50_START <= t
            maj = majority_frac(metrics)
            min_majority_50 = min(min_majority_50, maj)
        end

        # Track dominant at end
        if t == T_TOTAL
            dominant_after_50 = dominant_strat(metrics)
        end

        # ── Record trajectory ──
        if record_trajectory
            phase = if t <= PHASE1_END
                "emergence"
            elseif t <= FORCE_20_END
                "force_20"
            elseif t <= PHASE3_END
                "recovery_20"
            elseif t <= FORCE_50_END
                "force_50"
            else
                "recovery_50"
            end

            push!(trajectory, (
                condition = cond.label,
                tick = metrics.tick,
                phase = phase,
                fraction_A = metrics.fraction_A,
                coordination_rate = metrics.coordination_rate,
                num_crystallised = metrics.num_crystallised,
                mean_norm_strength = metrics.mean_norm_strength,
                num_enforcements = metrics.num_enforcements,
                belief_error = metrics.belief_error,
                belief_variance = metrics.belief_variance,
            ))
        end
    end

    # ── Compute norm metrics for 20% perturbation (DM_full only) ──
    local norm_surv::Float64
    local recryst::Float64
    if cond.model == :dm && cond.is_normative
        norm_surv, recryst = compute_norm_metrics(agents, forced_ids_20,
                                                   pre_crystal_snapshot_20,
                                                   mid_crystal_snapshot_20)
    else
        norm_surv = NaN
        recryst = NaN
    end

    # ── Check norm flip for 50% perturbation ──
    norm_flip = dominant_before_50 != NO_NORM && dominant_after_50 != dominant_before_50

    # ── Build final metrics ──
    lm = finalize_tracker(tracker;
        recovery_ticks = recovery_tick_20,
        perturbation_depth = min_majority_20,
        norm_survival_rate = norm_surv,
        recrystallisation_rate = recryst,
        norm_flip = norm_flip,
    )

    return TrialResult(lm, trajectory)
end

# ══════════════════════════════════════════════════════════════
# Main — run all conditions × trials
# ══════════════════════════════════════════════════════════════

function main()
    conditions = make_conditions()
    outdir = joinpath(@__DIR__, "..", "results")
    mkpath(outdir)

    # ── Results storage ──
    results_lock = ReentrantLock()
    all_metrics = NamedTuple[]
    all_trajectories = NamedTuple[]

    println("="^70)
    println("Lifecycle Benchmark — $N_TRIALS trials × $(length(conditions)) conditions")
    println("="^70)

    for cond in conditions
        println("\n  Running $(cond.label) × $N_TRIALS trials...")
        cond_metrics = Vector{NamedTuple}(undef, N_TRIALS)
        cond_trajectories = NamedTuple[]

        Threads.@threads for trial in 1:N_TRIALS
            seed = Int(hash((BASE_SEED, cond.label, trial)) % typemax(Int))
            record_traj = (trial == 1)
            result = run_trial(cond, seed; record_trajectory=record_traj)

            lm = result.metrics
            row = (
                condition = cond.label,
                trial = trial,
                symmetry_break_tick = lm.symmetry_break_tick,
                agent_divergence_at_break = lm.agent_divergence_at_break,
                first_mover_mean_C = lm.first_mover_mean_C,
                diffusion_duration = lm.diffusion_duration,
                crystal_cascade_bursts = lm.crystal_cascade_bursts,
                steady_state_tick = lm.steady_state_tick,
                enforcement_at_steady = lm.enforcement_at_steady,
                recovery_ticks = lm.recovery_ticks,
                perturbation_depth = lm.perturbation_depth,
                norm_survival_rate = lm.norm_survival_rate,
                recrystallisation_rate = lm.recrystallisation_rate,
                norm_flip = lm.norm_flip,
            )
            cond_metrics[trial] = row

            if record_traj && result.trajectory !== nothing
                lock(results_lock) do
                    append!(cond_trajectories, result.trajectory)
                end
            end
        end

        lock(results_lock) do
            append!(all_metrics, cond_metrics)
            append!(all_trajectories, cond_trajectories)
        end

        # Quick per-condition summary
        conv_count = count(r -> r.steady_state_tick > 0, cond_metrics)
        med_break = let vals = [r.symmetry_break_tick for r in cond_metrics if r.symmetry_break_tick > 0]
            isempty(vals) ? "—" : string(round(Int, median(vals)))
        end
        med_recovery = let vals = [r.recovery_ticks for r in cond_metrics if r.recovery_ticks > 0]
            isempty(vals) ? "—" : string(round(Int, median(vals)))
        end
        n_flip = count(r -> r.norm_flip, cond_metrics)
        println("    Reached steady: $conv_count/$N_TRIALS | Break tick: $med_break | Recovery: $med_recovery | Flips: $n_flip")
    end

    # ── Write CSVs ──
    metrics_path = joinpath(outdir, "lifecycle_metrics.csv")
    CSV.write(metrics_path, all_metrics)
    println("\nSaved $(length(all_metrics)) metric rows to $metrics_path")

    traj_path = joinpath(outdir, "lifecycle_trajectories.csv")
    CSV.write(traj_path, all_trajectories)
    println("Saved $(length(all_trajectories)) trajectory rows to $traj_path")

    # ══════════════════════════════════════════════════════════════
    # Console summary
    # ══════════════════════════════════════════════════════════════
    println("\n" * "="^120)
    println("LIFECYCLE BENCHMARK SUMMARY")
    println("="^120)

    # ── Emergence ──
    println("\n--- EMERGENCE ---")
    println(rpad("Condition", 14),
            rpad("Break Tick", 14),
            rpad("Divergence", 14),
            rpad("FirstMover C", 14))
    println("-"^56)
    for cond in conditions
        sub = [r for r in all_metrics if r.condition == cond.label]
        breaks = [r.symmetry_break_tick for r in sub if r.symmetry_break_tick > 0]
        divs = [r.agent_divergence_at_break for r in sub if !isnan(r.agent_divergence_at_break)]
        fmc = [r.first_mover_mean_C for r in sub if !isnan(r.first_mover_mean_C)]

        med_b = isempty(breaks) ? "—" : "$(round(Int, median(breaks)))"
        med_d = isempty(divs) ? "—" : "$(round(median(divs), digits=3))"
        med_c = isempty(fmc) ? "—" : "$(round(median(fmc), digits=3))"
        println(rpad(cond.label, 14), rpad(med_b, 14), rpad(med_d, 14), rpad(med_c, 14))
    end

    # ── Diffusion ──
    println("\n--- DIFFUSION ---")
    println(rpad("Condition", 14),
            rpad("Duration", 14),
            rpad("Cascade Bursts", 16))
    println("-"^44)
    for cond in conditions
        sub = [r for r in all_metrics if r.condition == cond.label]
        durs = [r.diffusion_duration for r in sub if r.diffusion_duration > 0]
        bursts = [r.crystal_cascade_bursts for r in sub]

        med_dur = isempty(durs) ? "—" : "$(round(Int, median(durs)))"
        med_burst = "$(round(median(bursts), digits=1))"
        println(rpad(cond.label, 14), rpad(med_dur, 14), rpad(med_burst, 16))
    end

    # ── Steady State ──
    println("\n--- STEADY STATE ---")
    println(rpad("Condition", 14),
            rpad("Steady Tick", 14),
            rpad("Enforcement", 14))
    println("-"^42)
    for cond in conditions
        sub = [r for r in all_metrics if r.condition == cond.label]
        steady = [r.steady_state_tick for r in sub if r.steady_state_tick > 0]
        enforc = [r.enforcement_at_steady for r in sub if r.steady_state_tick > 0]

        med_s = isempty(steady) ? "—" : "$(round(Int, median(steady)))"
        med_e = isempty(enforc) ? "—" : "$(round(median(enforc), digits=2))"
        println(rpad(cond.label, 14), rpad(med_s, 14), rpad(med_e, 14))
    end

    # ── Perturbation Recovery ──
    println("\n--- PERTURBATION RECOVERY (20%) ---")
    println(rpad("Condition", 14),
            rpad("Recovery", 14),
            rpad("Depth", 14),
            rpad("NormSurv", 14),
            rpad("Recryst", 14))
    println("-"^70)
    for cond in conditions
        sub = [r for r in all_metrics if r.condition == cond.label]
        rec = [r.recovery_ticks for r in sub if r.recovery_ticks > 0]
        depth = [r.perturbation_depth for r in sub]
        surv = [r.norm_survival_rate for r in sub if !isnan(r.norm_survival_rate)]
        rcr = [r.recrystallisation_rate for r in sub if !isnan(r.recrystallisation_rate)]

        med_rec = isempty(rec) ? "—" : "$(round(Int, median(rec)))"
        med_dep = "$(round(median(depth), digits=3))"
        med_surv = isempty(surv) ? "NaN" : "$(round(median(surv), digits=3))"
        med_rcr = isempty(rcr) ? "NaN" : "$(round(median(rcr), digits=3))"
        println(rpad(cond.label, 14), rpad(med_rec, 14), rpad(med_dep, 14),
                rpad(med_surv, 14), rpad(med_rcr, 14))
    end

    # ── Norm Flip (50%) ──
    println("\n--- EXTREME PERTURBATION (50%) ---")
    println(rpad("Condition", 14), rpad("Flip Rate", 14))
    println("-"^28)
    for cond in conditions
        sub = [r for r in all_metrics if r.condition == cond.label]
        n_flip = count(r -> r.norm_flip, sub)
        println(rpad(cond.label, 14), rpad("$n_flip/$N_TRIALS", 14))
    end

    println("\nDone.")
end

main()
