#!/usr/bin/env julia
#
# Tipping Point Sweep — find the critical forcing fraction for norm reversal
#
# For each (condition × forcing_fraction), run 50 trials:
#   Phase 1: Emergence (ticks 1–600, with early-exit check at 600)
#   Phase 2: Forcing   (ticks 601–610, K=10)
#   Phase 3: Recovery  (ticks 611–1110, 500 ticks)
#
# Sweep: forcing_fraction ∈ {0.05, 0.10, 0.15, …, 0.50}
# 5 conditions × 10 fractions × 50 trials = 2500 runs
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

const T_EMERGE = 600
const K_FORCE = 10
const T_RECOVER = 500
const T_TOTAL = T_EMERGE + K_FORCE + T_RECOVER   # 1110

const FRACTIONS = collect(0.05:0.05:0.50)   # 10 levels
const N_TRIALS = 50
const BASE_SEED = 7000
const BEST_LAMBDA = 10.0

# ══════════════════════════════════════════════════════════════
# Condition definitions (same as lifecycle benchmark)
# ══════════════════════════════════════════════════════════════

struct Condition
    label::String
    model::Symbol
    is_normative::Bool
    params_fn::Function
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
# Per-trial result
# ══════════════════════════════════════════════════════════════

struct TippingResult
    converged::Bool          # reached steady state before forcing?
    flip::Bool               # dominant strategy changed?
    recovery_ticks::Int      # ticks to re-achieve 95% (0 = never)
    perturbation_depth::Float64   # min majority during+after forcing
    norm_survival::Float64   # NaN for non-DM_full
    recrystallisation::Float64  # NaN for non-DM_full
    dominant_before::Int8
    dominant_after::Int8
end

# ══════════════════════════════════════════════════════════════
# Custom tick functions (reused from lifecycle benchmark)
# ══════════════════════════════════════════════════════════════

function dm_tick!(t::Int, agents::Vector{AgentState}, ws::TickWorkspace,
                  history::Vector{TickMetrics}, tick_count::Int,
                  params::SimulationParams, rng::AbstractRNG, probes,
                  forced_ids::Vector{Int}, forced_strategy::Int8)
    stage_1_pair_and_act!(agents, ws, params, rng)
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

function ewa_tick!(t::Int, agents::Vector{EWAAgent}, ws::EWAWorkspace,
                   history::Vector{TickMetrics}, tick_count::Int,
                   params::EWAParams, rng::AbstractRNG,
                   forced_ids::Vector{Int}, forced_strategy::Int8)
    ewa_pair_and_act!(agents, ws, params, rng)
    if !isempty(forced_ids)
        override_actions!(ws, params.N, forced_ids, forced_strategy)
    end
    ewa_update_attractions!(agents, ws, params)
    metrics = ewa_metrics(agents, ws, t, history, tick_count, params)
    history[tick_count + 1] = metrics
    return metrics
end

# ══════════════════════════════════════════════════════════════
# Single trial
# ══════════════════════════════════════════════════════════════

function run_tipping_trial(cond::Condition, seed::Int, forcing_fraction::Float64)
    params = cond.params_fn(seed)
    N = params.N
    empty_ids = Int[]

    # Initialize
    if cond.model == :ewa
        agents, ws, history, rng = ewa_initialize(params)
    else
        agents, ws, history, rng = initialize(params)
        probes = ProbeSet()
        init_probes!(probes, params, rng)
    end

    tick_count = 0
    steady_counter = 0

    # ── Phase 1: Emergence (1–600) ──
    local last_metrics::TickMetrics
    for t in 1:T_EMERGE
        if cond.model == :ewa
            last_metrics = ewa_tick!(t, agents, ws, history, tick_count, params, rng, empty_ids, STRATEGY_A)
        else
            last_metrics = dm_tick!(t, agents, ws, history, tick_count, params, rng, probes, empty_ids, STRATEGY_A)
        end
        tick_count += 1

        maj = max(last_metrics.fraction_A, 1.0 - last_metrics.fraction_A)
        if maj >= STEADY_MAJORITY_THRESHOLD
            steady_counter += 1
        else
            steady_counter = 0
        end
    end

    # Check convergence
    converged = steady_counter >= STEADY_WINDOW
    if !converged
        # Didn't reach steady state — still run but flag it
    end

    # Record dominant strategy before forcing
    dominant_before = last_metrics.fraction_A >= 0.5 ? STRATEGY_A : STRATEGY_B
    forced_strategy = dominant_before == STRATEGY_A ? STRATEGY_B : STRATEGY_A

    # Select forced agents
    n_forced = round(Int, N * forcing_fraction)
    forced_ids = sort(sample(rng, 1:N, n_forced; replace=false))

    # Snapshot crystallised (DM_full only)
    pre_crystal = cond.is_normative ? snapshot_crystallised(agents) : Dict{Int,Int8}()

    # ── Phase 2: Forcing (601–610) ──
    for t in (T_EMERGE + 1):(T_EMERGE + K_FORCE)
        if cond.model == :ewa
            last_metrics = ewa_tick!(t, agents, ws, history, tick_count, params, rng, forced_ids, forced_strategy)
        else
            last_metrics = dm_tick!(t, agents, ws, history, tick_count, params, rng, probes, forced_ids, forced_strategy)
        end
        tick_count += 1
    end

    # Mid-forcing snapshot (end of forcing)
    mid_crystal = cond.is_normative ? snapshot_crystallised(agents) : Dict{Int,Int8}()

    # ── Phase 3: Recovery (611–1110) ──
    min_majority = 1.0
    recovery_tick = 0

    # Also track min_majority during forcing
    for idx in (T_EMERGE + 1):tick_count
        m = history[idx]
        maj = max(m.fraction_A, 1.0 - m.fraction_A)
        min_majority = min(min_majority, maj)
    end

    for t in (T_EMERGE + K_FORCE + 1):T_TOTAL
        if cond.model == :ewa
            last_metrics = ewa_tick!(t, agents, ws, history, tick_count, params, rng, empty_ids, STRATEGY_A)
        else
            last_metrics = dm_tick!(t, agents, ws, history, tick_count, params, rng, probes, empty_ids, STRATEGY_A)
        end
        tick_count += 1

        maj = max(last_metrics.fraction_A, 1.0 - last_metrics.fraction_A)
        min_majority = min(min_majority, maj)

        if recovery_tick == 0 && maj >= STEADY_MAJORITY_THRESHOLD
            recovery_tick = t - (T_EMERGE + K_FORCE)
        end
    end

    # Dominant after
    dominant_after = last_metrics.fraction_A >= 0.5 ? STRATEGY_A : STRATEGY_B
    flip = dominant_before != dominant_after

    # Norm metrics (DM_full only)
    if cond.is_normative
        surv, rcr = compute_norm_metrics(agents, forced_ids, pre_crystal, mid_crystal)
    else
        surv, rcr = NaN, NaN
    end

    return TippingResult(converged, flip, recovery_tick, min_majority, surv, rcr,
                          dominant_before, dominant_after)
end

# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

function main()
    conditions = make_conditions()
    outdir = joinpath(@__DIR__, "..", "results")
    mkpath(outdir)

    n_total = length(conditions) * length(FRACTIONS) * N_TRIALS
    println("="^70)
    println("Tipping Point Sweep — $n_total total trials")
    println("  $(length(FRACTIONS)) fractions × $N_TRIALS trials × $(length(conditions)) conditions")
    println("="^70)

    all_rows = Vector{NamedTuple}(undef, n_total)
    row_idx = Threads.Atomic{Int}(0)

    for cond in conditions
        println("\n  $(cond.label):")
        for frac in FRACTIONS
            results = Vector{TippingResult}(undef, N_TRIALS)

            Threads.@threads for trial in 1:N_TRIALS
                seed = Int(hash((BASE_SEED, cond.label, frac, trial)) % typemax(Int))
                results[trial] = run_tipping_trial(cond, seed, frac)
            end

            # Store rows
            for trial in 1:N_TRIALS
                r = results[trial]
                idx = Threads.atomic_add!(row_idx, 1) + 1
                all_rows[idx] = (
                    condition = cond.label,
                    forcing_fraction = frac,
                    trial = trial,
                    converged = r.converged,
                    flip = r.flip,
                    recovery_ticks = r.recovery_ticks,
                    perturbation_depth = r.perturbation_depth,
                    norm_survival = r.norm_survival,
                    recrystallisation = r.recrystallisation,
                )
            end

            # Quick inline summary
            n_conv = count(r -> r.converged, results)
            n_flip = count(r -> r.flip, results)
            rec_vals = [r.recovery_ticks for r in results if r.recovery_ticks > 0]
            med_rec = isempty(rec_vals) ? "—" : string(round(Int, median(rec_vals)))
            dep_vals = [r.perturbation_depth for r in results]
            med_dep = round(median(dep_vals); digits=3)
            print("    frac=$(lpad(round(frac; digits=2), 4)) → flip=$n_flip/$n_conv  ")
            print("recovery=$med_rec  depth=$med_dep")
            if cond.is_normative
                surv_vals = [r.norm_survival for r in results if !isnan(r.norm_survival)]
                rcr_vals = [r.recrystallisation for r in results if !isnan(r.recrystallisation)]
                ms = isempty(surv_vals) ? "—" : string(round(median(surv_vals); digits=2))
                mr = isempty(rcr_vals) ? "—" : string(round(median(rcr_vals); digits=2))
                print("  surv=$ms  recryst=$mr")
            end
            println()
        end
    end

    # ── Write CSV ──
    csv_path = joinpath(outdir, "tipping_point.csv")
    CSV.write(csv_path, all_rows)
    println("\nSaved $(length(all_rows)) rows to $csv_path")

    # ══════════════════════════════════════════════════════════════
    # Summary table: flip rate S-curve
    # ══════════════════════════════════════════════════════════════
    println("\n" * "="^90)
    println("TIPPING POINT — FLIP RATE BY FORCING FRACTION")
    println("="^90)

    # Header
    print(rpad("Fraction", 10))
    for cond in conditions
        print(rpad(cond.label, 14))
    end
    println()
    println("-"^(10 + 14 * length(conditions)))

    for frac in FRACTIONS
        print(rpad(string(round(frac; digits=2)), 10))
        for cond in conditions
            sub = [r for r in all_rows if r.condition == cond.label && r.forcing_fraction == frac]
            n_conv = count(r -> r.converged, sub)
            n_flip = count(r -> r.flip && r.converged, sub)
            rate = n_conv > 0 ? round(n_flip / n_conv; digits=2) : NaN
            print(rpad("$n_flip/$n_conv ($rate)", 14))
        end
        println()
    end

    # ══════════════════════════════════════════════════════════════
    # Recovery time table
    # ══════════════════════════════════════════════════════════════
    println("\n" * "="^90)
    println("MEDIAN RECOVERY TICKS (converged trials only, 0=never recovered)")
    println("="^90)
    print(rpad("Fraction", 10))
    for cond in conditions
        print(rpad(cond.label, 14))
    end
    println()
    println("-"^(10 + 14 * length(conditions)))

    for frac in FRACTIONS
        print(rpad(string(round(frac; digits=2)), 10))
        for cond in conditions
            sub = [r for r in all_rows if r.condition == cond.label &&
                   r.forcing_fraction == frac && r.converged]
            vals = [r.recovery_ticks for r in sub]
            med = isempty(vals) ? "—" : string(round(Int, median(vals)))
            print(rpad(med, 14))
        end
        println()
    end

    # ══════════════════════════════════════════════════════════════
    # DM_full norm metrics table
    # ══════════════════════════════════════════════════════════════
    println("\n" * "="^60)
    println("DM_full NORM METRICS BY FORCING FRACTION")
    println("="^60)
    println(rpad("Fraction", 10), rpad("Survival", 14), rpad("Recryst", 14), rpad("Depth", 14))
    println("-"^52)
    for frac in FRACTIONS
        sub = [r for r in all_rows if r.condition == "DM_full" && r.forcing_fraction == frac && r.converged]
        surv = [r.norm_survival for r in sub if !isnan(r.norm_survival)]
        rcr = [r.recrystallisation for r in sub if !isnan(r.recrystallisation)]
        dep = [r.perturbation_depth for r in sub]
        ms = isempty(surv) ? "—" : string(round(median(surv); digits=3))
        mr = isempty(rcr) ? "—" : string(round(median(rcr); digits=3))
        md = isempty(dep) ? "—" : string(round(median(dep); digits=3))
        println(rpad(string(round(frac; digits=2)), 10), rpad(ms, 14), rpad(mr, 14), rpad(md, 14))
    end

    println("\nDone.")
end

main()
