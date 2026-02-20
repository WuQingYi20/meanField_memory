#!/usr/bin/env julia
#
# DM_full Tipping Point Parameter Exploration
#
# One-at-a-time sweeps of w_max, K, Phi, V to find how each shifts the
# critical forcing fraction for norm reversal.
#
# 4 sweeps × ~5 values × 9 fractions × 30 trials = 5400 runs
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

const T_EMERGE  = 600
const T_RECOVER = 500
const K_DEFAULT = 10

const FRACTIONS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
const N_TRIALS  = 30
const BASE_SEED = 8000

# ══════════════════════════════════════════════════════════════
# Sweep definitions
# ══════════════════════════════════════════════════════════════

struct SweepDef
    name::String            # sweep name for CSV
    param_name::String      # parameter being varied
    values::Vector{Float64} # parameter values (stored as Float64, cast as needed)
    default::Float64        # default value
end

const SWEEPS = [
    SweepDef("w_max", "w_max", [4.0, 6.0, 10.0, 15.0, 20.0], 6.0),
    SweepDef("K",     "K",     [3.0, 5.0, 10.0, 20.0, 30.0], 10.0),
    SweepDef("Phi",   "Phi",   [0.0, 0.5, 1.0, 2.0, 3.0],    1.0),
    SweepDef("V",     "V",     [0.0, 1.0, 3.0, 5.0, 8.0],    3.0),
]

# ══════════════════════════════════════════════════════════════
# Build SimulationParams for a given sweep point
# ══════════════════════════════════════════════════════════════

function make_params(sweep::SweepDef, value::Float64, seed::Int, K::Int)
    T_TOTAL = T_EMERGE + K + T_RECOVER
    w_max_val = sweep.param_name == "w_max" ? round(Int, value) : 6
    Phi_val   = sweep.param_name == "Phi"   ? value : 1.0
    V_val     = sweep.param_name == "V"     ? round(Int, value) : 3

    return SimulationParams(
        N=100, T=T_TOTAL, seed=seed,
        w_base=2, w_max=w_max_val,
        enable_normative=true,
        V=V_val, Phi=Phi_val,
    )
end

function get_K(sweep::SweepDef, value::Float64)
    return sweep.param_name == "K" ? round(Int, value) : K_DEFAULT
end

# ══════════════════════════════════════════════════════════════
# Per-trial result
# ══════════════════════════════════════════════════════════════

struct TippingResult
    converged::Bool
    flip::Bool
    recovery_ticks::Int
    perturbation_depth::Float64
    norm_survival::Float64
    recrystallisation::Float64
end

# ══════════════════════════════════════════════════════════════
# Custom tick function (DM_full only)
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
    stage_4_normative!(agents, ws, params, rng)
    stage_5_enforce!(agents, ws, params)
    metrics = stage_6_metrics(agents, ws, t, history, tick_count, params)
    history[tick_count + 1] = metrics
    record_probes!(probes, t, agents, ws, metrics)
    return metrics
end

# ══════════════════════════════════════════════════════════════
# Single trial
# ══════════════════════════════════════════════════════════════

function run_trial(sweep::SweepDef, value::Float64, seed::Int, forcing_fraction::Float64)
    K = get_K(sweep, value)
    params = make_params(sweep, value, seed, K)
    N = params.N
    T_TOTAL = T_EMERGE + K + T_RECOVER
    empty_ids = Int[]

    agents, ws, history, rng = initialize(params)
    probes = ProbeSet()
    init_probes!(probes, params, rng)

    tick_count = 0
    steady_counter = 0

    # ── Phase 1: Emergence (1–T_EMERGE) ──
    local last_metrics::TickMetrics
    for t in 1:T_EMERGE
        last_metrics = dm_tick!(t, agents, ws, history, tick_count, params, rng, probes, empty_ids, STRATEGY_A)
        tick_count += 1

        maj = max(last_metrics.fraction_A, 1.0 - last_metrics.fraction_A)
        if maj >= STEADY_MAJORITY_THRESHOLD
            steady_counter += 1
        else
            steady_counter = 0
        end
    end

    converged = steady_counter >= STEADY_WINDOW

    # Record dominant strategy before forcing
    dominant_before = last_metrics.fraction_A >= 0.5 ? STRATEGY_A : STRATEGY_B
    forced_strategy = dominant_before == STRATEGY_A ? STRATEGY_B : STRATEGY_A

    # Select forced agents
    n_forced = round(Int, N * forcing_fraction)
    forced_ids = sort(sample(rng, 1:N, n_forced; replace=false))

    # Snapshot crystallised before forcing
    pre_crystal = snapshot_crystallised(agents)

    # ── Phase 2: Forcing (T_EMERGE+1 to T_EMERGE+K) ──
    for t in (T_EMERGE + 1):(T_EMERGE + K)
        last_metrics = dm_tick!(t, agents, ws, history, tick_count, params, rng, probes, forced_ids, forced_strategy)
        tick_count += 1
    end

    # Mid-forcing snapshot (end of forcing)
    mid_crystal = snapshot_crystallised(agents)

    # ── Phase 3: Recovery (T_EMERGE+K+1 to T_TOTAL) ──
    min_majority = 1.0
    recovery_tick = 0

    # Track min_majority during forcing
    for idx in (T_EMERGE + 1):tick_count
        m = history[idx]
        maj = max(m.fraction_A, 1.0 - m.fraction_A)
        min_majority = min(min_majority, maj)
    end

    for t in (T_EMERGE + K + 1):T_TOTAL
        last_metrics = dm_tick!(t, agents, ws, history, tick_count, params, rng, probes, empty_ids, STRATEGY_A)
        tick_count += 1

        maj = max(last_metrics.fraction_A, 1.0 - last_metrics.fraction_A)
        min_majority = min(min_majority, maj)

        if recovery_tick == 0 && maj >= STEADY_MAJORITY_THRESHOLD
            recovery_tick = t - (T_EMERGE + K)
        end
    end

    # Dominant after
    dominant_after = last_metrics.fraction_A >= 0.5 ? STRATEGY_A : STRATEGY_B
    flip = dominant_before != dominant_after

    # Norm metrics
    surv, rcr = compute_norm_metrics(agents, forced_ids, pre_crystal, mid_crystal)

    return TippingResult(converged, flip, recovery_tick, min_majority, surv, rcr)
end

# ══════════════════════════════════════════════════════════════
# Estimate critical fraction (linear interpolation of 50% flip)
# ══════════════════════════════════════════════════════════════

function estimate_critical_fraction(flip_rates::Vector{Float64}, fractions::Vector{Float64})
    # Find first crossing of 0.5
    for i in 2:length(flip_rates)
        if flip_rates[i-1] < 0.5 && flip_rates[i] >= 0.5
            # Linear interpolation
            f1, f2 = fractions[i-1], fractions[i]
            r1, r2 = flip_rates[i-1], flip_rates[i]
            return f1 + (0.5 - r1) / (r2 - r1) * (f2 - f1)
        end
    end
    # If always below 0.5
    if all(r -> r < 0.5, flip_rates)
        return NaN  # tipping point above sweep range
    end
    # If always above 0.5
    if flip_rates[1] >= 0.5
        return NaN  # tipping point below sweep range
    end
    return NaN
end

# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

function main()
    outdir = joinpath(@__DIR__, "..", "results")
    mkpath(outdir)

    n_conditions = sum(length(s.values) for s in SWEEPS)
    n_total = n_conditions * length(FRACTIONS) * N_TRIALS
    println("="^70)
    println("DM_full Tipping Point Parameter Exploration — $n_total total trials")
    println("  $(length(SWEEPS)) sweeps, $n_conditions conditions, $(length(FRACTIONS)) fractions, $N_TRIALS trials")
    println("="^70)

    all_rows = Vector{NamedTuple}()
    sizehint!(all_rows, n_total)

    for sweep in SWEEPS
        println("\n" * "─"^70)
        println("SWEEP: $(sweep.name) ($(sweep.param_name))")
        println("─"^70)

        # Collect flip rates for heatmap
        flip_rate_matrix = zeros(length(sweep.values), length(FRACTIONS))

        for (vi, value) in enumerate(sweep.values)
            K = get_K(sweep, value)
            println("\n  $(sweep.param_name) = $(sweep.param_name == "K" || sweep.param_name == "w_max" || sweep.param_name == "V" ? round(Int, value) : value)  (K=$K, T=$(T_EMERGE+K+T_RECOVER))")

            for (fi, frac) in enumerate(FRACTIONS)
                results = Vector{TippingResult}(undef, N_TRIALS)

                Threads.@threads for trial in 1:N_TRIALS
                    seed = Int(hash((BASE_SEED, sweep.name, value, frac, trial)) % typemax(Int))
                    results[trial] = run_trial(sweep, value, seed, frac)
                end

                # Store rows
                for trial in 1:N_TRIALS
                    r = results[trial]
                    push!(all_rows, (
                        sweep = sweep.name,
                        param_name = sweep.param_name,
                        param_value = value,
                        forcing_fraction = frac,
                        trial = trial,
                        converged = r.converged,
                        flip = r.flip,
                        recovery_ticks = r.recovery_ticks,
                        perturbation_depth = r.perturbation_depth,
                        norm_survival = r.norm_survival,
                        recrystallisation = r.recrystallisation,
                    ))
                end

                # Inline summary
                n_conv = count(r -> r.converged, results)
                n_flip = count(r -> r.flip && r.converged, results)
                rate = n_conv > 0 ? n_flip / n_conv : 0.0
                flip_rate_matrix[vi, fi] = rate

                rec_vals = [r.recovery_ticks for r in results if r.recovery_ticks > 0]
                med_rec = isempty(rec_vals) ? "—" : string(round(Int, median(rec_vals)))
                dep_vals = [r.perturbation_depth for r in results]
                med_dep = round(median(dep_vals); digits=3)
                surv_vals = [r.norm_survival for r in results if !isnan(r.norm_survival)]
                ms = isempty(surv_vals) ? "—" : string(round(median(surv_vals); digits=2))

                print("    frac=$(lpad(string(round(frac; digits=2)), 4)) → ")
                print("flip=$n_flip/$n_conv ($(round(rate; digits=2)))  ")
                print("recovery=$med_rec  depth=$med_dep  surv=$ms")
                println()
            end
        end

        # ── Heatmap table ──
        println("\n  " * "="^70)
        println("  HEATMAP: $(sweep.name) — Flip Rate")
        println("  " * "="^70)

        # Header
        print("  ", rpad(sweep.param_name, 8))
        for frac in FRACTIONS
            print(rpad(string(round(frac; digits=2)), 8))
        end
        print("  critical")
        println()
        print("  ", "-"^(8 + 8 * length(FRACTIONS) + 10))
        println()

        for (vi, value) in enumerate(sweep.values)
            label = (sweep.param_name in ["K", "w_max", "V"]) ? string(round(Int, value)) : string(value)
            print("  ", rpad(label, 8))
            for fi in 1:length(FRACTIONS)
                r = flip_rate_matrix[vi, fi]
                print(rpad(string(round(r; digits=2)), 8))
            end

            # Critical fraction estimate
            rates = flip_rate_matrix[vi, :]
            crit = estimate_critical_fraction(rates, FRACTIONS)
            crit_str = isnan(crit) ? ">0.50" : string(round(crit; digits=3))
            print("  ", crit_str)
            println()
        end
    end

    # ── Write CSV ──
    csv_path = joinpath(outdir, "dm_tipping_params.csv")
    CSV.write(csv_path, all_rows)
    println("\n" * "="^70)
    println("Saved $(length(all_rows)) rows to $csv_path")

    # ── Summary: critical fractions ──
    println("\n" * "="^70)
    println("CRITICAL FRACTION SUMMARY (50% flip point)")
    println("="^70)
    for sweep in SWEEPS
        println("\n  $(sweep.name) sweep ($(sweep.param_name)):")
        for value in sweep.values
            sub = [r for r in all_rows if r.sweep == sweep.name && r.param_value == value]
            rates = Float64[]
            for frac in FRACTIONS
                frac_rows = [r for r in sub if r.forcing_fraction == frac]
                n_conv = count(r -> r.converged, frac_rows)
                n_flip = count(r -> r.flip && r.converged, frac_rows)
                push!(rates, n_conv > 0 ? n_flip / n_conv : 0.0)
            end
            crit = estimate_critical_fraction(rates, FRACTIONS)
            label = (sweep.param_name in ["K", "w_max", "V"]) ? string(round(Int, value)) : string(value)
            crit_str = isnan(crit) ? ">0.50" : string(round(crit; digits=3))
            println("    $(sweep.param_name)=$(rpad(label, 5)) → critical fraction ≈ $crit_str")
        end
    end

    println("\nDone.")
end

main()
