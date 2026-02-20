#!/usr/bin/env julia
#
# DualMemory vs EWA — Trajectory Comparison
#
# 5 conditions: 3 EWA variants (RL, mixed, belief-learning) + 2 DualMemory variants
# Output 1: representative trajectories (seed=42, one run per condition)
# Output 2: multi-trial summary (50 trials per condition)
#

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using DualMemoryABM
using CSV
using Statistics

# ── Configuration ──

# UPDATE THIS after running sweep_ewa_params.jl (C4)
const BEST_LAMBDA = 10.0   # placeholder — replace with C4 result

const N_SUMMARY_TRIALS = 50
const BASE_SEED = 8000

# ── Condition definitions ──

struct Condition
    label::String
    model::Symbol          # :ewa or :dm
    params_fn::Function    # seed::Int -> params object
end

function make_conditions()
    return [
        Condition("EWA_RL", :ewa, seed -> EWAParams(
            N=100, T=3000, seed=seed,
            delta=0.0, phi=0.9, rho=0.9, lambda=BEST_LAMBDA,
        )),
        Condition("EWA_MIX", :ewa, seed -> EWAParams(
            N=100, T=3000, seed=seed,
            delta=0.5, phi=0.9, rho=0.9, lambda=BEST_LAMBDA,
        )),
        Condition("EWA_BL", :ewa, seed -> EWAParams(
            N=100, T=3000, seed=seed,
            delta=1.0, phi=0.9, rho=0.9, lambda=BEST_LAMBDA,
        )),
        Condition("DM_base", :dm, seed -> SimulationParams(
            N=100, T=3000, seed=seed,
            w_base=2, w_max=6,
            enable_normative=false,
        )),
        Condition("DM_full", :dm, seed -> SimulationParams(
            N=100, T=3000, seed=seed,
            w_base=2, w_max=6,
            enable_normative=true,
            V=3, Phi=1.0,
        )),
    ]
end

# ── Helpers ──

function run_condition(cond::Condition, seed::Int)
    params = cond.params_fn(seed)
    if cond.model == :ewa
        return ewa_run!(params)
    else
        return run!(params)
    end
end

function get_layers(cond::Condition, result)
    params = result.params
    if cond.model == :ewa
        return ewa_first_tick_per_layer(result.history, params.N, params)
    else
        return first_tick_per_layer(result.history, params.N, params)
    end
end

# ── Output 1: Representative trajectories ──

function write_trajectories(conditions)
    println("="^60)
    println("Output 1: Representative trajectories (seed=42)")
    println("="^60)

    rows = []
    for cond in conditions
        println("  Running $( cond.label)...")
        result = run_condition(cond, 42)

        for m in result.history
            push!(rows, (
                condition          = cond.label,
                tick               = m.tick,
                fraction_A         = m.fraction_A,
                mean_confidence    = m.mean_confidence,
                coordination_rate  = m.coordination_rate,
                num_crystallised   = m.num_crystallised,
                mean_norm_strength = m.mean_norm_strength,
                num_enforcements   = m.num_enforcements,
                belief_error       = m.belief_error,
                belief_variance    = m.belief_variance,
                frac_dominant_norm = m.frac_dominant_norm,
                convergence_counter = m.convergence_counter,
            ))
        end

        println("    $(length(result.history)) ticks recorded")
    end

    outdir = joinpath(@__DIR__, "..", "results")
    mkpath(outdir)
    outpath = joinpath(outdir, "comparison_trajectories.csv")
    CSV.write(outpath, rows)
    println("\nSaved $(length(rows)) trajectory rows to $outpath")
end

# ── Output 2: Multi-trial summary ──

function write_summary(conditions)
    println("\n" * "="^60)
    println("Output 2: Multi-trial summary ($N_SUMMARY_TRIALS trials per condition)")
    println("="^60)

    rows = []
    for cond in conditions
        println("  Running $(cond.label) × $N_SUMMARY_TRIALS trials...")

        Threads.@threads for trial in 1:N_SUMMARY_TRIALS
            seed = hash((BASE_SEED, cond.label, trial)) % typemax(Int)
            result = run_condition(cond, Int(seed))
            s = summarize(result)
            layers = get_layers(cond, result)

            push!(rows, (
                condition              = cond.label,
                trial                  = trial,
                convergence_tick       = s.convergence_tick,
                converged              = s.converged,
                first_tick_behavioral  = layers.behavioral,
                first_tick_belief      = layers.belief,
                first_tick_crystal     = layers.crystal,
                first_tick_all_met     = layers.all_met,
                final_fraction_A       = s.final_fraction_A,
                final_coordination_rate = result.history[end].coordination_rate,
                final_mean_confidence  = s.final_mean_confidence,
                final_num_crystallised = s.final_num_crystallised,
                final_mean_norm_strength = s.final_mean_norm_strength,
                total_ticks            = s.total_ticks,
            ))
        end
    end

    outdir = joinpath(@__DIR__, "..", "results")
    mkpath(outdir)
    outpath = joinpath(outdir, "comparison_summary.csv")
    CSV.write(outpath, rows)
    println("\nSaved $(length(rows)) summary rows to $outpath")

    # ── Console summary table ──
    println("\n" * "="^100)
    println("COMPARISON SUMMARY")
    println("="^100)
    println(rpad("Condition", 14),
            rpad("Conv Rate", 12),
            rpad("Mean Speed", 12),
            rpad("Mean Coord", 12),
            rpad("Behav Tick", 12),
            rpad("Belief Tick", 12),
            rpad("Crystal Tick", 14))
    println("-"^100)

    conditions_order = unique([r.condition for r in rows])
    for label in conditions_order
        sub = [r for r in rows if r.condition == label]
        n_conv = count(r -> r.converged, sub)
        rate = n_conv / length(sub)

        conv_ticks = [r.convergence_tick for r in sub if r.converged]
        mean_speed = length(conv_ticks) > 0 ? round(Int, mean(conv_ticks)) : -1

        mean_coord = round(mean([r.final_coordination_rate for r in sub]), digits=3)

        behav_vals = [r.first_tick_behavioral for r in sub if r.first_tick_behavioral > 0]
        mean_behav = length(behav_vals) > 0 ? round(Int, mean(behav_vals)) : -1

        belief_vals = [r.first_tick_belief for r in sub if r.first_tick_belief > 0]
        mean_belief = length(belief_vals) > 0 ? round(Int, mean(belief_vals)) : -1

        crystal_vals = [r.first_tick_crystal for r in sub if r.first_tick_crystal > 0]
        mean_crystal = length(crystal_vals) > 0 ? round(Int, mean(crystal_vals)) : -1

        println(rpad(label, 14),
                rpad("$n_conv/$(length(sub))", 12),
                rpad(mean_speed == -1 ? "—" : string(mean_speed), 12),
                rpad(mean_coord, 12),
                rpad(mean_behav == -1 ? "—" : string(mean_behav), 12),
                rpad(mean_belief == -1 ? "—" : string(mean_belief), 12),
                rpad(mean_crystal == -1 ? "—" : string(mean_crystal), 14))
    end
end

# ── Main ──

function main()
    conditions = make_conditions()
    write_trajectories(conditions)
    write_summary(conditions)
    println("\nDone.")
end

main()
