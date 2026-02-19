#!/usr/bin/env julia
#
# 2×2 Factorial Experiment: Memory Type × Normative Layer
#
# Conditions:
#   A: Baseline       — Fixed window (w=5),    normative OFF
#   B: Lock-in Only   — Dynamic window [2,6],  normative OFF
#   C: Normative Only — Fixed window (w=5),    normative ON
#   D: Full Model     — Dynamic window [2,6],  normative ON
#
# All conditions use V=0, Phi=0.0, N=100, T=3000, 50 trials each.
# End condition: all 3 layers converged for convergence_window ticks, or T reached.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using DualMemoryABM
using CSV

# ── Configuration ──

const N_TRIALS = 50
const BASE_SEED = 1000

const CONDITIONS = [
    # (label, w_base, w_max, enable_normative)
    ("A_baseline",       5, 5, false),
    ("B_lockin_only",    2, 6, false),
    ("C_normative_only", 5, 5, true),
    ("D_full_model",     2, 6, true),
]

# ── Run ──

function run_factorial()
    rows = []

    for (label, wb, wm, enorm) in CONDITIONS
        println("Running condition: $label ($N_TRIALS trials)")

        Threads.@threads for trial in 1:N_TRIALS
            seed = hash((BASE_SEED, label, trial)) % typemax(Int)
            p = SimulationParams(
                N       = 100,
                T       = 3000,
                seed    = Int(seed),
                w_base  = wb,
                w_max   = wm,
                enable_normative = enorm,
            )
            result = run!(p)
            s = summarize(result)

            # First tick each layer threshold was met (0 = never)
            layers = first_tick_per_layer(result.history, p.N, p)

            push!(rows, (
                condition               = label,
                w_base                  = wb,
                w_max                   = wm,
                enable_normative        = enorm,
                trial                   = trial,
                convergence_tick        = s.convergence_tick,
                converged               = s.converged,
                first_tick_behavioral   = layers.behavioral,
                first_tick_belief       = layers.belief,
                first_tick_crystal      = layers.crystal,
                first_tick_all_met      = layers.all_met,
                final_fraction_A        = s.final_fraction_A,
                final_mean_confidence   = s.final_mean_confidence,
                final_num_crystallised  = s.final_num_crystallised,
                final_mean_norm_strength = s.final_mean_norm_strength,
                final_frac_dominant_norm = s.final_frac_dominant_norm,
                total_ticks             = s.total_ticks,
            ))
        end

        println("  Done.")
    end

    # ── Save ──
    outdir = joinpath(@__DIR__, "..", "results")
    mkpath(outdir)
    outpath = joinpath(outdir, "factorial_2x2.csv")
    CSV.write(outpath, rows)
    println("\nResults saved to $outpath")
    println("Total rows: $(length(rows))")
end

run_factorial()
