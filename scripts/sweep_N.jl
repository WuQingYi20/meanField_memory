#!/usr/bin/env julia
#
# Sweep population size N × 4 conditions (2×2 factorial)
# Check whether the qualitative conclusions hold across scales.
#

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using DualMemoryABM
using CSV
using Statistics

# ── Configuration ──

const N_RANGE   = [20, 50, 100, 200, 500]
const N_TRIALS  = 30
const BASE_SEED = 2000

const CONDITIONS = [
    # (label, w_base, w_max, enable_normative)
    ("A_baseline",       5, 5, false),
    ("B_lockin_only",    2, 6, false),
    ("C_normative_only", 5, 5, true),
    ("D_full_model",     2, 6, true),
]

# ── Run ──

function run_sweep()
    rows = []

    for N in N_RANGE
        for (label, wb, wm, enorm) in CONDITIONS
            println("N=$N  $label  ($N_TRIALS trials)")

            Threads.@threads for trial in 1:N_TRIALS
                seed = hash((BASE_SEED, N, label, trial)) % typemax(Int)
                p = SimulationParams(
                    N       = N,
                    T       = 3000,
                    seed    = Int(seed),
                    w_base  = wb,
                    w_max   = wm,
                    enable_normative = enorm,
                )
                result = run!(p)
                s = summarize(result)
                layers = first_tick_per_layer(result.history, p.N, p)

                push!(rows, (
                    N                       = N,
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
        end
    end

    # ── Save ──
    outdir = joinpath(@__DIR__, "..", "results")
    mkpath(outdir)
    outpath = joinpath(outdir, "sweep_N.csv")
    CSV.write(outpath, rows)
    println("\nSaved $(length(rows)) rows to $outpath")

    # ── Summary table ──
    println("\n" * "="^100)
    println("SUMMARY: behavioral layer (first tick ≥95% majority)")
    println("="^100)
    println(rpad("N", 6),
            rpad("A_baseline", 22), rpad("B_lockin", 22),
            rpad("C_norm_only", 22), rpad("D_full", 22))
    println("-"^100)

    for N in N_RANGE
        parts = []
        for (label, _, _, _) in CONDITIONS
            sub = [r for r in rows if r.N == N && r.condition == label]
            vals = [r.first_tick_behavioral for r in sub if r.first_tick_behavioral > 0]
            if length(vals) > 0
                s = "$(length(vals))/$(length(sub)) μ=$(round(Int, mean(vals)))"
            else
                s = "0/$(length(sub))"
            end
            push!(parts, s)
        end
        println(rpad(N, 6), rpad(parts[1], 22), rpad(parts[2], 22),
                rpad(parts[3], 22), rpad(parts[4], 22))
    end

    println("\n" * "="^100)
    println("SUMMARY: convergence (all layers met + stable $(50) ticks)")
    println("="^100)
    println(rpad("N", 6),
            rpad("A_baseline", 22), rpad("B_lockin", 22),
            rpad("C_norm_only", 22), rpad("D_full", 22))
    println("-"^100)

    for N in N_RANGE
        parts = []
        for (label, _, _, _) in CONDITIONS
            sub = [r for r in rows if r.N == N && r.condition == label]
            n_conv = count(r -> r.converged, sub)
            conv_ticks = [r.convergence_tick for r in sub if r.converged]
            if n_conv > 0
                s = "$(n_conv)/$(length(sub)) μ=$(round(Int, mean(conv_ticks)))"
            else
                s = "0/$(length(sub))"
            end
            push!(parts, s)
        end
        println(rpad(N, 6), rpad(parts[1], 22), rpad(parts[2], 22),
                rpad(parts[3], 22), rpad(parts[4], 22))
    end

    # ── Speed ratios ──
    println("\n" * "="^100)
    println("SPEED RATIOS (behavioral layer, mean tick)")
    println("="^100)
    println(rpad("N", 6), rpad("B/A", 12), rpad("C/A", 12), rpad("D/A", 12), rpad("D/C", 12))
    println("-"^60)

    for N in N_RANGE
        means = Dict{String, Float64}()
        for (label, _, _, _) in CONDITIONS
            sub = [r for r in rows if r.N == N && r.condition == label]
            vals = [r.first_tick_behavioral for r in sub if r.first_tick_behavioral > 0]
            means[label] = length(vals) > 0 ? mean(vals) : NaN
        end
        a = means["A_baseline"]
        b = means["B_lockin_only"]
        c = means["C_normative_only"]
        d = means["D_full_model"]
        println(rpad(N, 6),
                rpad(isnan(a) || isnan(b) ? "N/A" : string(round(a/b, digits=1), "x"), 12),
                rpad(isnan(a) || isnan(c) ? "N/A" : string(round(a/c, digits=1), "x"), 12),
                rpad(isnan(a) || isnan(d) ? "N/A" : string(round(a/d, digits=1), "x"), 12),
                rpad(isnan(c) || isnan(d) ? "N/A" : string(round(c/d, digits=1), "x"), 12))
    end
end

run_sweep()
