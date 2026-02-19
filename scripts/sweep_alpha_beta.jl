#!/usr/bin/env julia
#
# Sweep alpha/beta ratio (confidence asymmetry) × 4 conditions
# Fix alpha=0.1, vary beta from 0.15 to 0.7.
# Higher beta/alpha → faster confidence drop → more volatile windows → ???
#

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using DualMemoryABM
using CSV
using Statistics

# ── Configuration ──

const ALPHA = 0.1
const BETA_RANGE = [0.15, 0.2, 0.3, 0.5, 0.7]   # ratio = 1.5, 2, 3, 5, 7
const N_TRIALS   = 30
const BASE_SEED  = 4000

const CONDITIONS = [
    ("A_baseline",       5, 5, false),
    ("B_lockin_only",    2, 6, false),
    ("C_normative_only", 5, 5, true),
    ("D_full_model",     2, 6, true),
]

# ── Run ──

function run_sweep()
    rows = []

    for beta in BETA_RANGE
        ratio = round(beta / ALPHA, digits=1)
        for (label, wb, wm, enorm) in CONDITIONS
            println("β=$beta (ratio=$ratio)  $label  ($N_TRIALS trials)")

            Threads.@threads for trial in 1:N_TRIALS
                seed = hash((BASE_SEED, beta, label, trial)) % typemax(Int)
                p = SimulationParams(
                    N       = 100,
                    T       = 3000,
                    seed    = Int(seed),
                    w_base  = wb,
                    w_max   = wm,
                    enable_normative = enorm,
                    alpha   = ALPHA,
                    beta    = beta,
                )
                result = run!(p)
                s = summarize(result)
                layers = first_tick_per_layer(result.history, p.N, p)

                push!(rows, (
                    alpha                   = ALPHA,
                    beta                    = beta,
                    ratio                   = ratio,
                    condition               = label,
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
    outpath = joinpath(outdir, "sweep_alpha_beta.csv")
    CSV.write(outpath, rows)
    println("\nSaved $(length(rows)) rows to $outpath")

    # ── Summary: Behavioral layer ──
    println("\n" * "="^110)
    println("α/β SWEEP — Behavioral layer (first tick ≥95% majority)")
    println("="^110)
    println(rpad("β (ratio)", 14),
            rpad("A_baseline", 22), rpad("B_lockin", 22),
            rpad("C_norm_only", 22), rpad("D_full", 22))
    println("-"^110)

    for beta in BETA_RANGE
        ratio = round(beta / ALPHA, digits=1)
        parts = []
        for (label, _, _, _) in CONDITIONS
            sub = [r for r in rows if r.beta == beta && r.condition == label]
            vals = [r.first_tick_behavioral for r in sub if r.first_tick_behavioral > 0]
            if length(vals) > 0
                push!(parts, "$(length(vals))/$(length(sub)) μ=$(lpad(round(Int, mean(vals)), 4))")
            else
                push!(parts, "0/$(length(sub))")
            end
        end
        println(rpad("$beta ($(ratio)x)", 14),
                rpad(parts[1], 22), rpad(parts[2], 22),
                rpad(parts[3], 22), rpad(parts[4], 22))
    end

    # ── Summary: Convergence ──
    println("\n" * "="^110)
    println("α/β SWEEP — Convergence (all layers + 50 ticks)")
    println("="^110)
    println(rpad("β (ratio)", 14),
            rpad("A_baseline", 22), rpad("B_lockin", 22),
            rpad("C_norm_only", 22), rpad("D_full", 22))
    println("-"^110)

    for beta in BETA_RANGE
        ratio = round(beta / ALPHA, digits=1)
        parts = []
        for (label, _, _, _) in CONDITIONS
            sub = [r for r in rows if r.beta == beta && r.condition == label]
            n_conv = count(r -> r.converged, sub)
            conv_ticks = [r.convergence_tick for r in sub if r.converged]
            if n_conv > 0
                push!(parts, "$(n_conv)/$(length(sub)) μ=$(lpad(round(Int, mean(conv_ticks)), 4))")
            else
                push!(parts, "0/$(length(sub))")
            end
        end
        println(rpad("$beta ($(ratio)x)", 14),
                rpad(parts[1], 22), rpad(parts[2], 22),
                rpad(parts[3], 22), rpad(parts[4], 22))
    end

    # ── Mean confidence at end ──
    println("\n" * "="^80)
    println("α/β SWEEP — Final mean confidence")
    println("="^80)
    println(rpad("β (ratio)", 14),
            rpad("A", 16), rpad("B", 16), rpad("C", 16), rpad("D", 16))
    println("-"^80)

    for beta in BETA_RANGE
        ratio = round(beta / ALPHA, digits=1)
        parts = []
        for (label, _, _, _) in CONDITIONS
            sub = [r for r in rows if r.beta == beta && r.condition == label]
            mc = mean([r.final_mean_confidence for r in sub])
            push!(parts, round(mc, digits=3))
        end
        println(rpad("$beta ($(ratio)x)", 14),
                rpad(parts[1], 16), rpad(parts[2], 16),
                rpad(parts[3], 16), rpad(parts[4], 16))
    end
end

run_sweep()
