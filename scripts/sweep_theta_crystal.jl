#!/usr/bin/env julia
#
# Sweep theta_crystal (crystallisation threshold) × 4 conditions
# Expected: sweet spot — too low → premature/wrong norms; too high → never crystallises.
#

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using DualMemoryABM
using CSV
using Statistics

# ── Configuration ──

const THETA_RANGE = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0]
const N_TRIALS    = 30
const BASE_SEED   = 3000

# Only normative conditions matter, but include A as a control
const CONDITIONS = [
    ("A_baseline",       5, 5, false),
    ("C_normative_only", 5, 5, true),
    ("D_full_model",     2, 6, true),
]

# ── Run ──

function run_sweep()
    rows = []

    for theta in THETA_RANGE
        for (label, wb, wm, enorm) in CONDITIONS
            println("θ_crystal=$theta  $label  ($N_TRIALS trials)")

            Threads.@threads for trial in 1:N_TRIALS
                seed = hash((BASE_SEED, theta, label, trial)) % typemax(Int)
                p = SimulationParams(
                    N       = 100,
                    T       = 3000,
                    seed    = Int(seed),
                    w_base  = wb,
                    w_max   = wm,
                    enable_normative = enorm,
                    theta_crystal    = theta,
                )
                result = run!(p)
                s = summarize(result)
                layers = first_tick_per_layer(result.history, p.N, p)

                # Count norm direction agreement with behavioral majority
                last_m = result.history[end]
                behavioral_A = last_m.fraction_A > 0.5

                push!(rows, (
                    theta_crystal           = theta,
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
    outpath = joinpath(outdir, "sweep_theta_crystal.csv")
    CSV.write(outpath, rows)
    println("\nSaved $(length(rows)) rows to $outpath")

    # ── Summary ──
    println("\n" * "="^110)
    println("θ_crystal SWEEP — Convergence rate & speed")
    println("="^110)
    println(rpad("θ", 8),
            rpad("A behav.", 18), rpad("C behav.", 18), rpad("D behav.", 18),
            rpad("C conv", 18), rpad("D conv", 18))
    println("-"^110)

    for theta in THETA_RANGE
        parts = []
        for (label, _, _, _) in CONDITIONS
            sub = [r for r in rows if r.theta_crystal == theta && r.condition == label]
            vals = [r.first_tick_behavioral for r in sub if r.first_tick_behavioral > 0]
            if length(vals) > 0
                push!(parts, "$(length(vals))/$(length(sub)) μ=$(lpad(round(Int, mean(vals)), 4))")
            else
                push!(parts, "0/$(length(sub))")
            end
        end
        # Convergence for C and D
        for label in ["C_normative_only", "D_full_model"]
            sub = [r for r in rows if r.theta_crystal == theta && r.condition == label]
            n_conv = count(r -> r.converged, sub)
            conv_ticks = [r.convergence_tick for r in sub if r.converged]
            if n_conv > 0
                push!(parts, "$(n_conv)/$(length(sub)) μ=$(lpad(round(Int, mean(conv_ticks)), 4))")
            else
                push!(parts, "0/$(length(sub))")
            end
        end

        println(rpad(theta, 8),
                rpad(parts[1], 18), rpad(parts[2], 18), rpad(parts[3], 18),
                rpad(parts[4], 18), rpad(parts[5], 18))
    end

    # ── Norm quality: frac_dominant_norm shows if norms are "correct" ──
    println("\n" * "="^80)
    println("θ_crystal SWEEP — Norm quality (mean frac_dominant_norm at end)")
    println("="^80)
    println(rpad("θ", 8), rpad("C dom_norm", 20), rpad("D dom_norm", 20),
            rpad("C n_cryst", 20), rpad("D n_cryst", 20))
    println("-"^80)

    for theta in THETA_RANGE
        parts = []
        for label in ["C_normative_only", "D_full_model"]
            sub = [r for r in rows if r.theta_crystal == theta && r.condition == label]
            dn = [r.final_frac_dominant_norm for r in sub]
            nc = [r.final_num_crystallised for r in sub]
            push!(parts, round(mean(dn), digits=3))
            push!(parts, round(mean(nc), digits=1))
        end
        println(rpad(theta, 8),
                rpad(parts[1], 20), rpad(parts[3], 20),
                rpad(parts[2], 20), rpad(parts[4], 20))
    end
end

run_sweep()
