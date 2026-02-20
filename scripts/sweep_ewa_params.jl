#!/usr/bin/env julia
#
# C4: EWA Parameter Sensitivity — sweep delta × lambda
#
# Finds the best (delta, lambda) configuration for pure coordination
# under the EWA baseline model. Key insight: the symmetric equilibrium
# p=0.5 is stable for lambda <= 2 (sigmoid derivative condition), so
# only lambda > 2 produces asymmetric convergence.
#

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using DualMemoryABM
using CSV
using Statistics

# ── Configuration ──

const DELTA_RANGE  = [0.0, 0.25, 0.5, 0.75, 1.0]
const LAMBDA_RANGE = [0.5, 1.0, 2.0, 5.0, 10.0]
const N_TRIALS     = 20
const BASE_SEED    = 9000

# ── Run ──

function run_sweep()
    rows = []

    for delta in DELTA_RANGE
        for lambda in LAMBDA_RANGE
            println("delta=$delta  lambda=$lambda  ($N_TRIALS trials)")

            Threads.@threads for trial in 1:N_TRIALS
                seed = hash((BASE_SEED, delta, lambda, trial)) % typemax(Int)
                p = EWAParams(
                    N      = 100,
                    T      = 3000,
                    seed   = Int(seed),
                    delta  = delta,
                    phi    = 0.9,
                    rho    = 0.9,
                    lambda = lambda,
                )
                result = ewa_run!(p)
                s = summarize(result)
                layers = ewa_first_tick_per_layer(result.history, p.N, p)

                push!(rows, (
                    delta                  = delta,
                    lambda                 = lambda,
                    trial                  = trial,
                    convergence_tick       = s.convergence_tick,
                    converged              = s.converged,
                    first_tick_behavioral  = layers.behavioral,
                    first_tick_belief      = layers.belief,
                    first_tick_all_met     = layers.all_met,
                    final_fraction_A       = s.final_fraction_A,
                    final_coordination_rate = result.history[end].coordination_rate,
                    total_ticks            = s.total_ticks,
                ))
            end
        end
    end

    # ── Save ──
    outdir = joinpath(@__DIR__, "..", "results")
    mkpath(outdir)
    outpath = joinpath(outdir, "sweep_ewa_params.csv")
    CSV.write(outpath, rows)
    println("\nSaved $(length(rows)) rows to $outpath")

    # ═══════════════════════════════════════════════════════════════
    # Summary 1: Convergence rate grid (delta × lambda)
    # ═══════════════════════════════════════════════════════════════
    println("\n" * "="^90)
    println("EWA SWEEP — Convergence rate (n_converged / n_trials)")
    println("="^90)
    println(rpad("δ \\ λ", 10), [rpad("λ=$l", 16) for l in LAMBDA_RANGE]...)
    println("-"^90)

    for delta in DELTA_RANGE
        parts = []
        for lambda in LAMBDA_RANGE
            sub = [r for r in rows if r.delta == delta && r.lambda == lambda]
            n_conv = count(r -> r.converged, sub)
            push!(parts, "$n_conv/$(length(sub))")
        end
        println(rpad("δ=$delta", 10), [rpad(p, 16) for p in parts]...)
    end

    # ═══════════════════════════════════════════════════════════════
    # Summary 2: Mean convergence speed (among converged trials)
    # ═══════════════════════════════════════════════════════════════
    println("\n" * "="^90)
    println("EWA SWEEP — Mean convergence tick (converged trials only)")
    println("="^90)
    println(rpad("δ \\ λ", 10), [rpad("λ=$l", 16) for l in LAMBDA_RANGE]...)
    println("-"^90)

    for delta in DELTA_RANGE
        parts = []
        for lambda in LAMBDA_RANGE
            sub = [r for r in rows if r.delta == delta && r.lambda == lambda]
            conv_ticks = [r.convergence_tick for r in sub if r.converged]
            if length(conv_ticks) > 0
                push!(parts, "μ=$(lpad(round(Int, mean(conv_ticks)), 4))")
            else
                push!(parts, "—")
            end
        end
        println(rpad("δ=$delta", 10), [rpad(p, 16) for p in parts]...)
    end

    # ═══════════════════════════════════════════════════════════════
    # Summary 3: Mean final coordination rate
    # ═══════════════════════════════════════════════════════════════
    println("\n" * "="^90)
    println("EWA SWEEP — Mean final coordination rate")
    println("="^90)
    println(rpad("δ \\ λ", 10), [rpad("λ=$l", 16) for l in LAMBDA_RANGE]...)
    println("-"^90)

    for delta in DELTA_RANGE
        parts = []
        for lambda in LAMBDA_RANGE
            sub = [r for r in rows if r.delta == delta && r.lambda == lambda]
            cr = mean([r.final_coordination_rate for r in sub])
            push!(parts, round(cr, digits=3))
        end
        println(rpad("δ=$delta", 10), [rpad(p, 16) for p in parts]...)
    end

    # ═══════════════════════════════════════════════════════════════
    # Best configuration
    # ═══════════════════════════════════════════════════════════════
    println("\n" * "="^90)
    println("BEST CONFIGURATION")
    println("="^90)

    best_rate = -1.0
    best_speed = Inf
    best_delta = 0.0
    best_lambda = 0.0

    for delta in DELTA_RANGE
        for lambda in LAMBDA_RANGE
            sub = [r for r in rows if r.delta == delta && r.lambda == lambda]
            n_conv = count(r -> r.converged, sub)
            rate = n_conv / length(sub)
            conv_ticks = [r.convergence_tick for r in sub if r.converged]
            speed = length(conv_ticks) > 0 ? mean(conv_ticks) : Inf

            if rate > best_rate || (rate == best_rate && speed < best_speed)
                best_rate = rate
                best_speed = speed
                best_delta = delta
                best_lambda = lambda
            end
        end
    end

    println("  delta  = $best_delta")
    println("  lambda = $best_lambda")
    println("  convergence rate = $best_rate")
    if best_speed < Inf
        println("  mean convergence tick = $(round(Int, best_speed))")
    else
        println("  mean convergence tick = N/A (no convergences)")
    end
end

run_sweep()
