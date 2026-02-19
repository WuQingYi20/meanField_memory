#!/usr/bin/env julia
#
# Sweep V (visibility) × Φ (enforcement strength) — conditions C and D
#
# V controls how many additional pair interactions an agent observes per tick.
# Φ controls the strength of partner-directed normative enforcement signals.
#
# NOTE: These results are PRELIMINARY / for reference only.
# The exact mechanisms by which V and Φ interact with the dual-memory system
# are not yet fully theorised. This sweep provides empirical patterns to
# guide subsequent mechanism analysis.
#

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using DualMemoryABM
using CSV
using Statistics

# ── Configuration ──

const V_RANGE   = [0, 1, 3, 5, 10]
const PHI_RANGE = [0.0, 0.5, 1.0, 2.0]
const N_TRIALS  = 30
const BASE_SEED = 7000

# Only normative conditions — V/Φ have no effect without normative layer
const CONDITIONS = [
    ("C_normative_only", 5, 5, true),
    ("D_full_model",     2, 6, true),
]

# ── Run ──

function run_sweep()
    rows = []

    for V in V_RANGE
        for Phi in PHI_RANGE
            for (label, wb, wm, enorm) in CONDITIONS
                println("V=$V  Φ=$Phi  $label  ($N_TRIALS trials)")

                Threads.@threads for trial in 1:N_TRIALS
                    seed = hash((BASE_SEED, V, Phi, label, trial)) % typemax(Int)
                    p = SimulationParams(
                        N       = 100,
                        T       = 3000,
                        seed    = Int(seed),
                        w_base  = wb,
                        w_max   = wm,
                        enable_normative = enorm,
                        V       = V,
                        Phi     = Phi,
                    )
                    result = run!(p)
                    s = summarize(result)
                    layers = first_tick_per_layer(result.history, p.N, p)

                    push!(rows, (
                        V                       = V,
                        Phi                     = Phi,
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
    end

    # ── Save ──
    outdir = joinpath(@__DIR__, "..", "results")
    mkpath(outdir)
    outpath = joinpath(outdir, "sweep_V_Phi.csv")
    CSV.write(outpath, rows)
    println("\nSaved $(length(rows)) rows to $outpath")

    # ═══════════════════════════════════════════════════════════════
    # Summary 1: Behavioral layer speed (mean tick to ≥95% majority)
    # ═══════════════════════════════════════════════════════════════
    for (label, _, _, _) in CONDITIONS
        println("\n" * "="^90)
        println("V × Φ SWEEP — Behavioral layer speed: $label")
        println("="^90)
        println(rpad("V \\ Φ", 10), [rpad("Φ=$p", 18) for p in PHI_RANGE]...)
        println("-"^82)

        for V in V_RANGE
            parts = []
            for Phi in PHI_RANGE
                sub = [r for r in rows if r.V == V && r.Phi == Phi && r.condition == label]
                vals = [r.first_tick_behavioral for r in sub if r.first_tick_behavioral > 0]
                if length(vals) > 0
                    push!(parts, "$(length(vals))/$(length(sub)) μ=$(lpad(round(Int, mean(vals)), 4))")
                else
                    push!(parts, "0/$(length(sub))")
                end
            end
            println(rpad("V=$V", 10), [rpad(p, 18) for p in parts]...)
        end
    end

    # ═══════════════════════════════════════════════════════════════
    # Summary 2: Convergence (all layers + stable window)
    # ═══════════════════════════════════════════════════════════════
    for (label, _, _, _) in CONDITIONS
        println("\n" * "="^90)
        println("V × Φ SWEEP — Convergence: $label")
        println("="^90)
        println(rpad("V \\ Φ", 10), [rpad("Φ=$p", 18) for p in PHI_RANGE]...)
        println("-"^82)

        for V in V_RANGE
            parts = []
            for Phi in PHI_RANGE
                sub = [r for r in rows if r.V == V && r.Phi == Phi && r.condition == label]
                n_conv = count(r -> r.converged, sub)
                conv_ticks = [r.convergence_tick for r in sub if r.converged]
                if n_conv > 0
                    push!(parts, "$(n_conv)/$(length(sub)) μ=$(lpad(round(Int, mean(conv_ticks)), 4))")
                else
                    push!(parts, "0/$(length(sub))")
                end
            end
            println(rpad("V=$V", 10), [rpad(p, 18) for p in parts]...)
        end
    end

    # ═══════════════════════════════════════════════════════════════
    # Summary 3: Norm quality (frac_dominant_norm)
    # ═══════════════════════════════════════════════════════════════
    for (label, _, _, _) in CONDITIONS
        println("\n" * "="^90)
        println("V × Φ SWEEP — Norm quality (mean frac_dominant_norm): $label")
        println("="^90)
        println(rpad("V \\ Φ", 10), [rpad("Φ=$p", 14) for p in PHI_RANGE]...)
        println("-"^66)

        for V in V_RANGE
            parts = []
            for Phi in PHI_RANGE
                sub = [r for r in rows if r.V == V && r.Phi == Phi && r.condition == label]
                dn = mean([r.final_frac_dominant_norm for r in sub])
                push!(parts, round(dn, digits=3))
            end
            println(rpad("V=$V", 10), [rpad(p, 14) for p in parts]...)
        end
    end

    # ═══════════════════════════════════════════════════════════════
    # Summary 4: Crystal layer speed
    # ═══════════════════════════════════════════════════════════════
    for (label, _, _, _) in CONDITIONS
        println("\n" * "="^90)
        println("V × Φ SWEEP — Crystal layer speed: $label")
        println("="^90)
        println(rpad("V \\ Φ", 10), [rpad("Φ=$p", 18) for p in PHI_RANGE]...)
        println("-"^82)

        for V in V_RANGE
            parts = []
            for Phi in PHI_RANGE
                sub = [r for r in rows if r.V == V && r.Phi == Phi && r.condition == label]
                vals = [r.first_tick_crystal for r in sub if r.first_tick_crystal > 0]
                if length(vals) > 0
                    push!(parts, "$(length(vals))/$(length(sub)) μ=$(lpad(round(Int, mean(vals)), 4))")
                else
                    push!(parts, "0/$(length(sub))")
                end
            end
            println(rpad("V=$V", 10), [rpad(p, 18) for p in parts]...)
        end
    end

    # ═══════════════════════════════════════════════════════════════
    # Summary 5: Enforcement count effect (D only, final norm strength)
    # ═══════════════════════════════════════════════════════════════
    println("\n" * "="^90)
    println("V × Φ SWEEP — Final mean norm strength: D_full_model")
    println("="^90)
    println(rpad("V \\ Φ", 10), [rpad("Φ=$p", 14) for p in PHI_RANGE]...)
    println("-"^66)

    for V in V_RANGE
        parts = []
        for Phi in PHI_RANGE
            sub = [r for r in rows if r.V == V && r.Phi == Phi && r.condition == "D_full_model"]
            ns = mean([r.final_mean_norm_strength for r in sub])
            push!(parts, round(ns, digits=3))
        end
        println(rpad("V=$V", 10), [rpad(p, 14) for p in parts]...)
    end
end

run_sweep()
