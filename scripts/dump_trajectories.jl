#!/usr/bin/env julia
# Dump full tick-by-tick trajectories for one representative run per condition.
# Also compute tipping-point analysis: Δfraction_A per tick, find max-acceleration tick.

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using DualMemoryABM
using CSV
using Statistics

const CONDITIONS = [
    ("A_baseline",       5, 5, false),
    ("B_lockin_only",    2, 6, false),
    ("C_normative_only", 5, 5, true),
    ("D_full_model",     2, 6, true),
]

const SEED = 42

# ── 1. Dump full trajectories ──

traj_rows = []
for (label, wb, wm, enorm) in CONDITIONS
    p = SimulationParams(N=100, T=3000, seed=SEED,
                         w_base=wb, w_max=wm, enable_normative=enorm)
    result = run!(p)

    for m in result.history
        push!(traj_rows, (
            condition          = label,
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
    println(label, ": ", result.tick_count, " ticks")
end

outdir = joinpath(@__DIR__, "..", "results")
mkpath(outdir)
CSV.write(joinpath(outdir, "trajectories_seed42.csv"), traj_rows)
println("\nSaved $(length(traj_rows)) trajectory rows")

# ── 2. Tipping-point analysis ──

println("\n" * "="^70)
println("TIPPING POINT ANALYSIS (seed=$SEED)")
println("="^70)

for (label, wb, wm, enorm) in CONDITIONS
    p = SimulationParams(N=100, T=3000, seed=SEED,
                         w_base=wb, w_max=wm, enable_normative=enorm)
    result = run!(p)
    h = result.history

    # Compute |fraction_A - 0.5| = deviation from symmetry
    dev = [abs(m.fraction_A - 0.5) for m in h]

    # Find the "takeoff" tick: first tick where deviation exceeds 0.1 and never
    # returns below 0.1 again (sustained departure from 50-50)
    takeoff = 0
    for t in 1:length(dev)
        if dev[t] >= 0.1
            # check it stays above 0.05 from here on
            sustained = true
            for t2 in t:length(dev)
                if dev[t2] < 0.05
                    sustained = false
                    break
                end
            end
            if sustained
                takeoff = h[t].tick
                break
            end
        end
    end

    # Find max single-tick swing in fraction_A
    max_delta = 0.0
    max_delta_tick = 0
    for t in 2:length(h)
        d = abs(h[t].fraction_A - h[t-1].fraction_A)
        if d > max_delta
            max_delta = d
            max_delta_tick = h[t].tick
        end
    end

    # Print window around takeoff
    println("\n--- $label ---")
    println("  Takeoff tick (|fA - 0.5| ≥ 0.1 sustained): $takeoff")
    println("  Max Δfraction_A: $(round(max_delta, digits=3)) at tick $max_delta_tick")

    if takeoff > 0
        # Show 10 ticks before through 20 ticks after takeoff
        lo = max(1, takeoff - 10)
        hi = min(length(h), takeoff + 20)

        println("\n  Tick | frac_A | Δfrac_A  | mean_C  | coord  | n_cryst | σ_norm | dom_norm")
        println("  " * "-"^88)
        for t in lo:hi
            idx = t  # h is 1-indexed, h[t].tick == t
            fA = h[idx].fraction_A
            dF = idx > 1 ? fA - h[idx-1].fraction_A : 0.0
            C  = h[idx].mean_confidence
            cr = h[idx].coordination_rate
            nc = h[idx].num_crystallised
            ns = h[idx].mean_norm_strength
            dn = h[idx].frac_dominant_norm

            marker = ""
            if h[idx].tick == takeoff
                marker = " ← TAKEOFF"
            elseif h[idx].tick == max_delta_tick
                marker = " ← MAX Δ"
            end

            println("  $(lpad(h[idx].tick, 4)) | $(lpad(round(fA, digits=2), 5)) | " *
                    "$(lpad(round(dF, digits=3), 7)) | " *
                    "$(lpad(round(C, digits=3), 6)) | " *
                    "$(lpad(round(cr, digits=2), 5)) | " *
                    "$(lpad(nc, 7)) | " *
                    "$(lpad(round(ns, digits=3), 6)) | $(lpad(round(dn, digits=2), 5))$marker")
        end
    end
end
