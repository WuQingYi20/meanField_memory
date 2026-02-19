#!/usr/bin/env julia
# Analyze factorial_2x2.csv: per-layer milestones and convergence

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CSV, Statistics

rows = CSV.File(joinpath(@__DIR__, "..", "results", "factorial_2x2.csv"))
conditions = ["A_baseline", "B_lockin_only", "C_normative_only", "D_full_model"]

function summarize_col(sub, col)
    vals = [getproperty(r, col) for r in sub]
    reached = [x for x in vals if x > 0]
    nr = length(reached)
    n = length(sub)
    if nr > 0
        mn = round(mean(reached); digits=1)
        md = round(median(Float64.(reached)); digits=1)
        return "$(nr)/$(n) | mean=$(lpad(mn,6)) | med=$(lpad(md,6)) | [$(minimum(reached)), $(maximum(reached))]"
    else
        return "0/$(n)"
    end
end

println("="^90)
println("FACTORIAL 2×2 RESULTS — Layer Milestone Analysis")
println("="^90)

for cond in conditions
    sub = [r for r in rows if r.condition == cond]
    n = length(sub)
    n_converged = count(r -> r.converged, sub)

    println("\n--- $cond ($n trials, $n_converged converged) ---")
    println("  behavioral:  ", summarize_col(sub, :first_tick_behavioral))
    println("  belief:      ", summarize_col(sub, :first_tick_belief))
    println("  crystal:     ", summarize_col(sub, :first_tick_crystal))
    println("  all_met:     ", summarize_col(sub, :first_tick_all_met))
    println("  conv_tick:   ", summarize_col(sub, :convergence_tick))

    # Gap analysis: belief - behavioral (how long after behavioral does belief catch up?)
    gaps = Int[]
    for r in sub
        if r.first_tick_behavioral > 0 && r.first_tick_belief > 0
            push!(gaps, r.first_tick_belief - r.first_tick_behavioral)
        end
    end
    if length(gaps) > 0
        println("  gap(belief - behavioral): mean=$(round(mean(gaps), digits=1)), med=$(median(gaps)), range=[$(minimum(gaps)), $(maximum(gaps))]")
    end

    # Gap: crystal - behavioral
    gaps_c = Int[]
    for r in sub
        if r.first_tick_behavioral > 0 && r.first_tick_crystal > 0
            push!(gaps_c, r.first_tick_crystal - r.first_tick_behavioral)
        end
    end
    if length(gaps_c) > 0
        println("  gap(crystal - behavioral): mean=$(round(mean(gaps_c), digits=1)), med=$(median(gaps_c)), range=[$(minimum(gaps_c)), $(maximum(gaps_c))]")
    end

    # Gap: all_met - behavioral
    gaps_a = Int[]
    for r in sub
        if r.first_tick_behavioral > 0 && r.first_tick_all_met > 0
            push!(gaps_a, r.first_tick_all_met - r.first_tick_behavioral)
        end
    end
    if length(gaps_a) > 0
        println("  gap(all_met - behavioral): mean=$(round(mean(gaps_a), digits=1)), med=$(median(gaps_a)), range=[$(minimum(gaps_a)), $(maximum(gaps_a))]")
    end

    # Gap: convergence_tick - all_met (should be exactly convergence_window=50)
    gaps_conv = Int[]
    for r in sub
        if r.convergence_tick > 0 && r.first_tick_all_met > 0
            push!(gaps_conv, r.convergence_tick - r.first_tick_all_met)
        end
    end
    if length(gaps_conv) > 0
        println("  gap(conv_tick - all_met):  mean=$(round(mean(gaps_conv), digits=1)), med=$(median(gaps_conv)), range=[$(minimum(gaps_conv)), $(maximum(gaps_conv))]")
    end
end

# ── Cross-condition comparison ──
println("\n" * "="^90)
println("CROSS-CONDITION: Behavioral layer speed")
println("="^90)

for (c1, c2, label) in [
    ("A_baseline", "B_lockin_only", "A vs B (lock-in effect)"),
    ("C_normative_only", "D_full_model", "C vs D (lock-in + normative)"),
    ("A_baseline", "C_normative_only", "A vs C (normative effect, fixed window)"),
    ("B_lockin_only", "D_full_model", "B vs D (normative effect, dynamic window)"),
]
    s1 = [r.first_tick_behavioral for r in rows if r.condition == c1 && r.first_tick_behavioral > 0]
    s2 = [r.first_tick_behavioral for r in rows if r.condition == c2 && r.first_tick_behavioral > 0]
    if length(s1) > 0 && length(s2) > 0
        println("  $label:")
        println("    $(c1): mean=$(round(mean(s1), digits=1)), n=$(length(s1))")
        println("    $(c2): mean=$(round(mean(s2), digits=1)), n=$(length(s2))")
        println("    ratio: $(round(mean(s1)/mean(s2), digits=2))x")
    end
end
