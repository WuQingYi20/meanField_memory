#!/usr/bin/env julia
#
# Visualize the normative cascade mechanism
# Generates multi-panel figures from cascade diagnostic data
#

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using CSV, DataFrames, Plots, Statistics

const RESULTS = joinpath(@__DIR__, "..", "results")
const FIGURES = joinpath(@__DIR__, "..", "figures")
mkpath(FIGURES)

# ══════════════════════════════════════════════════════════════
# Load data
# ══════════════════════════════════════════════════════════════

diag = CSV.read(joinpath(RESULTS, "cascade_diagnostic.csv"), DataFrame)
comp = CSV.read(joinpath(RESULTS, "cascade_comparison.csv"), DataFrame)
trans = CSV.read(joinpath(RESULTS, "cascade_transitions.csv"), DataFrame)

both = filter(r -> r.condition == "Both", comp)
exponly = filter(r -> r.condition == "Exp_only", comp)

# ══════════════════════════════════════════════════════════════
# Figure 1: Both vs Exp_only — fraction_A over time
# ══════════════════════════════════════════════════════════════

function fig1_comparison()
    p = plot(both.tick, both.fraction_A,
        label="Dual Memory (Both)", lw=2.5, color=:royalblue,
        xlabel="Tick", ylabel="Fraction playing A",
        title="Convergence: Dual Memory vs Experiential Only (seed=42)",
        legend=:bottomright, size=(800, 450),
        ylims=(0.35, 1.05), xlims=(0, 300),
        grid=true, gridalpha=0.3,
        fontfamily="sans-serif", titlefontsize=12)
    plot!(p, exponly.tick, exponly.fraction_A,
        label="Experiential Only", lw=2.5, color=:coral, ls=:dash)

    # Mark cascade region
    vspan!(p, [50, 75], color=:royalblue, alpha=0.08, label="Cascade phase")

    # Annotations
    annotate!(p, 62, 0.45, text("Cascade\n(15 ticks)", 9, :royalblue))
    annotate!(p, 220, 0.85, text("Slow diffusion\n(~300 ticks)", 9, :coral))

    hline!(p, [0.95], color=:gray, ls=:dot, lw=1, label="95% threshold")

    savefig(p, joinpath(FIGURES, "fig1_both_vs_exponly.png"))
    println("  Saved fig1_both_vs_exponly.png")
    return p
end

# ══════════════════════════════════════════════════════════════
# Figure 2: Cascade anatomy — crystallization group dynamics
# ══════════════════════════════════════════════════════════════

function fig2_cascade_anatomy()
    d = diag[diag.tick .<= 120, :]

    p = plot(d.tick, d.n_cryst_A,
        label="Crystallized-A", lw=2.5, color=:royalblue, fill=(0, 0.15),
        xlabel="Tick", ylabel="Number of agents",
        title="Cascade Anatomy: Agent Group Dynamics (N=100, seed=42)",
        legend=:right, size=(800, 450),
        ylims=(0, 105), grid=true, gridalpha=0.3,
        fontfamily="sans-serif", titlefontsize=12)
    plot!(p, d.tick, d.n_cryst_B,
        label="Crystallized-B", lw=2.5, color=:crimson, fill=(0, 0.15))
    plot!(p, d.tick, d.n_uncr,
        label="Uncrystallized", lw=2.0, color=:gray, ls=:dash)

    # Phase markers
    vline!(p, [7], color=:black, ls=:dot, lw=0.8, label="")
    vline!(p, [30], color=:black, ls=:dot, lw=0.8, label="")
    vline!(p, [55], color=:black, ls=:dot, lw=0.8, label="")
    vline!(p, [72], color=:black, ls=:dot, lw=0.8, label="")

    annotate!(p, 3, 98, text("①", 10, :black))
    annotate!(p, 18, 98, text("② Symmetric\nCrystallization", 8, :black))
    annotate!(p, 42, 98, text("③ Tipping", 8, :black))
    annotate!(p, 63, 98, text("④ Cascade", 8, :darkorange))
    annotate!(p, 95, 98, text("⑤ Lock-in", 8, :black))

    savefig(p, joinpath(FIGURES, "fig2_cascade_anatomy.png"))
    println("  Saved fig2_cascade_anatomy.png")
    return p
end

# ══════════════════════════════════════════════════════════════
# Figure 3: Feedback loop — sigma, anomaly, b_exp divergence
# ══════════════════════════════════════════════════════════════

function fig3_feedback_loop()
    d = diag[diag.tick .<= 100, :]

    # Panel a: sigma divergence
    pa = plot(d.tick, d.sigma_crA,
        label="σ (A-norm)", lw=2, color=:royalblue,
        ylabel="Norm strength σ", xlabel="",
        title="(a) Norm Strength: Asymmetric Decay",
        legend=:topright, ylims=(0, 1.0),
        grid=true, gridalpha=0.3, titlefontsize=10)
    plot!(pa, d.tick, d.sigma_crB,
        label="σ (B-norm)", lw=2, color=:crimson)
    hline!(pa, [0.1], color=:gray, ls=:dot, lw=1, label="σ_min (dissolution)")

    # Panel b: anomaly divergence
    pb = plot(d.tick, d.anom_crA,
        label="Anomaly (A-norm)", lw=2, color=:royalblue,
        ylabel="Mean anomaly count", xlabel="",
        title="(b) Anomaly Accumulation: Minority Suffers More",
        legend=:topleft, ylims=(0, 12),
        grid=true, gridalpha=0.3, titlefontsize=10)
    plot!(pb, d.tick, d.anom_crB,
        label="Anomaly (B-norm)", lw=2, color=:crimson)
    hline!(pb, [10], color=:gray, ls=:dot, lw=1, label="θ_crisis = 10")

    # Panel c: b_exp and fraction_A
    pc = plot(d.tick, d.fraction_A,
        label="fraction_A (actual)", lw=2, color=:black, ls=:solid,
        ylabel="Probability / fraction", xlabel="",
        title="(c) Experiential Belief Tracks the Field",
        legend=:bottomright, ylims=(0.3, 1.05),
        grid=true, gridalpha=0.3, titlefontsize=10)
    plot!(pc, d.tick, d.bexp_all,
        label="b_exp_A (mean)", lw=2, color=:forestgreen, ls=:dash)
    plot!(pc, d.tick, d.beff_all,
        label="b_eff_A (mean)", lw=2, color=:darkorange, ls=:dashdot)

    # Panel d: dissolutions per tick
    pd = bar(d.tick, d.dissolutions,
        label="Dissolutions / tick", color=:crimson, alpha=0.7, lw=0,
        ylabel="Count", xlabel="Tick",
        title="(d) Dissolution Events: The Avalanche",
        legend=:topright, ylims=(0, 10),
        grid=true, gridalpha=0.3, titlefontsize=10)
    bar!(pd, d.tick, d.new_cryst_A,
        label="New crystallize-to-A / tick", color=:royalblue, alpha=0.5, lw=0)

    p = plot(pa, pb, pc, pd, layout=(2, 2), size=(1000, 700),
        fontfamily="sans-serif", margin=5Plots.mm)
    savefig(p, joinpath(FIGURES, "fig3_feedback_loop.png"))
    println("  Saved fig3_feedback_loop.png")
    return p
end

# ══════════════════════════════════════════════════════════════
# Figure 4: Per-group effective belief — the amplifier
# ══════════════════════════════════════════════════════════════

function fig4_amplifier()
    d = diag[diag.tick .<= 100, :]

    p = plot(d.tick, d.beff_crA,
        label="b_eff (A-norm agents)", lw=2.5, color=:royalblue,
        xlabel="Tick", ylabel="Effective belief b_eff_A",
        title="Signal Amplification: Norm → Action Probability",
        legend=:right, size=(800, 450),
        ylims=(-0.05, 1.1), grid=true, gridalpha=0.3,
        fontfamily="sans-serif", titlefontsize=12)
    plot!(p, d.tick, d.beff_crB,
        label="b_eff (B-norm agents)", lw=2.5, color=:crimson)
    plot!(p, d.tick, d.beff_uncr,
        label="b_eff (uncrystallized)", lw=2, color=:gray, ls=:dash)
    plot!(p, d.tick, d.bexp_all,
        label="b_exp_A (field estimate)", lw=1.5, color=:forestgreen, ls=:dot)

    # Show the amplification gap
    annotate!(p, 20, 0.93, text("A-norm agents\nact A ~85%", 8, :royalblue))
    annotate!(p, 20, 0.07, text("B-norm agents\nact A ~12%", 8, :crimson))
    annotate!(p, 50, 0.58, text("Field ≈ 0.55\nbut output ≈ 0.85", 8, :forestgreen))

    savefig(p, joinpath(FIGURES, "fig4_amplifier.png"))
    println("  Saved fig4_amplifier.png")
    return p
end

# ══════════════════════════════════════════════════════════════
# Figure 5: Individual agent lifecycle — B → dissolution → A
# ══════════════════════════════════════════════════════════════

function fig5_agent_lifecycle()
    # Find all agents that went crB -> uncr -> crA
    agents_B_to_A = Int[]
    crB_ticks = Dict{Int,Int}()
    diss_ticks = Dict{Int,Int}()
    recryst_ticks = Dict{Int,Int}()

    grouped = groupby(trans, :agent)
    for g in grouped
        aid = g.agent[1]
        trs = sort(g, :tick)
        for i in 1:nrow(trs)
            if trs.to[i] == "crB"
                crB_tick = trs.tick[i]
                for j in (i+1):nrow(trs)
                    if trs.from[j] == "crB" && trs.to[j] == "uncr"
                        dt = trs.tick[j]
                        for k in (j+1):nrow(trs)
                            if trs.from[k] == "uncr" && trs.to[k] == "crA"
                                push!(agents_B_to_A, aid)
                                crB_ticks[aid] = crB_tick
                                diss_ticks[aid] = dt
                                recryst_ticks[aid] = trs.tick[k]
                                break
                            end
                        end
                        break
                    end
                end
                break  # only first B-crystallization per agent
            end
        end
    end

    # Sort by dissolution tick
    sort!(agents_B_to_A, by=a -> diss_ticks[a])

    # Plot: horizontal timeline per agent
    n = length(agents_B_to_A)
    p = plot(xlabel="Tick", ylabel="Agent (sorted by dissolution time)",
        title="Agent Lifecycle: B-crystallized → Dissolution → Re-crystallize to A (N=$n agents)",
        size=(900, 600), legend=:bottomright,
        xlims=(0, 100), ylims=(0, n+1),
        grid=true, gridalpha=0.3,
        fontfamily="sans-serif", titlefontsize=11,
        yticks=nothing)

    for (idx, aid) in enumerate(agents_B_to_A)
        t_crB = crB_ticks[aid]
        t_diss = diss_ticks[aid]
        t_recryst = recryst_ticks[aid]

        # B-crystallized phase (red)
        plot!(p, [t_crB, t_diss], [idx, idx], lw=3, color=:crimson,
            label=(idx == 1 ? "B-crystallized" : ""))
        # Uncrystallized phase (gray)
        plot!(p, [t_diss, t_recryst], [idx, idx], lw=3, color=:gray,
            label=(idx == 1 ? "Uncrystallized" : ""))
        # A-crystallized phase (blue)
        plot!(p, [t_recryst, 100], [idx, idx], lw=3, color=:royalblue,
            label=(idx == 1 ? "A-crystallized" : ""))

        # Dissolution marker
        scatter!(p, [t_diss], [idx], color=:black, ms=3, label="",
            markershape=:circle)
    end

    savefig(p, joinpath(FIGURES, "fig5_agent_lifecycle.png"))
    println("  Saved fig5_agent_lifecycle.png")
    return p
end

# ══════════════════════════════════════════════════════════════
# Figure 6: Causal chain — the full cascade mechanism
# ══════════════════════════════════════════════════════════════

function fig6_causal_chain()
    d = diag[diag.tick .<= 100, :]

    # Compute the "minority fraction" = crB / (crA + crB)
    minority_frac = [
        (d.n_cryst_A[i] + d.n_cryst_B[i]) > 0 ?
            d.n_cryst_B[i] / (d.n_cryst_A[i] + d.n_cryst_B[i]) : 0.5
        for i in 1:nrow(d)
    ]

    # Panel a: field (fraction_A) drives crystallization asymmetry
    pa = plot(d.tick, d.fraction_A,
        label="fraction_A (field)", lw=2, color=:forestgreen,
        ylabel="Fraction", xlabel="",
        title="(a) Field breaks symmetry",
        legend=:right, ylims=(0.4, 1.05),
        grid=true, gridalpha=0.3, titlefontsize=10)
    plot!(pa, d.tick, 1.0 .- minority_frac,
        label="A-norm share of crystallized", lw=2, color=:royalblue, ls=:dash)

    # Panel b: sigma divergence caused by field asymmetry
    pb = plot(d.tick, d.sigma_crA .- coalesce.(d.sigma_crB, NaN),
        label="σ_A - σ_B", lw=2, color=:darkorange,
        ylabel="Δσ", xlabel="",
        title="(b) Field asymmetry → norm strength divergence",
        legend=:topleft,
        grid=true, gridalpha=0.3, titlefontsize=10)
    hline!(pb, [0], color=:gray, ls=:dot, lw=1, label="")

    # Panel c: dissolution feeds re-crystallization
    cumul_diss = cumsum(d.dissolutions)
    cumul_crA = cumsum(d.new_cryst_A)
    pc = plot(d.tick, cumul_diss,
        label="Cumulative dissolutions", lw=2, color=:crimson,
        ylabel="Cumulative count", xlabel="",
        title="(c) Dissolution → re-crystallize to A",
        legend=:topleft,
        grid=true, gridalpha=0.3, titlefontsize=10)
    plot!(pc, d.tick, cumul_crA,
        label="Cumulative new A-crystallizations", lw=2, color=:royalblue, ls=:dash)

    # Panel d: the acceleration — crB collapse rate
    # Compute Δ(crB) per 5 ticks
    pd = plot(d.tick, d.n_cryst_B,
        label="crB count", lw=2.5, color=:crimson, fill=(0, 0.2),
        ylabel="Agents", xlabel="Tick",
        title="(d) B-norm collapse: exponential acceleration",
        legend=:topright,
        grid=true, gridalpha=0.3, titlefontsize=10)
    plot!(pd, d.tick, d.n_cryst_A,
        label="crA count", lw=2.5, color=:royalblue)

    p = plot(pa, pb, pc, pd, layout=(2, 2), size=(1000, 700),
        fontfamily="sans-serif", margin=5Plots.mm)
    savefig(p, joinpath(FIGURES, "fig6_causal_chain.png"))
    println("  Saved fig6_causal_chain.png")
    return p
end

# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

function main()
    println("Generating cascade mechanism figures...")
    println("  Output directory: $FIGURES\n")

    fig1_comparison()
    fig2_cascade_anatomy()
    fig3_feedback_loop()
    fig4_amplifier()
    fig5_agent_lifecycle()
    fig6_causal_chain()

    println("\nAll 6 figures saved to figures/")
end

main()
