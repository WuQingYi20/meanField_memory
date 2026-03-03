#!/usr/bin/env julia
#
# Comprehensive Cascade Anatomy Figure
# Generates a 4-panel figure showing all cascade phases, early warning signals,
# and enforcement-capable agents (σ > θ_enforce).
#
# N = 100, seed = 42, Dynamic + Normative condition
#

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using DualMemoryABM
using CSV, DataFrames, Plots, Statistics

const N_AGENTS = 100
const T_MAX    = 200
const SEED     = 42
const FIGURES  = joinpath(@__DIR__, "..", "figures")
mkpath(FIGURES)

# ══════════════════════════════════════════════════════════════
# Run simulation & collect extended per-tick diagnostics
# ══════════════════════════════════════════════════════════════

function run_and_collect()
    params = SimulationParams(
        N = N_AGENTS, T = T_MAX, seed = SEED,
        w_base = 2, w_max = 6,
        enable_normative = true,
        V = 0, Phi = 0.0,
    )
    agents, ws, history, rng = initialize(params)
    probes = ProbeSet()
    init_probes!(probes, params, rng)
    tick_count = 0

    prev_norms = [agents[i].r for i in 1:N_AGENTS]
    rows = []

    for t in 1:T_MAX
        run_tick!(t, agents, ws, history, tick_count, params, rng, probes)
        tick_count += 1

        # ── Classify agents ──
        ids_uncr = Int[]
        ids_crA  = Int[]
        ids_crB  = Int[]
        for i in 1:N_AGENTS
            if agents[i].r == DualMemoryABM.NO_NORM
                push!(ids_uncr, i)
            elseif agents[i].r == DualMemoryABM.STRATEGY_A
                push!(ids_crA, i)
            else
                push!(ids_crB, i)
            end
        end

        n_uncr = length(ids_uncr)
        n_crA  = length(ids_crA)
        n_crB  = length(ids_crB)

        # ── Transitions ──
        new_cryst_A = 0; new_cryst_B = 0; dissolutions = 0
        for i in 1:N_AGENTS
            prev = prev_norms[i]
            curr = agents[i].r
            if prev == DualMemoryABM.NO_NORM && curr == DualMemoryABM.STRATEGY_A
                new_cryst_A += 1
            elseif prev == DualMemoryABM.NO_NORM && curr == DualMemoryABM.STRATEGY_B
                new_cryst_B += 1
            elseif prev != DualMemoryABM.NO_NORM && curr == DualMemoryABM.NO_NORM
                dissolutions += 1
            end
            prev_norms[i] = curr
        end

        # ── Per-group beliefs ──
        bexp_all = mean(agents[i].b_exp_A for i in 1:N_AGENTS)
        beff_all = mean(agents[i].b_eff_A for i in 1:N_AGENTS)
        bexp_crA = n_crA > 0 ? mean(agents[i].b_exp_A for i in ids_crA) : NaN
        beff_crA = n_crA > 0 ? mean(agents[i].b_eff_A for i in ids_crA) : NaN
        beff_crB = n_crB > 0 ? mean(agents[i].b_eff_A for i in ids_crB) : NaN
        beff_uncr = n_uncr > 0 ? mean(agents[i].b_eff_A for i in ids_uncr) : NaN

        # ── Belief variance (early warning) ──
        bexp_var = var([agents[i].b_exp_A for i in 1:N_AGENTS]; corrected=false)
        beff_var = var([agents[i].b_eff_A for i in 1:N_AGENTS]; corrected=false)

        # ── Norm strength per group ──
        sigma_crA = n_crA > 0 ? mean(agents[i].sigma for i in ids_crA) : NaN
        sigma_crB = n_crB > 0 ? mean(agents[i].sigma for i in ids_crB) : NaN

        # ── Enforceable agents (σ > 0.7) ──
        n_enforce_A = count(i -> agents[i].sigma > 0.7, ids_crA)
        n_enforce_B = count(i -> agents[i].sigma > 0.7, ids_crB)

        # ── TickMetrics ──
        m = history[tick_count]

        push!(rows, (
            tick = t,
            fraction_A = m.fraction_A,
            coordination = m.coordination_rate,
            n_uncr = n_uncr, n_crA = n_crA, n_crB = n_crB,
            new_cryst_A = new_cryst_A, new_cryst_B = new_cryst_B,
            dissolutions = dissolutions,
            bexp_all = bexp_all, beff_all = beff_all,
            bexp_crA = bexp_crA, beff_crA = beff_crA,
            beff_crB = beff_crB, beff_uncr = beff_uncr,
            sigma_crA = sigma_crA, sigma_crB = sigma_crB,
            n_enforce_A = n_enforce_A, n_enforce_B = n_enforce_B,
            bexp_var = bexp_var, beff_var = beff_var,
            belief_variance = m.belief_variance,
            num_enforcements = m.num_enforcements,
        ))
    end

    return DataFrame(rows)
end

# ══════════════════════════════════════════════════════════════
# Compute rolling variance (early warning signal)
# ══════════════════════════════════════════════════════════════

function rolling(x::Vector{Float64}, window::Int)
    n = length(x)
    result = fill(NaN, n)
    for i in window:n
        result[i] = mean(x[i-window+1:i])
    end
    return result
end

# ══════════════════════════════════════════════════════════════
# Generate the comprehensive 4-panel figure
# ══════════════════════════════════════════════════════════════

function plot_comprehensive(df::DataFrame)
    d = df[df.tick .<= 120, :]
    ticks = d.tick

    # ── Phase boundary ticks (from existing cascade analysis) ──
    t_phase2 = 6    # first crystallizations begin
    t_phase3 = 30   # asymmetric tipping
    t_phase4 = 37   # cascade onset
    t_phase5 = 55   # lock-in begins

    # Common phase annotation function
    function add_phases!(p; y_top=nothing)
        vline!(p, [t_phase2], color=:black, ls=:dot, lw=0.8, label="")
        vline!(p, [t_phase3], color=:black, ls=:dot, lw=0.8, label="")
        vline!(p, [t_phase4], color=:black, ls=:dot, lw=0.8, label="")
        vline!(p, [t_phase5], color=:black, ls=:dot, lw=0.8, label="")
    end

    # ════════════════════════════════════════════════
    # Panel (a): Agent group dynamics + phase labels
    # ════════════════════════════════════════════════
    pa = plot(ticks, d.n_crA,
        label="Crystallized-A", lw=2.5, color=:royalblue, fill=(0, 0.12),
        ylabel="Agents", xlabel="",
        title="(a) Population Dynamics",
        legend=:right, ylims=(0, 108),
        grid=true, gridalpha=0.3, titlefontsize=10)
    plot!(pa, ticks, d.n_crB,
        label="Crystallized-B", lw=2.5, color=:crimson, fill=(0, 0.12))
    plot!(pa, ticks, d.n_uncr,
        label="Uncrystallized", lw=2.0, color=:gray, ls=:dash)
    add_phases!(pa)
    # Phase labels at top
    annotate!(pa, 3,   104, text("①", 9, :black))
    annotate!(pa, 17,  104, text("② Drift", 8, :black))
    annotate!(pa, 33,  104, text("③", 8, :black))
    annotate!(pa, 46,  104, text("④ Cascade", 8, :darkorange))
    annotate!(pa, 85,  104, text("⑤ Lock-in", 8, :black))

    # ════════════════════════════════════════════════
    # Panel (b): Signal amplification (b_exp → b_eff)
    # ════════════════════════════════════════════════
    pb = plot(ticks, d.fraction_A,
        label="fraction_A (actual)", lw=2, color=:black,
        ylabel="Belief / Fraction", xlabel="",
        title="(b) Signal Amplification: Experiential → Effective Belief",
        legend=:right, ylims=(-0.05, 1.1),
        grid=true, gridalpha=0.3, titlefontsize=10)
    plot!(pb, ticks, d.bexp_all,
        label="b_exp (field signal)", lw=2, color=:forestgreen, ls=:dash)
    plot!(pb, ticks, d.beff_crA,
        label="b_eff (A-norm)", lw=2, color=:royalblue)
    plot!(pb, ticks, d.beff_crB,
        label="b_eff (B-norm)", lw=2, color=:crimson)
    plot!(pb, ticks, d.beff_uncr,
        label="b_eff (uncrystallized)", lw=1.5, color=:gray, ls=:dashdot)
    add_phases!(pb)
    # Amplification annotation
    annotate!(pb, 25, 0.92, text("amplified ↑", 8, :royalblue))
    annotate!(pb, 25, 0.10, text("suppressed ↓", 8, :crimson))

    # ════════════════════════════════════════════════
    # Panel (c): Early warning — belief variance spike
    # ════════════════════════════════════════════════
    # Rolling variance of b_exp across agents (already computed per-tick)
    bvar = Float64.(d.bexp_var)
    # Also compute rolling mean of fraction_A volatility (Δfraction_A)
    frac_A = Float64.(d.fraction_A)
    delta_frac = [0.0; abs.(diff(frac_A))]
    rolling_delta = rolling(delta_frac, 5)

    # Skip FIFO warm-up: focus on tick >= 5 for y-axis scaling
    bvar_post_warmup = bvar[ticks .>= 5]
    ymax_c = maximum(filter(!isnan, bvar_post_warmup)) * 1.4

    pc = plot(ticks, bvar,
        label="Belief variance (b_exp)", lw=2.5, color=:darkorange,
        ylabel="Variance / Volatility", xlabel="",
        title="(c) Early Warning Signals",
        legend=:topright, ylims=(0, ymax_c),
        grid=true, gridalpha=0.3, titlefontsize=10)
    plot!(pc, ticks, rolling_delta,
        label="Action volatility (5-tick MA)", lw=2, color=:purple, ls=:dash)
    add_phases!(pc)
    # Mark the pre-cascade variance rise (skip initial FIFO warm-up ticks)
    post_warmup_idx = findfirst(t -> t >= 10, ticks)
    peak_idx = post_warmup_idx + argmax(bvar[post_warmup_idx:end]) - 1
    peak_tick = ticks[peak_idx]
    annotate!(pc, min(peak_tick + 8, 110), bvar[peak_idx] * 0.95,
        text("← pre-cascade\n   variance rise", 7, :darkorange))

    # ════════════════════════════════════════════════
    # Panel (d): Enforcement capacity (high-σ agents)
    # ════════════════════════════════════════════════
    pd = plot(ticks, d.n_enforce_A,
        label="Enforceable A-norm (σ > 0.7)", lw=2.5, color=:royalblue,
        fill=(0, 0.15),
        ylabel="Number of agents", xlabel="Tick",
        title="(d) Enforcement Capacity: Agents with σ > θ_enforce = 0.7",
        legend=:right, ylims=(0, max(maximum(d.n_enforce_A) + 10, 15)),
        grid=true, gridalpha=0.3, titlefontsize=10)
    plot!(pd, ticks, d.n_enforce_B,
        label="Enforceable B-norm (σ > 0.7)", lw=2.5, color=:crimson,
        fill=(0, 0.15))
    # Total crystallized per group (thin lines for context)
    plot!(pd, ticks, d.n_crA,
        label="Total cryst-A", lw=1.2, color=:royalblue, ls=:dot, alpha=0.5)
    plot!(pd, ticks, d.n_crB,
        label="Total cryst-B", lw=1.2, color=:crimson, ls=:dot, alpha=0.5)
    add_phases!(pd)

    # ════════════════════════════════════════════════
    # Combine
    # ════════════════════════════════════════════════
    p = plot(pa, pb, pc, pd,
        layout=(4, 1), size=(900, 1100),
        fontfamily="sans-serif",
        left_margin=8Plots.mm,
        bottom_margin=3Plots.mm,
        top_margin=2Plots.mm)

    figpath = joinpath(FIGURES, "fig_cascade_comprehensive.png")
    savefig(p, figpath)
    println("Saved → $figpath")
    return p
end

# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

function main()
    println("Running simulation (N=$N_AGENTS, seed=$SEED)...")
    df = run_and_collect()
    println("Collected $(nrow(df)) ticks of diagnostic data.")

    println("Generating comprehensive cascade figure...")
    plot_comprehensive(df)

    # Also save extended diagnostic CSV for reference
    outpath = joinpath(@__DIR__, "..", "results", "cascade_diagnostic_extended.csv")
    CSV.write(outpath, df)
    println("Extended diagnostic saved → $outpath")

    println("\nDone.")
end

main()
