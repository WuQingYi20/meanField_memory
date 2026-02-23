#!/usr/bin/env julia
#
# 2×2 Ablation: Experiential (Dynamic) Memory × Normative Memory
# Sweep over population sizes N ∈ {20, 100, 500}
#
# 4 conditions:
#   Neither    — no experiential learning, no normative → pure random baseline
#   Exp only   — experiential FIFO + confidence, no normative
#   Norm only  — normative crystallization, b_exp frozen at 0.5
#   Both       — full dual-memory model
#
# All conditions: V=0, Phi=0.0, 50 trials per (N, condition)
# Convergence: behavioral-only (majority ≥ 0.95 sustained 50 ticks)
#

using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

using DualMemoryABM
using CSV
using Statistics
using Random

# ══════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════

const N_VALUES  = [20, 100, 500]
const T_MAX     = 3000
const N_TRIALS  = 50
const BASE_SEED = 6000

# Behavioral convergence: majority ≥ 0.95 for 50 consecutive ticks
const BEH_THRESH  = 0.95
const BEH_WINDOW  = 50

# ══════════════════════════════════════════════════════════════
# Condition definition
# ══════════════════════════════════════════════════════════════

struct AblationCondition
    label::String
    enable_experiential::Bool
    enable_normative::Bool
end

const CONDITIONS = [
    AblationCondition("Neither",  false, false),
    AblationCondition("Exp_only", true,  false),
    AblationCondition("Norm_only",false, true),
    AblationCondition("Both",     true,  true),
]

# ══════════════════════════════════════════════════════════════
# Custom run loop with ablation
# ══════════════════════════════════════════════════════════════

function run_ablation(N::Int, cond::AblationCondition, seed::Int)
    params = SimulationParams(
        N = N, T = T_MAX, seed = seed,
        w_base = 2, w_max = 6,
        enable_normative = cond.enable_normative,
        V = 0, Phi = 0.0,
    )
    agents, ws, history, rng = initialize(params)
    probes = ProbeSet()
    init_probes!(probes, params, rng)
    tick_count = 0

    # Track behavioral convergence (model-agnostic)
    beh_counter = 0
    beh_converged = false
    beh_conv_tick = 0

    for t in 1:T_MAX
        run_tick!(t, agents, ws, history, tick_count, params, rng, probes)
        tick_count += 1

        # ── Ablation: disable experiential learning ──
        if !cond.enable_experiential
            for i in 1:N
                agents[i].b_exp_A = 0.5
            end
        end

        # ── Behavioral convergence check ──
        m = history[tick_count]
        majority = max(m.fraction_A, 1.0 - m.fraction_A)
        if majority >= BEH_THRESH
            beh_counter += 1
            if !beh_converged && beh_counter >= BEH_WINDOW
                beh_converged = true
                beh_conv_tick = t - BEH_WINDOW + 1
            end
        else
            beh_counter = 0
        end

        # Early termination after convergence + buffer
        if beh_converged && t >= beh_conv_tick + 200
            break
        end
    end

    return (
        history    = history[1:tick_count],
        params     = params,
        converged  = beh_converged,
        conv_tick  = beh_conv_tick,
        tick_count = tick_count,
    )
end

# ══════════════════════════════════════════════════════════════
# Post-hoc metrics
# ══════════════════════════════════════════════════════════════

function compute_metrics(result, N::Int)
    hist = result.history
    n = length(hist)

    final = hist[end]
    final_coord = final.coordination_rate
    final_frac_A = final.fraction_A
    final_majority = max(final_frac_A, 1.0 - final_frac_A)
    final_cryst = final.num_crystallised
    final_norm_str = final.mean_norm_strength
    final_belief_err = final.belief_error

    first_beh_tick = 0
    for m in hist
        if max(m.fraction_A, 1.0 - m.fraction_A) >= BEH_THRESH
            first_beh_tick = m.tick
            break
        end
    end

    first_crystal_50 = 0
    for m in hist
        if m.num_crystallised >= N ÷ 2
            first_crystal_50 = m.tick
            break
        end
    end

    mean_enforce = 0.0
    if result.converged
        ct = result.conv_tick
        start_idx = 0
        for idx in 1:n
            if hist[idx].tick >= ct
                start_idx = idx
                break
            end
        end
        if start_idx > 0
            end_idx = min(start_idx + 99, n)
            mean_enforce = mean(hist[idx].num_enforcements for idx in start_idx:end_idx)
        end
    end

    return (
        converged         = result.converged,
        conv_tick         = result.conv_tick,
        first_beh_tick    = first_beh_tick,
        first_crystal_50  = first_crystal_50,
        final_coord       = final_coord,
        final_majority    = final_majority,
        final_cryst       = final_cryst,
        final_norm_str    = final_norm_str,
        final_belief_err  = final_belief_err,
        mean_enforce      = mean_enforce,
        total_ticks       = result.tick_count,
    )
end

# ══════════════════════════════════════════════════════════════
# Multi-trial summary across N values
# ══════════════════════════════════════════════════════════════

function run_sweep()
    summary_rows = []
    lk = ReentrantLock()

    for N in N_VALUES
        println("=" ^ 60)
        println("N = $N")
        println("=" ^ 60)

        for cond in CONDITIONS
            println("  $(cond.label) × $N_TRIALS trials...")

            Threads.@threads for trial in 1:N_TRIALS
                seed = hash((BASE_SEED, N, cond.label, trial)) % typemax(Int)
                result = run_ablation(N, cond, Int(seed))
                met = compute_metrics(result, N)

                lock(lk) do
                    push!(summary_rows, (
                        N              = N,
                        condition      = cond.label,
                        exp_on         = cond.enable_experiential,
                        norm_on        = cond.enable_normative,
                        trial          = trial,
                        converged      = met.converged,
                        conv_tick      = met.conv_tick,
                        first_beh_tick = met.first_beh_tick,
                        first_cryst_50 = met.first_crystal_50,
                        final_coord    = met.final_coord,
                        final_majority = met.final_majority,
                        final_cryst    = met.final_cryst,
                        final_norm_str = met.final_norm_str,
                        final_belief_err = met.final_belief_err,
                        mean_enforce   = met.mean_enforce,
                        total_ticks    = met.total_ticks,
                    ))
                end
            end
        end
    end

    outpath = joinpath(@__DIR__, "..", "results", "ablation_summary.csv")
    CSV.write(outpath, summary_rows)
    println("\nSaved → $outpath")

    return summary_rows
end

# ══════════════════════════════════════════════════════════════
# Console summary table
# ══════════════════════════════════════════════════════════════

function print_summary(rows)
    labels = ["Neither", "Exp_only", "Norm_only", "Both"]

    for N in N_VALUES
        nrows = [r for r in rows if r.N == N]
        println("\n" * "=" ^ 90)
        println("N = $N   ($(N_TRIALS) trials per condition)")
        println("=" ^ 90)

        data = Dict{String, Any}()
        for label in labels
            sub = [r for r in nrows if r.condition == label]
            ns = length(sub)
            n_conv = count(r -> r.converged, sub)

            conv_ticks = [r.conv_tick for r in sub if r.converged]
            mean_spd = length(conv_ticks) > 0 ? round(mean(conv_ticks), digits=1) : NaN
            med_spd  = length(conv_ticks) > 0 ? round(median(conv_ticks), digits=1) : NaN

            mean_coord = round(mean([r.final_coord for r in sub]), digits=4)
            mean_final_cryst = round(mean([r.final_cryst for r in sub]), digits=1)

            data[label] = (
                conv_rate = "$n_conv/$ns",
                mean_speed = isnan(mean_spd) ? "—" : string(mean_spd),
                med_speed = isnan(med_spd) ? "—" : string(med_spd),
                mean_coord = string(mean_coord),
                mean_final_cryst = string(mean_final_cryst),
            )
        end

        println("\n", rpad("Metric", 26),
                rpad("Neither", 14), rpad("Exp only", 14),
                rpad("Norm only", 14), rpad("Both", 14))
        println("-" ^ 82)

        function row(metric, field)
            vals = [getfield(data[l], field) for l in labels]
            println(rpad(metric, 26), [rpad(v, 14) for v in vals]...)
        end

        row("Conv rate", :conv_rate)
        row("Mean conv tick", :mean_speed)
        row("Median conv tick", :med_speed)
        row("Final coordination", :mean_coord)
        row("Final crystallised", :mean_final_cryst)

        # 2×2 grid
        println("\n  ── 2×2 Grid (conv rate / mean tick) ──\n")
        println("  ", rpad("", 16), rpad("Norm OFF", 22), rpad("Norm ON", 22))
        println("  ", "-" ^ 60)

        d_n = data["Neither"]; d_e = data["Exp_only"]
        d_no = data["Norm_only"]; d_b = data["Both"]

        println("  ", rpad("Exp OFF", 16),
                rpad("$(d_n.conv_rate) / $(d_n.mean_speed)", 22),
                rpad("$(d_no.conv_rate) / $(d_no.mean_speed)", 22))
        println("  ", rpad("Exp ON", 16),
                rpad("$(d_e.conv_rate) / $(d_e.mean_speed)", 22),
                rpad("$(d_b.conv_rate) / $(d_b.mean_speed)", 22))
    end
end

# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

function main()
    mkpath(joinpath(@__DIR__, "..", "results"))
    rows = run_sweep()
    print_summary(rows)
    println("\n\nDone.")
end

main()
