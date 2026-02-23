#!/usr/bin/env julia
#
# 3×2 Ablation: Experiential Memory Level × Normative Memory
# Sweep over population sizes N ∈ {20, 100, 500}
#
# Experiential Memory (3 levels):
#   None    — b_exp_A frozen at 0.5 every tick (no learning)
#   Fixed   — standard FIFO learning, fixed window w=5
#   Dynamic — standard FIFO learning, dynamic window w ∈ [2,6]
#
# Normative Memory (2 levels): OFF / ON
#
# 6 conditions total:
#   None_noNorm    None_Norm
#   Fixed_noNorm   Fixed_Norm
#   Dyn_noNorm     Dyn_Norm
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
const N_TRIALS  = 100
const BASE_SEED = 7000

const BEH_THRESH  = 0.95
const BEH_WINDOW  = 50

# ══════════════════════════════════════════════════════════════
# Condition definition
# ══════════════════════════════════════════════════════════════

struct Condition
    label::String
    exp_level::Symbol       # :none, :fixed, :dynamic
    enable_normative::Bool
    w_base::Int
    w_max::Int
end

const CONDITIONS = [
    # exp_level    norm    w_base  w_max
    Condition("None_noNorm",  :none,    false, 5, 5),
    Condition("None_Norm",    :none,    true,  5, 5),
    Condition("Fixed_noNorm", :fixed,   false, 5, 5),
    Condition("Fixed_Norm",   :fixed,   true,  5, 5),
    Condition("Dyn_noNorm",   :dynamic, false, 2, 6),
    Condition("Dyn_Norm",     :dynamic, true,  2, 6),
]

# ══════════════════════════════════════════════════════════════
# Custom run loop with ablation
# ══════════════════════════════════════════════════════════════

function run_ablation(N::Int, cond::Condition, seed::Int)
    params = SimulationParams(
        N = N, T = T_MAX, seed = seed,
        w_base = cond.w_base, w_max = cond.w_max,
        enable_normative = cond.enable_normative,
        V = 0, Phi = 0.0,
    )
    agents, ws, history, rng = initialize(params)
    probes = ProbeSet()
    init_probes!(probes, params, rng)
    tick_count = 0

    beh_counter = 0
    beh_converged = false
    beh_conv_tick = 0

    for t in 1:T_MAX
        run_tick!(t, agents, ws, history, tick_count, params, rng, probes)
        tick_count += 1

        # ── Ablation: freeze experiential learning ──
        if cond.exp_level == :none
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
        final_coord       = final.coordination_rate,
        final_majority    = max(final.fraction_A, 1.0 - final.fraction_A),
        final_cryst       = final.num_crystallised,
        final_norm_str    = final.mean_norm_strength,
        final_belief_err  = final.belief_error,
        mean_enforce      = mean_enforce,
        total_ticks       = result.tick_count,
    )
end

# ══════════════════════════════════════════════════════════════
# Run sweep
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
                        exp_level      = string(cond.exp_level),
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

    outpath = joinpath(@__DIR__, "..", "results", "ablation_3x2_summary.csv")
    CSV.write(outpath, summary_rows)
    println("\nSaved → $outpath")

    return summary_rows
end

# ══════════════════════════════════════════════════════════════
# Console summary table
# ══════════════════════════════════════════════════════════════

function print_summary(rows)
    exp_levels = ["none", "fixed", "dynamic"]
    exp_labels = ["None", "Fixed(w=5)", "Dynamic[2,6]"]

    for N in N_VALUES
        nrows = [r for r in rows if r.N == N]
        println("\n" * "=" ^ 90)
        println("N = $N   ($N_TRIALS trials per condition)")
        println("=" ^ 90)

        # Collect data for grid
        grid = Dict{Tuple{String,Bool}, Any}()

        for exp_lv in exp_levels
            for norm_on in [false, true]
                sub = [r for r in nrows if r.exp_level == exp_lv && r.norm_on == norm_on]
                ns = length(sub)
                n_conv = count(r -> r.converged, sub)

                conv_ticks = [r.conv_tick for r in sub if r.converged]
                mean_spd = length(conv_ticks) > 0 ? round(mean(conv_ticks), digits=1) : NaN
                med_spd  = length(conv_ticks) > 0 ? round(median(conv_ticks), digits=1) : NaN

                mean_coord = round(mean([r.final_coord for r in sub]), digits=4)
                mean_final_cryst = round(mean([r.final_cryst for r in sub]), digits=1)

                grid[(exp_lv, norm_on)] = (
                    conv_rate = "$n_conv/$ns",
                    mean_speed = isnan(mean_spd) ? "—" : string(mean_spd),
                    med_speed = isnan(med_spd) ? "—" : string(med_spd),
                    mean_coord = string(mean_coord),
                    mean_final_cryst = string(mean_final_cryst),
                )
            end
        end

        # Print 3×2 grid
        println("\n  ── 3×2 Grid: Experiential Level × Normative ──\n")
        println("  ", rpad("", 18), rpad("Norm OFF", 28), rpad("Norm ON", 28))
        println("  ", "-" ^ 74)

        for (i, exp_lv) in enumerate(exp_levels)
            d_off = grid[(exp_lv, false)]
            d_on  = grid[(exp_lv, true)]
            println("  ", rpad(exp_labels[i], 18),
                    rpad("$(d_off.conv_rate) / $(d_off.mean_speed)", 28),
                    rpad("$(d_on.conv_rate) / $(d_on.mean_speed)", 28))
        end

        # Detailed table
        println("\n  ── Detailed metrics ──\n")
        all_labels = ["None_noNorm", "None_Norm", "Fixed_noNorm", "Fixed_Norm", "Dyn_noNorm", "Dyn_Norm"]
        short_labels = ["None-", "None+", "Fix-", "Fix+", "Dyn-", "Dyn+"]

        println("  ", rpad("Metric", 20), [rpad(s, 12) for s in short_labels]...)
        println("  ", "-" ^ 92)

        data_by_label = Dict{String, Any}()
        for label in all_labels
            sub = [r for r in nrows if r.condition == label]
            ns = length(sub)
            n_conv = count(r -> r.converged, sub)
            conv_ticks = [r.conv_tick for r in sub if r.converged]
            mean_spd = length(conv_ticks) > 0 ? round(mean(conv_ticks), digits=1) : NaN
            mean_coord = round(mean([r.final_coord for r in sub]), digits=3)
            mean_cryst = round(mean([r.final_cryst for r in sub]), digits=1)

            data_by_label[label] = (
                conv_rate = "$n_conv/$ns",
                mean_speed = isnan(mean_spd) ? "—" : string(mean_spd),
                mean_coord = string(mean_coord),
                mean_cryst = string(mean_cryst),
            )
        end

        function prow(metric, field)
            vals = [getfield(data_by_label[l], field) for l in all_labels]
            println("  ", rpad(metric, 20), [rpad(v, 12) for v in vals]...)
        end

        prow("Conv rate", :conv_rate)
        prow("Mean conv tick", :mean_speed)
        prow("Final coord", :mean_coord)
        prow("Final cryst", :mean_cryst)

        # Speedup from Norm
        println("\n  ── Norm ON speedup (mean tick: OFF → ON) ──")
        for (i, exp_lv) in enumerate(exp_levels)
            d_off = grid[(exp_lv, false)]
            d_on  = grid[(exp_lv, true)]
            off_str = d_off.mean_speed
            on_str  = d_on.mean_speed
            if off_str != "—" && on_str != "—"
                ratio = round(parse(Float64, off_str) / parse(Float64, on_str), digits=1)
                println("  ", rpad(exp_labels[i], 18), "$off_str → $on_str  ($(ratio)× speedup)")
            else
                println("  ", rpad(exp_labels[i], 18), "$off_str → $on_str")
            end
        end
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
